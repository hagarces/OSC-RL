#!/usr/bin/env python

import sys
import rospy 
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal as torchNormal
from copy import deepcopy
from scipy.spatial.transform import Rotation 
from franka_msgs.msg import FrankaState
from franka_control_test.srv import SetPointXYZ, SetPointXYZRequest, SetPointRL, SetPointRLResponse



#  ~~~~~~~~~~~~~~~~~~ Funciones auxiliares para la red neuronal ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Actor(nn.Module):
    # Clase actor que toma como input la observacion del environment
    # y retorna la accion a efectuar por el agente, se puede setear para
    # diferentes algoritmos, si se ocupan algoritmos estocasticos la red
    # retorna la media y desviacion esandar, si se ocupan algoritmos
    # deterministicos la red retorna el valor exacto de la accion a tomar.
    
        
    def __init__(self, algorithm_name, net_architecture, last_dim,  action_dim, 
                 max_range, min_std = 0.01, max_std = 1):

        super(Actor, self).__init__()
        self.net = net_architecture
        self.action_dim = action_dim
        self.max_range = max_range
        self.min_std = min_std
        self.max_std = max_std
        self.last_dim = last_dim
    
        # {algo_name: (stochastic_bool)}
        self.algo_name = algorithm_name
        self.algo_dict = {"VPG": (True,), "A2C": (True,), "DDPG": (False,),
                          "TD3": (False,), "SAC": (True,), "NPG": (True,) }
        
        if algorithm_name not in self.algo_dict.keys():
            print("Algorithm named: " + str(algorithm_name) + \
                  "not implemented yet, the available algorithms are: " + \
                  print(list(self.algo_dict.keys)))
            raise KeyError
        
        # differentiate between stochastic architecture
        if self.algo_dict[self.algo_name][0]:
            self.mu_layer = nn.Linear(last_dim, action_dim)
            self.std_layer = nn.Linear(last_dim, action_dim)
            self.std_activation = nn.Sigmoid()
        else:
            self.output_layer = nn.Linear(last_dim, action_dim)
            # --- Ver si es necesario agregar mas funciones de activacion! ---
            self.output_activation = nn.Tanh()
      
    # [V5.0] que el mismo actor retorne sus parametros
    def get_args(self):
        return [self.algo_name, deepcopy(self.net), self.last_dim,
                self.action_dim, self.max_range, self.min_std, self.max_std]   
     
    def forward(self, x):            
        # if the algorithm is stochastic
        if self.algo_dict[self.algo_name][0]:
            x = self.net(x)
            
            # mu
            mu_net = self.mu_layer(x)
            
            # std
            std_net = self.std_activation(self.std_layer(x)) * \
                self.max_std + self.min_std
            
            return mu_net, std_net
        
        else:
            x = self.net(x)
            x = self.output_layer(x)
            actions = self.output_activation(x) * self.max_range
            
            return actions, None
            
    def sample(self, x, epsilon, stability_const = 0.00001, squashed = True,
               min_logprob = -5, max_logprob = 2):
        # [V5.0]: min_logprob = -5: 0.6% de prob
        #         max_logprob = 2: 738% de prob (recordar que no es estrictamente una prob)
        # en vez de sumar las log-probabilidades se promediaran, de esta forma
        # se volvera inmune(?) a los cambios en las DOF
        
        # Solo usado para los algoritmos estocasticos
        
        # Se obtiene el mu, std, luego se limita inferiormente el std mediante
        # el parametro epsilon, esto para incentivar la exploracion.
        mu_net, std_net = self.forward(x)
        std_net = torch.clamp(std_net, min = epsilon) 
        
        # Se toma una accion aleatoria de la distribucion, se usa rsample para
        # poder hacer la etapa de backpropagation y pasar por la distribucion.
        action_normal = torchNormal(mu_net, std_net)
        action_sample = action_normal.rsample()
        
        if squashed:
            # se aplasta (squashed) la distribucion para que quede en el rango, 
            # al momento de obtener la prob se compensa el efecto de la
            # funcion Tanh. 
            actions = nn.Tanh()(action_sample) * self.max_range
            
            log_prob = action_normal.log_prob(action_sample) - \
                        torch.log(self.max_range**2 - actions.pow(2) + stability_const)
                        
            mu_net = nn.Tanh()(mu_net) * self.max_range
        else:
            actions = torch.clamp(action_sample, min = -self.max_range,
                                  max=self.max_range)
            log_prob = action_normal.log_prob(action_sample)
        
        # Se limitan los valores de las probabilidades (esto para darle mas
        # estabilidad a los algoritmos que utilizan log_prob para ajustar el 
        # paso de aprendizaje)
        log_prob = torch.clamp(log_prob, min = min_logprob, max = max_logprob)
        log_prob = log_prob.mean(1, keepdim=True)

        return mu_net, std_net, actions, log_prob
        
    def __str__(self):
        return self.algo_name + " Actor"
        
def load_net(filepath):
    # Para cargar una red guardada en el computador
    checkpoint = torch.load(filepath + ".pth", map_location='cpu')
    model = Actor(*checkpoint['model_args'])
    model.load_state_dict(deepcopy(checkpoint['state_dict']))
    return model


#  ~~~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class NeuralPlanner():
    
    
    def __init__(self, angle_dist_thresh, pure_OSC_thresh, neural_rate, xyz_pos_lim, xyz_neg_lim):
        rospy.init_node('NeuralPlanner')
        rospy.loginfo("Inicializando clase de planner")
        
        self.q = None  # (que se entrega a la red como sen(q), cos(q)), osea dim 14
        self.dq = None # dim 7
        self.target_xyz_delta = None # dim 3
        self.target_angle = None # dim 1
        self.angle_dist_thresh = angle_dist_thresh
        self.pure_OSC_thresh = pure_OSC_thresh
        self.neural_rate = neural_rate
        
        self.curr_xyz = None
        self.xyz_goal = None
        self.curr_dist = None
        
        self.xyz_pos_lim = xyz_pos_lim
        self.xyz_neg_lim = xyz_neg_lim
        
        self.RL_agent = load_net("/home/ral/catkin_ws/src/franka_control_test/scripts/NEURAL_BRAIN")
        
        rospy.Subscriber("/franka_state_controller/franka_states", FrankaState, self.franka_states_callback)
        rospy.Service('/RLMotionPlanner/setpoint', SetPointRL, self.RL_setpoint_server_handler)
        self.OSC_setpoint_handler = rospy.ServiceProxy('/robot_OSC_controller/setpoint', SetPointXYZ)
        
        
        
        
    def RL_setpoint_server_handler(self, req):
        
        delta_xyz = np.array(req.xyz_delta_ref)
        self.xyz_goal = self.curr_xyz.copy() + delta_xyz
        
        response = SetPointRLResponse()
        response.success = True
        response.message = "Setpoint for RL MOTION PLANNER correctly set"
        return response
        
        
    def franka_states_callback(self, data):
    
        self.q = np.array(data.q)
        self.dq = np.array(data.dq)
        self.curr_xyz = np.array([data.O_T_EE[12], data.O_T_EE[13], data.O_T_EE[14]])
        
        rot_matrix = np.array([[data.O_T_EE[0], data.O_T_EE[4], data.O_T_EE[8]],
                               [data.O_T_EE[1], data.O_T_EE[5], data.O_T_EE[9]],
                               [data.O_T_EE[2], data.O_T_EE[6], data.O_T_EE[10]]])
        r = Rotation.from_matrix(rot_matrix)
        
        #rospy.loginfo("xyz : " + str(np.round(self.curr_xyz, 2)))
        
        # tomando la posicion inicial del robot con sus ejes de referencia
        # angle[0] es la rotacion en el eje z (pero con signo cambiado y parte en 0) 
        # angle[1] es la rotacion en el eje y (si mira hacia abajo es 0 y tambien esta al reves
        # angle[2] es la rotacion en el eje x (parte en 180/-180 y es producido mayoritariamente por la articulacion 5)
        
        angles = r.as_euler("zyx", degrees=False)
        
        if np.any(self.xyz_goal):
        
            self.target_xyz_delta = self.xyz_goal - self.curr_xyz
            self.ee_orientation = np.array([np.cos(-angles[0]) * np.sin(angles[1]),
                                            np.sin(-angles[0]) * np.sin(angles[1]),
                                            -np.cos(angles[1])])
            
            self.curr_dist = np.linalg.norm(self.target_xyz_delta)
                                            
            
            
            if self.curr_dist <= angle_dist_thresh:
                self.target_angle = 0
            
            else:
                # calculo de angulo entre orientacion efector y target
                line1 = self.target_xyz_delta/self.curr_dist
                line2 = self.ee_orientation
        
                angle = abs(np.arccos(np.clip(np.dot(line1, line2), -1.0, 1.0)))
                self.target_angle = angle
             
                
                
    def send_motion_planner_signal(self):
        
        request = SetPointXYZRequest()
        delta = self.xyz_goal - self.curr_xyz
        
        if self.pure_OSC_thresh >= self.curr_dist:
        
            #rospy.loginfo("current xyz: " + str(np.round(self.curr_xyz, 2)))
            rospy.loginfo("curr  delta: " + str(np.round(delta, 2)))
            
            request.x_delta_ref = delta[0]
            request.y_delta_ref = delta[1]
            request.z_delta_ref = delta[2]
            
            request.tau_null_bool = False
        else:
            # PONER RED NEURONAL ACA
            OBS_STATE = np.concatenate([np.sin(self.q), np.cos(self.q), self.dq, delta, np.array([self.target_angle])])      
            state = torch.FloatTensor(np.array(OBS_STATE, dtype = float))
            state = torch.unsqueeze(state, 0)
            epsilon = 0
            mu_net, _, _, _ = self.RL_agent.sample(state, epsilon = 0, squashed = True)
            
            actions = mu_net.detach().numpy()[0]
            
            xyz_delta_RL_ref = actions[0:3]
            tau_null_RL = actions[3:10]
            
            
            # Por miedo a la red neuronal se limita un poco por mientras los delta cartesianos
            # [WARNING]: recordar que el maximo limite delta con el que se entreno la red es de 1.3 delta xyz!!!!
            # Segun las reales limitaciones del robot este deberia tener un maximo delta aprox de 0.5
            
            safe_factor = 0.35
            tau_null_safe_factor = 0.05#0.1
            
            xyz_delta_RL_ref *= safe_factor
            tau_null_RL *= tau_null_safe_factor
            
            
            desired_xyz = np.clip(self.curr_xyz + xyz_delta_RL_ref, self.xyz_neg_lim, self.xyz_pos_lim)
            xyz_delta_RL_ref = desired_xyz - self.curr_xyz
            
            #rospy.loginfo("current xyz: " + str(np.round(self.curr_xyz, 2)))
            rospy.loginfo("xyz delta RL: " + str(np.round(xyz_delta_RL_ref, 2)))
            rospy.loginfo("curr   delta: " + str(np.round(delta, 2)))
            
            request.x_delta_ref = xyz_delta_RL_ref[0]
            request.y_delta_ref = xyz_delta_RL_ref[1]
            request.z_delta_ref = xyz_delta_RL_ref[2]
            
            request.tau_null = list(tau_null_RL)
            request.tau_null_bool = True
            
            
            
            
            
        try:
            response = self.OSC_setpoint_handler(request)
        except rospy.ServiceException as error:
             rospy.logerr("Service call failed: %s" % error)
        

    
    def run(self):
        ros_rate = int(1.0/self.neural_rate) # ros trabaja con Hz
        rate = rospy.Rate(ros_rate) # rosrate a 2Hz
        
        #self.xyz_goal = np.array([0.7, 0.3, 0.8])
        
        while not rospy.is_shutdown():
            if np.any(self.xyz_goal) and np.any(self.curr_dist):
                self.send_motion_planner_signal()
            rate.sleep()
    

if __name__ == '__main__':

    # LIMITES REALES DEL FRANKA EMIKA RESEARCH 3
    
    # POSE INICIAL: 0.3, 0, 0.4
    # LIM POSITIVO: 0.85, 0.8, 1.1 
    # LIM NEGATIVO:-0.6, -0.7, 0.1
    
    # delta max inicial : 0.55, 0.8, 0.6
    # delta min inicial : -0.9, -0.7, -0.3
    
    
    # Buena prueba!: ir y volver de: (0.4, 0.3, 0.4)
    
    xyz_pos_lim = np.array([0.85, 0.8, 1.1])
    xyz_neg_lim = np.array([-0.6, -0.7, 0.1])
    
    # ESPACIO EN X: PARTE EN 0.31
    
    pure_OSC_thresh = 0.2
    angle_dist_thresh = 0.2
    neural_rate = 0.5 # red neuronal fue entrenada para enviar setpoints cada 0.5[s]
    
    
    Planner = NeuralPlanner(angle_dist_thresh = angle_dist_thresh,
                            pure_OSC_thresh = pure_OSC_thresh,
                            neural_rate = neural_rate,
                            xyz_pos_lim = xyz_pos_lim,
                            xyz_neg_lim = xyz_neg_lim)
    

    
    Planner.run()
    
    
    

