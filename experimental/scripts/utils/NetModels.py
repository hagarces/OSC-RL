import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal as torchNormal
from copy import deepcopy
from collections import deque
from random import sample


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

            


class DummyActor():
    # actor sin cerebro solo para usar de benchmark
    def __init__(self, env, action_dim, max_range):
        
        self.env = env
        self.action_dim = action_dim
        self.max_range = max_range
        
    # como reemplazo del forward, retorna el target del ambiente + 0s hasta 
    # completar la dimension definida
    def __call__(self, *args, **kwargs):
        
        target = self.env.get_body_com("target") - self.env.get_body_com("EE")
        target = torch.Tensor(np.pad(target, (0, self.action_dim - 3)))
        target = torch.unsqueeze(target, 0)
        
        return target, None 
    
    # como reemplazo del actor sample
    def sample(self, *args, **kwargs):
        
        one_tensor = torch.Tensor([1] *self.action_dim)
        one_tensor = torch.unsqueeze(one_tensor, 0)
        target = self.env.get_body_com("target") - self.env.get_body_com("EE")
        target = torch.Tensor(np.pad(target, (0, self.action_dim - 3)))
        target = torch.unsqueeze(target, 0)

        return target, one_tensor, target, one_tensor
    
    def __str__(self):
        return "DUMMY OSC ACTOR"

class ImmobileActor():
    # actor sin cerebro solo para usar de benchmark
    def __init__(self, env, action_dim, max_range):
        
        self.env = env
        self.action_dim = action_dim
        self.max_range = max_range
        
    # como reemplazo del forward, retorna el target del ambiente + 0s hasta 
    # completar la dimension definida
    def __call__(self, *args, **kwargs):
        
        target =  torch.zeros(self.action_dim, 1)
        return target, None 
    
    # como reemplazo del actor sample
    def sample(self, *args, **kwargs):
        
        one_tensor = torch.Tensor([1] *self.action_dim)
        one_tensor = torch.unsqueeze(one_tensor, 0)
        target =  torch.zeros(1, self.action_dim)

        return target, one_tensor, target, one_tensor
    
    def __str__(self):
        return "DUMMY IMMOBILE ACTOR"

class Critic(nn.Module):
    # red para estimar Q o V (depende de la dimension de la entrada)
    # Clase Crtico que toma como input la observacion + accion del environment
    # y retorna el valor estimado de dicha accion / estado, se puede setear para
    # estimar V o Q (depende de la dimension de la entrada).
    
    
    def __init__(self, algorithm_name, net_architecture, hidden_dim):
        super(Critic, self).__init__()
        self.net = net_architecture
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.algo_name = algorithm_name
        
        # {algo_name: (stochastic_bool)}
        self.algo_dict = {"VPG": (True,), "A2C": (True,), "DDPG": (False,),
                          "TD3": (False,), "SAC": (True,), "NPG": (True,) }
        
        if algorithm_name not in self.algo_dict.keys():
            print("Algorithm named: " + str(algorithm_name) + \
                  "not implemented yet, the available algorithms are: " + \
                  print(list(self.algo_dict.keys)))
            raise KeyError
    
    def forward(self, x):
        x = self.net(x)
        output = self.output_layer(x)
        
        return output
    
    def __str__(self):
        return self.algo_name + " Critic"

class Agent():
    
    
    def __init__(self, algorithm_name, max_mem_lenght,
                 max_score_mem, max_stats_mem, batch_size, max_steps,
                 state_dim, action_dim, dist_thresh, device, epsilon = 0, gamma = 0):
        
        # Setear las caracteristicas del algoritmo seleccionado
        # {algo_name: (stochastic_bool)}
        self.algo_name = algorithm_name
        self.algo_dict = {"VPG": (True,), "A2C": (True,), "DDPG": (False,),
                          "TD3": (False,), "SAC": (True,), "NPG": (True,) }
        
        if algorithm_name not in self.algo_dict.keys():
            print("Algorithm named: " + str(algorithm_name) + \
                  "not implemented yet, the available algorithms are: " + \
                  print(list(self.algo_dict.keys)))
            raise KeyError
        
        self.algo_params = self.algo_dict[self.algo_name]
        self.device = device 
        
        # Variables de la memoria del agente
        self.replay_memory = deque([], max_mem_lenght)
        self.max_mem_lenght = max_mem_lenght
        
        # Hiper parametros
        self.batch_size = batch_size     
        self.max_steps = max_steps - 1 # -1 porque la interaccion inicial no se guarda
        self.action_dim = action_dim 
        self.state_dim = state_dim
        self.epsilon =  epsilon
        self.gamma = gamma
        
        # Variables del estado del agente
        self.state = None
        self.current_trayectory = None # matriz que va guardando la trayectoria
        self.current_step = 0 # cont que lleva la cuenta del paso actual de la trayectoria
        self.current_score = 0 # variable que guarda el reward acumulado del episodio
        
        # Mediciones de desempehno
        self.max_score_mem = max_score_mem
        self.test_score = deque([], max_score_mem) # ultimos scores
        self.goal_reached_deque = deque([], max_score_mem) # % de que el efector llegue al objetivo
        self.collision_deque = deque([], max_score_mem) # ultimas colisiones
        self.manipu_deque = deque([], max_score_mem) # ultimos volumenes de elipsoides de manipulabilidad
        self.dist_deque = deque([], max_score_mem) # ultimas distancias acumuladas
        self.energy_deque = deque([], max_score_mem) # ultimas cantidades de energia usadas
        self.time_deque = deque([], max_score_mem) # ultimas tiempos de llegada
        
        #[V4.0] - Guardar las ultimas acciones del agente para saber si aumentar el rango
        self.last_actions = deque([], max_score_mem * self.max_steps)
        
        # si el algoritmo es estocastico se guardan el historial de std
        if self.algo_params[0]:
            self.mean_std_deque = deque([], max_stats_mem * self.max_steps)
        
        # Variables auxiliares 
        self.dist_thresh = dist_thresh # distancia maxima para considerar que el agente paso por la meta
        self.goal_reached = False # Si el agente llego a la meta

        
    def check_mem(self):
        # Chequear si la memoria se lleno
        len(self.replay_memory) >= self.max_mem_lenght
        
    def show_score(self):
        #Retorna el score promedio
        return round(sum(self.test_score)/len(self.test_score),2)
    
    def get_std_mean(self):
        # Retorna el std promedio
        if self.algo_params[0]:
            return round(sum(self.mean_std_deque)/len(self.mean_std_deque), 4)
        else:
            return "NaN"
        
    #[V4.0] - Guardar las ultimas acciones del agente para saber si aumentar el rango
    def get_actions_range_excess(self, DOF, max_range, excess_thresh):
        # [WARNING]: ojo cuando se modifique las acciones del agente!
        actions_range_excess = (np.abs(np.array(
            self.last_actions)[:, -DOF:]) > np.array(max_range.cpu()) * excess_thresh)
        
        actions_range_excess_perc = np.sum(actions_range_excess, 0)/len(self.last_actions)
        return torch.Tensor(actions_range_excess_perc).to(self.device)
    
    def get_goal_mean(self):
        # retorna el % de llegada promedio
        return round(sum(self.goal_reached_deque)/len(
                self.goal_reached_deque) * 100, 2)
    
    def get_time(self):
        # retorna el tiempo de llegada promedio (de los que llegaron)
        
        if sum(self.time_deque) == 0:
            return np.inf
        # retorna la integral de dist promedio
        return round(sum(self.time_deque)/ sum(np.array(self.time_deque) > 0) ,3)
        
    def get_dist(self):
        # retorna la integral de dist promedio
        return round(sum(self.dist_deque)/len(self.dist_deque),3)
    
    def get_energy(self):
        # retorna la integral de energia promedio
        return round(sum(self.energy_deque)/len(self.energy_deque),3)
    
    def get_manipulability(self):
        # retorna la integral de manipulacion promedio
        return round(sum(self.manipu_deque)/len(self.manipu_deque),5)
    
    def get_collisions(self):
        # retorna las colisiones promedio
        return round(sum(self.collision_deque)/len(self.collision_deque),5)
    
    def reset(self, env, start_action = None, area_perc = 1, repeat_perc = 0,
              custom_target_delta = None):
        # Resetear el enviroment y el reward obtenido,
        # tambien obtener el estado inicial.
        env.reset(area_perc = area_perc, repeat_perc = repeat_perc,
                  custom_target_delta = custom_target_delta)
        
        # si se define la accion inicial se aplica, si no se aplica un array de 0s
        if start_action:
            action = start_action
        else:
            action = np.zeros(self.action_dim)
        
        # [V5.0] lamentablemente si no hace nada por 0.5 segundos igual se debe
        # considerar en el reward
        state, ini_reward, _, _ = env.step(action)
        self.state = state
       
        # matriz que guarda las interacciones estado, accion, reward
        self.current_trayectory = np.zeros((self.max_steps,
                                            self.state_dim + self.action_dim + 1))
        self.current_score = ini_reward
        self.current_step = 0
        self.goal_reached = False
        
    def policy(self, net, device, stability_const = 0.00001,
               squashed = True, min_logprob = -10, max_logprob = 10):
        
        state = torch.FloatTensor(np.array(self.state, dtype = float))
        state = torch.unsqueeze(state, 0).to(device)
        
        if self.algo_params[0]:
            
            _, std_red, actions, _ = net.sample(state, self.epsilon,
                                                stability_const, squashed,
                                                min_logprob, max_logprob)
            
            std_red = std_red.cpu().detach()
            actions = actions.cpu().detach().numpy()[0]

            # Para despues monitorear como se comporta la std en el entrenamiento.
            self.mean_std_deque.append(float(torch.mean(std_red)))
            
        else:
            actions = net(state).cpu().detach()
            actions += torch.normal(torch.Tensor([0] * self.action_dim)) * \
                self.epsilon
            # --- Asume que la red tiene rangos simetricos! ---
            actions = torch.clamp(actions, min= -net.max_range, max=net.max_range)
            actions = actions.numpy()[0]
            
        return actions
    
    def interact(self, env, net, device, curr_epoch = 0, stability_const = 0.00001,
                 squashed = True, min_logprob = -10, max_logprob = 10,
                 exploring = False):
        
        
        if not exploring:
            # Se consulta la politica
            actions = self.policy(net, device, stability_const, squashed,
                                  min_logprob, max_logprob)
        else:
            # Se toma una accion al azar
            actions = env.action_space.sample()
        
        # Se interactua con el ambiente
        new_state, reward, done, aux_dict = env.step(actions)
            
        # Se va guardando la trayectoria actual en la matriz
        memory_step = np.concatenate([self.state, actions, [reward]])
        
        self.current_trayectory[self.current_step] = memory_step
        self.current_step += 1
        self.state = new_state
        self.current_score += reward
        
        #[V4.0] - Guardar las ultimas acciones del agente para saber si aumentar el rango
        self.last_actions.append(actions)
        
        # En el caso de que se haya acabado el episodio se guarda en la memoria
        if done:
            if env.env.fruit_adquired: # es un exito si agarro la fruta
                self.goal_reached = True
                
            self.collision_deque.append(env.collision)
            self.manipu_deque.append(env.manipu_measure)
            self.dist_deque.append(env.dist_measure)
            self.energy_deque.append(env.energy_measure)    
            self.test_score.append(self.current_score)
            self.goal_reached_deque.append(self.goal_reached)
            self.time_deque.append(env.env.success_time)
            self.update_memory()
            
        return done
    
    def update_memory(self):
        
        # [V7.0] se usa la forma (s, a, r, s', done) para guardar las memorias
        # asi sale en todos los papers!
        # se convierte de (1, max_T, state - action - reward) a
        # (1, state - action - reward - new state * max_T)
        # [OJO]: se le tiene que agregar un booleano que indique si la 
        # interaccion es terminal, o si no despues no se podra saber!
        
        
        next_states = np.zeros((self.current_step, self.state_dim))
        done_signal = np.zeros((self.current_step, 1))
        done_signal[self.current_step - 1, 0] = 1
        next_states[:-1, :] = self.current_trayectory[1:self.current_step, 0:self.state_dim]
        
        
        self.aux = np.concatenate((self.current_trayectory[:self.current_step, :],
                                   next_states, done_signal), axis = 1)
        
        self.replay_memory += np.split(self.aux, self.current_step , axis = 0)
        

        
    
    def sample_memory(self, rho):

        batch = sample(self.replay_memory, k = int(self.batch_size))
        unified_batch = np.concatenate(batch, axis = 0)
        unified_batch = torch.FloatTensor(unified_batch)
        
        return unified_batch
    
    def test(self, env, actor, device, stability_const = 0.00001,
                 squashed = True, min_logprob = -10, max_logprob = 10,
                 critic = None, q_critic = None, stop_early = True):
        # Funcion para testear al agente (sin aprendizaje), no se requiere de
        # una clase critica para usa este modelo, pero se le pueden pasar para
        # observar su comportamiento.
        
        state = torch.FloatTensor(np.array(self.state, dtype = float))   
        state = torch.unsqueeze(state, 0).to(device)
        epsilon = 0
        
        if self.algo_params[0]:
            #print("NUEVO PRUEBA!")
            # [OJO CAMBIANDO EL COMPORTAMIENTO DEL TESTEO Y SU RUIDO]
            #actions, std, _, _ = actor.sample(state, epsilon, stability_const,
            #                                     squashed, min_logprob,
            #                                     max_logprob)
            _, std, actions, _ = actor.sample(state, epsilon, stability_const,
                                                 squashed, min_logprob,
                                                 max_logprob)
            
            
        
            actions = actions.cpu().detach()
            std = std.cpu().detach().numpy()[0]
            
        else:
            std = [0]
            actions = actor(state).cpu().detach()
        
        # estimacion de los criticos
        V = 0
        Q = 0
        if critic:
            V = critic(state)
            V = round(float(V.detach().cpu().numpy()[0]), 3)

        if q_critic:
            q_state = torch.cat((state.cpu(), actions), 1)
            Q = q_critic(q_state)
            Q = round(float(Q.detach().cpu().numpy()[0]), 3)

        actions = actions.numpy()[0]
        new_state, reward, done, aux_dict = env.step(actions, stop_early = stop_early)
        
        #[V4.0] - Guardar las ultimas acciones del agente para saber si aumentar el rango
        self.last_actions.append(actions)
        
        self.state = new_state
        self.current_score += reward
        
        if done:
            
            if env.env.fruit_adquired:
                self.goal_reached = True
            
            self.collision_deque.append(env.collision)
            self.manipu_deque.append(env.manipu_measure)
            self.dist_deque.append(env.dist_measure)
            self.energy_deque.append(env.energy_measure)    
            self.test_score.append(self.current_score)
            self.goal_reached_deque.append(self.goal_reached)
            self.time_deque.append(env.env.success_time)
    
        # Para imprimir de mejor manera
        mu = list(actions)
        std = list(std)
        
        # redondear para que se vea mejor
        mu = [round(num, 5) for num in mu]
        std = [round(num, 5) for num in std]
        total_reward = round(self.current_score, 3)
        
        return mu, std, V, Q, total_reward, done