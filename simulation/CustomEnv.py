import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


class Env(mujoco_env.MujocoEnv, utils.EzPickle):

    
    def __init__(self, file_name, max_episode_steps = 100,
                 DOF = 0, FINGER_DOF = 0, initial_pose = None):

        
        self._max_episode_steps = max_episode_steps
        self.DOF = DOF
        self.FINGER_DOF = FINGER_DOF
        
        # Initial pose
        if initial_pose is not None:
            self.initial_pose = initial_pose
        else:
            self.initial_pose = np.zeros(self.DOF)
        
        # Set initial parameters
        self.iternum = 0
        self._dict = {}
        self.collision = False
        self.q = None
        self.dq = None
        self.fixed_ini = False
        self.fruit_adquired = False
        
        # Performance metrics
        self.energy_measure  = 0
        self.distance_measure  = 0
        self.angle2target = 0
        self.success_time = 0
        
        # Interal parameters
        self.angle_dist_thresh = 0.15
        self.area_perc = 1
        self.safe_perc = 1
        self.last_configuration = None
        self.repeat_perc = 0

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, file_name, frame_skip = 1) 
        
        # Joints limit
        self.q_limit_low = self.model.jnt_range[
                -(self.DOF + self.FINGER_DOF): -self.FINGER_DOF, 0]
        
        self.q_limit_high = self.model.jnt_range[
                -(self.DOF + self.FINGER_DOF): -self.FINGER_DOF, 1]
            
    def check_collisions(self):
        # Checks if the robot has collided with itself or with an enviromental
        # geom.
        
        geom_names = self.model.geom_names

        robot_geoms = [self.model.geom_name2id(
            i) for i in list(geom_names) if ("robo" in i)]
        
        collision_ = False
        for contact_data in self.data.contact:
            if contact_data.geom1 in robot_geoms or contact_data.geom2 in robot_geoms:
                collision_ = True
                break
                
            else:
                if contact_data.geom1 == 0 and contact_data.geom2 == 0:
                    break

        self.collision = collision_
        return collision_
    
    def check_angle(self):
        # This function calculates the angle between the "hand link" and the
        #target, if the end effector is close enough to the target then the 
        # angle is considered zero.
        
        line1 = self.get_body_com("EE") - self.get_body_com("hand_link")
        line2 = self.get_body_com("target") - self.get_body_com("hand_link")
        
        if np.linalg.norm(line2) <= self.angle_dist_thresh: 
            self.angle2target = 0
            return None
        
        line1 = line1/np.linalg.norm(line1)
        line2 = line2/np.linalg.norm(line2)
        angle = abs(np.arccos(np.clip(np.dot(line1, line2), -1.0, 1.0)))
        
        self.angle2target = angle
        
    def step(self, actions, ctrl_const = -1, crash_const = -5, sucess_const = 1, time_const = -1,
             finger_act = None, fruit_grabbed = False, testing = False):
        
        self.iternum += 1
        done =  False
        reward_crash = 0
        reward_sucess = 0
        self.check_angle()
        
        if finger_act is not None:
            actions = np.concatenate((actions, finger_act))
        else:
            actions_ = np.ones(self.DOF + self.FINGER_DOF) * self.action_space.finger_h * 0.1
            
            actions_[:self.DOF] = actions
            actions = actions_.copy()

        self.do_simulation(actions, self.frame_skip) 
        
        # checks if task is completed, whether it is because of success or 
        # because the time runned out.
        if fruit_grabbed:
            self.fruit_adquired = True
            self.success_time = self.data.time
            reward_sucess = sucess_const
            done = True
            self.last_configuration = self.sim.data.qpos[
                -(self.DOF + self.FINGER_DOF): -self.FINGER_DOF]
        elif (self.iternum >= self._max_episode_steps):
            done = True
            self.last_configuration = self.sim.data.qpos[
                -(self.DOF + self.FINGER_DOF): -self.FINGER_DOF]
            
        target = self.get_body_com("target")
        target_dist = np.linalg.norm(target - self.get_body_com("EE"))
        ob = self._get_obs(fruit_grabbed)
        
        # --- Reward design ---
        
        # Penalize for time
        reward_time =  time_const * self.sim.model.opt.timestep
        
        # Penalize for energy
        reward_ctrl =  np.abs(actions[:self.DOF] * self.dq).sum()  * self.sim.model.opt.timestep
        
        # Penalize for collision
        if not self.collision:
            reward_crash = self.check_collisions() * crash_const
        
        reward = reward_time + reward_ctrl * ctrl_const  + reward_crash + reward_sucess
        
        # Update performance metrics
        self.energy_measure += reward_ctrl
        self.distance_measure  += target_dist  * self.sim.model.opt.timestep
        
        dict_ = {"target_pos": target,
                 "target_dist": target_dist,
                 "target_angle": self.angle2target, 
                 "finger_angles": self.sim.data.qpos[-self.FINGER_DOF:]}
        
        self._dict = dict_
        
        return ob, reward, done, dict_
    
    def reset_model(self):

        # Variables reset
        self.iternum = 0
        self.collision = False
        self.energy_measure = 0
        self.distance_measure = 0
        self.angle2target = 0
        self.success_time = 0
        self.fruit_adquired = False
    
        qpos = np.zeros(self.model.nq) 
        qvel = np.zeros(self.model.nv)
        
        # Set the initial configuration
        qpos[-(self.DOF + self.FINGER_DOF): -self.FINGER_DOF] = self.initial_pose
        
        # If the robot does not start in the fixed intial configuration then
        # it starts with a random one.
        if not self.fixed_ini:
             
             mid_range = (self.q_limit_low + self.q_limit_high)/2
            
             ini_joint = self.np_random.uniform(
                 low = mid_range - (mid_range - self.q_limit_low)  * self.safe_perc ,
                 high = mid_range + (self.q_limit_high - mid_range)  * self.safe_perc ,
                 size = self.DOF)

             qpos[-(self.DOF + self.FINGER_DOF): -self.FINGER_DOF] = ini_joint
        
        # There is also a posibility that the next episode the robot starts 
        # with the same configuration as the final configuration of the last
        # episode.
        elif self.repeat_perc >= np.random.uniform():
            qpos[-(self.DOF + self.FINGER_DOF): -self.FINGER_DOF] = self.last_configuration
        
        self.set_state(qpos, qvel)
            
        x, y, z = self.get_body_com("EE")
        ang = np.arctan2(y, x)
        delta = 0 * (np.pi/180)
             
        # The posible target spawn space expands as the training progresses
        aux = self.np_random.uniform(low= -1, high= 1, size=1)
        theta = aux * (np.pi - delta) * self.area_perc + delta * np.sign(aux) + ang
        
        # The target spawn space is cylindrical
        h = self.np_random.uniform(low=0.25, high=0.7, size=1)
        r = self.np_random.uniform(low=0.22, high=0.6, size=1)
        
        target_pos = np.array([r * np.cos(theta),
                              r * np.sin(theta),
                              h])[:, 0]
                    
        # The target is defined as a mocap
        self.data.mocap_pos[1] = np.array(target_pos) 

        self.set_state(qpos, qvel)
        
        self._dict = {"target_pos": self.get_body_com("target"),
                      "target_dist": 1,
                      "target_angle": self.angle2target,
                      "finger_angles": self.sim.data.qpos[-self.FINGER_DOF:]}
                      
        return self._get_obs()
    
    def _get_obs(self, fruit_grabbed = False):
        
        # The observation state is designed as: 
        # ob = [sen(q), cos(q), q_dot, dist2target, angle2target]

        angles = self.sim.data.qpos[-(self.DOF + self.FINGER_DOF):-self.FINGER_DOF]
        angle_vel = self.sim.data.qvel[-(self.DOF + self.FINGER_DOF):-self.FINGER_DOF]
        self.q = angles 
        self.dq = angle_vel
        
        aux1 = np.sin(angles)
        aux2 = np.cos(angles)
        limited_angles = np.concatenate([aux1, aux2])
        dist2target = self.get_body_com("target") - self.get_body_com("EE")
        
        return np.concatenate([limited_angles, angle_vel, dist2target,
                               np.array([self.angle2target])])

    def _set_action_space(self):

        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        
        self.action_space = spaces.Box(low=low[:self.DOF], high=high[:self.DOF],
                                       dtype=np.float32)
        self.action_space.finger_h = high[-1]
        return self.action_space
