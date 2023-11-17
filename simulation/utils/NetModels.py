import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal as torchNormal
from copy import deepcopy
from collections import deque
from random import sample


class Actor(nn.Module):

    
    def __init__(self, algorithm_name, net_architecture, last_dim,  action_dim, 
                 max_range, min_std = 0.01, max_std = 1):

        super(Actor, self).__init__()
        self.net = net_architecture
        self.action_dim = action_dim
        self.max_range = max_range
        self.min_std = min_std
        self.max_std = max_std
        self.last_dim = last_dim
        self.algo_name = algorithm_name
        self.mu_layer = nn.Linear(last_dim, action_dim)
        self.std_layer = nn.Linear(last_dim, action_dim)
        self.std_activation = nn.Sigmoid()
      
    def get_args(self):
        return [self.algo_name, deepcopy(self.net), self.last_dim,
                self.action_dim, self.max_range, self.min_std, self.max_std]   
     
    def forward(self, x):            
        x = self.net(x)
        mu_net = self.mu_layer(x)

        std_net = self.std_activation(self.std_layer(x)) * \
            self.max_std + self.min_std
        
        return mu_net, std_net
        
    def sample(self, x, epsilon, stability_const = 0.00001, squashed = True,
               min_logprob = -5, max_logprob = 2):

        mu_net, std_net = self.forward(x)
        std_net = torch.clamp(std_net, min = epsilon) 
        
        action_normal = torchNormal(mu_net, std_net)
        action_sample = action_normal.rsample()
        
        if squashed:
            actions = nn.Tanh()(action_sample) * self.max_range
            
            log_prob = action_normal.log_prob(action_sample) - \
                        torch.log(self.max_range**2 - actions.pow(2) + \
                                  stability_const)
                        
            mu_net = nn.Tanh()(mu_net) * self.max_range
        else:
            actions = torch.clamp(action_sample, min = -self.max_range,
                                  max=self.max_range)
            log_prob = action_normal.log_prob(action_sample)
        
        log_prob = torch.clamp(log_prob, min = min_logprob, max = max_logprob)
        log_prob = log_prob.mean(1, keepdim=True)

        return mu_net, std_net, actions, log_prob
        
    def __str__(self):
        return self.algo_name + " Actor"


class DummyActor():
    # Test actor that only outputs the delta Cartesian target position
    def __init__(self, env, action_dim, max_range):
        
        self.env = env
        self.action_dim = action_dim
        self.max_range = max_range
        
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
    # Test actor that does not move
    def __init__(self, env, action_dim, max_range):
        
        self.env = env
        self.action_dim = action_dim
        self.max_range = max_range
        
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

    
    def __init__(self, algorithm_name, net_architecture, hidden_dim):
        super(Critic, self).__init__()
        self.net = net_architecture
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.algo_name = algorithm_name  
    
    def forward(self, x):
        x = self.net(x)
        output = self.output_layer(x)
        
        return output
    
    def __str__(self):
        return self.algo_name + " Critic"


class Agent():
    
    
    def __init__(self, algorithm_name, max_mem_lenght, max_score_mem,
                 max_stats_mem, batch_size, max_steps, state_dim, action_dim,
                 dist_thresh, device, epsilon = 0, gamma = 0):
        
        self.algo_name = algorithm_name
        self.device = device 
        
        # Variables de la memoria del agente
        self.replay_memory = deque([], max_mem_lenght)
        self.max_mem_lenght = max_mem_lenght
        
        # Hiper parametros
        self.batch_size = batch_size     
        self.max_steps = max_steps - 1
        self.action_dim = action_dim 
        self.state_dim = state_dim
        self.epsilon =  epsilon
        self.gamma = gamma
        
        # Variables del estado del agente
        self.state = None
        self.current_trayectory = None
        self.current_step = 0 
        self.current_score = 0 
        
        # Mediciones de desempehno
        self.max_score_mem = max_score_mem
        self.test_score = deque([], max_score_mem) 
        self.goal_reached_deque = deque([], max_score_mem) 
        self.collision_deque = deque([], max_score_mem)
        self.dist_deque = deque([], max_score_mem) 
        self.energy_deque = deque([], max_score_mem) 
        self.time_deque = deque([], max_score_mem) 
        self.mean_std_deque = deque([], max_stats_mem * self.max_steps)
        self.last_actions = deque([], max_score_mem * self.max_steps)

        # Variables auxiliares 
        self.dist_thresh = dist_thresh
        self.goal_reached = False
        
    def check_mem(self):
        # Chequear si la memoria se lleno
        len(self.replay_memory) >= self.max_mem_lenght
        
    def show_score(self):
        #Retorna el score promedio
        return round(sum(self.test_score)/len(self.test_score),2)
    
    def get_std_mean(self):
        return round(sum(self.mean_std_deque)/len(self.mean_std_deque), 4)
   
    def reset_stats_mem(self):
        self.test_score.clear()
        self.goal_reached_deque.clear()
        self.collision_deque.clear()
        self.dist_deque.clear()
        self.energy_deque.clear()
        self.time_deque.clear()
        
    def get_goal_mean(self):
        return round(sum(self.goal_reached_deque)/len(
                self.goal_reached_deque) * 100, 2)
    
    def get_time(self):
        if sum(self.time_deque) == 0:
            return np.inf
        return round(sum(self.time_deque)/ sum(np.array(self.time_deque) > 0) , 3)
        
    def get_dist(self):
        return round(sum(self.dist_deque)/len(self.dist_deque), 3)
    
    def get_energy(self):
        return round(sum(self.energy_deque)/len(self.energy_deque), 3)
    
    def get_collisions(self):
        return round(sum(self.collision_deque)/len(self.collision_deque), 5)
    
    def reset(self, env, area_perc = 1, repeat_perc = 0,
              custom_target_delta = None):

        env.reset(area_perc = area_perc, repeat_perc = repeat_perc,
                  custom_target_delta = custom_target_delta)
        
        action = np.zeros(self.action_dim)
        state, ini_reward, _, _ = env.step(action)
        self.state = state
       
        self.current_trayectory = np.zeros((self.max_steps,
                                            self.state_dim + self.action_dim + 1))
        self.current_score = ini_reward
        self.current_step = 0
        self.goal_reached = False
        
    def policy(self, net, device, stability_const = 0.00001,
               squashed = True, min_logprob = -10, max_logprob = 10):
        
        state = torch.FloatTensor(np.array(self.state, dtype = float))
        state = torch.unsqueeze(state, 0).to(device)
    
        _, std_red, actions, _ = net.sample(state, self.epsilon,
                                            stability_const, squashed,
                                            min_logprob, max_logprob)
        std_red = std_red.cpu().detach()
        actions = actions.cpu().detach().numpy()[0]

        self.mean_std_deque.append(float(torch.mean(std_red)))
        return actions
    
    def interact(self, env, net, device, curr_epoch = 0,
                 stability_const = 0.00001, squashed = True, min_logprob = -10,
                 max_logprob = 10, exploring = False):
        
        if not exploring:
            actions = self.policy(net, device, stability_const, squashed,
                                  min_logprob, max_logprob)
        else:
            actions = env.action_space.sample()
        
        new_state, reward, done, aux_dict = env.step(actions)
            
        memory_step = np.concatenate([self.state, actions, [reward]])
        
        self.current_trayectory[self.current_step] = memory_step
        self.current_step += 1
        self.state = new_state
        self.current_score += reward
        
        self.last_actions.append(actions)
        
        if done:
            if env.env.fruit_adquired: 
                self.goal_reached = True
                
            self.collision_deque.append(env.collision)
            self.dist_deque.append(env.dist_measure)
            self.energy_deque.append(env.energy_measure)    
            self.test_score.append(self.current_score)
            self.goal_reached_deque.append(self.goal_reached)
            self.time_deque.append(env.env.success_time)
            self.update_memory()
            
        return done
    
    def update_memory(self):

        next_states = np.zeros((self.current_step, self.state_dim))
        done_signal = np.zeros((self.current_step, 1))
        done_signal[self.current_step - 1, 0] = 1
        next_states[:-1, :] = self.current_trayectory[1:self.current_step,
                                                      0:self.state_dim]
        
        self.aux = np.concatenate((self.current_trayectory[:self.current_step,
                                                           :],
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
        
        state = torch.FloatTensor(np.array(self.state, dtype = float))   
        state = torch.unsqueeze(state, 0).to(device)
        epsilon = 0
        
        _, std, actions, _ = actor.sample(state, epsilon, stability_const,
                                             squashed, min_logprob,
                                             max_logprob)
        
        actions = actions.cpu().detach()
        std = std.cpu().detach().numpy()[0]
        
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
        
        self.last_actions.append(actions)
        
        self.state = new_state
        self.current_score += reward
        
        if done:
            
            if env.env.fruit_adquired:
                self.goal_reached = True
            
            self.collision_deque.append(env.collision)
            self.dist_deque.append(env.dist_measure)
            self.energy_deque.append(env.energy_measure)    
            self.test_score.append(self.current_score)
            self.goal_reached_deque.append(self.goal_reached)
            self.time_deque.append(env.env.success_time)

        mu = [round(num, 5) for num in list(actions)]
        std = [round(num, 5) for num in list(std)]
        total_reward = round(self.current_score, 3)
        
        return mu, std, V, Q, total_reward, done