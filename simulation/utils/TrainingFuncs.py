import numpy as np
import torch
from collections import deque
import datetime
from copy import deepcopy
from utils.AuxFuncs import soft_update, save_net
from utils.NetModels import Actor


def print_func(algo_name, *print_args, initial_print = False, stats_list = []):

    if initial_print:
        print_args = ["Epochs", "Score", "Act_loss", "Alpha", "Std",
                      "Crit_loss", "Time_elapsed", "% Goal_reached"]

    else:
        
        aux = []
        for item in print_args:
            if type(item) == list:
                aux += item[:2]
            else:
                aux.append(str(item))
        
        print_args = aux[0:3] + list(map(lambda x: stats_list[x], [1, 2]))  + aux[3:]
            
    print_string = '| {:<18}{:<12}{:<16}'
    print_string += '{:<15}'*3 + '{:<16}{:<8} |'

    print(print_string.format(*print_args))
    
def learning_function(ep, lr_step, loss_deques, nets, targ_nets, net_losses,
                      opt_list, actor_lr_interval, critic_lr_interval, tau):

    
    if ep % max(1, actor_lr_interval) == 0:
        if lr_step < int(1/actor_lr_interval):
            # Aprende el actor
            actor_opt = opt_list[0]
            actor_loss = net_losses[0]
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            loss_deques[0].append(float(actor_loss))
            if targ_nets[0]:
                soft_update(targ_nets[0], nets[0], tau)
            
    if ep % max(1, critic_lr_interval) == 0:
        if lr_step < int(1/critic_lr_interval):
            # Aprenden los criticos
            for cont in range(1, 3):
                critic_opt = opt_list[cont]
                critic_loss = net_losses[cont]
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()
                loss_deques[cont].append(float(critic_loss))
                if targ_nets[cont]:
                    soft_update(targ_nets[cont], nets[cont], tau)
                    
    alpha_opt = opt_list[-1]
    alpha_loss = net_losses[-1]
    alpha_opt.zero_grad()
    alpha_loss.backward()
    alpha_opt.step()
    return loss_deques
            
def train_agent(rl_algorithm, agent, env, device, epochs, exp_epochs,
                print_interval, save_interval, save_file_name,
                nets_list, targ_nets_list, opt_list,
                actor_lr_interval = 1, critic_lr_interval = 1,
                rho = 0, gamma = 1,
                tau = 1, eta = 1, target_alpha = 0.1, log_alpha = None,
                step_size = 1, max_gradient = 1,
                deque_loss_mem = 30, repeat_max_perc = 0,
                agent_kwargs = None,  example_epochs = None):
    
    ini_epoch = 1
    name_addition = ""

    area_percs = [0.25, 0.5, 0.75, 1]
    area_cont = 0
    area_score_thresh = -3.0
    repeat_perc = 0
    repeat_ini_epoch = epochs - 1
    algo_name = agent.algo_name

    collision_history = []
    energy_history = []
    dist_history = []
    score_history = []
    goal_history = []
    final_angle = []

    # Initialize variables
    epsilon = agent.epsilon 
    best_nets = None
    best_reward = -np.inf
    best_epoch = None

    lr_steps = max(int(1/actor_lr_interval), int(1/critic_lr_interval), 1)
    loss_deques = [deque([0], deque_loss_mem) for i in range(len(nets_list))]

    # Exploration phase
    exp_epochs = max(exp_epochs, agent.batch_size/20)
    print("Generating initial memories...")
    for ep in range(exp_epochs):
        done = False
        agent.reset(env)
        while not done:
            exploring = (ini_epoch == 1)
            done = agent.interact(env, nets_list[0], device,
                                  exploring = exploring )
        if agent.check_mem():
            break

    print("The exploration phase has finished!")
    print("Average score: {}".format(agent.show_score()))

    t_i_ = datetime.datetime.now()
    print("Starting training...")
    try:
        
        for ep in range(ini_epoch, epochs):
            
            agent.reset(env, area_perc = area_percs[area_cont],
                        repeat_perc = repeat_perc)
                        
            for lr_step in range(lr_steps):
                done = False
                while not done:
        
                    done, net_losses = rl_algorithm(agent, env, device, ep,
                                                    lr_step, rho, gamma, eta,
                                                    log_alpha, target_alpha,
                                                    step_size, nets_list,
                                                    targ_nets_list,
                                                    agent_kwargs)
                    
                loss_deques = learning_function(ep, lr_step, loss_deques,
                                                nets_list, targ_nets_list,
                                                net_losses, opt_list,
                                                actor_lr_interval,
                                                critic_lr_interval, tau)

            alpha = float(log_alpha.exp())
            agent.epsilon = epsilon
            

            repeat_perc = max(((ep - repeat_ini_epoch)/(
                epochs - repeat_ini_epoch)) * repeat_max_perc, 0)
            
            if ep % save_interval == 0:
                aux_net = Actor(*nets_list[0].get_args()) 
                aux_net.load_state_dict(deepcopy(best_nets[0]))
                
                print("Saving net in a local file...")
                if ep == save_interval:
                    name_addition = save_net(save_file_name + "_fN", 
                                             nets_list[0], Actor,
                                             overwrite=False)
                    
                    save_net(save_file_name + "_bN" + name_addition,
                             aux_net, Actor, overwrite=True) 
                
                else:
                    save_net(save_file_name + "_fN" + name_addition,
                             nets_list[0], Actor, overwrite=True)
                    save_net(save_file_name + "_bN" + name_addition,
                             aux_net, Actor, overwrite=True)   
                
                torch.save({'model': nets_list[1]}, 'q1critic.pth')
                torch.save({'model': nets_list[2]}, 'q2critic.pth')
                torch.save({'model': targ_nets_list[1]}, 'target_q1critic.pth')
                torch.save({'model': targ_nets_list[2]}, 'target_q2critic.pth')
    
            if ep % print_interval == 0:
                
                t_f_ = datetime.datetime.now()
                if ep == print_interval:
                    total_secs = epochs/ep * (t_f_ - t_i_).total_seconds()
                    
                    est_hours = int(total_secs // 3600)
                    est_mins  = int((total_secs % 3600) // 60)
                    print("| ---              Estimated time to finish training: {} hours, {} minutes\
                                         --- |".format(est_hours, est_mins))
                    print_func(algo_name, initial_print = True)
                    
                curr_losses = [round(sum(loss_deques[i])/len(loss_deques[i]),
                                     5) for i in range(len(nets_list))]
                goal_prom = str(agent.get_goal_mean()) + "%"
                score = agent.show_score()
                std = agent.get_std_mean()
                alpha_str = str(round(alpha, 3))
                eps = str(round(epsilon, 3))
                stats_list = [eps, alpha_str, std]
                epoch_str = "[" + str(ep) + " / " + str(epochs) + "]"
                time_str = str(round((t_f_ - t_i_).total_seconds(), 1)) + " [s]"
                
                print_func(algo_name, epoch_str, score, curr_losses, time_str,
                           goal_prom, initial_print = False, 
                           stats_list = stats_list)
                print("Estimated task finish time: " + str(agent.get_time() ))
                
                if score >= area_score_thresh:
                    
                    if repeat_perc == 0:
                        repeat_ini_epoch = ep
                    
                    if area_cont < (len(area_percs) - 1):
                        print("~ ~ Se expande el area de entrenamiento! ~ ~")
                        area_cont += 1
                        
                if agent.show_score() > best_reward:
                    best_reward = agent.show_score()
                    best_nets = list(map(lambda x: deepcopy(x.state_dict()),
                                         nets_list))
                    best_epoch = ep
                    
                collision_history.append(agent.get_collisions())
                energy_history.append(agent.get_energy())
                dist_history.append(agent.get_dist())
                goal_history.append(agent.get_goal_mean())
                score_history.append(agent.show_score())
                final_angle.append(env.env.angle2target)
                
    except (KeyboardInterrupt, Exception) as err:
        print(err)
    print("The training has finished!")
    print("Recovering the best agent of epoch: " + str(best_epoch) + " ...")
    
    aux_net = Actor(*nets_list[0].get_args()) 
    aux_net.load_state_dict(deepcopy(best_nets[0]))
                
    save_net(save_file_name + "_fN" + name_addition,
                             nets_list[0], Actor, overwrite=True)
    save_net(save_file_name + "_bN" + name_addition,
             aux_net, Actor, overwrite=True)
    
    final_nets = list(map(lambda x: deepcopy(x.state_dict()), nets_list))
    stats_history = [score_history, goal_history, dist_history,
                     collision_history, energy_history,
                     final_angle]
    
    return (stats_history, best_nets, final_nets)