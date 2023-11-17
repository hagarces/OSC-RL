import numpy as np
import torch
from collections import deque
import datetime
from copy import deepcopy
from utils.AuxFuncs import soft_update, save_net
from utils.NetModels import Actor


def print_func(algo_name, *print_args, initial_print = False, stats_list = []):
    # Funcion para imprimir el progreso del entrenamiento
    
    # stats_lists = [epsilon, alpha, std]
    # {algo_name: (campos extra para printear, indices de la lista de stats)}

    # ep - score - actor_loss - crit_losses - net_params - time - goal%
    algo_dict = {"VPG": (1, [2]), "A2C": (3, [2]), "DDPG": (2, [0]),
                 "TD3": (2, [0]), "SAC": (3, [1, 2]), "NPG": (2, [2]) }
    
    if initial_print:
        print_args = ["Epochs", "Score", "Act_loss"]
        if algo_name == "VPG":
            print_args += ["Std"]
        
        elif algo_name == "A2C":
            print_args += [ "Std", "Crit_loss", "QCrit_loss"]
        
        elif algo_name in ["DDPG", "TD3"]:
            print_args += ["Epsilon", "QCrit_loss"]
        
        elif algo_name == "SAC":
            print_args += ["Alpha", "Std", "Crit_loss" ]
        
        elif algo_name == "NPG":
            print_args += ["Std", "Crit_loss"]
        print_args += ["Time_elapsed", "% Goal_reached"]
    else:
        
        aux = []
        for item in print_args:
            if type(item) == list:
                aux += item[:1 + (algo_dict[algo_name][0] - len(algo_dict[algo_name][1]))]
            else:
                aux.append(str(item))
        
        print_args = aux[0:3] + list(map(lambda x: stats_list[x],
                           algo_dict[algo_name][1]))  + aux[3:]
            
    print_string = '| {:<18}{:<12}{:<16}'
    print_string += '{:<15}'*algo_dict[algo_name][0] + '{:<16}{:<8} |'

    print(print_string.format(*print_args))
    
def learning_function(ep, lr_step, loss_deques, nets, targ_nets, net_losses,
                      opt_list, actor_lr_interval, critic_lr_interval, tau):
    # recordar que si tau es 1 es un hard_update!
    # opt_list = actor, critic, q_critic, alpha     
    # se hace tantas actualizaciones de redes como el intervalo indique
    # [V7.0] autotunear alpha
    
    # [OJO] estas funciones estan optimizadas para SAC, en el caso de que
    # se use otro algoritmo si o si hay que tener un diccionario con lo que
    # necesita cada uno.
    
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
                    
    #[V7.0] se tunea alpha
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
                ini_rho = 1, end_rho = 0, gamma = 1,
                tau = 1, eta = 1, target_alpha = 0.1, log_alpha = None,
                step_size = 1, max_gradient = 1,
                deque_loss_mem = 30, repeat_max_perc = 0,
                agent_kwargs = None, example_actor = None,
                example_epochs = None, ini_epoch = 1):
    
    # [V5.0] guardado automatico
    ini_epoch = max(ini_epoch, 1)
    if ini_epoch > 1:
        name_addition = "newly_trained"
    else:
        name_addition = "" # en caso de overwriting de archivos
    
    
    # [V3.0] - el entrenamiento se divide por porciones!
    # Trabajo futuro: curriculum learning debe estar afuera.
    
    #  [0.25 - 0.5 - 0.75 - 1]
    area_percs = [0.25, 0.5, 0.75, 1]
    area_cont = 0
    area_score_thresh = -3.0
    
    # [V6.0] - conforme va avanzando el entrenamiento el robot tiene la 
    # posibilidad de partir de la configuracion final del episodio pasado
    repeat_perc = 0
    repeat_ini_epoch = epochs - 1
    # [error!] env._max_episode_steps = agent.max_steps
    
    # --- (Aun no se implementa el clipping de gradientes!) ---
    algo_name = agent.algo_name

    # Para poder crear graficos
    # Se registrara la cantidad de choques/energia/manipulabilidad/distancia/
    # score/final_angle.
    collision_history = []
    energy_history = []
    manipu_history = []
    dist_history = []
    score_history = []
    goal_history = []
    final_angle = []

    # Iniciar variables
    epsilon = agent.epsilon 
    
    rho = ini_rho
    best_nets = None
    best_reward = -np.inf
    best_epoch = None
    
    # Para entrenar al actor o al agente mas de 1 vez por episodio    
    lr_steps = max(int(1/actor_lr_interval), int(1/critic_lr_interval), 1)
    
    # Para guardar las perdidas promedio de las diversas redes
    loss_deques = [deque([0], deque_loss_mem) for i in range(len(nets_list))]

    # Fase de exploracion (NO hay aprendizaje!), solo para
    # rellenar parcialmente la memoria.
    exp_epochs = max(exp_epochs, agent.batch_size/20)
    print("Generando memorias iniciales...")
    for ep in range(exp_epochs):
        done = False
        agent.reset(env)
        while not done:
            
            exploring = (ini_epoch == 1)
            done = agent.interact(env, nets_list[0], device, exploring = exploring )

        if agent.check_mem():
            break

    print("ha terminado la fase de exploracion sin aprendizaje")
    print("Average score: {}".format(agent.show_score()))

    t_i_ = datetime.datetime.now()
    print("Comienza el entrenamiento!")
    try:
        
        for ep in range(ini_epoch, epochs):
            
            agent.reset(env, area_perc = area_percs[area_cont], repeat_perc = repeat_perc)
            
            # De esta forma se permite que el agente aprenda con controladores
            # auxiliares!
            if example_actor:
                if ep > example_epochs:
                    example_actor = None
            
            for lr_step in range(lr_steps):
                done = False
                while not done:
                    
                    # El agente interactua y retorna las funciones de perdida
                    done, net_losses = rl_algorithm(agent, env, device, ep, lr_step,
                                                    rho, gamma, eta, log_alpha,
                                                    target_alpha, step_size,
                                                    nets_list, targ_nets_list,
                                                    agent_kwargs, example_actor)
                    
                # Se actualizan los pesos de las redes
                loss_deques = learning_function(ep, lr_step, loss_deques, nets_list,
                                                targ_nets_list, net_losses,
                                                opt_list, actor_lr_interval,
                                                critic_lr_interval, tau)
                
            
                    
            
            # Se actualizan las variables de memoria y exploracion.
            alpha = float(log_alpha.exp())
            rho = ini_rho - (ini_rho - end_rho) * (ep/epochs)
            agent.epsilon = epsilon
            
            # [V6]: se actualiza el porcentaje de probabilidad de repeticion
            repeat_perc = max(( 
                (ep - repeat_ini_epoch)/(epochs - repeat_ini_epoch)) * repeat_max_perc, 0)
            
            if ep % save_interval == 0:
                aux_net = Actor(*nets_list[0].get_args()) # FEO
                aux_net.load_state_dict(deepcopy(best_nets[0]))
                
                print("Guardando redes en archivo local...")
                if ep == save_interval:
                    name_addition = save_net(save_file_name + "_fN",
                                             nets_list[0], Actor, overwrite=False)
                    
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
                
                # Para printear el tiempo estimado de termino
                if ep == print_interval:
                    total_segs = epochs/ep * (t_f_ - t_i_).total_seconds()
                    
                    est_hours = int(total_segs // 3600)
                    est_mins  = int((total_segs % 3600) // 60)
                    print("| ---              Estimated time to finish training: {} hours, {} minutes\
                                         --- |".format(est_hours, est_mins))
                    print_func(algo_name, initial_print = True)
                    
                # Se obtiene las perdidas promedio.
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
                
                # Se imprime el estado del entrenamiento
                print_func(algo_name, epoch_str, score, curr_losses, time_str,
                           goal_prom, initial_print = False, stats_list = stats_list)
                print("tiempo de llegada prom: " + str(agent.get_time() ))
                
                
                # Si el agente lo hace lo suficientemente bien se aumenta la dificultad
                if score >= area_score_thresh:
                    
                    if repeat_perc == 0:
                        repeat_ini_epoch = ep
                    
                    if area_cont < (len(area_percs) - 1):
                    
                        print("~ ~ ~ Se expande el area de entrenamiento! ~ ~ ~")
                        area_cont += 1
                        

                # Se guarda el mejor agente si es que su reward obtenida es la mejor.
                if agent.show_score() > best_reward:
                    best_reward = agent.show_score()
                    best_nets = list(map(lambda x: deepcopy(x.state_dict()), nets_list))
                    best_epoch = ep
                    
                collision_history.append(agent.get_collisions())
                energy_history.append(agent.get_energy())
                manipu_history.append(agent.get_manipulability())
                dist_history.append(agent.get_dist())
                goal_history.append(agent.get_goal_mean())
                score_history.append(agent.show_score())
                final_angle.append(env.env.angle2target)
                
    except (KeyboardInterrupt, Exception) as err:
        print(err)
    print("Entrenamiento Finalizado!")
    print("Recuperando al mejor agente de la epoca: " + str(best_epoch) + " ...")
    
    # Se guarda automaticamente en los archivos locales
    aux_net = Actor(*nets_list[0].get_args()) # FEO
    aux_net.load_state_dict(deepcopy(best_nets[0]))
                
    save_net(save_file_name + "_fN" + name_addition,
                             nets_list[0], Actor, overwrite=True)
    save_net(save_file_name + "_bN" + name_addition,
             aux_net, Actor, overwrite=True)
    
    final_nets = list(map(lambda x: deepcopy(x.state_dict()), nets_list))
    
    stats_history = [score_history, goal_history, dist_history,
                     collision_history, energy_history, manipu_history,
                     final_angle]
    
    return (stats_history, best_nets, final_nets)