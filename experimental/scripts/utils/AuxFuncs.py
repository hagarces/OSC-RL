import numpy as np
import torch
from torch.distributions import Normal as torchNormal
from copy import deepcopy
from IPython.display import clear_output
from collections import deque
import os


def save_net(file_name, net, net_class, overwrite = False):
    # Para guardar la red en el computador 
    # [V5.0] Ahora es imposible sobreescribir un archivo guardado
    checkpoint = {'model': net_class(*net.get_args()),
                  'state_dict': net.state_dict()}
    
    name_addition = ""
    if not overwrite:
        warning_cont = 1
        while True:
            if (file_name + name_addition + '.pth') in os.listdir():
                warning_cont += 1
                name_addition = "V" + str(warning_cont)           
            else:
                if warning_cont > 1:
                    file_name += name_addition
                    print("[WARNING]: ya existen archivos con ese nombre!, se guardara como V" + str(
                        warning_cont))
                break
    
    torch.save(checkpoint, file_name + '.pth')
    return name_addition

def compare_weights(model1, model2):
    # Para comparar si dos modelos tienen exactamente los mismos parametros
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True
    
    
def load_net(filepath):
    # Para cargar una red guardada en el computador
    checkpoint = torch.load(filepath + ".pth")
    model = checkpoint['model']
    model.load_state_dict(deepcopy(checkpoint['state_dict']))
    return model

def soft_update(target, source, tau):
    # Para ir actualizando los datos de forma suave, un tau = 0 indica que los
    # parametros iniciales no variaran nunca (sin aprendizaje) un tau = 1 indica
    # que se reemplazan completamente los parametros (hard update) el rango 
    # entremedio indica actualizaciones suaves.
    
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def test_models(agent, env, device, test_steps, num_plays, model_list,
               model_args_list):
    # Funcion para comparar distintas redes "actores", en donde retorna
    # estadisticas de la recompensa obtenida por cada una de ellas
    # y tambien la red con mejor desempenho
    
    # [Vsim2real]
    try:
        # [V5.0] mostrar tambien el desempehno en colisiones
        
        print("probando modelos")
        n_models = len(model_list)
        
        # modificaciones temporales
        agent_prev_mem = agent.max_score_mem
        agent.goal_reached_deque = deque([], num_plays)
        agent.collision_deque = deque([], num_plays)
        agent.energy_deque = deque([], num_plays)
        agent.time_deque = deque([], num_plays)
        
        env._max_episode_steps = test_steps
        env.render = False
        
        # definicion de matrices para guardar informacion
        scores = np.zeros((n_models, num_plays))
        goal_scores = np.zeros(n_models)
        collision_score = np.zeros(n_models)
        time_score = np.zeros(n_models)
        energy_score = np.zeros(n_models)
        
        # definicion de matrices para saber episodios interesantes!
        worst_episodes = np.zeros(n_models)
        crash_episodes = []
        failed_episodes = []
        
        model_cont = 0
        for model in model_list:
            
            crash_list = []
            failed_list = []
            worst_reward = 10
            worst_episode = -1
            
            for num_play in range(num_plays):
                
                clear_output(wait=True)
                str_num =  str(
                    round(((num_play + num_plays * model_cont) /(num_plays * len(model_list)))*100, 1))
                print(" ~~~~~ {}% completado ~~~~~".format(str_num))
        
                env.seed(num_play)
                agent.reset(env)
                
                # test_steps - 1 para compensar que el agente no hizo nada por un
                # step (esto ocurre al hacer agent.reset).
                for t in range(test_steps - 1): 
                    _, _, _, _, total_reward, done = agent.test(
                        env, model, device, *model_args_list)
                    if done:
                        break
                
                if not agent.goal_reached:
                    failed_list.append(num_play)
                    
                if env.collision:
                     crash_list.append(num_play)
                    
                scores[model_cont, num_play] = total_reward
                
                if total_reward < worst_reward:
                    worst_reward = total_reward
                    worst_episode = num_play
            
            worst_episodes[model_cont] = worst_episode
            
                
            
            crash_episodes.append(crash_list)
            failed_episodes.append(failed_list)
                
            goal_scores[model_cont] = agent.get_goal_mean()
            collision_score[model_cont] = agent.get_collisions()
            energy_score[model_cont] = agent.get_energy()
            time_score[model_cont] = agent.get_time()
            model_cont += 1 
    
        # Resultados
        model_cont = 0
        score_prom = scores.mean(axis = 1).round(3)
        score_std = scores.std(axis = 1).round(3)
        score_min = scores.min(axis = 1).round(3)
        score_max = scores.max(axis = 1).round(3)
        best_model = model_list[np.argmax(score_prom)]
    
        for model in model_list:
            print("""
    Name: {}, Reward Prom: {}, Reward Std: {},
    Reward Min: {}, Reward Max: {}, goal perc: {}%,
    colli perc: {}%, Energy Prom: {}, Time Prom: {}""" .format(
                  str(model), str(score_prom[model_cont]), str(score_std[model_cont]),
                  str(score_min[model_cont]), str(score_max[model_cont]),
                  str(goal_scores[model_cont]), str(collision_score[model_cont] * 100),
                  str(energy_score[model_cont]), str(time_score[model_cont])))
            
            print("Failed episodes: " + str(failed_episodes[model_cont]))
            print("Crashed episodes: " + str(crash_episodes[model_cont]))
            print("Worst episode: " + str(worst_episodes[model_cont]))
            
            model_cont += 1
            
        env.render = True
        
        # realmente es necesario hacer esto?
        agent.goal_reached_deque = deque([], agent_prev_mem)
        
        print("")
        print("BEST MODEL IS: " + str(best_model))
    except Exception as err:
        print(err)
        print("Problemas con el modelo {}, en la seed {}".format(model_cont, num_play))
        
    return score_prom, score_std, score_min, score_max,  goal_scores, best_model

# -------- (Para algoritmos de region de confiaza NO 100% PROBADOS!) ----------
def kl_divergence(pi_old, pi_new, states):
    # Es la esperanza de la divergencia KL por el batch de estados, recordar 
    # que DKL(P, Q) != DKL(Q, P), se busca DKL(pi_old || pi_new). 
    # Notar que en NPG se usa DKL(theta_k, theta) evaluado en theta, esto
    # a pesar de que es 0, y su derivada es 0 su 2da derivada no lo es!,
    # porque [F = nabla^2 DKL] !
    
    mu, std = pi_new(states)
    new_policy = torchNormal(mu, std)
    
    mu_old, std_old = pi_old(states)
    mu_old = mu_old.detach()
    std_old = std_old.detach()
    old_policy = torchNormal(mu_old, std_old)
    
    kl = torch.distributions.kl_divergence(old_policy, new_policy)
     
    return kl.sum(1, keepdim=True)

def hessian_product(pi, states, vector):
    # Un truco para calcular el hessiano, sin tener que computarlo,
    # esto es mediante un truco que indica que es mas facil obtener
    # H * vector que H
    
    # Se calcula DKL (que debe ser 0)
    kl = kl_divergence(pi, pi, states).mean()

    # Se obtiene la 1ra derivada (debe ser 0)
    kl_grad = torch.autograd.grad(kl, pi.parameters(), create_graph=True)
    
    # Se concatenan los gradientes
    kl_grad = torch.cat([aux.view(-1) for aux in kl_grad])
    
    # Se multiplica por el vector
    kl_grad_vect = (kl_grad * vector).sum()
    
    # se deriva nuevamente y se concatena
    kl_hessian_vect = torch.autograd.grad(kl_grad_vect, pi.parameters())
    kl_hessian_vect = torch.cat([aux.view(-1) for aux in kl_hessian_vect])
    
    # por estabilidad se le agrega un pequehno factor
    kl_hessian_vect += vector * 0.001
    
    return kl_hessian_vect

def conjugated_gradient(pi, states, gradient, n_steps, device, err_tol = 10^-10):
    # para calcular H^-1 * g sin tener que invertir el hessiano
    
    H_inv_g = torch.zeros(gradient.size()).to(device)
    r = gradient.clone().to(device)
    p = gradient.clone().to(device)
    rdotr = torch.dot(r, r).to(device)
    for i in range(n_steps):
        
        Hp = hessian_product(pi, states, p).to(device)

        alpha = rdotr / torch.dot(p, Hp)
        
        H_inv_g += alpha * p
        r -= alpha * Hp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < err_tol:
            break
    return H_inv_g

# -----------------------------------------------------------------------------