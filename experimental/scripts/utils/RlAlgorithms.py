import torch
import torch.nn as nn


##############################################################################
# [Importante]: Los algoritmos que utilicen redes que estiman q, DEBEN 
# usar los q_estados de forma (action, state) NO (state, action), esto debido
# a temas de compatibilidad para procesar imagenes.
##############################################################################


def VPG(agent, env, device, ep, lr_step, rho, gamma, eta, step_size, nets_list,
        targ_nets_list, agent_kwargs = {"squashed": False}, example_actor = None):
    
    # Ojo con las recompensas negativas, ya que estas provocan que el agente
    # "escape" de acciones inconvenientes, pero es menos eficiente que 
    # incentivar al agente a que vaya por acciones "buenas".
    # parametros: rho, gamma
    # redes: [actor]
    # redes target: []
    
    state_dim = agent.state_dim
    actor = nets_list[0]
    
    if lr_step < 1:
        if example_actor:
            done = agent.interact(env, example_actor, device, curr_epoch = ep,
                                  exploring = False, **agent_kwargs)
        else:
            done = agent.interact(env, actor, device, curr_epoch = ep,
                                  exploring = False, **agent_kwargs)
    else:
        done = True
    
    if done:
            batch = agent.sample_memory(rho)
            
            states = batch[:, :, 0:state_dim].to(device)
            rewards = batch[:, :, -1].to(device)
            
            gamma_tensor = torch.pow(gamma, torch.arange(agent.max_steps)).to(device) # para realizar el descuento de las rewards
            gamma_tensor_aux = 1 / gamma_tensor
            
            # se hace la suma de rewards descontadas
            disc_rewards = torch.flip(gamma_tensor * rewards, [1])
            disc_rewards = torch.flip(torch.cumsum(disc_rewards, dim = 1), [1])
            
            # Se obtiene un estimado sin sesgo de Q o V
            Q_hat = (disc_rewards * gamma_tensor_aux).flatten()
            
            # Se concatenan entre trayectorias (para pasar solo un gran batch a la red,
            # porque es mas eficiente)
            target_states = states.reshape(-1, state_dim)
            
            # Se pasan los estados a la red y se obtienen las probabilidades
            _, _, _, log_prob = actor.sample(target_states, agent.epsilon,
                                             **agent_kwargs)
            
            loss = torch.mean(-Q_hat * log_prob)
            
            return done, [loss]
    else:
         return done, None
    

def A2C(agent, env, device, ep, lr_step, rho, gamma, eta, alpha, step_size, nets_list,
        targ_nets_list, agent_kwargs = {"squashed": False}, example_actor = None):
    
    # parametros: rho, gamma
    # redes: [actor, critic, q_critic]
    # redes target: [targ_actor, targ_critic, targ_q_critic]
    
    state_dim = agent.state_dim
    actor = nets_list[0]
    critic = nets_list[1]
    q_critic = nets_list[2]
    #targ_actor =  targ_nets_list[0]
    targ_critic =  targ_nets_list[1]
    targ_q_critic =  targ_nets_list[2]
    
    if lr_step < 1:
        if example_actor:
            done = agent.interact(env, example_actor, device, curr_epoch = ep,
                                  exploring = False, **agent_kwargs)
        else:
            done = agent.interact(env, actor, device, curr_epoch = ep,
                                  exploring = False, **agent_kwargs)
    else:
        done = True
    
    if done:
            batch = agent.sample_memory(rho)
            N, T, X = batch.shape
            
            states = batch[:, :, 0:state_dim].to(device)
            actions = batch[:, :, state_dim: -1].to(device)
            rewards = batch[:, :, -1].to(device).flatten()
        
            # Se concatenan entre trayectorias (para pasar solo un gran batch a la red,
            # porque es mas eficiente)
            target_states = states.reshape(-1, state_dim)
            target_actions = actions.reshape(-1, X - state_dim - 1)
            q_critic_state = torch.cat((target_actions, target_states), 1)
            
            # Se pasan los estados a las redes criticas
            v_target = targ_critic(target_states).flatten().detach()
            q_target = targ_q_critic(q_critic_state).flatten().detach()
            
            # Loop para calcular las funciones de ventaja, se van calculando
            # los TD errors de atras para adelante (ver formula de GAE).
            advantages = torch.zeros(rewards.shape).to(device)
            disc_rewards = torch.zeros(rewards.shape).to(device)
            advantage = 0
            max_i = rewards.shape[0]
            
            # EN EL CASO QUE SE QUIERA USAR GAE (AL MENOS A MI NO ME FUNCIONO).
            # SI SE DESEA PROBAR USAR LAMB > 0
            lamb = 0 
        
            for i in reversed(range(max_i)) :
        
                # El ultimo step de cada trayectoria
                if (i + 1) % T == 0:
                    td_err = q_target[i] - v_target[i]
                    advantage = td_err
                    disc_reward = rewards[i]
                else:
                    td_err = q_target[i] - v_target[i]
                    advantage = td_err + gamma * lamb * advantage
                    disc_reward = rewards[i] + gamma * disc_reward
        
                advantages[i] = advantage 
                disc_rewards[i] = disc_reward
            
            critic_advantages = disc_rewards
            
            # Se retornan las estimaciones de los criticos, que despues sera usada
            # para hacer back propagation.
            critic_estimates = critic(target_states).flatten()
            q_critic_estimates = q_critic(q_critic_state).flatten()
                
            # Se pasan los estados a la red y se obtienen las probabilidades
            _, _, _, log_prob = actor.sample(target_states, agent.epsilon,
                                             **agent_kwargs)
            
            # Correccion de probabilidades para las recompensas negativas
            # (Ver apuntes o efectuar backpropagation a mano para entender)
            # equivalente a cambiar las prob a 1 - prob, pero con valores limitados.
            log_prob = log_prob.flatten()
            neg_adv = advantages < 0
            prob = torch.exp(log_prob).detach()     
            corr_factor =  prob/(1 - prob)
            corr_factor = torch.clamp(corr_factor, max = 0.5)
            
            log_prob = log_prob * (neg_adv.logical_not() + neg_adv * corr_factor)
            
            actor_loss = torch.mean(-advantage * log_prob)
            critic_loss = nn.MSELoss()(critic_advantages, critic_estimates)
            q_critic_loss = nn.MSELoss()(critic_advantages, q_critic_estimates)
            return done, [actor_loss, critic_loss, q_critic_loss]
    else:
         return done, None
     
def DDPG(agent, env, device, ep, lr_step, rho, gamma, eta, alpha, step_size, nets_list,
        targ_nets_list, agent_kwargs = {}, example_actor = None):
    
    # parametros: rho, gamma
    # redes: [actor, q_critic]
    # redes target: [targ_actor, targ_q_critic]
    
    state_dim = agent.state_dim
    actor = nets_list[0]
    q_critic = nets_list[1]
    targ_actor =  targ_nets_list[0]
    targ_q_critic =  targ_nets_list[1]
    
    if lr_step < 1:
        if example_actor:
            done = agent.interact(env, example_actor, device, curr_epoch = ep,
                                  exploring = False, **agent_kwargs)
        else:
            done = agent.interact(env, targ_actor, device, curr_epoch = ep,
                                  exploring = False, **agent_kwargs)
    else:
        done = True
    
    if done:
        
        # batch = matrix(N, tau, obs_dim + act_dim + reward)!
        batch = agent.sample_memory(rho)
     
        N, T, X = batch.shape
    
        states = batch[:, :, 0:state_dim].to(device)
        actions = batch[:, :, state_dim: -1].to(device)
        rewards = batch[:, :, -1].to(device)
    
        # Se concatenan entre trayectorias (para pasar solo un gran batch a la red,
        # porque es mas eficiente)
        target_states = states.reshape(-1, state_dim)
        
        # Se usan las acciones target por el actor target para estimar la mejor posible
        # accion siguiente, las acciones predichas son para la funcion de perdida del 
        # actor, las acciones de memoria son para calcular el q basal (TD(0): r + q(target) - q(mem)).
        pred_actions = actor(target_states)
        targ_actions = targ_actor(target_states)
        mem_actions = actions.reshape(-1, X - state_dim - 1)
        
        q_critic_state = torch.cat((mem_actions, target_states), 1)
        q_critic_target_state = torch.cat((targ_actions, target_states), 1)
        
        # se elimina la primera fila y la ultima se reemplaza por ceros (
        # para restar directamente las matrices)
        q_target = targ_q_critic(q_critic_target_state).detach().reshape(N, T)[:, 1:]
        q_target = torch.cat((q_target, torch.zeros(N, 1).to(device)), dim = 1)
        q = q_critic(q_critic_state).reshape(N, T)
        
        # Actor loss: q(actor(s))
        q_actor = q_critic(torch.cat((target_states, pred_actions), 1))
        
        # critic loss: r + gamma * q(s´, a´) - q(s, a)
        critic_targ = rewards + gamma * q_target 
        
        # Se computan las funciones de perdida
        actor_loss = -torch.mean(q_actor)
        q_critic_loss = nn.MSELoss()(critic_targ, q)
        
        return done, [actor_loss, q_critic_loss]

    else:
         return done, None
     
def TD3(agent, env, device, ep, lr_step, rho, gamma, eta, alpha, step_size,
        nets_list, targ_nets_list, agent_kwargs = {}, example_actor = None):
    
    # parametros: rho, eta
    # redes: [actor, q1_critic, q2_critic]
    # redes target: [targ_actor, targ_critic1, targ_q_critic2]
    
    state_dim = agent.state_dim
    actor = nets_list[0]
    q1_critic = nets_list[1]
    q2_critic = nets_list[2]
    targ_actor =  targ_nets_list[0]
    targ_q1_critic =  targ_nets_list[1]
    targ_q2_critic =  targ_nets_list[2]
    
    if lr_step < 1:
        if example_actor:
            done = agent.interact(env, example_actor, device, curr_epoch = ep,
                                  exploring = False, **agent_kwargs)
        else:
            done = agent.interact(env, targ_actor, device, curr_epoch = ep,
                                  exploring = False, **agent_kwargs)
    else:
        done = True
    
    if done:
        # batch = matrix(N, tau, obs_dim + act_dim + reward)
        batch = agent.sample_memory(rho)
        
        N, T, X = batch.shape
    
        states = batch[:, :, 0:state_dim].to(device)
        actions = batch[:, :, state_dim: -1].to(device)
        rewards = batch[:, :, -1].to(device)
    
        # Se concatenan entre trayectorias (para pasar solo un gran batch a la red,
        # porque es mas eficiente)
        target_states = states.reshape(-1, state_dim)
        
        # Se usan las acciones target por el actor target para estimar la mejor posible
        # accion siguiente, las acciones predichas son para la funcion de perdida del 
        # actor, las acciones de memoria son para calcular el q basal (TD(0): r + q(target) - q(mem)).
        pred_actions = actor(target_states)
        mem_actions = actions.reshape(-1, X - state_dim - 1)
        targ_actions = targ_actor(target_states).detach()
        
        # Se agrega ruido a la red de target (truco de TD3)
        targ_actions += (torch.rand(targ_actions.shape).to(device) * 2 - 1) * eta
        targ_actions = torch.clamp(targ_actions, min = -actor.max_range, max = actor.max_range)
        
        q_state = torch.cat((mem_actions, target_states), 1)
        q_target_state = torch.cat((targ_actions, target_states), 1)
        
        q1_target = targ_q1_critic(q_target_state).detach().reshape(N, T)[:, 1:]
        q2_target = targ_q2_critic(q_target_state).detach().reshape(N, T)[:, 1:]
        
        # Se toma el target menor entre las redes (Truco de TD3)
        q_target = torch.min(q1_target, q2_target)
        q_target = torch.cat((q_target, torch.zeros(N, 1).to(device)), dim = 1)
    
        # Se calculan los valores de q de ambas redes.
        q1 = q1_critic(q_state).reshape(N, T)
        q2 = q2_critic(q_state).reshape(N, T)
        
        # Actor loss: q(actor(s))
        q_actor = q1_critic(torch.cat((pred_actions, target_states), 1))
        
        # critic loss: r + gamma * q(s´, a´) - q(s, a)
        critic_targ = rewards + gamma * q_target
        
        # Se computan las funciones de perdida
        actor_loss = -torch.mean(q_actor)
        q1_critic_loss = nn.MSELoss()(critic_targ, q1)
        q2_critic_loss = nn.MSELoss()(critic_targ, q2)
        
        return done, [actor_loss, q1_critic_loss, q2_critic_loss]
    
    else:
         return done, None
     
                
def SAC(agent, env, device, ep, lr_step, rho, gamma, eta, log_alpha, target_alpha,
        step_size, nets_list, targ_nets_list, agent_kwargs = {"squashed": True},
        example_actor = None):
    
    # [V7.0]: Vesion que samplea (s, a, r, s', done) aleatorios en vez de trayectorias aleatorias
    # [V7.0]: Se actualiza alpha tambien!! (asumo que se usa log_prob, para
    # forzar a que alpha no sea negativo)
    alpha = float(log_alpha.exp())  
    # [IMPORTANTE]: que tan necesario es que se haga dos veces net(s) y net(s'),
    # no es mejor literal simplemente trabajar con net(s') ?
    # supongo que para evitar hacer net(s_terminal) ?
    
    # parametros: rho, gamma, alpha
    # redes: [actor, q1_critic, q2_critic]
    # redes target: [targ_actor, targ_q_critic, targ_q_critic2]
    
    state_dim = agent.state_dim
    action_dim = agent.action_dim
    actor = nets_list[0]
    q1_critic = nets_list[1]
    q2_critic = nets_list[2]
    #targ_actor =  targ_nets_list[0]
    targ_q1_critic =  targ_nets_list[1]
    targ_q2_critic =  targ_nets_list[2]
    
    if lr_step < 1:
        if example_actor:
            done = agent.interact(env, example_actor, device, curr_epoch = ep,
                                  exploring = False, **agent_kwargs)
        else:
            done = agent.interact(env, actor, device, curr_epoch = ep,
                                  exploring = False, **agent_kwargs)
    else:
        done = True
    
    if done:
        # batch = matrix(N, tau, obs_dim + act_dim + reward)
        batch = agent.sample_memory(rho)

        N, X = batch.shape # N: batch, X: (state, action, reward, next_state)
    
        states = batch[:, 0:state_dim].to(device)
        mem_actions = batch[:, state_dim: (state_dim + action_dim)].to(device)
        rewards = batch[:, (state_dim + action_dim): (state_dim + action_dim + 1)].to(device)
        target_states = batch[:, (state_dim + action_dim + 1): -1].to(device)
        done_signal = batch[:, -1:].to(device)
        
        _, _, target_actions, target_logprob = actor.sample(target_states, agent.epsilon, **agent_kwargs)
        _, _, net_actions, logprob = actor.sample(states, agent.epsilon, **agent_kwargs)
        
        # q_state: para calcular q_s, q_target_state: para calcular q_s',
        # actor_state: para calcular q(actor)
        q_state = torch.cat((mem_actions, states), 1)
        q_target_state = torch.cat((target_actions.detach(), target_states), 1)
        actor_state = torch.cat((net_actions, states), 1)
        
        q1_target = targ_q1_critic(q_target_state).detach()
        q2_target = targ_q2_critic(q_target_state).detach()

        # Se toma el target menor entre las redes (Truco de TD3 y SAC)
        q_target = torch.min(q1_target, q2_target)
        
        # Se calcula la entropia (Truco de SAC)
        actor_entropy = -logprob 
        target_entropy = -target_logprob.detach() 
            
        # Se calculan los valores de q de ambas redes.
        q1 = q1_critic(q_state)
        q2 = q2_critic(q_state)
        
        # critic loss: [r + gamma * (q(s´, a´) + H(s'))] - q(s, a)
        critic_targ = rewards + gamma * (1 - done_signal) * (q_target + target_entropy * alpha)
        
        # Actor loss: q(actor(s))
        q1_actor = q1_critic(actor_state)
        q2_actor = q2_critic(actor_state)
        
        # Se usa el minimo de ambas redes (Truco SAC)
        q_actor = torch.min(q1_actor, q2_actor) + actor_entropy * alpha
        
        # [V7.0]: Se actualiza alpha tambien!
        alpha_loss = -(log_alpha * (actor_entropy + target_alpha).detach()).mean()
        
        # Se computan las funciones de perdida
        actor_loss = -torch.mean(q_actor)
        q1_critic_loss = nn.MSELoss()(critic_targ, q1)
        q2_critic_loss = nn.MSELoss()(critic_targ, q2)
        
        return done, [actor_loss, q1_critic_loss, q2_critic_loss, alpha_loss]
    
    else:
         return done, None