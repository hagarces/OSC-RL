import torch
import torch.nn as nn

                
def SAC(agent, env, device, ep, lr_step, rho, gamma, eta, log_alpha, 
        target_alpha, step_size, nets_list, targ_nets_list, 
        agent_kwargs = {"squashed": True}, example_actor = None):
    

    alpha = float(log_alpha.exp())  
    state_dim = agent.state_dim
    action_dim = agent.action_dim
    actor = nets_list[0]
    q1_critic = nets_list[1]
    q2_critic = nets_list[2]
    targ_q1_critic =  targ_nets_list[1]
    targ_q2_critic =  targ_nets_list[2]
    
    if lr_step < 1:
        done = agent.interact(env, actor, device, curr_epoch = ep,
                              exploring = False, **agent_kwargs)
    else:
        done = True
    
    if done:
        batch = agent.sample_memory(rho)

        N, X = batch.shape 
    
        states = batch[:, 0:state_dim].to(device)
        mem_actions = batch[:, state_dim: (state_dim + action_dim)].to(device)
        rewards = batch[:,(state_dim + action_dim): (
            state_dim + action_dim + 1)].to(device)
        target_states = batch[:, (state_dim + action_dim + 1): -1].to(device)
        done_signal = batch[:, -1:].to(device)
        
        _, _, target_actions, target_logprob = actor.sample(target_states,
                                                            agent.epsilon,
                                                            **agent_kwargs)
        _, _, net_actions, logprob = actor.sample(states, agent.epsilon,
                                                  **agent_kwargs)
        
        q_state = torch.cat((mem_actions, states), 1)
        q_target_state = torch.cat((target_actions.detach(), target_states), 1)
        actor_state = torch.cat((net_actions, states), 1)
        
        q1_target = targ_q1_critic(q_target_state).detach()
        q2_target = targ_q2_critic(q_target_state).detach()

        q_target = torch.min(q1_target, q2_target)

        actor_entropy = -logprob 
        target_entropy = -target_logprob.detach() 
            
        q1 = q1_critic(q_state)
        q2 = q2_critic(q_state)
        
        critic_targ = rewards + gamma * (1 - done_signal) * (q_target + target_entropy * alpha)
        
        q1_actor = q1_critic(actor_state)
        q2_actor = q2_critic(actor_state)
        
        q_actor = torch.min(q1_actor, q2_actor) + actor_entropy * alpha
        
        alpha_loss = -(log_alpha * (actor_entropy + target_alpha).detach()).mean()
        
        actor_loss = -torch.mean(q_actor)
        q1_critic_loss = nn.MSELoss()(critic_targ, q1)
        q2_critic_loss = nn.MSELoss()(critic_targ, q2)
        
        return done, [actor_loss, q1_critic_loss, q2_critic_loss, alpha_loss]
    
    else:
         return done, None