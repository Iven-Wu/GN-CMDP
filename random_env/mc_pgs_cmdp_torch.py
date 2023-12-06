import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import pdb
import time 
import os
from tqdm import tqdm
import torch

from mc_env_torch import MC_Env

if __name__ == '__main__':
    np.random.seed(10)
    torch.manual_seed(10)

    device = 'cuda'

    num_state = 20
    num_action = 10
    policy_type = 'softmax'
    gamma = 0.9
    lam = 0.

    mc_env = MC_Env(num_state,num_action,policy_type,gamma=gamma,device=device)

    ell_star = mc_env.get_optimum()

    num_MC_sims = 5000
    horizon = 100
    '''
    Policy gradient in action
    '''
    num_iter = 1000
    record_interval = 1
    # Parameters for line search
    alpha = 0.1
    beta = 0.1
    b = 4.

    theta = torch.rand(num_state, num_action).to(device)
    # theta = torch.zeros(num_state,num_action).to(device)
    # theta = np.zeros((num_state,num_action))
    gnpg_gap = []
    gnpg_violation = []
    start_time = time.time()

    for k in tqdm(range(num_iter)):
        grad_sum = torch.zeros_like(theta).to(device)  # Replace np.zeros_like
        cum_rewards = torch.zeros(num_MC_sims).to(device)  # Replace np.zeros
        cum_constrains = torch.zeros(num_MC_sims).to(device)  # Replace np.zeros
        states = torch.randint(0, num_state, (num_MC_sims,)).to(device) 

        rewards_list = []
        constrains_list = []
        actions_list = []
        states_list = []
        for i in range(horizon):
            actions = mc_env.get_action(theta,states)

            next_states, rewards,utilities = mc_env.env_step(states,actions)

            ### compute gradient
            # d_softmax = mc_env.theta_to_policy(theta)[states].clone()
            # d_softmax[np.arange(num_MC_sims),actions] -= 1
            # # cum_rewards = rewards + gamma * cum_rewards
            # cum_rewards = cum_rewards + gamma**i *rewards
            # # cum_constrains = utilities + gamma * cum_constrains
            # cum_constrains = cum_constrains + gamma ** i * utilities

            # final_rewards=  cum_rewards+lam*cum_constrains
            # d_theta = d_softmax * final_rewards.reshape(-1,1)
            # grad_sum[states] += torch.mean(d_theta,dim=0)

            states_list.append(states)
            actions_list.append(actions)
            rewards_list.append(rewards)
            constrains_list.append(utilities)
            ### next state
            states = next_states

        cum_returns_list = mc_env.compute_returns(rewards_list,constrains_list,lam)

        for i in range(horizon):
            states = states_list[i]
            actions = actions_list[i]
            d_softmax = mc_env.theta_to_policy(theta)[states].clone()
            d_softmax[np.arange(num_MC_sims),actions] -= 1
            d_theta = d_softmax * cum_returns_list[i].unsqueeze(-1)
            grad_sum[states] += torch.mean(d_theta,dim=0)

        grad = grad_sum / torch.norm(grad_sum)
        # grad[theta<-50] = 0
        theta -= alpha * grad
        lam = torch.maximum(lam - beta * (cum_constrains.mean() - b).mean(), torch.tensor(0.0).to(device))

        if k % record_interval == 0:
            avg_reward,avg_constrain = mc_env.ell_approx(theta,num_MC_sims,horizon)
            gnpg_gap.append((ell_star-avg_reward).item())
            gnpg_violation.append((b-avg_constrain).item())


    theta = torch.rand(num_state, num_action).to(device)
    # theta = torch.zeros(num_state,num_action).to(device)
    lam = 0.0
    pg_gap = []
    pg_violation = []
    start_time = time.time()
    for k in tqdm(range(num_iter)):
        grad_sum = torch.zeros_like(theta).to(device)  # Replace np.zeros_like
        cum_rewards = torch.zeros(num_MC_sims).to(device)  # Replace np.zeros
        cum_constrains = torch.zeros(num_MC_sims).to(device)  # Replace np.zeros
        states = torch.randint(0, num_state, (num_MC_sims,)).to(device)

        rewards_list = []
        constrains_list = []
        actions_list = []
        states_list = []
        for i in range(horizon):
            actions = mc_env.get_action(theta,states)

            next_states, rewards,utilities = mc_env.env_step(states,actions)
            ### compute gradient
            # d_softmax = mc_env.theta_to_policy(theta)[states].clone()
            # # pdb.set_trace()
            # d_softmax[np.arange(num_MC_sims),actions] -= 1
            # cum_rewards = rewards + gamma * cum_rewards
            # cum_rewards = cum_rewards + gamma**i *rewards
            # cum_constrains = utilities + gamma * cum_constrains
            # cum_constrains = cum_constrains + gamma ** i * utilities


            # final_rewards =  cum_rewards+lam*cum_constrains
            # d_theta = d_softmax * final_rewards.unsqueeze(-1)
            # grad_sum[states] += torch.mean(d_theta,dim=0)

            states_list.append(states)
            actions_list.append(actions)
            rewards_list.append(rewards)
            constrains_list.append(utilities)

            ### next states
            states = next_states
        cum_returns_list = mc_env.compute_returns(rewards_list,constrains_list,lam)

        for i in range(horizon):
            states = states_list[i]
            actions = actions_list[i]
            d_softmax = mc_env.theta_to_policy(theta)[states].clone()
            d_softmax[np.arange(num_MC_sims),actions] -= 1
            d_theta = d_softmax * cum_returns_list[i].unsqueeze(-1)
            grad_sum[states] += torch.mean(d_theta,dim=0)

        grad = grad_sum 
        theta -= alpha * grad
        lam = torch.maximum(lam - beta * (cum_constrains.mean() - b).mean(), torch.tensor(0.0).to(device))


        if k % record_interval == 0:
            avg_reward,avg_constrain = mc_env.ell_approx(theta,num_MC_sims,horizon)
            pg_gap.append((ell_star-avg_reward).item())
            pg_violation.append((b-avg_constrain).item())

    mc_env.plot_curve([pg_gap,gnpg_gap],
                      [pg_violation,gnpg_violation],['gnpg','pg'],method='softmax')
