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
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(128,128)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

def npg_mc(mc_env,num_state,num_action,gamma,num_MC_sims,horizon,num_iter,alpha,beta,record_interval,b):
    npg_gap = []
    npg_violation= []
    lam = 0.5
    theta = torch.tensor(np.random.uniform(0,1,size=(num_state,num_action))).float().to(device)
    values = torch.zeros(num_state).to(device)
    q_values = torch.zeros((num_state,num_action)).to(device)
    for k in tqdm(range(num_iter)):
        for init_state in range(num_action):
            cum_constrains = torch.zeros((num_action, num_MC_sims)).to(device)
            states = torch.full((num_action, num_MC_sims), init_state, dtype=torch.int64).to(device)
            actions = torch.arange(num_action).unsqueeze(1).repeat(1, num_MC_sims).to(device)
            
            rewards_list = []
            actions_list = []
            states_list = []
            constrains_list = []
            for i in range(horizon):
                if i > 0:
                    actions = mc_env.get_action(theta, states.reshape(-1)).view(num_action, num_MC_sims)
                next_states, rewards, utilities = mc_env.env_step(states.reshape(-1), actions.reshape(-1))
                next_states = next_states.view(num_action, num_MC_sims)
                rewards = rewards
                utilities = utilities

                states_list.append(states)
                actions_list.append(actions)
                rewards_list.append(rewards)
                constrains_list.append(utilities)
                states = next_states
            cum_returns_list = mc_env.compute_returns(rewards_list,constrains_list,lam)
            cum_returns_final = cum_returns_list[0].reshape(num_action,-1)
            for action in range(num_action):
                q_values[init_state,action] = cum_returns_final[action].mean()
        
        values = torch.sum(q_values * mc_env.theta_to_policy(theta),dim=1)
        advantage_mat = q_values - values[:,None]
        
        V_g_rho = 0
        cum_constrains = 0
        init_states = torch.multinomial(mc_env.rho, num_MC_sims, replacement=True)
        states = init_states
        for i in range(horizon):
            actions = mc_env.get_action(theta,states)
            next_states, rewards, utilities = mc_env.env_step(states,actions)
            # cum_rewards = gamma*cum_rewards + rewards
            cum_constrains += utilities
            states = next_states
        V_g_rho = cum_constrains.mean()
        ### update with gradient
        theta += alpha*advantage_mat/(1-gamma)
        theta = mc_env.theta_to_policy(theta)
        lam = torch.maximum(lam-beta*(V_g_rho-b),torch.tensor(0.0).to(device))

        if k % record_interval == 0:
            avg_reward,avg_constrain = mc_env.ell_theta(theta)
            npg_gap.append(ell_star-avg_reward)
            npg_violation.append(b-avg_constrain)
    return npg_gap,npg_violation

def pg_mc(mc_env,policy_net,num_state,num_action,gamma,num_MC_sims,horizon,num_iter,alpha,beta,record_interval,b):


    theta = torch.tensor(np.random.uniform(0,1,size=(num_state,num_action))).float().to(device)
    lam = 0.5
    pg_gap = []
    pg_violation = []
    start_time = time.time()

    optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
    for k in tqdm(range(num_iter)):
        cum_constrains = torch.zeros(num_MC_sims).to(device)
        states = torch.multinomial(mc_env.rho, num_samples=num_MC_sims,replacement=True).squeeze()
        rewards_list = []
        constrains_list = []
        actions_list = []
        states_list = []
        for i in range(horizon):
            # actions = mc_env.get_action(theta,states)
            probs = policy_net(theta[states])
            actions = torch.multinomial(probs, num_samples=1).squeeze()
            next_states, rewards,utilities = mc_env.env_step(states,actions)
            ### compute gradient
            states_list.append(states)
            actions_list.append(actions)
            rewards_list.append(rewards)
            constrains_list.append(utilities)
            ### next states
            states = next_states
        cum_returns_list,cum_constrains = mc_env.compute_returns(rewards_list,constrains_list,lam)

        cum_loss = 0
        # for i in range(horizon):
        i = 0
        states = states_list[i]
        actions = actions_list[i]
        probs = policy_net(theta[states].clone())
        m = torch.distributions.Categorical(probs)
        loss = -m.log_prob(actions) * cum_returns_list[i].unsqueeze(-1)
        cum_loss += loss.mean()

        # cum_loss/= horizon
        # grad = grad_sum
        # theta -= alpha * grad
        optimizer.zero_grad()
        cum_loss.backward()
        optimizer.step()

        lam = torch.maximum(lam - beta * (cum_constrains[0].mean() - b).mean(), torch.tensor(0.0).to(device))

        if k % record_interval == 0:
            # avg_reward,avg_constrain = mc_env.ell_theta(theta)
            avg_reward,avg_constrain = mc_env.ell_network(theta,policy_net)
            if k % 50==0:
                print((ell_star-avg_reward).item())
            pg_gap.append((ell_star-avg_reward).item())
            pg_violation.append((b-avg_constrain).item())
    return pg_gap,pg_violation

def gnpg_mc(mc_env,num_state,num_action,gamma,num_MC_sims,horizon,num_iter,alpha,beta,record_interval,b):
    lam = 0.5
    # theta = torch.rand(num_state, num_action).to(device)
    theta = torch.tensor(np.random.uniform(0,1,size=(num_state,num_action))).float().to(device)
    gnpg_gap = []
    gnpg_violation = []
    start_time = time.time()

    for k in tqdm(range(num_iter)):
        grad_sum = torch.zeros_like(theta).to(device)  
        cum_constrains = torch.zeros(num_MC_sims).to(device)  
        states = torch.randint(0, num_state, (num_MC_sims,)).to(device) 

        rewards_list = []
        constrains_list = []
        actions_list = []
        states_list = []
        for i in range(horizon):
            actions = mc_env.get_action(theta,states)

            next_states, rewards,utilities = mc_env.env_step(states,actions)

            ### compute gradient
            states_list.append(states)
            actions_list.append(actions)
            rewards_list.append(rewards)
            constrains_list.append(utilities)
            ### next state
            states = next_states

        cum_returns_list = mc_env.compute_returns(rewards_list,constrains_list,lam)
        # pdb.set_trace()
        for i in range(horizon):
            states = states_list[i]
            actions = actions_list[i]
            probs = mc_env.theta_to_policy(theta)[states].clone()
            log_probs = torch.log(probs)
            log_probs[torch.arange(num_MC_sims).to(device),actions] -= 1
            d_theta = log_probs * cum_returns_list[i].unsqueeze(-1)
            grad_sum[states,actions] += d_theta[torch.arange(num_MC_sims).to(device),actions]/num_MC_sims/horizon

        grad = grad_sum / torch.norm(grad_sum)
        theta -= alpha * grad
        lam = torch.maximum(lam - beta * (cum_constrains.mean() - b).mean(), torch.tensor(0.0).to(device))

        if k % record_interval == 0:
            avg_reward,avg_constrain = mc_env.ell_approx(theta,num_MC_sims,horizon)
            gnpg_gap.append((ell_star-avg_reward).item())
            gnpg_violation.append((b-avg_constrain).item())

    return gnpg_gap,gnpg_violation

if __name__ == '__main__':
    np.random.seed(10)
    torch.manual_seed(10)

    device = 'cuda'

    num_state = 10
    num_action = 5
    policy_type = 'softmax'
    gamma = 0.9

    mc_env = MC_Env(num_state,num_action,policy_type,gamma=gamma,device=device)

    ell_star = mc_env.get_optimum()

    num_MC_sims = 1000
    horizon = 50
    '''
    Policy gradient in action
    '''
    num_iter = 3000
    record_interval = 1
    # Parameters for line search
    alpha = 0.01
    beta = 0.01
    b = 5.

    policy_net = PolicyNetwork().to(device)

    gnpg_gap, gnpg_violation = gnpg_mc(mc_env,num_state,num_action,gamma,num_iter=0,num_MC_sims=num_MC_sims,horizon=horizon,alpha=alpha,beta=beta,record_interval=record_interval,b=b)
    pg_gap, pg_violation = pg_mc(mc_env,policy_net,num_state,num_action,gamma,num_iter=num_iter,num_MC_sims=num_MC_sims,horizon=horizon,alpha=alpha,beta=beta,record_interval=record_interval,b=b)
    npg_gap, npg_violation = npg_mc(mc_env,num_state,num_action,gamma,num_iter=0,num_MC_sims=num_MC_sims,horizon=horizon,alpha=alpha,beta=beta,record_interval=record_interval,b=b)

    mc_env.plot_curve([pg_gap,gnpg_gap,npg_gap],
                      [pg_violation,gnpg_violation,npg_violation],['gnpg','pg','npg'],method='softmax')
