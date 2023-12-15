import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import pdb
import time 
import os
from tqdm import tqdm

from mc_env_acc import MC_Env


def compute_returns(rewards, utility_list, gamma):
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    constrains = []
    U = 0
    for utility in reversed(utility_list):
        U = utility + gamma * U
        constrains.insert(0, U)
    return returns, constrains


if __name__ == '__main__':
    np.random.seed(10)

    num_state = 10
    num_action = 5
    policy_type = 'softmax'
    gamma = 0.9
    lam = 0.

    mc_env = MC_Env(num_state,num_action,policy_type,gamma=gamma)

    ell_star = mc_env.get_optimum()

    num_MC_sims = 500
    horizon = 20
    '''
    Policy gradient in action
    '''
    num_iter = 1000
    record_interval = 1
    # Parameters for line search
    alpha = 0.2
    beta = 0.2
    b = 4.

    theta = np.random.uniform(0,1,size=(num_state,num_action)) ### information for policy compute
    # theta = np.zeros((num_state,num_action))
    gnpg_gap = []
    gnpg_violation = []
    start_time = time.time()

    for k in tqdm(range(num_iter)):
        grad_sum = np.zeros_like(theta)
        cum_rewards = np.zeros(num_MC_sims)
        cum_constrains = np.zeros(num_MC_sims)
        states = np.random.choice(range(num_state),num_MC_sims)

        for i in range(horizon):
            actions = mc_env.get_action(theta,states)

            next_states, rewards,utilities = mc_env.env_step(states,actions)
            ### compute gradient
            d_softmax = mc_env.theta_to_policy(theta)[states].copy()
            
            d_softmax[actions] -= 1
            # cum_rewards = rewards + gamma * cum_rewards
            cum_rewards = cum_rewards + gamma**i * rewards
            # cum_constrains = utilities + gamma * cum_constrains
            cum_constrains = cum_constrains + gamma**i * utilities

            final_rewards=  cum_rewards+lam*cum_constrains
            d_theta = d_softmax * final_rewards.reshape(-1,1)
            grad_sum[states] += np.mean(d_theta,axis=0)
            ### next state
            states = next_states
        # grad_avg = grad_avg/np.linalg.norm(grad_avg,axis=1,keepdims=True)
        # grad_avg = grad_avg/np.linalg.norm(grad_avg)
        grad = grad_sum/np.linalg.norm(grad_sum)
        theta -= alpha * grad
        lam = np.maximum( lam-beta*(cum_constrains.mean()-b).mean() ,0)

        if k % record_interval == 0:
            avg_reward,avg_constrain = mc_env.ell_approx(theta,num_MC_sims,horizon)
            gnpg_gap.append(ell_star-avg_reward)
            gnpg_violation.append(b-avg_constrain)


    theta = np.zeros((num_state,num_action))
    pg_gap = []
    pg_violation = []
    start_time = time.time()
    for k in tqdm(range(0)):
        grad_sum = np.zeros_like(theta)
        for _ in range(num_MC_sims):
            state = np.random.choice(range(num_state))
            reward_list = []
            action_list = []
            state_list = []
            util_list = []
            for i in range(horizon):
                action = mc_env.get_action(theta,state)
                next_state, reward,utility = mc_env.env_step(state,action)
                state_list.append(state)
                action_list.append(action)
                reward_list.append(reward)
                util_list.append(utility)
                state = next_state
            final_rewards, constrains = compute_returns(reward_list,util_list,gamma)
            states = np.array(state_list)
            for t, state in enumerate(states):
                probs = mc_env.theta_to_policy(theta)[state]
                action = action_list[t]
                d_softmax = probs.copy()
                d_softmax[action] -= 1
                d_theta = np.outer(d_softmax, final_rewards[t]+lam*constrains[t])
                grad_sum[state] += d_theta.reshape(-1)
        grad_avg = grad_sum/num_MC_sims
        # grad_avg = grad_avg/np.linalg.norm(grad_avg)
        theta -= alpha * grad_avg
        # pdb.set_trace()
        lam = np.maximum( lam-beta*(constrains[0]-b).mean() ,0)

        if k % record_interval == 0:
            avg_reward,avg_constrain = mc_env.ell_approx(theta,num_MC_sims,horizon)
            pg_gap.append(ell_star-avg_reward)
            pg_violation.append(b-avg_constrain)

    mc_env.plot_curve([pg_gap,gnpg_gap],
                      [pg_violation,gnpg_violation],['gnpg','pg'],method='softmax')


    # ## Saving the 'Optmality gap array'. This can be loaded to make the figure again.
    # f = plt.figure()
    # plt.plot(np.array(gap))
    # plt.title('Optimality gap during training')
    # plt.ylabel('Gap')
    # plt.xlabel('Iteration number/{}'.format(record_interval))
    # f.savefig("figs/MC_GNPG_CMDP_reward.jpg")

    # f = plt.figure()
    # plt.plot(np.array(violation))
    # plt.title('Optimality gap during violation')
    # plt.ylabel('Violation')
    # plt.xlabel('Iteration number/{}'.format(record_interval))
    # f.savefig("figs/MC_GNPG_CMDP_violation.jpg")