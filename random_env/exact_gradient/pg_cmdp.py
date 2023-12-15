import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import pdb
import time 

from env import Random_Env
from agent import Agent

from tqdm import tqdm

if __name__ == '__main__':
    num_state = 20
    num_action = 10
    gamma = 0.9
    np.random.seed(10) 
    policy_type = 'softmax'
    rm_env = Random_Env(num_state=num_state,num_action=num_action,gamma=gamma,policy_type=policy_type)

    PG_agent = Agent(num_state=num_state,num_action=num_action,type='pg',policy_type=policy_type,gamma=gamma)
    NPG_agent = Agent(num_state=num_state,num_action=num_action,type='npg',policy_type=policy_type,gamma=gamma)
    GNPG_agent = Agent(num_state=num_state,num_action=num_action,type='gnpg',policy_type=policy_type,gamma=gamma)

    re_type = 'reward'

    ell_star = rm_env.get_optimum(type=re_type)

    num_iter = 5000
    record_interval = 1
    # alpha is the lr for theta
    alpha = 0.05
    # beta is the lr for lamda
    beta = 0.05
    constrain_threshold = 6

    # theta = np.random.uniform(0,1,size=num_state*num_action) ### information for policy compute

    start_time = time.time()
    agent_list = [PG_agent,NPG_agent,GNPG_agent]
    # agent_list = [GNPG_agent]
    for agent in agent_list:
        for k in tqdm(range(num_iter)):
            ### prob is the probability for the policy
            # print(agent.theta.sum())
            prob = rm_env.theta_to_policy(agent.theta)

            qvals,q_constrain_vals = rm_env.get_q(prob)
            gradient = agent.compute_grad(prob,qvals,q_constrain_vals,type=re_type)
            agent.theta += alpha*gradient
            ### constrain_violation
            V_constrain_vals = (np.sum((q_constrain_vals * prob).reshape((num_state,num_action)),axis=1) * rm_env.rho).sum()
            violation = V_constrain_vals - constrain_threshold
            '''
            have problems
            '''
            agent.lam = np.maximum(agent.lam-(beta*violation).mean(),0)

            if re_type == 'all':
                avg_reward = rm_env.ell(qvals+q_constrain_vals,prob)
            else:
                avg_reward = rm_env.ell(qvals,prob)
            agent.reward_list.append(ell_star-avg_reward)
            agent.violation_list.append(-violation.mean())

    rm_env.plot_curve([agent.reward_list for agent in agent_list],
                      [agent.violation_list for agent in agent_list],[agent.type for agent in agent_list],method=policy_type)
