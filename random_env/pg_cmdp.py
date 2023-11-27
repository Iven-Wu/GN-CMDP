import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import pdb
import time 

from env import Random_Env
from agent import Agent



if __name__ == '__main__':
    np.random.seed(10) 
    rm_env = Random_Env(num_state=10,num_action=5)
    num_state = rm_env.num_state
    num_action = rm_env.num_action
    gamma = rm_env.gamma

    PG_agent = Agent(type='pg')
    NPG_agent = Agent(type='npg')
    GNPG_agent = Agent(type='gnpg')

    num_iter = 1000
    record_interval = 1
    # alpha is the lr for theta
    alpha = 0.2
    # beta is the lr for lamda
    beta = 0.1
    constrain_threshold = 4.

    # theta = np.random.uniform(0,1,size=num_state*num_action) ### information for policy compute

    start_time = time.time()
    agent_list = PG_agent,NPG_agent,GNPG_agent
    for agent in agent_list:
        for k in range(num_iter):
            ### prob is the probability for the policy
            prob = rm_env.theta_to_policy(agent.theta)
            qvals,q_constrain_vals = rm_env.get_q(prob)
            gradient = agent.compute_grad(prob,qvals,q_constrain_vals)
            agent.theta += alpha*gradient
            ### constrain_violation
            violation = q_constrain_vals - constrain_threshold
            agent.lam = np.maximum(agent.lam-beta*violation,0)

            avg_reward = rm_env.ell(qvals,prob)
            agent.reward_list.append(avg_reward)
            agent.violation_list.append(violation.mean())
    

    rm_env.plot_curve([agent.reward_list for agent in agent_list],
                      [agent.violation_list for agent in agent_list],[agent.type for agent in agent_list],method='')
