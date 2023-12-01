# Assume we have an environment with state space S and action space A
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import pdb
import time 
from tqdm import tqdm

## Random Seed
np.random.seed(10) 
## Problem Setup

num_state, num_action = 20, 10

raw_transition = np.random.uniform(0,1,size=(num_state*num_action,num_state))
prob_transition = raw_transition/raw_transition.sum(axis=1,keepdims=1)

#p(s'| s, a)
prob_transition = prob_transition.reshape((num_state,num_action,num_state))

r = np.random.uniform(0,1,size=((num_state,num_action)))
r = r.reshape((num_state, num_action))
g = np.random.uniform(0,1,size=((num_state,num_action)))
g = g.reshape((num_state, num_action))
b = 3.0

gamma = 0.8

rho = np.ones(num_state)/num_state

num_iter = 300
num_episodes = 50

def get_Pi(prob):
    #print(prob)
    prob_reshaped = prob[:, :, np.newaxis]
    # Initialize Pi as a zero array
    Pi = np.sum(prob_transition * prob_reshaped, axis=1) 
    return Pi

def policy_iter(vals,num_state,num_action):
    q_vals = r + gamma * np.sum(prob_transition * vals[np.newaxis, np.newaxis, :], axis=2)

    new_policy = np.zeros((num_state,num_action))
    
    max_idx = np.argmax(q_vals,axis=1)
    
    new_policy[np.arange(num_state),max_idx] = 1
    #new_policy = new_policy.flatten()
    return new_policy

def ell(vals,rho):
    # Compute ell using dot product
    ell = np.dot(vals, rho)
    return ell

raw_vec = np.random.uniform(0,1,size=(num_state,num_action))
prob_vec = raw_vec/raw_vec.sum(axis=1,keepdims=1)
init_policy = prob_vec.reshape((num_state, num_action))

curr_policy = np.random.uniform(0,1,size=(num_state*num_action)).reshape((num_state, num_action))
new_policy = init_policy
print('Starting policy',init_policy)

### use policy iteration to find out the optimal one
while np.count_nonzero(curr_policy.flatten() - new_policy.flatten()) > 0:
    curr_policy = new_policy
    print(curr_policy)

    Pi = get_Pi(curr_policy)
    mat = np.identity(num_state) - gamma*Pi

    R_pi = np.sum(r * curr_policy, axis=1)

    vals = np.dot(np.linalg.inv(mat),R_pi)
    #q_vals = np.dot(np.linalg.inv(mat),reward)
    new_policy = policy_iter(vals,num_state,num_action)
    
print('Final policy',new_policy)

ell_star = ell(vals,rho)
print('Optimal Reward',ell_star)



def theta_to_policy(theta):
    # pdb.set_trace()
    theta_reshaped = theta.reshape((num_state, num_action))
    # Compute the exponential of each element
    print(theta_reshaped)
    exp_theta = np.exp(theta_reshaped)
    # Normalize each row to get probabilities
    prob = exp_theta / np.sum(exp_theta, axis=1, keepdims=True)
    return prob

def get_action(prob, state):
    action = np.random.choice(range(num_action),p=prob[state])
    return action

def env_step(state,action):
    prob_next = prob_transition[state,action]
    next_state = np.random.choice(range(num_state),p=prob_next)
    reward = r[state,action]
    utility = g[state,action]
    return next_state,reward,utility


lam = 0
theta = np.zeros((num_state,num_action))
value_list = []
gap = []
violation_list = []
for t in tqdm(range(num_iter)):
    values = np.zeros(num_state)
    qvals = np.zeros((num_state,num_action))

    # geom_values = np.random.geometric(1-gamma, size=(num_episodes, num_state))
    # geom_values_g = np.random.geometric(1-gamma, size=(num_episodes, num_state))
    advantage = qvals - values[:, np.newaxis]
    values_g = np.zeros(num_state)
    prob = theta_to_policy(theta)

    for episode in range(num_episodes):
        values_temp = np.zeros(num_state)
        qvals_temp = np.zeros((num_state,num_action))

        for s in range(num_state):
            next_state_v = s
            next_state_q = s
            for i in range(np.random.geometric(1-gamma)):
                action = get_action(prob,next_state_v)
                next_state_v, reward, utility = env_step(next_state_v, action)
                values_temp[s] += (reward + lam*utility)

            for a in range(num_action):
                next_action = a
                for i in range(np.random.geometric(1-gamma)):
                    qvals_temp[s,a] += r[next_state_q, next_action] + lam*g[next_state_q, next_action]
                    next_state_q, reward, utility = env_step(next_state_q, next_action)
                    next_action = get_action(prob,next_state_q)

        values += 1.0/num_episodes*values_temp
        qvals += 1.0/num_episodes*qvals_temp


        s_vg = np.random.choice(np.arange(num_state), p=rho)
        a_vg = get_action(prob,s_vg)
        next_state_g = s_vg
        next_action_g = a_vg

        values_g_temp = np.zeros(num_state)
        for i in range(np.random.geometric(1-gamma)):
            values_g_temp[next_state_g] += g[next_state_g, next_action_g]
            next_state_g, reward, utility = env_step(next_state_g, next_action_g)
            next_action_g = get_action(prob,next_state_g)
        
        values_g += 1.0/num_episodes*values_g_temp

    advantage = qvals - values[:, np.newaxis]
    theta = theta + (1.0/(1-gamma))*advantage
    lam = max(lam-1.0*values_g)

    pi_reshaped = prob[:, :, np.newaxis]  # Reshape to (num_state, num_action, 1)

    # Element-wise multiplication of transition probabilities and reshaped policy
    P_pi = np.sum(prob_transition * pi_reshaped, axis=1)  # Sum along the action axis to get P_pi
    mat = np.identity(num_state) - gamma*P_pi
    R_pi = np.sum(r * prob, axis=1)
    avg_reward = ell(np.dot(np.linalg.inv(mat),R_pi), rho)

    gap.append(ell_star - avg_reward)

f = plt.figure()
plt.plot(np.array(gap))
plt.title('Optimality gap during training')
plt.ylabel('Gap')
plt.xlabel('Iteration number/{}'.format(1))
f.savefig("figs/Fig_Sample_based_NPG_CMDP.jpg")
f.clf()
