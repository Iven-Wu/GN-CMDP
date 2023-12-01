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
b = 3

gamma = 0.8

rho = np.ones(num_state)/num_state

alpha = 0.2
beta = 0.1

num_iter = 20
num_episodes = 50

def theta_to_policy(theta):
    # pdb.set_trace()
    theta_reshaped = theta.reshape((num_state, num_action))
    # Compute the exponential of each element
    exp_theta = np.exp(theta_reshaped)
    # Normalize each row to get probabilities
    prob = exp_theta / np.sum(exp_theta, axis=1, keepdims=True)
    return prob
    # Flatten the array back to 1D if needed
    # prob = prob.flatten()

def get_action(theta,state):
    #print(theta)
    prob = theta_to_policy(theta)
    #print(prob)
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
for t in tqdm(range(num_iter)):
    values = np.zeros(num_state)
    qvals = np.zeros((num_state,num_action))

    for episode in range(num_episodes):
        values_temp = np.zeros(num_state)
        qvals_temp = np.zeros((num_state,num_action))
        for s in range(num_state):
            next_state = s
            for i in range(np.random.geometric(1-gamma)):
                action = get_action(theta,next_state)
                next_state, reward, utility = env_step(next_state, action)
                values_temp[s] += (reward + lam*utility)

        for s in range(num_state):
            next_state = s
            for a in range(num_action):
                next_action = a
                for i in range(np.random.geometric(1-gamma)):
                    qvals_temp[s,a] += r[next_state, next_action] + lam*g[next_state, next_action]
                    next_state, reward, utility = env_step(next_state, next_action)
                    next_action = get_action(theta,next_state)
        
        values += 1.0/num_episodes*values_temp
        qvals += 1.0/num_episodes*qvals_temp

    advantage = qvals - values[:, np.newaxis]
    values_g = np.zeros(num_state)

    for episode in range(num_episodes):
        s = np.random.choice(np.arange(num_state), p=rho)
        a = get_action(theta,s)
        next_state = s
        next_action = a

        values_g_temp = np.zeros(num_state)
        for i in range(np.random.geometric(1-gamma)):
            values_g_temp[next_state] += g[next_state, next_action]
            next_state, reward, utility = env_step(next_state, next_action)
            next_action = get_action(theta,next_state)
        
        values_g += 1.0/num_episodes*values_g_temp

    theta = theta + (1.0/(1-gamma))*advantage
    lam = max(lam-1.0*values_g)

    pi = theta_to_policy(theta)
    pi_reshaped = pi[:, :, np.newaxis]  # Reshape to (num_state, num_action, 1)

    # Element-wise multiplication of transition probabilities and reshaped policy
    P_pi = np.sum(prob_transition * pi_reshaped, axis=1)  # Sum along the action axis to get P_pi
    mat = np.identity(num_state) - gamma*P_pi
    R_pi = np.sum(r * pi, axis=1)
    value_list.append(np.dot(np.linalg.inv(mat),R_pi).mean())

print(value_list)
f = plt.figure()
plt.plot(np.array(value_list))
plt.title('Value during training')
plt.ylabel('Value')
plt.xlabel('Iteration number/{}'.format(1))
f.savefig("figs/Fig_sample_based_npg_CMDP.jpg")
f.clf()

