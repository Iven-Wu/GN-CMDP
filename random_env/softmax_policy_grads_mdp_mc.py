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
gamma = 0.9
num_state, num_action = 10, 5
'''
Randomly generated probability transition matrix P((s,a) -> s') in R^{|S||A| x |S|}
Each row sums up to one. P[s,a,s']
'''
raw_transition = np.random.uniform(0,1,size=(num_state*num_action,num_state))
prob_transition = raw_transition/raw_transition.sum(axis=1,keepdims=1)
prob_transition = prob_transition.reshape((num_state,num_action,num_state))
# print(prob_transition)
'''
Random positive rewards
'''
reward_mat = np.random.uniform(0,1,size=((num_state,num_action)))
'''
Start state distribution
'''
rho = np.ones(num_state)/num_state

'''
Input: probability vector, state, action
Output: \nabla_{\theta} \pi_{\theta}(s,a)

States go from 0 to num_state-1 and actons from 0 to num_action-1
'''
def grad_state_action(prob,state,action):
    grad = np.zeros(num_state*num_action)
    # grad = np.zeros((num_state,num_action))
    for j in range(0,num_action):
        if j == action:
            grad[state*num_action+j] = prob[num_action*state + action]*(1-prob[num_action*state + j])
        else:
            grad[state*num_action+j] = -prob[num_action*state + action]*prob[num_action*state + j]
    return grad

def grad_state(qvals,prob,state):
    # grad = np.sum([qvals[state*num_action + i]*grad_state_action(prob,state,i) for i in range(0,num_action)],axis=0)
    s_grad_list = []
    for i in range(0,num_action):
        s_grad_list.append(qvals[state*num_action+i] * grad_state_action(prob,state,i))
    # pdb.set_trace()
    grad = np.sum(s_grad_list,axis=0)
    return grad

def grad(qvals,prob,d_pi):
    ### grad is (s,a)
    grad = np.sum([d_pi[i]*grad_state(qvals,prob,i) for i in range(0,num_state)],axis=0)
    return grad


'''
The overall reward_mat function \ell(\theta). 
'''
### for each state, summing over all the actions
### then multiply with initial state, get the overall reward_mat for this policy
### prob is the probability when agent in a state doing action a prob[s,a]
### qvals is the q value function
def ell(qvals,prob,rho):
    # V = np.zeros(num_state)
    # for i in range(num_state):
    #     V[i] = np.sum([qvals[i*num_action + j]*prob[i*num_action + j] for j in range(num_action)])
    # ell = np.dot(V,rho)
    qvals_reshaped = qvals.reshape(num_state, num_action)
    prob_reshaped = prob.reshape(num_state, num_action)

    # Compute V vector using vectorized operations
    V = np.sum(qvals_reshaped * prob_reshaped, axis=1)

    # Compute ell using dot product
    ell = np.dot(V, rho)
    return ell
'''
Input: theta as an array and 
Ouput: array of probabilites corresponding to each state: [\pi_{s_1}(.), ...., \pi_{s_n}(.)]
'''
def theta_to_policy(theta):
    # pdb.set_trace()
    theta_reshaped = theta.reshape((num_state, num_action))
    # Compute the exponential of each element
    exp_theta = np.exp(theta_reshaped)
    # Normalize each row to get probabilities
    prob = exp_theta / np.sum(exp_theta, axis=1, keepdims=True)
    # Flatten the array back to 1D if needed
    # prob = prob.flatten()

    return np.asarray(prob)
'''
Get \Pi_{\pi}((s) -> (s,a)) in R^{|S| x |S||A|} matrix corresponding to the policy \pi using the prob vector
'''
def get_Pi(prob,num_state,num_action):
    prob_reshaped = prob.reshape((num_state, num_action))
    # Initialize Pi as a zero array
    Pi = np.zeros((num_state, num_state * num_action))
    # Create an index array
    col_indices = np.arange(num_state * num_action).reshape((num_state, num_action))
    # Use advanced indexing to efficiently assign values
    Pi[np.arange(num_state)[:, None], col_indices] = prob_reshaped
    return Pi

'''
Backtracking line search
'''

def ell_theta(theta,rho):
    prob = theta_to_policy(theta)
    Pi = get_Pi(prob,num_state,num_action)
    mat = np.identity(num_state*num_action) - gamma*np.matmul(prob_transition,Pi)
    qvals = np.dot(np.linalg.inv(mat),reward_mat)
    return ell(qvals,prob,rho)
    
def find_step(theta,gradient,alpha,beta):
    step = alpha
    while ell_theta(theta - step*gradient,rho) > ell_theta(theta,rho) - (step/2)*np.linalg.norm(gradient):
        step = beta*step
    return step
'''
Policy Iteration to get the optimal policy
'''

raw_vec = np.random.uniform(0,1,size=(num_state,num_action))
prob_vec = raw_vec/raw_vec.sum(axis=1,keepdims=1)
init_policy = prob_vec.flatten()


#Policy iteration function
def policy_iter(q_vals,num_state,num_action):
    new_policy = np.zeros((num_state,num_action))
    max_idx = np.argmax(q_vals.reshape(num_state,num_action),axis=1)
    new_policy[np.arange(num_state),max_idx] = 1
    new_policy = new_policy.flatten()
    return new_policy

curr_policy = np.random.uniform(0,1,size=(num_state*num_action))
new_policy = init_policy

def get_action(theta,state):
    # prob = theta_to_policy(theta,num_state,num_action)
    # state_trans = np.einsum('ij,ijk->ik',prob,prob_transition)
    # next_state = np.random.choice(range(num_state),p=state_trans[state])
    prob = theta_to_policy(theta)
    action = np.random.choice(range(num_action),p=prob[state])
    return action
    
def env_step(state,action):
    # prob = theta_to_policy(theta,num_state,num_action)
    prob_next = prob_transition[state,action]
    next_state = np.random.choice(range(num_state),p=prob_next)
    reward = reward_mat[state,action]
    return next_state,reward

def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns

def compute_loss(probs, episode_actions, returns):
    # Convert episode_actions to a numpy array if it's not already
    episode_actions = np.array(episode_actions)

    # Get the probabilities of the actions taken
    action_probs = probs[np.arange(len(episode_actions)), episode_actions]

    # Compute the log probabilities
    log_probs = np.log(action_probs)

    # Compute the loss
    loss = -np.sum(log_probs * returns)

    return loss

num_MC_sims = 50
horizon = 20
def ell_approx(theta):
    total_reward = 0
    for sim_i in range(num_MC_sims):
        state = 0
        reward_list = []
        action_list = []
        state_list = []
        for i in range(horizon):
            action = get_action(theta,state)
            next_state, reward = env_step(state,action)
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)

            state = next_state
        final_rewards = compute_returns(reward_list,gamma)
        total_reward += final_rewards[0]
    avg_reward = total_reward/num_MC_sims

    return avg_reward

'''
Policy gradient in action
'''
num_iter = 1000
record_interval = 1
stepsize = 0.01
# Parameters for line search
alpha = 0.2
# horizon = 20
theta = np.random.uniform(0,1,size=(num_state,num_action)) ### information for policy compute
gap = []
start_time = time.time()
for k in tqdm(range(num_iter)):
    ### prob is the probability for the policy
    # prob = theta_to_policy(theta,num_state,num_action)
    # while not done:
    state = 0
    reward_list = []
    action_list = []
    state_list = []
    for i in range(horizon):
        action = get_action(theta,state)
        next_state, reward = env_step(state,action)
        state_list.append(state)
        action_list.append(action)
        reward_list.append(reward)

        state = next_state
    final_rewards = compute_returns(reward_list,gamma)

    states = np.array(state_list)
    # probs = theta_to_policy(theta)[states]
    # loss = compute_loss(probs,action_list,final_rewards)
    for t, state in enumerate(states):
        probs = theta_to_policy(theta)[state]
        action = action_list[t]
        d_softmax = probs.copy()
        d_softmax[action] -= 1
        d_theta = np.outer(d_softmax, final_rewards[t])
        # pdb.set_trace()
        theta[state] -= alpha * d_theta.reshape(-1)
    
    if k % record_interval == 0:
        avg_reward = ell_approx(theta)
        gap.append(avg_reward)


## Saving the 'Optmality gap array'. This can be loaded to make the figure again.
f = plt.figure()
plt.plot(np.array(gap))
plt.title('Optimality gap during training')
plt.ylabel('Gap')
plt.xlabel('Iteration number/{}'.format(record_interval))
f.savefig("figs/Fig_Policy_MDP.jpg")