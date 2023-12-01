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
lam = 0.5
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

constrain_mat = np.random.uniform(0,1,size=((num_state,num_action)))
'''
Start state distribution
'''
rho = np.ones(num_state)/num_state

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
# print('Starting policy',init_policy)

# # ### use policy iteration to find out the optimal one
while np.count_nonzero(curr_policy - new_policy) > 0:
    curr_policy = new_policy
    Pi = get_Pi(curr_policy,num_state,num_action)
    mat = np.identity(num_state*num_action) - gamma*np.matmul(prob_transition.reshape((num_state*num_action,num_state)),Pi)
    q_vals = np.dot(np.linalg.inv(mat),reward_mat.reshape((num_state*num_action)))
    new_policy = policy_iter(q_vals,num_state,num_action)
    
print('Final policy',new_policy)

ell_star = ell(q_vals,new_policy,rho)
print('Optimal reward_mat',ell_star)

def get_action(theta,state):
    # prob = theta_to_policy(theta,num_state,num_action)
    # state_trans = np.einsum('ij,ijk->ik',prob,prob_transition)
    # next_state = np.random.choice(range(num_state),p=state_trans[state])
    prob = theta_to_policy(theta)
    action = np.random.choice(range(num_action),p=prob[state])
    return action
    # prob_reshaped = prob[:, :, np.newaxis]

    # # Element-wise multiplication and sum over the action dimension
    # final_probability_a = np.sum(prob_reshaped * prob_transition, axis=1)


    # final_probability = np.zeros((num_state,num_state))
    # for prev_state in range(num_state):
    #     for next_state in range(num_state):
    #         final_probability[prev_state,next_state] = np.sum([prob[prev_state,a]* prob_transition[prev_state,a,next_state] for a in range(num_action)])
    
    # pdb.set_trace()
def env_step(state,action):
    # prob = theta_to_policy(theta,num_state,num_action)
    prob_next = prob_transition[state,action]
    next_state = np.random.choice(range(num_state),p=prob_next)
    reward = reward_mat[state,action]
    utility = constrain_mat[state,action]
    return next_state,reward,utility

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
    total_constrain = 0
    for sim_i in range(num_MC_sims):
        state = 0
        reward_list = []
        action_list = []
        state_list = []
        util_list = []
        for i in range(horizon):
            action = get_action(theta,state)
            next_state, reward, utility = env_step(state,action)
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            util_list.append(utility)

            state = next_state
        final_rewards,constrains = compute_returns(reward_list,util_list,gamma)

        total_reward += final_rewards[0] 
        total_constrain += constrains[0]
    avg_reward = total_reward/num_MC_sims
    avg_constrain = total_constrain/num_MC_sims

    return avg_reward,avg_constrain

'''
Policy gradient in action
'''
num_iter = 300
record_interval = 1
# Parameters for line search
alpha = 1
beta = 1
b = 4

theta = np.random.uniform(0,1,size=(num_state,num_action)) ### information for policy compute
gap = []
start_time = time.time()
num_traj = 5
for k in tqdm(range(num_iter)):
    ### prob is the probability for the policy
    # prob = theta_to_policy(theta,num_state,num_action)
    # while not done:
    grad_sum = np.zeros_like(theta)
    for _ in range(num_traj):
        state = 0
        reward_list = []
        action_list = []
        state_list = []
        util_list = []
        for i in range(horizon):
            action = get_action(theta,state)
            next_state, reward,utility = env_step(state,action)
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            util_list.append(utility)

            state = next_state
        final_rewards, constrains = compute_returns(reward_list,util_list,gamma)

        states = np.array(state_list)
        # probs = theta_to_policy(theta)[states]
        # loss = compute_loss(probs,action_list,final_rewards)
        for t, state in enumerate(states):
            probs = theta_to_policy(theta)[state]
            action = action_list[t]
            d_softmax = probs.copy()
            d_softmax[action] -= 1
            d_theta = np.outer(d_softmax, final_rewards[t]+lam*constrains[t])
            grad_sum[state] += d_theta.reshape(-1)
        grad_avg = grad_sum/num_traj
        grad_avg = grad_avg/np.linalg.norm(grad_avg)
        theta -= alpha * grad_avg
    # pdb.set_trace()
    lam = np.maximum( lam-beta*(constrains[0]-b).mean() ,0)

    if k % record_interval == 0:
        avg_reward,avg_constrain = ell_approx(theta)
        gap.append(ell_star-avg_reward)


## Saving the 'Optmality gap array'. This can be loaded to make the figure again.
f = plt.figure()
plt.plot(np.array(gap))
plt.title('Optimality gap during training')
plt.ylabel('Gap')
plt.xlabel('Iteration number/{}'.format(record_interval))
f.savefig("figs/Fig_GNPG_CMDP.jpg")