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
gamma = 0.8
num_state, num_action =  10,5
lam = 0
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
    # pdb.set_trace()
    raw_prob = theta_to_policy(theta)
    probs = raw_prob[state]
    # print(probs.min())
    actions = np.stack([np.random.choice(range(num_action),p=prob) for prob in probs])
    return actions

def env_step(states,actions):
    # prob = theta_to_policy(theta,num_state,num_action)
    # pdb.set_trace()
    probs_next = prob_transition[states,actions]
    # pdb.set_trace()
    next_states = np.stack([np.random.choice(range(num_state),p=prob_next) for prob_next in probs_next] )
    rewards = reward_mat[states,actions]
    utilities = constrain_mat[states,actions]
    return next_states,rewards,utilities

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

num_MC_sims = 25
horizon = 15
b = 4.
def ell_approx(theta):
    V_r_rho, V_g_rho = 0,0
    # for _ in range(num_MC_sims):
    init_states = np.stack([np.random.choice(range(len(rho)),p=rho) for _ in range(num_MC_sims)])
    states = init_states
    cum_rewards = 0
    cum_constrains = 0
    for i in range(horizon):
        actions = get_action(theta,states)
        next_states, rewards, utilities = env_step(states,actions)
        cum_rewards = gamma*cum_rewards + rewards
        cum_constrains = gamma*cum_constrains + utilities
        states = next_states
    V_r_rho = cum_rewards.mean()
    V_g_rho = cum_constrains.mean()
    return V_r_rho,V_g_rho

'''
Policy gradient in action
'''
num_iter = 300
record_interval = 1
stepsize = 0.01
# Parameters for line search
alpha = 0.1
beta = 0.2
b = 4
# horizon = 20
theta = np.random.uniform(0,1,size=(num_state,num_action)) ### information for policy compute
gap = []
constrain = []
start_time = time.time()

values = np.zeros(num_state)
q_values = np.zeros((num_state,num_action))
for k in tqdm(range(num_iter)):
    ### prob is the probability for the policy
    # prob = theta_to_policy(theta,num_state,num_action)
    # while not done:
    state = 0
    reward_list = []
    action_list = []
    state_list = []
    util_list = []

    for init_state in range(num_state):
        cum_rewards = np.zeros(num_MC_sims)
        cum_constrains = np.zeros(num_MC_sims)
        states = np.array([init_state]*num_MC_sims)
        for i in range(horizon):
            actions = get_action(theta,states)
            next_states, rewards, utilities = env_step(states,actions)
            cum_rewards = gamma*cum_rewards + rewards
            cum_constrains = gamma*cum_constrains + utilities
            states = next_states
        state_reward_avg, state_constrain_avg = cum_rewards.mean(), cum_constrains.mean()
        values[init_state] = state_reward_avg + lam * state_constrain_avg

    for init_state in range(num_action):
        for init_action in range(num_action):
            cum_rewards = np.zeros(num_MC_sims)
            cum_constrains = np.zeros(num_MC_sims)
            states = np.array([init_state]*num_MC_sims)
            actions = np.array([init_action]*num_MC_sims)
            for i in range(horizon):
                if i>0:
                    actions = get_action(theta,states)
                next_states, rewards,utilities = env_step(states,actions)
                cum_rewards = gamma*cum_rewards + rewards
                cum_constrains = gamma*cum_constrains + utilities
                states = next_states
            q_state_action_avg, g_state_action_avg = cum_rewards.mean(),cum_constrains.mean()
            q_values[init_state,init_action] = q_state_action_avg + lam*g_state_action_avg

    advantage_mat = q_values - values[:,np.newaxis]
    
    V_g_rho = 0
    cum_rewards = 0
    cum_constrains = 0
    init_states = np.stack([np.random.choice(range(len(rho)),p=rho) for _ in range(num_MC_sims)])
    states = init_states
    for i in range(horizon):
        actions = get_action(theta,states)
        next_states, rewards, utilities = env_step(states,actions)
        # cum_rewards = gamma*cum_rewards + rewards
        cum_constrains += utilities
        states = next_states
    V_g_rho = cum_constrains.mean()
    ### update with gradient
    theta += alpha*advantage_mat/(1-gamma)
    theta = theta_to_policy(theta)
    # pdb.set_trace()
    lam = np.maximum(lam-beta*(V_g_rho-b),0)

    if k % record_interval == 0:
        avg_reward,avg_constrain = ell_approx(theta)
        gap.append(ell_star-avg_reward)
        constrain.append(b-avg_constrain)


## Saving the 'Optmality gap array'. This can be loaded to make the figure again.
f = plt.figure()
plt.plot(np.array(gap),label='gap')
plt.title('Optimality gap during training')
plt.ylabel('Gap')
plt.xlabel('Iteration number/{}'.format(record_interval))
f.savefig("figs/MC_NPG_CMDP_reward.jpg")

f = plt.figure()
plt.plot(np.array(constrain),label='violation')
plt.title('Constrain during training')
plt.ylabel('Violation')
plt.xlabel('Iteration number/{}'.format(record_interval))
f.savefig("figs/MC_NPG_CMDP_constrain.jpg")