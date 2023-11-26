import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import pdb
import time 

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
'''
Random positive rewards
'''
reward = np.random.uniform(0,1,size=(num_state*num_action))
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

def grad_new(qvals,prob,d_pi):
    ### grad is (s,a)
    qvals_reshaped = qvals.reshape(num_state, num_action)
    prob_reshaped = prob.reshape(num_state, num_action)
    # grad_list = []
    # for state in range(0,num_state):
    #     ### grad_pi is diag(pi(s)) - pi(s)*pi(s)^T
    #     grad_pi = np.diag(prob_reshaped[state]) - prob_reshaped[[state]].T @ prob_reshaped[[state]]
    #     ### state_grad = grad_pi * q(s)
    #     state_grad_vec = grad_pi @ qvals_reshaped[[state]].T
    #     grad_list.append(d_pi[state]*state_grad_vec.T)
    # grad = np.concatenate(grad_list,axis=0)

    # This is a vectorized form of np.diag(prob_reshaped[state]) - np.outer(prob_reshaped[state], prob_reshaped[state])
    grad_pi = np.eye(num_action)[np.newaxis, :, :] * prob_reshaped[:, np.newaxis, :] - prob_reshaped[:, :, np.newaxis] * prob_reshaped[:, np.newaxis, :]
    # Compute state_grad for all states
    # Broadcasting is used to vectorize the computation
    state_grads = np.matmul(grad_pi, qvals_reshaped[:,:,np.newaxis])

    # Apply d_pi weighting and reshape
    weighted_state_grads = d_pi[:, np.newaxis, np.newaxis] * state_grads
    grad = weighted_state_grads.flatten()
    # Sum over states and squeeze to remove singleton dimension
    return grad

def grad_new_vectorized_v2(qvals, prob, d_pi):
    # Reshape qvals and prob for easier manipulation
    qvals_reshaped = qvals.reshape(num_state, num_action)
    prob_reshaped = prob.reshape(num_state, num_action)

    # Compute grad_pi for all states at once
    # This is a vectorized form of np.diag(prob_reshaped[state]) - np.outer(prob_reshaped[state], prob_reshaped[state])
    grad_pi = np.eye(num_action)[np.newaxis, :, :] * prob_reshaped[:, np.newaxis, :] - prob_reshaped[:, :, np.newaxis] * prob_reshaped[:, np.newaxis, :]
    # Compute state_grad for all states
    # Broadcasting is used to vectorize the computation
    state_grads = np.matmul(grad_pi, qvals_reshaped[:,:,np.newaxis])

    # Apply d_pi weighting and reshape
    weighted_state_grads = d_pi[:, np.newaxis, np.newaxis] * state_grads

    # Sum over states and squeeze to remove singleton dimension
    grad = np.sum(weighted_state_grads, axis=0).squeeze()

    return grad

def compute_policy_gradient(qvals, prob, d_pi,):
    grad = np.zeros_like(prob)  # Initialize gradient

    for i in range(num_state):
        for j in range(num_action):
            # Calculate the gradient for each state-action pair
            action_index = i * num_state + j
            grad[action_index] = d_pi[action_index] * qvals[action_index] * (1 - prob[action_index])
    return grad

'''
The overall reward function \ell(\theta). 
'''
### for each state, summing over all the actions
### then multiply with initial state, get the overall reward for this policy
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
def theta_to_policy(theta,num_state,num_action):
    # pdb.set_trace()
    theta_reshaped = theta.reshape((num_state, num_action))
    # Compute the exponential of each element
    exp_theta = np.exp(theta_reshaped)
    # Normalize each row to get probabilities
    prob = exp_theta / np.sum(exp_theta, axis=1, keepdims=True)
    # Flatten the array back to 1D if needed
    prob = prob.flatten()

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
    prob = theta_to_policy(theta,num_state,num_action)
    Pi = get_Pi(prob,num_state,num_action)
    mat = np.identity(num_state*num_action) - gamma*np.matmul(prob_transition,Pi)
    qvals = np.dot(np.linalg.inv(mat),reward)
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

'''
Policy iteration function
'''
def policy_iter(q_vals,num_state,num_action):
    new_policy = np.zeros((num_state,num_action))
    max_idx = np.argmax(q_vals.reshape(num_state,num_action),axis=1)
    new_policy[np.arange(num_state),max_idx] = 1
    new_policy = new_policy.flatten()
    return new_policy

curr_policy = np.random.uniform(0,1,size=(num_state*num_action))
new_policy = init_policy
print('Starting policy',init_policy)

### use policy iteration to find out the optimal one
while np.count_nonzero(curr_policy - new_policy) > 0:
    curr_policy = new_policy
    Pi = get_Pi(curr_policy,num_state,num_action)
    mat = np.identity(num_state*num_action) - gamma*np.matmul(prob_transition,Pi)
    q_vals = np.dot(np.linalg.inv(mat),reward)
    new_policy = policy_iter(q_vals,num_state,num_action)
    
print('Final policy',new_policy)

ell_star = ell(q_vals,new_policy,rho)
print('Optimal Reward',ell_star)

'''
Policy gradient in action
'''
num_iter = 10000
stepsize = 0.01
# Parameters for line search
alpha = 1
beta = 0.7
theta = np.random.uniform(0,1,size=num_state*num_action) ### information for policy compute
gap = []
start_time = time.time()
for k in range(num_iter):
    ### prob is the probability for the policy
    prob = theta_to_policy(theta,num_state,num_action)

    Pi = get_Pi(prob,num_state,num_action)
    mat = np.identity(num_state*num_action) - gamma*np.matmul(prob_transition,Pi)
    qvals = np.dot(np.linalg.inv(mat),reward)

    ### p_theta is the probability from state s to state s'
    P_theta = np.matmul(Pi,prob_transition)
    d_pi = (1-gamma)*np.dot(np.transpose((np.linalg.inv(np.identity(num_state) - gamma*P_theta))),rho)

    gradient = grad_new(qvals,prob,d_pi) / (1-gamma)
    # step = find_step(theta,gradient,alpha,beta)
    step = alpha
    theta += step*gradient
    if k % 50 == 0:
        avg_reward = ell(qvals,prob,rho)
        print('Optimality gap',ell_star - avg_reward)
        gap.append(ell_star - avg_reward)


## Saving the 'Optmality gap array'. This can be loaded to make the figure again.
np.save('Softmax.npy',gap)

f = plt.figure()
plt.plot(np.array(gap))
plt.title('Optimality gap during training')
plt.ylabel('Gap')
plt.xlabel('Iteration number/1000')
f.savefig("Fig_Softmax_Policy.jpg")
f.savefig("Fig_Softmax_Policy.pdf")