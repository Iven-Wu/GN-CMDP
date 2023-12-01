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

constrain = np.random.uniform(0,1,size=(num_state*num_action))

'''
Start state distribution
'''
rho = np.ones(num_state)/num_state

'''
Input: probability vector, state, action
Output: \nabla_{\theta} \pi_{\theta}(s,a)

States go from 0 to num_state-1 and actons from 0 to num_action-1
'''

def grad_new(qvals,prob,d_pi):
    ### grad is (s,a)
    qvals_reshaped = qvals.reshape(num_state, num_action)
    prob_reshaped = prob.reshape(num_state, num_action)

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

'''
The overall reward function \ell(\theta). 
'''
### for each state, summing over all the actions
### then multiply with initial state, get the overall reward for this policy
### prob is the probability when agent in a state doing action a prob[s,a]
### qvals is the q value function
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
num_iter = 100
record_interval = 1
stepsize = 0.1
# alpha is the lr for theta
alpha = 0.2
# beta is the lr for lamda
beta = 0.1
lam = 0.5
constrain_threshold = 4.

theta = np.random.uniform(0,1,size=num_state*num_action) ### information for policy compute
gap = []
violation_list = []
start_time = time.time()
for k in range(num_iter):
    ### prob is the probability for the policy
    prob = theta_to_policy(theta,num_state,num_action)

    Pi = get_Pi(prob,num_state,num_action)
    mat = np.identity(num_state*num_action) - gamma*np.matmul(prob_transition,Pi)
    qvals = np.dot(np.linalg.inv(mat),reward)

    q_constrain_vals = np.dot(np.linalg.inv(mat),constrain)

    ### p_theta is the probability from state s to state s'
    P_theta = np.matmul(Pi,prob_transition)
    d_pi = (1-gamma)*np.dot(np.transpose((np.linalg.inv(np.identity(num_state) - gamma*P_theta))),rho)

    gradient_q = grad_new(qvals,prob,d_pi) / (1-gamma)
    gradient_cons_q = grad_new(q_constrain_vals,prob,d_pi)/(1-gamma)
    
    gradient = gradient_q+lam*gradient_cons_q

    gradient = gradient/np.linalg.norm(gradient)
    theta += alpha*gradient

    ### constrain_violation
    violation = q_constrain_vals - constrain_threshold
    
    violation_list.append(violation.mean())
    lam = np.maximum(lam-beta*violation,0)

    if k % record_interval == 0:
        avg_reward = ell(qvals,prob,rho)
        # print('Optimality gap',ell_star - avg_reward)
        gap.append(ell_star - avg_reward)


## Saving the 'Optmality gap array'. This can be loaded to make the figure again.

f = plt.figure()
plt.plot(np.array(gap))
plt.title('Optimality gap during training')
plt.ylabel('Gap')
plt.xlabel('Iteration number/{}'.format(record_interval))
f.savefig("figs/Fig_Policy_GNPG_CMDP.jpg")
f.clf()

f = plt.figure()
plt.plot(np.array(violation_list))
plt.title('Violation during training')
plt.ylabel('Constrain Violation')
plt.xlabel('Iteration number/{}'.format(record_interval))
f.savefig("figs/Fig_Violation_GNPG_CMDP.jpg")
f.clf()

