import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import pdb
import time 
import os


class Random_Env():
    def __init__(self,num_state,num_action,policy_type,gamma=0.9):
        np.random.seed(10)
        self.gamma = gamma
        self.num_state, self.num_action = num_state,num_action

        raw_transition = np.random.uniform(0,1,size=(num_state*num_action,num_state))
        self.prob_transition = raw_transition/raw_transition.sum(axis=1,keepdims=1)

        self.reward = np.random.uniform(0,1,size=(num_state*num_action))

        self.constrain = np.random.uniform(0,1,size=(num_state*num_action))

        self.rho = np.ones(num_state)/num_state
        ### softmax or direct
        self.policy_type = policy_type
    
    ### compute the gradient for policy gradient algorithm
    def grad(self,qvals,prob,d_pi):
        ### grad is (s,a)
        qvals_reshaped = qvals.reshape(self.num_state, self.num_action)
        prob_reshaped = prob.reshape(self.num_state, self.num_action)
        # This is a vectorized form of np.diag(prob_reshaped[state]) - np.outer(prob_reshaped[state], prob_reshaped[state])
        if self.policy_type == 'softmax':
            grad_pi = np.eye(self.num_action)[np.newaxis, :, :] * prob_reshaped[:, np.newaxis, :] - prob_reshaped[:, :, np.newaxis] * prob_reshaped[:, np.newaxis, :]
        elif self.policy_type == 'direct':
            grad_pi = np.eye(self.num_action)[np.newaxis,:,:]
        # Compute state_grad for all states
        # Broadcasting is used to vectorize the computation
        state_grads = np.matmul(grad_pi, qvals_reshaped[:,:,np.newaxis])

        # Apply d_pi weighting and reshape
        weighted_state_grads = d_pi[:, np.newaxis, np.newaxis] * state_grads
        grad = weighted_state_grads.flatten()
        # Sum over states and squeeze to remove singleton dimension
        return grad
    
    def ell(self,qvals,prob):
        qvals_reshaped = qvals.reshape(self.num_state, self.num_action)
        prob_reshaped = prob.reshape(self.num_state, self.num_action)
        # pdb.set_trace()
        # Compute V vector using vectorized operations
        V = np.sum(qvals_reshaped * prob_reshaped, axis=1)

        # Compute ell using dot product
        ell = np.dot(V, self.rho)
        return ell

    def project_simplex(self,x):
        """ Take a vector x (with possible nonnegative entries and non-normalized)
            and project it onto the unit simplex.

            mask:   do not project these entries
                    project remaining entries onto lower dimensional simplex
        """

        xsorted = np.sort(x)[:,::-1]
        # entries need to sum up to 1 (unit simplex)
        sum_ = 1.0
        lambda_a = (np.cumsum(xsorted,axis=1) - sum_) / np.arange(1.0, xsorted.shape[1]+1.0)
        # pdb.set_trace()
        p = np.zeros(x.shape)
        for b_i in range(len(xsorted)):
            for i in range(len(lambda_a[0])-1):
                if lambda_a[b_i,i] >= xsorted[b_i,i+1]:
                    astar = i
                    break
            else:
                astar = -1
            p[b_i] = np.maximum(x[b_i]-lambda_a[b_i,astar],  0)
        return p

    def theta_to_policy(self,theta,):
        theta_reshaped = theta.reshape((self.num_state, self.num_action))
        if self.policy_type == 'softmax':
        # Compute the exponential of each element
            exp_theta = np.exp(theta_reshaped)
        elif self.policy_type == 'direct':
            # exp_theta = theta_reshaped
            exp_theta = self.project_simplex(theta_reshaped)
            # exp_theta[exp_theta<0] = 0
            # exp_theta = 1 / (1 + np.exp(-theta_reshaped))
            # pdb.set_trace()
        # Normalize each row to get probabilities
        prob = exp_theta / np.sum(exp_theta, axis=1, keepdims=True)
        # pdb.set_trace()
        # Flatten the array back to 1D if needed
        prob = prob.flatten()

        return np.asarray(prob)
    
    def get_Pi(self,prob):
        prob_reshaped = prob.reshape((self.num_state, self.num_action))
        # Initialize Pi as a zero array
        Pi = np.zeros((self.num_state,self.num_state * self.num_action))
        # Create an index array
        col_indices = np.arange(self.num_state * self.num_action).reshape((self.num_state, self.num_action))      
        # Use advanced indexing to efficiently assign values
        Pi[np.arange(self.num_state)[:, None], col_indices] = prob_reshaped
        return Pi
    
        
    def ell_theta(self,theta):
        prob = self.theta_to_policy(theta,self.num_state,self.num_action)
        Pi = self.get_Pi(prob,self.num_state,self.num_action)
        mat = np.identity(self.num_state*self.num_action) - self.gamma*np.matmul(self.prob_transition,Pi)
        qvals = np.dot(np.linalg.inv(mat),self.reward)
        return self.ell(qvals,prob,self.rho)
    
    def policy_iter(self,q_vals):
        new_policy = np.zeros((self.num_state,self.num_action))
        max_idx = np.argmax(q_vals.reshape(self.num_state,self.num_action),axis=1)
        new_policy[np.arange(self.num_state),max_idx] = 1
        new_policy = new_policy.flatten()
        return new_policy

    def get_q(self,prob):
        Pi = self.get_Pi(prob)
        mat = np.identity(self.num_state*self.num_action) - self.gamma*np.matmul(self.prob_transition,Pi)
        qvals = np.dot(np.linalg.inv(mat),self.reward)

        q_constrain_vals = np.dot(np.linalg.inv(mat),self.constrain)
        return qvals,q_constrain_vals
    
    def get_optimum(self,):
        raw_vec = np.random.uniform(0,1,size=(self.num_state,self.num_action))
        prob_vec = raw_vec/raw_vec.sum(axis=1,keepdims=1)
        init_policy = prob_vec.flatten()

        curr_policy = np.random.uniform(0,1,size=(self.num_state*self.num_action))
        new_policy = init_policy
        print('Starting policy',init_policy)

        ### use policy iteration to find out the optimal one
        while np.count_nonzero(curr_policy - new_policy) > 0:
            curr_policy = new_policy
            Pi = self.get_Pi(curr_policy)
            mat = np.identity(self.num_state*self.num_action) - self.gamma*np.matmul(self.prob_transition,Pi)
            q_vals = np.dot(np.linalg.inv(mat),self.reward)
            new_policy = self.policy_iter(q_vals)
        
        print('Final policy',new_policy)

        ell_star = self.ell(q_vals,new_policy)
        print('Optimal Reward',ell_star)
        return ell_star

    def plot_curve(self,reward,violation,label,record_interval=1,method='pg',out_dir='figs/'):
        os.makedirs(out_dir,exist_ok=True)

        f = plt.figure()
        for i in range(len(reward)):
            plt.plot(np.array(reward[i]),label=label[i])
        plt.legend()
        plt.title('Optimal Gap during training')
        plt.ylabel('Optimal Gap')
        plt.xlabel('Iteration number/{}'.format(record_interval))
        f.savefig("{}/Reward_{}_CMDP.jpg".format(out_dir,method.upper()))
        f.clf()

        f = plt.figure()
        for i in range(len(violation)):
            plt.plot(np.array(violation[i]),label=label[i])
        plt.legend()
        plt.title('Violation during training')
        plt.ylabel('Constrain Violation')
        plt.xlabel('Iteration number/{}'.format(record_interval))
        f.savefig("{}/Violation_{}_CMDP.jpg".format(out_dir,method.upper()))
        f.clf()