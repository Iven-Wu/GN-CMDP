import numpy as np
import os
import pdb
from env import Random_Env


class Agent(Random_Env):
    def __init__(self,num_state=10,num_action=5,type='pg'):
        super().__init__(num_state,num_action)
        self.num_state, self.num_action = num_state, num_action
        self.theta = np.random.uniform(0,1,size=num_state*num_action)
        self.lam = 0.5
        self.reward_list = []
        self.violation_list = []
        self.type = type
    
    def record(self,reward,violation):
        self.reward_list.append(reward)
        self.violation_list.append(violation)

    def compute_grad_raw(self,prob,qvals,q_constrain_vals):
        Pi = self.get_Pi(prob)  
        ### p_theta is the probability from state s to state s'
        P_theta = np.matmul(Pi,self.prob_transition)
        d_pi = (1-self.gamma)*np.dot(np.transpose((np.linalg.inv(np.identity(self.num_state) - self.gamma*P_theta))),self.rho)

        gradient_q = self.grad(qvals,prob,d_pi) / (1-self.gamma)
        gradient_cons_q = self.grad(q_constrain_vals,prob,d_pi)/(1-self.gamma)

        return gradient_q, gradient_cons_q

    def compute_fisher_information_matrix(self,prob):
        prob_reshaped = prob.reshape(self.num_state, self.num_action)
        FIM = np.zeros((self.num_state * self.num_action, self.num_state * self.num_action))

        for s in range(self.num_state):
            start = s*self.num_action
            end = (s+1)*self.num_action
            FIM[start:end,start:end] = np.diag(prob_reshaped[s]) - prob_reshaped[s][:,np.newaxis] @ prob_reshaped[s][np.newaxis]
        return FIM

    def compute_natural_gradient(self,gradient, FIM):
        # Regularize FIM for numerical stability
        # reg_FIM = FIM + 1e-7 * np.eye(FIM.shape[0])
        reg_FIM = FIM
        # Compute the inverse of the Fisher Information Matrix
        FIM_inv = np.linalg.pinv(reg_FIM)
        # Compute natural gradient
        natural_gradient = np.dot(FIM_inv, gradient)
        return natural_gradient

    def compute_grad(self,prob,qvals,q_constrain_vals):
        gradient_q, gradient_cons_q = self.compute_grad_raw(prob,qvals,q_constrain_vals)
        gradient_raw = gradient_q + self.lam * gradient_cons_q
        if self.type == 'pg':
            gradient = gradient_raw
        elif self.type == 'npg':
            FIM = self.compute_fisher_information_matrix(prob)
            gradient = self.compute_natural_gradient(gradient_raw,FIM)
        elif self.type == 'gnpg':
            gradient = gradient_raw/np.linalg.norm(gradient_raw)
        
        return gradient

    def get_Pi(self,prob):
        prob_reshaped = prob.reshape((self.num_state, self.num_action))
        # Initialize Pi as a zero array
        Pi = np.zeros((self.num_state,self.num_state * self.num_action))
        # Create an index array
        col_indices = np.arange(self.num_state * self.num_action).reshape((self.num_state, self.num_action))      
        # Use advanced indexing to efficiently assign values
        Pi[np.arange(self.num_state)[:, None], col_indices] = prob_reshaped
        return Pi
