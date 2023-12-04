import numpy as np
import os
import pdb
from env import Random_Env


class Agent(Random_Env):
    def __init__(self,num_state=10,num_action=5,type='pg',policy_type='softmax',gamma=0.9):
        super().__init__(num_state,num_action,policy_type,gamma=gamma)
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
        # pdb.set_trace()

        return gradient_q, gradient_cons_q

    def compute_fisher_information_matrix(self,prob):
        
        prob_reshaped = prob.reshape(self.num_state, self.num_action)
        FIM = np.zeros((self.num_state * self.num_action, self.num_state * self.num_action))
        for s in range(self.num_state):
            start = s*self.num_action
            end = (s+1)*self.num_action
            # FIM[start:end,start:end] = np.diag(prob_reshaped[s]) - prob_reshaped[s][:,np.newaxis] @ prob_reshaped[s][np.newaxis]
            FIM[start:end,start:end] = np.eye(len(prob_reshaped[s])) - prob_reshaped[s][np.newaxis].T

        # FIM = np.zeros((self.num_action,self.num_action))
        # for s in range(self.num_state):
        #     # pdb.set_trace()
        #     FIM += np.diag(prob_reshaped[s]) - prob_reshaped[[s]].T
        return FIM

    def compute_natural_gradient(self,gradient, FIM):
        # Regularize FIM for numerical stability
        # reg_FIM = FIM + 1e-7 * np.eye(FIM.shape[0])
        reg_FIM = FIM
        # Compute the inverse of the Fisher Information Matrix
        FIM_inv = np.linalg.pinv(reg_FIM)
        # Compute natural gradient
        # pdb.set_trace()
        natural_gradient = np.dot(FIM_inv, gradient)
        # natural_gradient = np.dot(FIM_inv,gradient.reshape((self.num_state,self.num_action)).T).T
        # natural_gradient = natural_gradient.flatten()
        return natural_gradient

    def compute_grad(self,prob,qvals,q_constrain_vals):
        gradient_q, gradient_cons_q = self.compute_grad_raw(prob,qvals,q_constrain_vals)
        gradient_raw = gradient_q + self.lam * gradient_cons_q
        # pdb.set_trace()
        # print(np.linalg.norm(gradient_raw))
        if self.type == 'pg':
            # pdb.set_trace()
            gradient = gradient_raw
        elif self.type == 'npg':
            # pdb.set_trace()
            # FIM = self.compute_fisher_information_matrix(prob)
            # gradient= self.compute_natural_gradient(gradient_raw,FIM)

            qvals_reshaped = qvals.reshape((self.num_state,self.num_action))
            prob_reshaped = prob.reshape((self.num_state,self.num_action))
            values = np.sum(qvals_reshaped*prob_reshaped,axis=1,keepdims=True)
            advantage = qvals_reshaped - values
            # pdb.set_trace()
            gradient = advantage.flatten()/(1-self.gamma)


            # qvals_reshaped = qvals.reshape((self.num_state,self.num_action))
            # prob_reshaped = prob.reshape((self.num_state,self.num_action))
            # V = np.sum(qvals_reshaped * prob_reshaped, axis=1)
            # # values = qvals_reshaped @ prob.reshape((self.nu))

            # gradient = (qvals_reshaped - V.reshape(-1,1))/(1-self.gamma)
            # gradient = gradient.flatten()
            # pdb.set_trace()
        elif self.type == 'gnpg':
            gradient_reshaped = gradient_raw.reshape((self.num_state,self.num_action))
            gradient_reshaped = gradient_reshaped/np.linalg.norm(gradient_reshaped,axis=1,keepdims=True)
            gradient = gradient_reshaped.flatten()
            
            # gradient = gradient_raw/np.linalg.norm(gradient_raw)
            # pdb.set_trace()
            # gradient = gradient_q/np.linalg.norm(gradient_q) + self.lam * gradient_cons_q/np.linalg.norm(gradient_cons_q)
        
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
