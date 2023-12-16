import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import pdb
import time 
import os
import torch
import torch.nn.functional as F

class MC_Env():
    def __init__(self, num_state, num_action, policy_type, gamma=0.9,device='cpu'):
        np.random.seed(10) 
        torch.manual_seed(10)  # Set seed for reproducibility
        self.gamma = gamma
        self.device = device
        self.num_state, self.num_action = num_state, num_action

        # Generate the raw transition probabilities
        raw_transition = torch.tensor(np.random.uniform(0,1,size=(num_state*num_action,num_state)),device=device).float()
        # raw_transition = torch.rand(size=(num_state * num_action, num_state)).to(self.device)
        prob_transition = raw_transition / raw_transition.sum(dim=1, keepdim=True)
        self.prob_transition = prob_transition.view(num_state, num_action, num_state)

        # Generate random rewards and constraints
        # self.reward_mat = torch.rand(size=(num_state, num_action)).to(self.device)
        self.reward_mat = torch.tensor(np.random.uniform(0,1,size=(num_state,num_action))).float().to(self.device)
        # self.constrain_mat = torch.rand(size=(num_state, num_action)).to(self.device)
        self.constrain_mat = torch.tensor(np.random.uniform(0,1,size=(num_state,num_action))).float().to(self.device)

        # Uniform distribution for rho
        self.rho = torch.ones(num_state).to(self.device) / num_state

        # Policy type (assuming it's a string or some other non-array type)
        self.policy_type = policy_type
    
    
    def ell(self, qvals, prob):
        qvals_reshaped = qvals.view(self.num_state, self.num_action)
        prob_reshaped = prob.view(self.num_state, self.num_action)

        # Compute V vector using vectorized operations
        V = torch.sum(qvals_reshaped * prob_reshaped, dim=1)

        # Compute ell using dot product
        ell = torch.dot(V, self.rho)
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
    
    def theta_to_policy(self, theta):
        # Assuming theta is a PyTorch tensor. If not, convert it using torch.tensor or torch.from_numpy
        theta_reshaped = theta.view(self.num_state, self.num_action)
        # Compute the exponential of each element
        exp_theta = torch.exp(theta_reshaped)
        # Normalize each row to get probabilities
        prob = exp_theta / torch.sum(exp_theta, dim=1, keepdim=True)
        return prob
        # log_prob = F.log_softmax(theta_reshaped, dim=1)
        # return prob
        # return log_prob
    
    def theta_to_logpolicy(self,theta):
        theta_reshaped = theta.view(self.num_state, self.num_action)
        log_prob = F.log_softmax(theta_reshaped, dim=1)
        # return prob
        return log_prob

    def get_Pi(self, prob):
        prob_reshaped = prob.view(self.num_state, self.num_action)
        # Initialize Pi as a zero tensor
        Pi = torch.zeros(self.num_state, self.num_state * self.num_action).to(self.device)
        # Create an index tensor
        col_indices = torch.arange(self.num_state * self.num_action).to(self.device).view(self.num_state, self.num_action)
        # Use advanced indexing to efficiently assign values
        Pi[torch.arange(self.num_state)[:, None].to(self.device), col_indices] = prob_reshaped
        return Pi

    
    def policy_iter(self, q_vals):
        new_policy = torch.zeros(self.num_state, self.num_action).to(self.device)
        q_vals_reshaped = q_vals.view(self.num_state, self.num_action)
        max_idx = torch.argmax(q_vals_reshaped, dim=1)
        new_policy[torch.arange(self.num_state).to(self.device), max_idx] = 1
        new_policy = new_policy.flatten()
        return new_policy
    


    def get_action(self,theta,states):
        start_time = time.time()
        raw_prob = self.theta_to_policy(theta)
        probs = raw_prob[states]
        # pdb.set_trace()
        actions = torch.multinomial(probs, num_samples=1).squeeze()
        return actions

    def env_step(self, states, actions):
        # Extracting the transition probabilities for the given states and actions
        probs_next = self.prob_transition[states, actions]

        # Sampling the next states using torch.multinomial
        next_states = torch.multinomial(probs_next,num_samples=1).squeeze()

        # Extracting rewards and utilities for the given states and actions
        rewards = self.reward_mat[states, actions]
        utilities = self.constrain_mat[states, actions]

        return next_states, rewards, utilities
    
    def ell_approx(self, theta, num_MC_sims, horizon):
        V_r_rho, V_g_rho = 0, 0

        # Initialize states using torch.multinomial for random sampling
        init_states = torch.multinomial(self.rho, num_MC_sims, replacement=True)
        states = init_states
        cum_rewards = torch.tensor(0.0).to(self.device)
        cum_constrains = torch.tensor(0.0).to(self.device)

        for i in range(horizon):
            actions = self.get_action(theta, states)
            next_states, rewards, utilities = self.env_step(states, actions)
            cum_rewards = self.gamma * cum_rewards + rewards
            cum_constrains = self.gamma * cum_constrains + utilities
            states = next_states

        # Compute the means
        V_r_rho = cum_rewards.mean()
        V_g_rho = cum_constrains.mean()

        return V_r_rho, V_g_rho

    def ell_theta(self,theta):
        # pdb.set_trace()
        prob = self.theta_to_policy(theta)
        Pi = self.get_Pi(prob)
        mat = torch.eye(self.num_state*self.num_action).cuda() - self.gamma*self.prob_transition.view(-1,self.num_state)@Pi

        qvals = torch.linalg.inv(mat)@self.reward_mat.view(-1)

        q_cons = torch.linalg.inv(mat)@self.constrain_mat.view(-1)
        return self.ell(qvals,prob), self.ell(q_cons,prob)

    def ell_network(self,theta,policy_net):
        # pdb.set_trace()
        # prob = self.theta_to_policy(theta)
        prob = policy_net(theta)
        Pi = self.get_Pi(prob)
        mat = torch.eye(self.num_state*self.num_action).cuda() - self.gamma*self.prob_transition.view(-1,self.num_state)@Pi

        qvals = torch.linalg.inv(mat)@self.reward_mat.view(-1)

        q_cons = torch.linalg.inv(mat)@self.constrain_mat.view(-1)
        return self.ell(qvals,prob), self.ell(q_cons,prob)


    def get_optimum(self,):
        raw_vec = torch.rand(self.num_state, self.num_action).to(self.device)
        prob_vec = raw_vec / raw_vec.sum(dim=1, keepdim=True)
        init_policy = prob_vec.flatten()

        curr_policy = torch.rand(self.num_state * self.num_action).to(self.device)
        new_policy = init_policy
        print('Starting policy', init_policy)

        ### use policy iteration to find out the optimal one
        while torch.count_nonzero(curr_policy - new_policy) > 0:
            curr_policy = new_policy
            Pi = self.get_Pi(curr_policy)
            identity = torch.eye(self.num_state * self.num_action).to(self.device)
            gamma_term = self.gamma * torch.matmul(self.prob_transition.view(self.num_state * self.num_action, self.num_state), Pi)
            mat = identity - gamma_term
            self.q_vals = torch.matmul(torch.linalg.inv(mat), (self.reward_mat).flatten() )
            new_policy = self.policy_iter(self.q_vals)
            self.policy = new_policy

        print('Final policy', new_policy)

        ell_star = self.ell(self.q_vals,new_policy)
        print('Optimal Reward',ell_star)
        return ell_star

    def compute_returns(self,rewards_list,constrains_list,lam):
        cum_returns_list = [0]
        cum_constrain_list = [0]
        for i in range(len(rewards_list)-1,-1,-1):
            # cum_R =(rewards_list[i]+ lam*constrains_list[i])+self.gamma*cum_returns_list[0]
            cum_R = rewards_list[i] + self.gamma*cum_returns_list[0]
            cum_returns_list.insert(0,cum_R)

            cum_Con = constrains_list[i] + self.gamma*cum_constrain_list[0]
            cum_constrain_list.insert(0,cum_Con)

        return cum_returns_list[:-1], cum_constrain_list[:-1]


    def plot_curve(self,reward,violation,label,record_interval=1,method='pg',out_dir='figs/'):
        os.makedirs(out_dir,exist_ok=True)

        f = plt.figure()
        for i in range(len(reward)):
            plt.plot(np.array(reward[i]),label=label[i])
        plt.legend()
        plt.title('Optimal Gap during training')
        plt.ylabel('Optimal Gap')
        plt.xlabel('Iteration number/{}'.format(record_interval))
        f.savefig("{}/MC_Reward_{}_CMDP.jpg".format(out_dir,method.upper()))
        f.clf()

        f = plt.figure()
        for i in range(len(violation)):
            plt.plot(np.array(violation[i]),label=label[i])
        plt.legend()
        plt.title('Violation during training')
        plt.ylabel('Constrain Violation')
        plt.xlabel('Iteration number/{}'.format(record_interval))
        f.savefig("{}/MC_Violation_{}_CMDP.jpg".format(out_dir,method.upper()))
        f.clf()