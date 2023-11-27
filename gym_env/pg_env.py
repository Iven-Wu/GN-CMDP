import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Neural network for the policy model
class PolicyNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# Function to select an action based on policy probabilities
def select_action(policy_net, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy_net(state)
    action = torch.multinomial(probs, 1).item()
    return action

# Function to compute returns
def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

# Main training loop
def train_policy_gradient(env_name='CartPole-v1', n_episodes=1000, gamma=0.99, lr=0.01):
    env = gym.make(env_name)
    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy_net = PolicyNetwork(n_inputs, n_actions)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    for episode in range(n_episodes):
        state = env.reset()
        saved_log_probs = []
        rewards = []
        done = False

        while not done:
            action = select_action(policy_net, state)
            state, reward, done, _ = env.step(action)
            saved_log_probs.append(torch.log(policy_net(torch.from_numpy(state).float())[action]))
            rewards.append(reward)

        returns = compute_returns(rewards, gamma)
        policy_loss = []
        for log_prob, G in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * G)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f'Episode {episode}/{n_episodes}: Total Reward: {sum(rewards)}')

    env.close()

# Run the training
train_policy_gradient()