import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pdb

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

def select_action(policy_net, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy_net(state)
    action = torch.multinomial(probs, 1).item()
    return action

def compute_returns(rewards, gamma):
    rewards = torch.Tensor(rewards)
    returns = torch.zeros_like(rewards)

    running_returns = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + gamma * running_returns 
        returns[t] = running_returns

    return returns

def train(env_name='CartPole-v1', n_episodes=5000, gamma=0.99, learning_rate=0.01):
    env = gym.make(env_name)
    policy_net = PolicyNetwork()
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    for episode in range(n_episodes):
        state,_ = env.reset()
        episode_states, episode_actions, episode_rewards = [], [], []

        while True:
            action = select_action(policy_net, state)
            next_state, reward, done, _,_ = env.step(action)
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            if done:
                break
            state = next_state

        returns = compute_returns(episode_rewards, gamma)
        optimizer.zero_grad()

        states = torch.from_numpy(np.stack(episode_states)).float()
        probs = policy_net(states)
        m = torch.distributions.Categorical(probs)
        loss = -m.log_prob(torch.tensor(episode_actions)) * returns
        loss = loss.mean()
        loss.backward()

        optimizer.step()
        if episode % 50 == 0:
            print(f'Episode {episode}: Total Reward: {sum(episode_rewards)}')

    env.close()

# Run the training
train()