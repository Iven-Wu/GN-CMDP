import gym
import torch
import pdb
env = gym.make('CartPole-v0')
n_state = env.observation_space.shape[0]
n_action = env.action_space.n

def run_episode(env, weight):
    state,_ = env.reset()
    grads = []
    total_reward = 0
    is_done = False
    while not is_done:
        state = torch.from_numpy(state).float()
        z = torch.matmul(state, weight)
        probs = torch.nn.Softmax()(z)
        action = int(torch.bernoulli(probs[1]).item())
        # pdb.set_trace()
        d_softmax = torch.diag(probs) -probs.view(-1, 1) * probs
        d_log = d_softmax[action] / probs[action]
        grad = state.view(-1, 1) * d_log
        grads.append(grad)
        state, reward, is_done, _ = env.step(action)
        total_reward += reward
        if is_done:
            break
    return total_reward, grads


n_episode = 1000
weight = torch.rand(n_state, n_action)
total_rewards = []
learning_rate = 0.001

for episode in range(n_episode):
    total_reward, gradients = run_episode(env, weight)
    print('Episode {}: {}'.format(episode + 1, total_reward))
    for i, gradient in enumerate(gradients):
        weight += learning_rate * gradient * (total_reward - i)
    total_rewards.append(total_reward)
