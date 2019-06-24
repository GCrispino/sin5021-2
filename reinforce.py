import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym
from collections import deque
import sys

class Policy(nn.Module):
    def __init__(self, state_size, action_size, n_hidden=1, hidden_size=15):
        super(Policy, self).__init__()
        self.l_input = nn.Linear(state_size, hidden_size)
        self.l_hidden = [
            nn.Linear(hidden_size, action_size) if i == n_hidden - 1
            else nn.Linear(hidden_size, hidden_size)
            for i in range(n_hidden)
        ]

    def forward(self, x):
        o_l_input = F.relu(self.l_input(x)) 
        new_x = o_l_input
        for l_h in self.l_hidden:
            new_x = l_h(new_x)
        
        return F.softmax(new_x, dim=0)

def learn_reinforce(env, device, n_episodes, p, gamma, alpha, render=True):
    optimizer = torch.optim.Adam(p.parameters(), lr=alpha)

    last_100_rewards = deque(maxlen=100)
    acc_rewards = np.zeros(n_episodes)
    for e in range(n_episodes):
        state = torch.Tensor(env.reset()).to(device)
        t = 0
        log_probs = []
        acc_reward = 0
        rewards = []
        while True:
            if render:
                env.render()
            probs = p(state)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)

            next_state, reward, done, _ = env.step(action.item())
            state = torch.Tensor(next_state)

            rewards.append(reward)

            t += 1
            
            if done:
                break

        last_100_rewards.append(np.sum(rewards))
        acc_rewards[e] = (np.sum(rewards))
        if (e + 1) % 100 == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(e + 1, np.mean(last_100_rewards))) 
        # print('Episode %d ended after %d timesteps.' % (e, t))
        #print('\tRewards sum: %d. ' % acc_rewards[e]) 

        # update policy
        for t, log_prob in enumerate(log_probs):
            optimizer.zero_grad()
            G = np.sum([
                (gamma ** (k - t)) * rewards[k]
                for k in range(t, len(log_probs))
            ])
            loss = -log_prob * G
            # print(-log_prob, G, loss)
            loss.backward()
            optimizer.step()

    env.close()

    result = {
        'acc_rewards': acc_rewards.tolist(),
        'episode_count': acc_rewards.tolist()
    }
    return result



if __name__ == '__main__':
    p = Policy(state_size = 4, action_size = 2)

    env = gym.make('CartPole-v1')
    n_episodes = 1000
    gamma = 1
    alpha = 1e-3

    learn_reinforce(env, n_episodes, p, gamma, alpha, render=False)
