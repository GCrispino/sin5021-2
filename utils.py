import numpy as np
import math
from collections import deque

# the lower the decay limit, the higher is the speed of decayment
def get_epsilon(_epsilon, t, log_decay_limit=10):
    return max(_epsilon, min(1.0, 1.0 - math.log10((t + 1) / log_decay_limit)))

def get_alpha(_alpha, t, log_decay_limit=10):
    return max(_alpha, min(1.0, 1.0 - math.log10((t + 1) / log_decay_limit)))


def e_greedy(env, epsilon, i_s, Q):
    rand = np.random.random() 
    if rand <= epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[i_s])

def get_feat_index(i, feat, discretized_states):
    discretized_feat_values = discretized_states[i]
    first, last = discretized_feat_values[0], discretized_feat_values[-1]
    if feat < first:
        return 0
    if feat > last:
        return -1
    delta = abs((first - last) / len(discretized_feat_values))
    
    feat_index = int((feat - first) / delta) - 1
    return feat_index

def get_index_state(state, discretized_states):
    feat_indices = tuple([get_feat_index(i,feat,discretized_states)  for i, feat in enumerate(state)])

    return tuple(feat_indices)

def learn(
    env, n_episodes, n_discrete_states, discretized_states, update_q, 
    _epsilon, gamma, _alpha, render=True
):
    
    nA = env.action_space.n
    Q = np.zeros(n_discrete_states + (env.action_space.n,))

    episode_count = np.zeros(n_episodes)
    acc_rewards = np.zeros(n_episodes)
    last_100_rewards = deque(maxlen=100)

    for e in range(n_episodes):
        t = 0
        state = env.reset()
        i_state = get_index_state(state, discretized_states)
        acc_reward = 0
        epsilon = get_epsilon(_epsilon, e)
        alpha = get_alpha(_alpha, e)
        rewards = []
        while True:
            if render:
                env.render()

            action = e_greedy(env, epsilon, i_state, Q)
            # print('current_state: ', i_state)
            # print('action: ', action)

            next_state, reward, done, _ = env.step(action)
            i_next_state = get_index_state(next_state, discretized_states)
            acc_reward += reward
            rewards.append(reward)
            update_q(Q, i_state, action, reward, i_next_state, gamma, alpha)

            state = next_state
            i_state = get_index_state(state, discretized_states)


            if render:
                print("t = ", t)
                print(action, next_state, reward)
            if done:
               # print("Episode %d finished after %d timesteps" % (e, t + 1))
               # print("\t Acc reward: %d" % np.sum(rewards))
               episode_count[e] = t + 1
               acc_rewards[e] = acc_reward
               last_100_rewards.append(np.sum(rewards))
               break

            t += 1

        if (e + 1) % 100 == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(e + 1, np.mean(last_100_rewards))) 
            
    env.close()

    result = {
        'acc_rewards': acc_rewards.tolist(),
        'episode_count': episode_count.tolist(),
    }

    return Q, result
