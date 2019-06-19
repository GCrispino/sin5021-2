import numpy as np
import math

def get_epsilon(_epsilon, t):
    return max(_epsilon, min(1, 1.0 - math.log10((t + 1) / 25)))

def get_alpha(_alpha, t):
    return max(_alpha, min(1.0, 1.0 - math.log10((t + 1) / 25)))


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
    
    feat_index = int((feat - first) / delta)
    return feat_index

def get_index_state(state, discretized_states):
    feat_indices = tuple([get_feat_index(i,feat,discretized_states)  for i, feat in enumerate(state)])

    return tuple(feat_indices)

def learn(
    env, n_episodes, start_state_index, n_discrete_states, discretized_states, update_q, 
    _epsilon, gamma, _alpha, render=True
):
    
    nA = env.action_space.n
    Q = np.zeros(n_discrete_states + (env.action_space.n,))

    episode_count = np.zeros(n_episodes)
    acc_rewards = np.zeros(n_episodes)

    for e in range(n_episodes):
        t = 0
        state = env.reset()
        i_state = get_index_state(state, discretized_states)
        acc_reward = 0
        epsilon = get_epsilon(_epsilon, e)
        print('epsilon: ',epsilon)
        alpha = get_alpha(_alpha, e)
        print('alpha: ',alpha)
        while True:
            if render:
                env.render()

            action = e_greedy(env, epsilon, i_state, Q)
            # print('current_state: ', i_state)
            # print('action: ', action)

            next_state, reward, done, _ = env.step(action)
            i_next_state = get_index_state(next_state, discretized_states)
            acc_reward += reward
            update_q(Q, i_state, action, reward, i_next_state, gamma, alpha)

            state = next_state
            i_state = get_index_state(state, discretized_states)

            t += 1

            if render:
                print("t = ", t)
                print(action, next_state, reward)
            if done:
               print("Episode %d finished after %d timesteps" % (e, t + 1))
               episode_count[e] = t + 1
               acc_rewards[e] = acc_reward
               break

            
    env.close()

    result = {
        'acc_rewards': acc_rewards,
        'episode_count': episode_count,
    }

    return Q, result


