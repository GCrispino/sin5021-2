import numpy as np

def e_greedy(env, epsilon, i_s, Q):
        if np.random.uniform() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(Q[i_s])

def get_feat_index(i, feat, discretized_states):
    discretized_feat_values = discretized_states[i]
    first, last = discretized_feat_values[0], discretized_feat_values[-1]
    delta = abs((first - last) / len(discretized_feat_values))
    
    feat_index = int((feat - first) / delta)
    return feat_index

def get_index_state(state, discretized_states):
    feat_indices = tuple([get_feat_index(i,feat,discretized_states)  for i, feat in enumerate(state)])

    return tuple(feat_indices)

def learn(
    env, n_episodes, start_state_index, discretized_states, update_q, 
    epsilon, gamma, alpha, render=True
):
    
    n_feats, n_discrete = discretized_states.shape[0], discretized_states.shape[1]
    nA = env.action_space.n
    Q = np.zeros((n_discrete,) * n_feats + (nA,))
    state = start_state_index

    episode_count = np.zeros(n_episodes)
    acc_rewards = np.zeros(n_episodes)

    for e in range(n_episodes):
        t = 0
        state = env.reset()
        i_state = get_index_state(state, discretized_states)
        acc_reward = 0
        while True:
            if render:
                env.render()

            print('current state: ', state)
            action = e_greedy(env, epsilon, i_state, Q)

            next_state, reward, done, info = env.step(action)
            acc_reward += reward

            if render:
                print("t = ", t)
                print(action, next_state, reward)
            if done:
               print("Episode %d finished after %d timesteps" % (e, t + 1))
               episode_count[e] = t + 1
               acc_rewards[e] = acc_reward
               break

            print('current state: ', state)
            print('next state: ', next_state)
            update_q(Q, state, action, reward, next_state, gamma, alpha)

            state = next_state

            t += 1

    env.close()

    result = {
        'acc_rewards': acc_rewards,
        'episode_count': episode_count
    }

    return Q, result


