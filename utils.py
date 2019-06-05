import numpy as np

def e_greedy(env, epsilon, s, Q):
        if np.random.uniform() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(Q[s])

def learn(
    env, n_episodes, start_state_index, update_q, 
    epsilon, gamma, alpha, render=True
):
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA))
    state = start_state_index

    episode_count = np.zeros(n_episodes)
    acc_rewards = np.zeros(n_episodes)

    for e in range(n_episodes):
        t = 0
        state = env.reset()
        acc_reward = 0
        while True:
            if render:
                env.render()

            action = e_greedy(env, epsilon, state, Q)

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

            update_q(Q, state, action, reward, next_state, gamma, alpha)

            state = next_state

            t += 1

    env.close()

    result = {
        'acc_rewards': acc_rewards,
        'episode_count': episode_count
    }

    return Q, result


