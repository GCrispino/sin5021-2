import sys
import numpy as np
import matplotlib.pyplot as plt
import gym
from qlearning import update_q as update_q_qlearning
from sarsa import update_q_factory as update_q_factory_sarsa
from utils import e_greedy, learn 

def get_values(Q):
    new_Q = np.array([ np.max(Q[s]) for s in range(Q.shape[0])])

    return new_Q.reshape(4,12)


def get_policy(Q):
    new_Q = np.array([ np.argmax(Q[s]) for s in range(Q.shape[0])])

    return new_Q.reshape(4,12)


if len(sys.argv) < 2:
    sys.exit("USAGE: python main.py <algorithm={0,1}>")

algorithm = int(sys.argv[1])

env = gym.make('CartPole-v1')

gamma = 0.9
decay = 0.9
alpha = 0.1
epsilon = 0.1
n_episodes = 2000

state_intervals = np.array([
    env.observation_space.low,
    env.observation_space.high
]).T

n_discrete_states = 50

cart_pos_vals = np.linspace(state_intervals[0][0], state_intervals[0][1], n_discrete_states)
cart_vel_vals = np.linspace(state_intervals[1][0], state_intervals[1][1], n_discrete_states)
pole_angle_vals = np.linspace(state_intervals[2][0], state_intervals[2][1], n_discrete_states)
pole_vel_vals = np.linspace(state_intervals[3][0], state_intervals[3][1], n_discrete_states)

discretized_states = np.array([
    cart_pos_vals, cart_vel_vals,
    pole_angle_vals, pole_vel_vals
])

# E = np.zeros((env.observation_space.n, env.action_space.n))
E = np.zeros((n_discrete_states, env.action_space.n))
update_q_sarsa = update_q_factory_sarsa(E, env, epsilon, decay)

start_state_index = env.start_state_index if hasattr(env,'start_state_index') else None

if algorithm == 0:
    Q, result = learn(
        env, n_episodes, start_state_index, discretized_states, update_q_qlearning, 
        epsilon, gamma, alpha, render=False
    )
elif algorithm == 1:
    Q, result = learn(
        env, n_episodes, start_state_index, discretized_states, update_q_sarsa,
        epsilon, gamma, alpha, render=False
    )
else:
    sys.exit("Invalid algorithm option!")

print(get_policy(Q))
print(get_values(Q))
print(Q)

acc_rewards = result['acc_rewards']
episode_count = result['episode_count']
plt.plot(acc_rewards)
plt.plot(episode_count)
plt.show()
