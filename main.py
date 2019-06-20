import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from qlearning import update_q as update_q_qlearning
from sarsa import update_q_factory as update_q_factory_sarsa
from utils import learn 

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

# TODO:
#       Implementar renderização de política aprendida
#       Implementar sarsa (se tiver saco)
decay = 0.9
alpha = 0.1
epsilon = 0.01
n_episodes = 1000

state_intervals = np.array([
    env.observation_space.low,
    env.observation_space.high
]).T

n_discrete_states = (1, 1, 10, 10)

cart_pos_vals = np.linspace(state_intervals[0][0], state_intervals[0][1], n_discrete_states[0])
cart_vel_vals = np.linspace(-0.5, 0.5, n_discrete_states[1])
pole_angle_vals = np.linspace(-np.radians(24), np.radians(24), n_discrete_states[2])
pole_vel_vals = np.linspace(-1, 1, n_discrete_states[3])

discretized_states = np.array([
    cart_pos_vals, cart_vel_vals,
    pole_angle_vals, pole_vel_vals
])

E = np.zeros(n_discrete_states + (env.action_space.n,))
update_q_sarsa = update_q_factory_sarsa(E, env, epsilon, decay)


if algorithm == 0:
    Q, result = learn(
        env, n_episodes, n_discrete_states, discretized_states, update_q_qlearning, 
        epsilon, gamma, alpha, render=False
    )
elif algorithm == 1:
    Q, result = learn(
        env, n_episodes, discretized_states, update_q_sarsa,
        epsilon, gamma, alpha, render=False
    )
else:
    sys.exit("Invalid algorithm option!")

#print(get_policy(Q))
#print(get_values(Q))
print(Q, Q.shape)

acc_rewards = result['acc_rewards']
episode_count = result['episode_count']
plt.plot(acc_rewards)
plt.plot(episode_count)
plt.show()
