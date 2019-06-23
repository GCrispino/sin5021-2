import sys
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import gym
from qlearning import update_q as update_q_qlearning
from sarsa import update_q_factory as update_q_factory_sarsa
from utils import learn 
from reinforce import Policy, learn_reinforce

np.random.seed(0)
torch.manual_seed(0) 

def get_values(Q):
    new_Q = np.array([ np.max(Q[s]) for s in range(Q.shape[0])])

    return new_Q.reshape(4,12)


def get_policy(Q):
    new_Q = np.array([ np.argmax(Q[s]) for s in range(Q.shape[0])])

    return new_Q.reshape(4,12)


if len(sys.argv) < 2:
    sys.exit("USAGE: python main.py <algorithm={0,1,2}>")

algorithm = int(sys.argv[1])
n_executions = int(sys.argv[2]) if len(sys.argv) > 2 else 1

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
env = gym.make('CartPole-v1')
env.action_space.seed(0)
env.seed(0)

# TODO:
#       Implementar renderização de política aprendida
#       Implementar sarsa (se tiver saco)
gamma = 1
decay = 0.9
alpha = 1e-3
epsilon = 0.01
n_episodes = 2000

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


results = [None] * n_executions

for i in range(n_executions):
    print('-> %dith execution: ' % i)
    E = np.zeros(n_discrete_states + (env.action_space.n,))
    Q = np.array([])
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
    elif algorithm == 2:
        p = Policy(state_size = 4, action_size = 2)
        p.to(device)

        result = learn_reinforce(
            env, device, n_episodes, p,
            gamma, alpha, render=False
        )
    else:
        sys.exit("Invalid algorithm option!")
    results[i] = result

results_obj = {
    'alg': algorithm,
    'gamma': gamma,
    'alpha': alpha,
    'epsilon': epsilon,
    'results': results,
    'decay': decay,
    'n_episodes': n_episodes,
}

#if acc_rewards.size > 0:
#    plt.plot(acc_rewards)
#if episode_count.size > 0:
#    plt.plot(episode_count)
#plt.show()
timestamp = datetime.datetime.now().timestamp()
filename = './results/result-%s-%s.json' % (algorithm, str(timestamp))
with open(filename, 'w') as fp:
    json.dump(results_obj, fp, indent=2)
