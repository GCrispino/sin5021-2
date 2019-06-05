import numpy as np
from utils import e_greedy

def _update_q(Q, E, env, s, a, r, s_, gamma, alpha, epsilon, decay): 
    a_ = e_greedy(env, epsilon, s_, Q)

    delta = r + gamma * Q[s_,a_] - Q[s,a]

    E *= gamma * decay
    E[s,a] += 1
    Q[s,a] += alpha * E[s,a] * delta


def update_q_factory(E, env, epsilon, decay):
    return lambda Q, s, a, r, s_, gamma, alpha: _update_q(Q, E, env, s, a, r, s_, gamma, alpha, epsilon, decay)


