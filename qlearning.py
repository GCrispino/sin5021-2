import numpy as np
from utils import e_greedy

def update_q(Q, s, a, r, s_, gamma, alpha): 
    q_max = np.max(Q[s_])
    delta = r + gamma * q_max - Q[s, a]

    Q[s, a] += alpha * delta

