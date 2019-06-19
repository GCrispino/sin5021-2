import numpy as np
from utils import e_greedy

def update_q(Q, s, a, r, s_, gamma, alpha): 
    q_max = np.max(Q[s_])
    delta = r + gamma * q_max - Q[s + (a,)]

    # print('aloaloaloalo ',s)
    # print(Q)
    #print('aloaloaloalo ',s, Q[s + (a,)], delta)
    #print('\t', state)
    #print('\t\t', [
    #    discretized_states[0][s[0] - 1],
    #    discretized_states[1][s[1] - 1],
    #    discretized_states[2][s[2] - 1#],
    #    discretized_states[3][s[3] - 1],
    #])
    #print('\t\t', [
    #    discretized_states[0][s[0]],
    #    discretized_states[1][s[1]],
    #    discretized_states[2][s[2]],
    #    discretized_states[3][s[3]],
    #])
    #print('\t\t', [
    #    discretized_states[0][s[0] + 1],
    #    discretized_states[1][s[1] + 1],
    #    discretized_states[2][s[2] + 1],
    #    discretized_states[3][s[3] + 1],
    #])

#    print(alpha, delta)
#    print(Q[s + (a,)])
    Q[s + (a,)] += alpha * delta
#    print(Q[s + (a,)])

#    print('====================')

