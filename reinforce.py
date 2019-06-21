"""
    TODO:
        - Criar rede neural
            - criar classe
            - 2 camadas
            - imlpementar função forward
                - Escolher função de ativação das camadas internas e camada de saída
        - Implementar algoritmo reinforce
            - para cada episódio
                - Coletar histórico até fim do episódio seguindo a política pi
                - para cada tempo t do episódio:
                    - acumular descontos e recompensas descontadas
                - atualizar pesos da rede
                    - definir loss function
                    - atualizar gradientes para essa funcao (loss.backwards())
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.l_input = nn.Linear(state_size, 10)
        self.l_hidden = nn.Linear(10, action_size)

    def forward(self, x):
        o_l_input = F.relu(self.l_input(x)) 
        o_l_hidden = F.relu(self.l_hidden(o_l_input))

        return F.softmax(o_l_hidden)

def learn_reinforce():
    pass

if __name__ == '__main__':
    p = Policy(state_size = 4, action_size = 2)

    env = gym.make('CartPole-v1')
    env.reset()
    action = env.action_space.sample()
    # interact with env with random action
    s, r, done, _ = env.step(action)

    # print network output
    print(p(torch.Tensor(s)))
