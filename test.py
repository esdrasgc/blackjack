from QLearning import QLearning
import gymnasium as gym
import numpy as np
from numpy import loadtxt
black = gym.make('Blackjack-v1', natural=False, sab=False)
q_table = loadtxt('data/q-table-sarsa-blackjack.csv', delimiter=',')

rewards = 0
n_episodes = 100
ganhou = 0
perdeu = 0
for i in range(0,n_episodes):    
    (state, _) = black.env.env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state[0]+ 32*state[1] + 32*11*state[2]])
        state, reward, done, _, info = black.env.env.step(action)
    if reward == 1:
        ganhou +=1
    if reward == -1:
        perdeu +=1
    rewards += reward
print(rewards/n_episodes)
print(ganhou, perdeu)
