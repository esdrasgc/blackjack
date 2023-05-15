import gymnasium as gym
from SARSA import Sarsa
from QLearning import QLearning

black = gym.make('Blackjack-v1', natural=False, sab=False)

# qlearn = QLearning(black.env, alpha=0.1, gamma=0.95, epsilon=0.8, epsilon_min=0.0001, epsilon_dec=0.99999, episodes=200000)
sarsa = Sarsa(black.env, alpha=0.2, gamma=0.95, epsilon=0.8, epsilon_min=0.0001, epsilon_dec=0.9999, episodes=200000)

# q_table = qlearn.train('data/q-table-blackjack.csv','results/blackjack_1')
q_table_sarsa = sarsa.train('data/q-table-sarsa-blackjack.csv','results/blackjack-sarsa_1')