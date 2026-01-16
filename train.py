import numpy as np
import matplotlib.pyplot as plt
from enviroment import TrafficEnvironment
from agent import QLearningAgent

env = TrafficEnvironment()
agent = QLearningAgent()

episodes = 500
rewards = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    for _ in range(50):
        action = agent.choose_action(state)
        next_state, reward = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    rewards.append(total_reward)

# 🔹 SMOOTHING (Moving Average)
window = 20
smoothed_rewards = np.convolve(
    rewards, np.ones(window)/window, mode='valid'
)

# 🔹 PLOTTING
plt.figure()
plt.plot(rewards, label="Original Reward")
plt.plot(smoothed_rewards, label="Smoothed Reward")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Learning Curve - Smart Traffic Control (Q-Learning)")
plt.legend()
plt.grid(True)
plt.show()
