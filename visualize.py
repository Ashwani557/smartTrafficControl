import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from enviroment import TrafficEnvironment
from agent import QLearningAgent

env = TrafficEnvironment()
agent = QLearningAgent(epsilon=0)  # no exploration during visualization

state = env.reset()

fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_title("Smart Traffic Control - RL Animation")

# Road lines
ax.plot([-10, 10], [0, 0])
ax.plot([0, 0], [-10, 10])

north_cars, = ax.plot([], [], 'bo', label="North")
south_cars, = ax.plot([], [], 'ro', label="South")
east_cars, = ax.plot([], [], 'go', label="East")
west_cars, = ax.plot([], [], 'yo', label="West")

ax.legend()

def update(frame):
    global state

    action = agent.choose_action(state)
    state, reward = env.step(action)

    n, s, e, w = state

    north_cars.set_data(np.zeros(n), np.linspace(1, 9, n))
    south_cars.set_data(np.zeros(s), np.linspace(-1, -9, s))
    east_cars.set_data(np.linspace(1, 9, e), np.zeros(e))
    west_cars.set_data(np.linspace(-1, -9, w), np.zeros(w))

    if action == 0:
        ax.set_title("Green: North-South")
    else:
        ax.set_title("Green: East-West")

    return north_cars, south_cars, east_cars, west_cars

ani = animation.FuncAnimation(fig, update, frames=200, interval=500)
plt.show()
