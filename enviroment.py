import numpy as np

class TrafficEnvironment:
    def __init__(self):
        self.state = np.random.randint(0, 10, size=4)

    def reset(self):
        self.state = np.random.randint(0, 10, size=4)
        return tuple(self.state)

    def step(self, action):
        # action: 0 = NS green, 1 = EW green
        if action == 0:
            self.state[0] = max(0, self.state[0] - 2)
            self.state[1] = max(0, self.state[1] - 2)
        else:
            self.state[2] = max(0, self.state[2] - 2)
            self.state[3] = max(0, self.state[3] - 2)

        # new cars arrive
        self.state += np.random.randint(0, 3, size=4)

        reward = -sum(self.state)
        return tuple(self.state), reward
