import random
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.max_size = int(max_size)
        self.storage = []
        self.next_idx = 0

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.storage) < self.max_size:
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data
            self.next_idx = (self.next_idx + 1) % self.max_size

    def sample(self, batch_size):
        idxs = random.sample(range(len(self.storage)), batch_size)
        batch = [self.storage[i] for i in idxs]
        states, actions, rewards, next_states, dones = map(lambda x: np.array(x), zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.storage)
