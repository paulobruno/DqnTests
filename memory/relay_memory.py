import numpy as np
from random import sample


class ReplayMemory:
    def __init__(self, state_shape):
        capacity = state_shape[0]
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def store(self, experience):
        s1, action, s2, isterminal, reward = experience
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return i, (self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i])

    def batch_update(self, id, abs_loss):
        pass
