import random
import numpy as np

class Memory:

    def __init__(self, memory_size):
        self.buffer = []
        self.count = 0
        self.max_memory_size = memory_size

    def _recalibrate(self):
        self.count = len(self.buffer)

    def remove_n(self, n):
        self.buffer = self.buffer[n-1:-1]
        self._recalibrate()

    def add(self, memory):
        self.buffer.append(memory)
        self.count += 1

        if self.count > self.max_memory_size:
            self.buffer.pop(0)
            self.count -= 1

    def get(self, batch_size=1):
        if self.count <= batch_size:
            return np.array(self.buffer)

        return np.array(random.sample(self.buffer, batch_size))
