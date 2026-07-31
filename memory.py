import random
from collections import deque

class Memory:
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)

    def append(self, experience):
        self.memory.append(experience)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
