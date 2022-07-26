import random
from collections import deque


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_paths = 0
        self.buffer = deque()

    def get_sample(self, sample_size):
        # Return all the paths from buffer if sample_size > num_paths_in_buffer
        if self.num_paths < sample_size:
            return random.sample(self.buffer, self.num_paths)
        # Else return sample_size paths
        else:
            return random.sample(self.buffer, sample_size)

    def size(self):
        return self.buffer_size

    def add(self, path):
        if self.num_paths < self.buffer_size:
            self.buffer.append(path)
            self.num_paths += 1
        else:
            self.buffer.popleft()
            self.buffer.append(path)

    def count(self):
        return self.num_paths

    def clear(self):
        self.buffer = deque()
        self.num_paths = 0


class RolloutWorker(object):
    def __init__(self, env, policy, params):
        self.env = env
        self.policy = policy

        # Get the params
        self.T = params['T']
        self.parallel_rollout_workers = params['parallel_rollout_workers']



