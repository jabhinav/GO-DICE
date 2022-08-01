import sys
import numpy as np
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.T = T
        self.sample_transitions = sample_transitions
        self.buffer_size = size_in_transitions // T
        self.buffer = {key: np.empty([self.buffer_size, *shape]) for key, shape in buffer_shapes.items()}

        self.current_size = 0  # Tracks size of the buffer in terms of number of episodes
        self.n_transitions_stored = 0  # Tracks size of the buffer in terms of number of transitions

        # for storing transitions that are not a whole batch
        self.current_outer_idx = 0
        self.current_inner_idx = 0

    def sample(self, batch_size):
        buffered_data = {}
        for key in self.buffer.keys():
            buffered_data[key] = self.buffer[key][:self.current_size]

        buffered_data['states_2'] = buffered_data['states'][:, 1:, :]
        buffered_data['achieved_goals_2'] = buffered_data['achieved_goals'][:, 1:, :]

        transitions = self.sample_transitions(buffered_data, batch_size)

        return transitions

    def __len__(self):
        return self.current_size

    @property
    def full(self):
        # with self.lock:
        return self.current_size == self.buffer_size

    def store_episode(self, episode_batch):
        """
            Store batch of episodes into replay buffer
            episode_batch: array(batch_size x (T or T+1) x dim)
        """
        batch_sizes = [episode_batch[key].shape[0] for key in episode_batch.keys()]
        batch_sizes = tf.convert_to_tensor(batch_sizes, dtype=tf.int32)
        try:
            assert tf.math.reduce_all(tf.math.equal(batch_sizes, batch_sizes[0]))
        except AssertionError:
            logger.error("The Buffer is storing an episode with unequal Batch Sizes of its tensors from rollout")
            sys.exit(-1)

        batch_size = batch_sizes[0].numpy()

        # Get indexes of the replay buffer to insert episode batch <---- This still uses numpy. TODO: Convert to TF
        idxs = self._get_storage_idx(batch_size)

        # load inputs into buffers. TODO: Convert to TF compatible since Tensor object doesn't support item assignment
        for key in self.buffer.keys():
            self.buffer[key][idxs] = episode_batch[key]

        self.n_transitions_stored += batch_size * self.T

    def store_transition(self, transition):
        for key in self.buffer.keys():
            self.buffer[key][self.current_outer_idx][self.current_inner_idx] = transition[key]

        # set the success flag of previous transition to 0 given current transition.
        # Success flag for single transition are absent
        if self.current_inner_idx:
            self.buffer['successes'][self.current_outer_idx][self.current_inner_idx - 1] = 0
        self.current_inner_idx += 1
        self.n_transitions_stored += 1

    def new_transition(self):
        self.current_inner_idx = 0
        self.current_outer_idx = self._get_storage_idx()

    def get_current_size_ep(self):
        return self.current_size

    def get_current_size_trans(self):
        return self.current_size * self.T

    def clear_buffer(self):
        self.current_size = 0

    def _get_storage_idx(self, num_to_ins=None):
        num_to_ins = num_to_ins or 1  # buffer size increment
        try:
            assert num_to_ins <= self.buffer_size
        except AssertionError:
            logger.error("Batch committed to replay is too large! {}>{}".format(num_to_ins, self.buffer_size))
            sys.exit(-1)

        # consecutively insert until you hit the end of the buffer, and then insert randomly.
        if self.current_size + num_to_ins <= self.buffer_size:
            idx = np.arange(self.current_size, self.current_size + num_to_ins)
        elif self.current_size < self.buffer_size:
            overflow = num_to_ins - (self.buffer_size - self.current_size)
            idx_a = np.arange(self.current_size, self.buffer_size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.buffer_size, num_to_ins)

        # update buffer size
        self.current_size = min(self.buffer_size, self.current_size + num_to_ins)

        if num_to_ins == 1:
            idx = idx[0]
        return idx

    def maintain_running_stats(self):
        pass

    def save_buffer_data(self, path):
        buffered_data = {}
        for key in self.buffer.keys():
            buffered_data[key] = self.buffer[key][:self.current_size]

        import pickle
        with open(path, 'wb') as handle:
            pickle.dump(buffered_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data_into_buffer(self, path_to_data):
        import pickle
        with open(path_to_data, 'rb') as handle:
            buffered_data = pickle.load(handle)
        self.clear_buffer()
        data_sizes = [len(buffered_data[key]) for key in buffered_data.keys()]
        assert np.all(np.array(data_sizes) == data_sizes[0])
        idxs = self._get_storage_idx(num_to_ins=data_sizes[0])
        for key in buffered_data.keys():
            self.buffer[key][idxs] = buffered_data[key]
