import logging
import sys
from typing import Dict, Tuple, List

import numpy as np
import tensorflow as tf
from tf_agents.replay_buffers.table import Table

from utils.custom import debug

logger = logging.getLogger(__name__)


class ReplayBufferTf:
	def __init__(self, buffer_shapes: Dict[str, Tuple[int, ...]], size_in_transitions, T, transition_fn=None):
		"""Creates a replay buffer.

		Args:
			buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
				buffer
			size_in_transitions (int): the size of the buffer, measured in transitions
			T (int): the time horizon for episodes
			transition_fn (function): a function that samples from the replay buffer
		"""
		self.T = tf.constant(T, dtype=tf.int32)
		self.buffer_size = tf.constant(size_in_transitions // T, dtype=tf.int32)
		
		self.current_size = tf.Variable(0, dtype=tf.int32)  # Size of buffer in terms of no. of episodes
		self.n_transitions_stored = tf.Variable(0, dtype=tf.int32)  # Size of buffer in terms of no. of transitions
		
		self.transition_fn = transition_fn
		self.buffer_keys: List[str] = [key for key in buffer_shapes.keys()]
		tensor_spec = [tf.TensorSpec(buffer_shapes[key], tf.float32, key) for key in self.buffer_keys]
		self.table = Table(tensor_spec, capacity=self.buffer_size)
	
	@tf.function  # Make sure batch_size passed here is a tf.constant to avoid retracing
	def sample_transitions(self, batch_size):
		debug(fn_name="buffer_sample")
		
		# buffered_data = self.sample_episodes(ep_start, ep_end, num_episodes)
		buffered_data = {}
		_data = self.table.read(rows=tf.range(self.current_size))
		for index, key in enumerate(self.buffer_keys):
			buffered_data[key] = _data[index]
		
		transitions = self.transition_fn(buffered_data, batch_size)
		return transitions
	
	@tf.function
	def sample_episodes(self, ep_start: int = None, ep_end: int = None, num_episodes: int = None):
		
		if ep_start is None or ep_end is None:
			if num_episodes:
				num_episodes = tf.math.minimum(tf.cast(num_episodes, dtype=self.current_size.dtype), self.current_size)
			else:
				num_episodes = self.current_size
			ep_range = tf.range(num_episodes)
		else:
			ep_range = tf.range(ep_start, ep_end)
		
		buffered_data = {}
		_data = self.table.read(rows=ep_range)
		for index, key in enumerate(self.buffer_keys):
			buffered_data[key] = _data[index]
		
		return buffered_data
	
	@tf.function
	def store_episode(self, episode_batch):
		"""
			Store each episode into replay buffer
			episode_batch: {"": array(1 x (T or T+1) x dim)}
		"""
		debug(fn_name="buffer_store_ep")
		
		idxs = self._get_storage_idxs(num_to_ins=tf.constant(1, dtype=tf.int32))
		values = [episode_batch[key] for key in self.buffer_keys if key in episode_batch.keys()]
		self.table.write(rows=idxs, values=values)
		self.n_transitions_stored.assign(self.n_transitions_stored + self.T)
	
	def store_episodes(self, episodes_batch):
		for ep_idx in tf.range(tf.shape(episodes_batch['actions'])[0]):
			episode_batch = {}
			for key in self.buffer_keys:
				episode_batch[key] = tf.gather(episodes_batch[key], ep_idx)
			self.store_episode(episode_batch)
	
	def _get_storage_idxs(self, num_to_ins=None):
		if num_to_ins is None:
			num_to_ins = tf.cast(1, dtype=tf.int32)
		
		# consecutively insert until you hit the end of the buffer, and then insert randomly.
		if self.current_size + num_to_ins <= self.buffer_size:
			idxs = tf.range(self.current_size, self.current_size + num_to_ins)
		elif self.current_size < self.buffer_size:
			overflow = num_to_ins - (self.buffer_size - self.current_size)
			idx_a = tf.range(self.current_size, self.buffer_size)
			idx_b = tf.experimental.numpy.random.randint(0, self.current_size, size=(overflow,), dtype=tf.int32)
			idxs = tf.concat([idx_a, idx_b], axis=0)
		else:
			idxs = tf.experimental.numpy.random.randint(0, self.buffer_size, size=(num_to_ins,), dtype=tf.int32)
		
		# update buffer size
		self.current_size.assign(tf.math.minimum(self.buffer_size, self.current_size + num_to_ins))
		return idxs
	
	def get_current_size_ep(self):
		return self.current_size
	
	def get_current_size_trans(self):
		return self.current_size * self.T
	
	def clear_buffer(self):
		self.current_size.assign(0)
	
	@property
	def full(self):
		return self.current_size == self.buffer_size
	
	def __len__(self):
		return self.current_size
	
	def save_buffer_data(self, path):
		buffered_data = {}
		_data = self.table.read(rows=tf.range(self.current_size))
		for index, key in enumerate(self.buffer_keys):
			buffered_data[key] = _data[index]
		
		import pickle
		with open(path, 'wb') as handle:
			pickle.dump(buffered_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	def load_data_into_buffer(self, path_to_data=None, clear_buffer=True, num_demos_to_load=None, buffered_data=None):
		
		import pickle
		
		if buffered_data is None and path_to_data:
			with open(path_to_data, 'rb') as handle:
				buffered_data = pickle.load(handle)
		elif path_to_data is None and buffered_data:
			pass
		elif buffered_data is None and path_to_data is None:
			raise ValueError("Either buffered_data or path_to_data must be provided")
		else:
			raise ValueError("Only one of buffered_data or path_to_data can be provided")
		
		if clear_buffer:
			self.clear_buffer()
		
		if num_demos_to_load is not None:
			
			# Randomly sample idxs to load
			idxs = np.random.choice(len(buffered_data['actions']), size=num_demos_to_load, replace=False).tolist()
			
			for key in buffered_data.keys():
				buffered_data[key] = tf.gather(buffered_data[key], idxs)
		
		# Check if all tensors are present in loaded data
		data_sizes = [len(buffered_data[key]) for key in self.buffer_keys]
		assert np.all(np.array(data_sizes) == data_sizes[0])
		
		idxs = self._get_storage_idxs(num_to_ins=data_sizes[0])
		values = [buffered_data[key] for key in self.buffer_keys]
		
		self.table.write(rows=idxs, values=values)
		self.n_transitions_stored.assign(self.n_transitions_stored + len(idxs) * self.T)
		logger.info("Loaded {} episodes into the buffer.".format(len(idxs)))


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
