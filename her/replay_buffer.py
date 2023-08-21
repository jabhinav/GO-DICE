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

		if buffered_data is None:
			raise ValueError("No buffered_data provided")

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
		# logger.info("Loaded {} episodes into the buffer.".format(len(idxs)))