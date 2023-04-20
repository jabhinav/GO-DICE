import logging
import random
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

logger = logging.getLogger(__name__)


def get_segmented_transitions(c_specific_mask: np.ndarray):
	t_samples = []
	past_t = []
	future_t = []
	for rollout_id in range(c_specific_mask.shape[0]):
		t_trg_c = np.where(c_specific_mask[rollout_id] == 1)[0]
		t_segments = np.split(t_trg_c, np.where(np.diff(t_trg_c) != 1)[0] + 1)
		if len(t_segments):
			min_t_segments: List = [c for c in t_segments if len(c) > 2]  # need at least 3 time steps to sample from
			if len(min_t_segments) > 0:
				t_segment: np.ndarray = random.choice(min_t_segments)
				
				# Select a time step randomly from the segment except the last one
				t_curr = random.choice(t_segment[1:-1])
				t_samples.append(t_curr)
				
				# Determine the length of future segment and then the future offset
				c_t_remaining = t_segment[t_segment > t_curr]
				rdm_future_offset_frac = random.uniform(0, 1)
				future_offset = np.floor(rdm_future_offset_frac * len(c_t_remaining)).astype(np.int32)
				t_future = c_t_remaining[future_offset]
				future_t.append(t_future)
				
				# Determine the length of past segment and then the past offset
				rdm_past_offset_frac = random.uniform(0, 1)
				c_t_remaining = t_segment[t_segment < t_curr]
				past_offset = np.floor(rdm_past_offset_frac * len(c_t_remaining)).astype(np.int32)
				t_past = c_t_remaining[-past_offset]
				past_t.append(t_past)
			else:
				# Just put the same time step for all
				t_segment: np.ndarray = random.choice(t_segments)
				t_curr = random.choice(t_segment)
				t_samples.append(t_curr)
				future_t.append(t_curr)
				past_t.append(t_curr)
			
		else:
			raise ValueError("No segments found for the rollout")
		
	return np.array(t_samples, dtype=np.int32), np.array(future_t, dtype=np.int32), np.array(past_t, dtype=np.int32)


@tf.function(experimental_relax_shapes=True)  # Imp otherwise code will be very slow
def get_obj_goal(obj_identifiers, goals, time_steps):
	# Collect object indices and goals based for the time steps
	obj_identifiers = tf.gather_nd(obj_identifiers, indices=time_steps)
	obj_identifiers = tf.cast(obj_identifiers, dtype=time_steps.dtype)
	goals = tf.gather_nd(goals, indices=time_steps)
	
	# Collect the goal for the computed goal_dim (each object's goal is 3-dimensional)
	obj_specific_goals = tf.map_fn(lambda x: x[0][x[1] * 3: (x[1] + 1) * 3], (goals, obj_identifiers),
								   fn_output_signature=tf.float32)
	
	return obj_specific_goals


def get_ep_term_idxs(episodic_data, batch_size):
	T = episodic_data['actions'].shape[1]
	successes = episodic_data['successes']
	# Get index at which episode terminated
	terminate_idxes = tf.math.argmax(successes, axis=-1)
	# If no success, set to last index
	mask_no_success = tf.math.equal(terminate_idxes, 0)
	terminate_idxes += tf.multiply((T - 1) * tf.ones_like(terminate_idxes),
								   tf.cast(mask_no_success, terminate_idxes.dtype))
	
	# Get episode idx for each transition to sample: more likely to sample from episodes which didn't end in success
	p = (terminate_idxes + 1) / tf.reduce_sum(terminate_idxes + 1)
	episode_idxs = tfp.distributions.Categorical(probs=p).sample(sample_shape=(batch_size,))
	episode_idxs = tf.cast(episode_idxs, dtype=terminate_idxes.dtype)
	# Get terminate index for the selected episodes
	terminate_idxes = tf.gather(terminate_idxes, episode_idxs)
	
	return episode_idxs, terminate_idxes
	
	
def sample_transitions(sample_style: str, state_to_goal=None, num_options: int = None):
	
	def sample_skills_random_transitions(episodic_data, batch_size_in_transitions=None):
		"""
		Sample transitions for each option
		"""
		T = episodic_data['actions'].shape[1]
		batch_size = batch_size_in_transitions  # Number of transitions to sample
		
		successes = episodic_data['successes']
		# Get index at which episode terminated
		terminate_idxes = tf.math.argmax(successes, axis=-1)
		# If no success, set to last index
		mask_no_success = tf.math.equal(terminate_idxes, 0)
		terminate_idxes += tf.multiply((T - 1) * tf.ones_like(terminate_idxes),
									   tf.cast(mask_no_success, terminate_idxes.dtype))
		
		# Get episode idx for each transition to sample: more likely to sample from episodes which didn't end in success
		p = (terminate_idxes + 1) / tf.reduce_sum(terminate_idxes + 1)
		
		option_transitions = {}
		# Select episodes for each options
		for i in range(num_options):
			option_transitions[i] = {}
			episode_idxs = tfp.distributions.Categorical(probs=p).sample(sample_shape=(batch_size,))
			episode_idxs = tf.cast(episode_idxs, dtype=terminate_idxes.dtype)
			# Get terminate index for the selected episodes
			terminate_idxes = tf.gather(terminate_idxes, episode_idxs)
			
			# Get the start and end index for each segment of the episode
			c = tf.gather(episodic_data['curr_skills'], episode_idxs)
			c = tf.argmax(c, axis=-1)
			t_mask_trg_c = tf.equal(c, i)
			t_mask_trg_c = tf.cast(t_mask_trg_c, dtype=episode_idxs.dtype)
			t_samples, t_samples_future, t_samples_init = tf.numpy_function(get_segmented_transitions, [t_mask_trg_c],
																			Tout=[tf.int32, tf.int32, tf.int32])
			t_samples = tf.cast(t_samples, dtype=episode_idxs.dtype)
			t_samples_future = tf.cast(t_samples_future, dtype=episode_idxs.dtype)
			t_samples_init = tf.cast(t_samples_init, dtype=episode_idxs.dtype)
			
			# --------------- 3) Select the batch of transitions corresponding to the current time steps ------------
			curr_indices = tf.stack((episode_idxs, t_samples), axis=-1)
			for key in episodic_data.keys():
				option_transitions[i][key] = tf.gather_nd(episodic_data[key], indices=curr_indices)
				
				
			# --------------- 4) Select the batch of transitions corresponding to the future time steps ------------
			future_indices = tf.stack((episode_idxs, t_samples_future), axis=-1)
			option_transitions[i]['her_goals'] = state_to_goal(
				states=tf.gather_nd(episodic_data['states'], indices=future_indices),
				obj_identifiers=None)
			
			# --------------- 5) Select the batch of transitions corresponding to the initial time steps ------------
			init_indices = tf.stack((episode_idxs, t_samples_init), axis=-1)
			option_transitions[i]['init_states'] = tf.gather_nd(episodic_data['states'], indices=init_indices)
		
		return option_transitions
	
	def sample_random_transitions(episodic_data, batch_size_in_transitions=None):
		"""
		Sample random transitions without HER.
		Functionality: Sample random time-steps from each episode: (g_t-1, c_t-1, s_t, g_t, c_t, a_t) for all episodes.
		"""

		batch_size = batch_size_in_transitions  # Number of transitions to sample
		episode_idxs, terminate_idxes = get_ep_term_idxs(episodic_data, batch_size)
		
		# ------------------------------------------------------------------------------------------------------------
		# --------------------------------- 2) Select which time steps + goals to use --------------------------------
		# Get the current time step
		t_samples_frac = tf.experimental.numpy.random.random(size=(batch_size,))
		t_samples = t_samples_frac * tf.cast(terminate_idxes, dtype=t_samples_frac.dtype)
		t_samples = tf.cast(tf.round(t_samples), dtype=terminate_idxes.dtype)
		
		# Get random init time step (before t_samples)
		# rdm_past_offset_frac = tf.experimental.numpy.random.random(size=(batch_size,))
		# Instead rdm_past_offset_frac should be 0
		rdm_past_offset_frac = tf.zeros_like(t_samples_frac)
		t_samples_init = rdm_past_offset_frac * tf.cast(t_samples, dtype=rdm_past_offset_frac.dtype)
		t_samples_init = tf.cast(tf.floor(t_samples_init), dtype=t_samples.dtype)
		
		# Get the future time step
		rdm_future_offset_frac = tf.experimental.numpy.random.random(size=(batch_size,))
		future_offset = rdm_future_offset_frac * tf.cast((terminate_idxes - t_samples), rdm_future_offset_frac.dtype)
		# future_offset = future_offset_frac * tf.cast((terminate_idxes - t_samples), future_offset_frac.dtype)
		future_offset = tf.cast(future_offset, terminate_idxes.dtype)
		t_samples_future = t_samples + future_offset
		
		# ------------------------------------------------------------------------------------------------------------
		# ----------------- 3) Select the batch of transitions corresponding to the current time steps ---------------
		curr_indices = tf.stack((episode_idxs, t_samples), axis=-1)
		transitions = {}
		for key in episodic_data.keys():
			transitions[key] = tf.gather_nd(episodic_data[key], indices=curr_indices)
		
		# Get initial state
		init_indices = tf.stack((episode_idxs, t_samples_init), axis=-1)
		transitions['init_states'] = tf.gather_nd(episodic_data['states'], indices=init_indices)
		
		# Get the goals to achieve for t0, t and t+1.
		future_indices = tf.stack((episode_idxs, t_samples_future), axis=-1)
		transitions['her_goals'] = state_to_goal(states=tf.gather_nd(episodic_data['states'], indices=future_indices),
												 obj_identifiers=None)  # Object identifiers are not used for unsegmented HER
		
		return transitions
	
	
	if sample_style == 'random_unsegmented':
		return sample_random_transitions
	elif sample_style == 'random_segmented':
		return sample_skills_random_transitions
	else:
		raise NotImplementedError
		
