import logging
import random
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from utils.custom import debug

logger = logging.getLogger(__name__)


def get_segmented_transitions(c_specific_mask: np.ndarray, future_offset_frac: float = None):
	t_samples = []
	future_t = []
	for rollout_id in range(c_specific_mask.shape[0]):
		t_trg_c = np.where(c_specific_mask[rollout_id] == 1)[0]
		t_segments = np.split(t_trg_c, np.where(np.diff(t_trg_c) != 1)[0] + 1)
		t_segments: List = [c for c in t_segments if len(c) > 1]  # remove single element segments
		t_segment: np.ndarray = random.choice(t_segments)
		
		# Select a time step randomly from the segment except the last one
		t_curr = random.choice(t_segment[:-1])
		t_samples.append(t_curr)

		if future_offset_frac:
			# Determine the length of remaining segment
			c_t_remaining = t_segment[t_segment > t_curr]
			# Determine the length of the future offset
			future_offset = int(future_offset_frac * len(c_t_remaining))
			# Select the future offset
			t_future = c_t_remaining[future_offset]
			future_t.append(t_future)
	return np.array(t_samples, dtype=np.int32), np.array(future_t, dtype=np.int32)


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


def sample_her_transitions(sample_style: str, state_to_goal, future_offset_frac: float = None):
	if future_offset_frac:
		future_offset_frac = tf.constant(future_offset_frac, dtype=tf.float32)
	
	# Following function is not tf.function compatible (TODO: Make it compatible later)
	# [used by buffer.sample_transitions which is tf.function compatible]
	def sample_skills_random_transitions(episodic_data, batch_size_in_transitions=None, num_options=3):
		debug(fn_name="_sample_options_transitions")
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
			c = tf.gather(episodic_data['latent_modes'], episode_idxs)
			c = tf.argmax(c, axis=-1)
			t_mask_trg_c = tf.equal(c, i)
			t_mask_trg_c = tf.cast(t_mask_trg_c, dtype=episode_idxs.dtype)
			t_samples, future_t = tf.numpy_function(get_segmented_transitions, [t_mask_trg_c, future_offset_frac],
													Tout=[tf.int32, tf.int32])
			t_samples = tf.cast(t_samples, dtype=episode_idxs.dtype)
			future_t = tf.cast(future_t, dtype=episode_idxs.dtype)
			
			# --------------- 3) Select the batch of transitions corresponding to the current time steps ------------
			curr_indices = tf.stack((episode_idxs, t_samples), axis=-1)
			for key in episodic_data.keys():
				option_transitions[i][key] = tf.gather_nd(episodic_data[key], indices=curr_indices)
			
			# if obj_space_as_goal:
			# 	curr_ag = tf.numpy_function(get_obj_goal, [episodic_data['obj_identifiers'],
			# 											   episodic_data['achieved_goals'],
			# 											   curr_indices], Tout=tf.float32)
			# 	option_transitions[i]['achieved_goals'] = curr_ag
			# 	curr_ag2 = tf.numpy_function(get_obj_goal, [episodic_data['obj_identifiers'],
			# 												episodic_data['achieved_goals_2'],
			# 												curr_indices], Tout=tf.float32)
			# 	option_transitions[i]['achieved_goals_2'] = curr_ag2
			#
			# # # ------------------------------------ 4) Determine future goal to achieve -----------------------------
			# t_samples_future = tf.stack((episode_idxs, future_t), axis=-1)
			# if obj_space_as_goal:
			# 	future_ag = tf.numpy_function(get_obj_goal, [episodic_data['obj_identifiers'],
			# 												 episodic_data['achieved_goals'],
			# 												 t_samples_future], Tout=tf.float32)
			# else:
			# 	future_ag = tf.gather_nd(episodic_data['achieved_goals'], indices=t_samples_future)
			# option_transitions[i]['inter_goals'] = future_ag
		
		return option_transitions
	
	def sample_random_transitions(episodic_data, batch_size_in_transitions=None):
		debug(fn_name="_sample_transitions")
		
		batch_size = batch_size_in_transitions  # Number of transitions to sample
		episode_idxs, terminate_idxes = get_ep_term_idxs(episodic_data, batch_size)
		
		# ------------------------------------------------------------------------------------------------------------
		# --------------------------------- 2) Select which time steps + goals to use --------------------------------
		# Get the current time step
		t_samples_frac = tf.experimental.numpy.random.random(size=(batch_size,))
		t_samples = t_samples_frac * tf.cast(terminate_idxes, dtype=t_samples_frac.dtype)
		t_samples = tf.cast(tf.round(t_samples), dtype=terminate_idxes.dtype)
		
		# Get random init time step (before t_samples)
		rdm_past_offset_frac = tf.experimental.numpy.random.random(size=(batch_size,))
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
												 obj_identifiers=None)
		
		# if obj_space_as_goal:
		# 	curr_ag = tf.numpy_function(get_obj_goal, [episodic_data['obj_identifiers'],
		# 											   episodic_data['achieved_goals'],
		# 											   curr_indices], Tout=tf.float32)
		# 	transitions['achieved_goals'] = curr_ag
		# 	curr_ag2 = tf.numpy_function(get_obj_goal, [episodic_data['obj_identifiers'],
		# 												episodic_data['achieved_goals_2'],
		# 												curr_indices], Tout=tf.float32)
		# 	transitions['achieved_goals_2'] = curr_ag2
		#
		# t_samples_future = tf.stack((episode_idxs, future_t), axis=-1)
		# if obj_space_as_goal:
		# 	future_ag = tf.numpy_function(get_obj_goal, [episodic_data['obj_identifiers'],
		# 												 episodic_data['achieved_goals'],
		# 												 t_samples_future], Tout=tf.float32)
		# else:
		# 	future_ag = tf.gather_nd(episodic_data['achieved_goals'], indices=t_samples_future)
		# transitions['inter_goals'] = future_ag
		return transitions
	
	def sample_all_transitions(episodic_data, num_episodes):
		"""
		Sample all transitions without HER.
		Functionality: Sample all time-steps from each episode: (s_t, a_t) for all episodes.
		"""
		debug(fn_name="_sample_all_transitions")
		
		T = episodic_data['actions'].shape[1]
		
		successes = episodic_data['successes']
		# Get index at which episode terminated
		terminate_idxes = tf.math.argmax(successes, axis=-1)
		# If no success, set to last index
		mask_no_success = tf.math.equal(terminate_idxes, 0)
		terminate_idxes += tf.multiply((T - 1) * tf.ones_like(terminate_idxes),
									   tf.cast(mask_no_success, terminate_idxes.dtype))
		
		curr_indices = tf.TensorArray(dtype=terminate_idxes.dtype, size=0, dynamic_size=True)
		transition_idx = 0
		# TODO: This is giving ValueError: None values not supported. (while tracing num_episodes=None)
		for ep in tf.range(num_episodes, dtype=terminate_idxes.dtype):
			for t in tf.range(0, terminate_idxes[ep] - 1, dtype=terminate_idxes.dtype):
				# Get t
				curr_index = tf.stack((ep, t), axis=-1)
				curr_indices = curr_indices.write(transition_idx, curr_index)
				transition_idx += 1
		
		curr_indices = curr_indices.stack()
		transitions = {}
		for key in episodic_data.keys():
			transitions[key] = tf.gather_nd(episodic_data[key], indices=curr_indices)
		
		if obj_space_as_goal:
			curr_ag = tf.numpy_function(get_obj_goal, [episodic_data['obj_identifiers'],
													   episodic_data['achieved_goals'],
													   curr_indices], Tout=tf.float32)
			transitions['achieved_goals'] = curr_ag
			curr_ag2 = tf.numpy_function(get_obj_goal, [episodic_data['obj_identifiers'],
														episodic_data['achieved_goals_2'],
														curr_indices], Tout=tf.float32)
			transitions['achieved_goals_2'] = curr_ag2
		
		return transitions
	
	if sample_style == 'random_unsegmented':
		return sample_random_transitions
	elif sample_style == 'all_unsegmented':
		return sample_all_transitions
	elif sample_style == 'random_segmented':
		return sample_skills_random_transitions
	else:
		raise NotImplementedError
	
	
def sample_non_her_transitions(sample_style: str):
	
	def sample_skills_random_transitions(episodic_data, batch_size_in_transitions=None, num_skills=3):
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
		
		segmented_transitions = {}
		# Select episodes for each options
		for i in range(num_skills):
			segmented_transitions[i] = {}
			episode_idxs = tfp.distributions.Categorical(probs=p).sample(sample_shape=(batch_size,))
			episode_idxs = tf.cast(episode_idxs, dtype=terminate_idxes.dtype)
			# Get terminate index for the selected episodes
			terminate_idxes = tf.gather(terminate_idxes, episode_idxs)
			
			# Get the start and end index for each skill segment of the episode
			c = tf.gather(episodic_data['curr_skills'], episode_idxs)
			c = tf.argmax(c, axis=-1)
			t_mask_trg_c = tf.equal(c, i)
			t_mask_trg_c = tf.cast(t_mask_trg_c, dtype=episode_idxs.dtype)
			t_samples, _ = tf.numpy_function(get_segmented_transitions, [t_mask_trg_c],
											 Tout=[tf.int32, tf.int32])
			t_samples = tf.cast(t_samples, dtype=episode_idxs.dtype)
			
			# --------------- 3) Select the batch of transitions corresponding to the current time steps ------------
			curr_indices = tf.stack((episode_idxs, t_samples), axis=-1)
			for key in episodic_data.keys():
				segmented_transitions[i][key] = tf.gather_nd(episodic_data[key], indices=curr_indices)
		
		return segmented_transitions
	
	def sample_skills_all_transitions(episodic_data, num_episodes, num_skills=3):
		"""
		Sample transitions for each option
		"""
		T = episodic_data['actions'].shape[1]
		successes = episodic_data['successes']
		# Get index at which episode terminated
		terminate_idxes = tf.math.argmax(successes, axis=-1)
		# If no success, set to last index
		mask_no_success = tf.math.equal(terminate_idxes, 0)
		terminate_idxes += tf.multiply((T - 1) * tf.ones_like(terminate_idxes),
									   tf.cast(mask_no_success, terminate_idxes.dtype))
		
		# Create empty TensorArray to store transitions for each skill
		segmented_transitions = {}
		for i in range(num_skills):
			segmented_transitions[i] = {}
			curr_indices = tf.TensorArray(dtype=terminate_idxes.dtype, size=0, dynamic_size=True)
			transition_idx = 0
			for ep in tf.range(num_episodes, dtype=terminate_idxes.dtype):
				for t in tf.range(0, terminate_idxes[ep] - 1, dtype=terminate_idxes.dtype):
					if tf.argmax(episodic_data['curr_skills'][ep][t], axis=-1) == i:
						curr_indices = curr_indices.write(transition_idx, [ep, t])
						transition_idx += 1
			curr_indices = curr_indices.stack()
			for key in episodic_data.keys():
				segmented_transitions[i][key] = tf.gather_nd(episodic_data[key], indices=curr_indices)
				
		return segmented_transitions
	
	def sample_random_transitions(episodic_data, batch_size_in_transitions=None):
		"""
		Sample random transitions without HER.
		Functionality: Sample random time-steps from each episode: (g_t-1, c_t-1, s_t, g_t, c_t, a_t) for all episodes.
		"""
		debug(fn_name="_sample_transitions")
		
		T = episodic_data['actions'].shape[1]
		batch_size = batch_size_in_transitions  # Number of transitions to sample
		
		successes = episodic_data['successes']
		# Get index at which episode terminated
		terminate_idxes = tf.math.argmax(successes, axis=-1)
		# If no success, set to last index
		mask_no_success = tf.math.equal(terminate_idxes, 0)
		terminate_idxes += tf.multiply((T - 1) * tf.ones_like(terminate_idxes), tf.cast(mask_no_success, terminate_idxes.dtype))
		
		# Get episode idx for each transition to sample: more likely to sample from episodes which didn't end in success
		p = (terminate_idxes + 1) / tf.reduce_sum(terminate_idxes + 1)
		episode_idxs = tfp.distributions.Categorical(probs=p).sample(sample_shape=(batch_size,))
		episode_idxs = tf.cast(episode_idxs, dtype=terminate_idxes.dtype)
		# Get terminate index for the selected episodes
		terminate_idxes = tf.gather(terminate_idxes, episode_idxs)
		
		# ------------------------------------------------------------------------------------------------------------
		# --------------------------------- 2) Select which time steps + goals to use --------------------------------
		# Get the current time step
		t_samples_frac = tf.experimental.numpy.random.random(size=(batch_size,))
		t_samples = t_samples_frac * tf.cast(terminate_idxes, dtype=t_samples_frac.dtype)
		t_samples = tf.cast(tf.round(t_samples), dtype=terminate_idxes.dtype)
		
		# Get random init time step (before t_samples)
		rdm_past_offset_frac = tf.experimental.numpy.random.random(size=(batch_size,))
		t_samples_init = rdm_past_offset_frac * tf.cast(t_samples, dtype=rdm_past_offset_frac.dtype)
		t_samples_init = tf.cast(tf.floor(t_samples_init), dtype=t_samples.dtype)
		
		# ------------------------------------------------------------------------------------------------------------
		# ----------------- 3) Select the batch of transitions corresponding to the current time steps ---------------
		curr_indices = tf.stack((episode_idxs, t_samples), axis=-1)
		transitions = {}
		for key in episodic_data.keys():
			transitions[key] = tf.gather_nd(episodic_data[key], indices=curr_indices)
			
		init_indices = tf.stack((episode_idxs, t_samples_init), axis=-1)
		transitions['init_states'] = tf.gather_nd(episodic_data['states'], indices=init_indices)

		return transitions
	 
	def sample_all_transitions(episodic_data, num_episodes):
		"""
		Sample all transitions without HER.
		Functionality: Sample all time-steps from each episode: (g_t-1, c_t-1, s_t, g_t, c_t, a_t) for all episodes.
		"""
		debug(fn_name="_sample_all_transitions")
		
		T = episodic_data['actions'].shape[1]
		successes = episodic_data['successes']
		# Get index at which episode terminated
		terminate_idxes = tf.math.argmax(successes, axis=-1)
		# If no success, set to last index
		mask_no_success = tf.math.equal(terminate_idxes, 0)
		terminate_idxes += tf.multiply((T - 1) * tf.ones_like(terminate_idxes),
									   tf.cast(mask_no_success, terminate_idxes.dtype))
		
		curr_indices = tf.TensorArray(dtype=terminate_idxes.dtype, size=0, dynamic_size=True)
		transition_idx = 0
		for ep in tf.range(num_episodes, dtype=terminate_idxes.dtype):
			for t in tf.range(0, terminate_idxes[ep] - 1, dtype=terminate_idxes.dtype):
				# Get t
				curr_index = tf.stack((ep, t), axis=-1)
				curr_indices = curr_indices.write(transition_idx, curr_index)
				transition_idx += 1
		
		curr_indices = curr_indices.stack()
		transitions = {}
		for key in episodic_data.keys():
			transitions[key] = tf.gather_nd(episodic_data[key], indices=curr_indices)
			
		return transitions
	
	
	if sample_style == 'random_unsegmented':
		return sample_random_transitions
	elif sample_style == 'all_unsegmented':
		return sample_all_transitions
	elif sample_style == 'random_segmented':
		return sample_skills_random_transitions
	elif sample_style == 'all_segmented':
		return sample_skills_all_transitions
	else:
		raise NotImplementedError
		
