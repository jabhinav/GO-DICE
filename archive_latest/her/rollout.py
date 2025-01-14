import logging
import sys
from collections import deque

import numpy as np
import tensorflow as tf
from mujoco_py import MujocoException
import time

from domains.PnP import MyPnPEnvWrapperForGoalGAIL
from utils.debug import debug

logger = logging.getLogger(__name__)


class RolloutWorker:
	def __init__(self, env: MyPnPEnvWrapperForGoalGAIL, policy, T, rollout_terminate=True, compute_c=False,
				 exploit=False, noise_eps=0., random_eps=0., compute_Q=False, use_target_net=False, render=False,
				 history_len=100, is_expert_worker=False):
		"""
		Rollout worker generates experience by interacting with policy.
		Args:
			env: List of environments.
			policy: the policy that is used to act
			T: Horizon Length
			rollout_terminate: If true, rollout is terminated when a goal is reached. Otherwise, it continues
			exploit: whether to explore (random action sampling) or exploit (greedy)
			noise_eps: scale of the additive Gaussian noise
			random_eps: probability of selecting a completely random action
			compute_Q: Whether to compute Q Values
			use_target_net: whether to use the target net for action selection
			history_len (int): length of history for statistics smoothing
		"""
		self.env = env
		self.compute_c = compute_c
		self.policy = policy
		self.horizon = T
		self.rollout_terminate = rollout_terminate
		# self.dims = dims
		
		self.n_episodes = 0
		self.success_history = deque(maxlen=history_len)
		self.Q_history = deque(maxlen=history_len)
		
		if not exploit:
			self.noise_eps = noise_eps
			self.random_eps = random_eps
		else:
			self.noise_eps = 0.
			self.random_eps = 0.
		
		self.compute_Q = compute_Q
		self.use_target_net = use_target_net
		
		self.render = render
		self.resume_state = None
		
		self.is_expert_worker = is_expert_worker
	
	@tf.function
	def reset_rollout(self, reset=True):
		"""
			Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
			and `g` arrays accordingly.
		"""
		if reset:
			curr_state, curr_ag, curr_g = tf.numpy_function(func=self.env.reset, inp=[self.render],
															Tout=(tf.float32, tf.float32, tf.float32))
		else:
			if self.resume_state is None:
				curr_state, curr_ag, curr_g = tf.numpy_function(func=self.env.reset, inp=[self.render],
																Tout=(tf.float32, tf.float32, tf.float32))
			else:
				# The env won't be reset, the environment would be at its last reached position
				curr_state = self.resume_state.copy()
				curr_ag = self.env.transform_to_goal_space(curr_state)
				curr_g = self.env.current_goal
		
		return curr_state, curr_ag, curr_g
	
	def force_reset_rollout(self, init_state_dict):
		# curr_state, curr_ag, curr_g = tf.numpy_function(func=self.env.forced_reset, inp=[init_state_dict, self.render],
		#                                                 Tout=(tf.float32, tf.float32, tf.float32))
		curr_state, curr_ag, curr_g = self.env.forced_reset(init_state_dict, self.render)
		return curr_state, curr_ag, curr_g
	
	@tf.function
	def generate_expert_rollout(self, reset=True):
		# Get the tuple (s_t, s_t+1, g_t, g_t-1, c_t, c_t-1, a_t)
		prev_goals = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  # g_t-1
		prev_skills = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  # c_t-1
		states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  # s_t
		env_goal = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  # g_env
		curr_goals = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  # g_t
		curr_skills = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  # c_t
		obj_identifiers = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  # obj_t
		actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  # a_t
		states_2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  # s_t+1
		
		successes = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		distances = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		
		curr_state, _, g_env = self.reset_rollout(reset=reset)  # Get s_0 and g_env
		curr_goal = curr_state[:3]  # g_-1 = s_0
		curr_skill = tf.squeeze(tf.one_hot(np.array([0]), depth=3, dtype=tf.float32))  # c_-1 = [1, 0, 0]
		init_state_dict = self.env.get_state_dict()
		
		for t in range(self.horizon):
			# Sleep to slow down the simulation
			# time.sleep(0.1)
			
			curr_state = tf.reshape(curr_state, shape=(1, -1))
			g_env = tf.reshape(g_env, shape=(1, -1))
			
			# Save g_t-1, c_t-1, s_t, g_env
			prev_goals = prev_goals.write(t, tf.squeeze(tf.cast(curr_goal, dtype=tf.float32)))
			prev_skills = prev_skills.write(t, tf.squeeze(tf.cast(curr_skill, dtype=tf.float32)))
			states = states.write(t, tf.squeeze(curr_state))
			env_goal = env_goal.write(t, tf.squeeze(g_env))
			
			curr_goal, curr_skill, curr_obj, action = self.policy.act(curr_state, g_env)
			
			# Save g_t, c_t, obj_t, a_t
			curr_goals = curr_goals.write(t, tf.squeeze(tf.cast(curr_goal, dtype=tf.float32)))
			curr_skills = curr_skills.write(t, tf.squeeze(tf.cast(curr_skill, dtype=tf.float32)))
			obj_identifiers = obj_identifiers.write(t, tf.squeeze(tf.cast(curr_obj, dtype=tf.float32)))
			actions = actions.write(t, tf.squeeze(tf.cast(action, dtype=tf.float32)))
			
			try:
				curr_state, _, g_env, done, distance = tf.numpy_function(func=self.env.step,
																		 inp=[action, self.render],
																		 Tout=[tf.float32, tf.float32,
																		 tf.float32, tf.int32, tf.float32])
				
				if self.rollout_terminate:
					done = int(done)
				else:
					done = 0
				
				states_2 = states_2.write(t, tf.squeeze(curr_state))  # s_t+1
				successes = successes.write(t, float(done))
				distances = distances.write(t, distance)
				
			except MujocoException:
				self.generate_expert_rollout(reset=True)
				
		# Save the terminal state and corresponding achieved goal
		states = states.write(self.horizon, tf.squeeze(curr_state))
		env_goal = env_goal.write(self.horizon, tf.squeeze(g_env))
		
		# Stack the arrays
		prev_goals = prev_goals.stack()
		prev_skills = prev_skills.stack()
		states = states.stack()
		env_goal = env_goal.stack()
		curr_goals = curr_goals.stack()
		curr_skills = curr_skills.stack()
		obj_identifiers = obj_identifiers.stack()
		actions = actions.stack()
		states_2 = states_2.stack()
		successes = successes.stack()
		distances = distances.stack()
		
		episode = dict(
			prev_goals=tf.expand_dims(prev_goals, axis=0),
			prev_skills=tf.expand_dims(prev_skills, axis=0),
			states=tf.expand_dims(states, axis=0),
			env_goals=tf.expand_dims(env_goal, axis=0),
			curr_goals=tf.expand_dims(curr_goals, axis=0),
			curr_skills=tf.expand_dims(curr_skills, axis=0),
			obj_identifiers=tf.expand_dims(obj_identifiers, axis=0),
			actions=tf.expand_dims(actions, axis=0),
			states_2=tf.expand_dims(states_2, axis=0),
			successes=tf.expand_dims(successes, axis=0),
			distances=tf.expand_dims(distances, axis=0)
		)
		
		# success_rate = tf.reduce_mean(tf.cast(successes, tf.float32)) #
		if tf.math.equal(tf.argmax(successes), 0):  # We want to check if goal is achieved or not i.e. binary
			success = 0
		else:
			success = 1
		self.success_history.append(tf.cast(success, tf.float32))
		
		# Log stats here to make these two functions part of computation graph of generate_rollout since *history vars
		# can't be used when calling the funcs from outside as *history vars are populated in generate_rollout's graph
		stats = {}
		success_rate = self.current_success_rate()
		stats['success_rate'] = success_rate
		
		self.n_episodes += 1
		
		stats['init_state_dict'] = init_state_dict
		return episode, stats
		
	@tf.function
	def generate_rollout(self, reset=True, slice_goal=None, init_state_dict=None, get_obj=False):
		debug("generate_rollout")
		
		# generate episodes
		states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		achieved_goals = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		states_2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		achieved_goals_2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		goals = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		
		latent_modes = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		obj_identifiers = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		
		successes = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		quality = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		distances = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		
		# Initialize the environment
		if init_state_dict is not None:
			try:
				curr_state, curr_ag, curr_g = self.force_reset_rollout(init_state_dict)
			
			except MujocoException as _:
				logging.error("Some Error occurred while loading initial state in the environment!")
				sys.exit(-1)
		
		else:
			curr_state, curr_ag, curr_g = self.reset_rollout(reset=reset)
			init_state_dict = self.env.get_state_dict()
		
		# print(init_state_dict['goal'])
		# tf.print("Rollout: {}".format(init_state_dict['goal']))
		
		# # Hack: Push this to env. reset
		if not self.is_expert_worker:
			curr_ag = curr_ag[:3]
		
		# Initialise other variables that will be computed
		action, value = tf.zeros(shape=(self.env.action_space.shape[0],)), tf.zeros(shape=(1,))
		latent_mode = tf.numpy_function(func=self.env.get_init_latent_mode, inp=[], Tout=tf.float32)
		latent_mode = tf.expand_dims(latent_mode, axis=0)
		
		# time.sleep(0.5)
		for t in range(self.horizon):
			# Sleep to slow down the simulation
			# time.sleep(0.1)
			
			# Convert state into a batched tensor (batch size = 1)
			curr_state = tf.reshape(curr_state, shape=(1, -1))
			curr_ag = tf.reshape(curr_ag, shape=(1, -1))
			curr_g = tf.reshape(curr_g, shape=(1, -1))
			
			states = states.write(t, tf.squeeze(curr_state))
			achieved_goals = achieved_goals.write(t, tf.squeeze(curr_ag))
			goals = goals.write(t, tf.squeeze(curr_g))
			
			# Run the model and to get action probabilities and critic value
			op = self.policy.act(curr_state, curr_ag, curr_g, prev_latent_mode=latent_mode,
								 compute_Q=self.compute_Q, compute_c=self.compute_c,
								 noise_eps=self.noise_eps, random_eps=self.random_eps,
								 use_target_net=self.use_target_net, get_obj=get_obj)
			
			if self.compute_Q and self.compute_c and get_obj:
				action, value, latent_mode, picked_obj = op
				actions = actions.write(t, tf.squeeze(tf.cast(action, dtype=tf.float32)))
				quality = quality.write(t, tf.squeeze(tf.cast(value, dtype=tf.float32)))
				latent_modes = latent_modes.write(t, tf.squeeze(tf.cast(latent_mode, dtype=tf.float32)))
				obj_identifiers = obj_identifiers.write(t, tf.squeeze(tf.cast(picked_obj, dtype=tf.float32)))
			
			elif self.compute_Q and self.compute_c:
				action, value, latent_mode = op
				actions = actions.write(t, tf.squeeze(tf.cast(action, dtype=tf.float32)))
				quality = quality.write(t, tf.squeeze(tf.cast(value, dtype=tf.float32)))
				latent_modes = latent_modes.write(t, tf.squeeze(tf.cast(latent_mode, dtype=tf.float32)))
			
			elif self.compute_Q:
				action, value = op
				actions = actions.write(t, tf.squeeze(tf.cast(action, dtype=tf.float32)))
				quality = quality.write(t, tf.squeeze(tf.cast(value, dtype=tf.float32)))
			
			elif self.compute_c and get_obj:
				action, latent_mode, picked_obj = op
				actions = actions.write(t, tf.squeeze(tf.cast(action, dtype=tf.float32)))
				latent_modes = latent_modes.write(t, tf.squeeze(tf.cast(latent_mode, dtype=tf.float32)))
				obj_identifiers = obj_identifiers.write(t, tf.squeeze(tf.cast(picked_obj, dtype=tf.float32)))
			
			elif self.compute_c:
				action, latent_mode = op
				actions = actions.write(t, tf.squeeze(tf.cast(action, dtype=tf.float32)))
				latent_modes = latent_modes.write(t, tf.squeeze(tf.cast(latent_mode, dtype=tf.float32)))
			else:
				action = op
				actions = actions.write(t, tf.squeeze(tf.cast(action, dtype=tf.float32)))
			if not self.is_expert_worker:
				next_goal = curr_ag + value
			# Apply action to the environment to get next state and reward
			try:
				curr_state, curr_ag, curr_g, done, distance = tf.numpy_function(func=self.env.step,
																				inp=[action, self.render],
																				Tout=[tf.float32, tf.float32,
																					  tf.float32, tf.int32, tf.float32])
				if self.rollout_terminate:
					done = int(done)
				else:
					done = 0
				
				if not self.is_expert_worker:
					curr_ag = next_goal
				
				states_2 = states_2.write(t, tf.squeeze(curr_state))
				achieved_goals_2 = achieved_goals_2.write(t, tf.squeeze(curr_ag))
				successes = successes.write(t, float(done))
				distances = distances.write(t, distance)
			
			# # We will save the done signals even after the episode terminates in order to maintain traj. length
			# if tf.cast(done, tf.bool):
			#     print("Terminate")
			#     break
			
			except MujocoException:
				self.generate_rollout(reset=True, slice_goal=slice_goal)
		
		# Save the terminal state and corresponding achieved goal
		states = states.write(self.horizon, tf.squeeze(curr_state))
		achieved_goals = achieved_goals.write(self.horizon, tf.squeeze(curr_ag))
		
		# Save the last state for the next episode [if reqd.]
		self.resume_state = curr_state
		
		states = states.stack()
		achieved_goals = achieved_goals.stack()
		states_2 = states_2.stack()
		achieved_goals_2 = achieved_goals_2.stack()
		goals = goals.stack()
		actions = actions.stack()
		
		latent_modes = latent_modes.stack()
		obj_identifiers = obj_identifiers.stack()
		
		quality = quality.stack()
		successes = successes.stack()
		distances = distances.stack()  # The distance between achieved goal and desired goal
		
		episode = dict(states=tf.expand_dims(states, axis=0),
					   achieved_goals=tf.expand_dims(achieved_goals, axis=0),
					   states_2=tf.expand_dims(states_2, axis=0),
					   achieved_goals_2=tf.expand_dims(achieved_goals_2, axis=0),
					   goals=tf.expand_dims(goals, axis=0),
					   actions=tf.expand_dims(actions, axis=0),
					   successes=tf.expand_dims(successes, axis=0),
					   distances=tf.expand_dims(distances, axis=0))
		if self.compute_Q:
			episode['quality'] = tf.expand_dims(quality, axis=0)
		
		if self.compute_c:
			episode['latent_modes'] = tf.expand_dims(latent_modes, axis=0)
		
		if get_obj:
			episode['obj_identifiers'] = tf.expand_dims(obj_identifiers, axis=0)
		
		# success_rate = tf.reduce_mean(tf.cast(successes, tf.float32)) #
		if tf.math.equal(tf.argmax(successes), 0):  # We want to check if goal is achieved or not i.e. binary
			success = 0
		else:
			success = 1
		self.success_history.append(tf.cast(success, tf.float32))
		
		if self.compute_Q:
			self.Q_history.append(tf.reduce_mean(quality))
		
		# Log stats here to make these two functions part of computation graph of generate_rollout since *history vars
		# can't be used when calling the funcs from outside as *history vars are populated in generate_rollout's graph
		stats = {}
		success_rate = self.current_success_rate()
		stats['success_rate'] = success_rate
		if self.compute_Q:
			mean_Q = self.current_mean_Q()
			stats['mean_Q'] = mean_Q
		
		self.n_episodes += 1
		
		stats['init_state_dict'] = init_state_dict
		return episode, stats
	
	def clear_history(self):
		"""
			Clears all histories that are used for statistics
		"""
		self.success_history.clear()
		self.Q_history.clear()
		self.n_episodes = 0
	
	def current_success_rate(self):
		return tf.add_n(self.success_history) / tf.cast(len(self.success_history), dtype=tf.float32)
	
	def current_mean_Q(self):
		return tf.add_n(self.Q_history) / tf.cast(len(self.Q_history), dtype=tf.float32)
