import collections
import logging
import time
from typing import Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

OBJ_MAPPING = {
	'0': 0,
	'1': 1,
	'-1': 2,  # Use when no object is left to be picked up
}

SKILL_MAPPING = {
	'pick': 0,  # Go with open gripper (should be capable of going anywhere just by conditioning on goal)
	'grab': 1,  # Grab the object gradually by closing th gripper (condition only on object's position)
	'drop': 2  # Go with closed gripper (should be capable of going anywhere just by conditioning on goal)
}


def Step(observation, reward, done, **kwargs):
	"""
	Convenience method creating a namedtuple with the results of the
	environment's step method.
	Put extra diagnostic info in the kwargs
	"""
	_Step = collections.namedtuple("Step", ["observation", "reward", "done", "info"])
	return _Step(observation, reward, done, kwargs)


def clip_action(ac):
	return np.clip(ac, -1, 1)


class PnPEnv(object):
	def __init__(self, full_space_as_goal=False, goal_weight=1., terminal_eps=0.01, feasible_hand=True, two_obj=False,
				 first_in_place=False, stacking=False, target_in_the_air=True, fix_goal=False):
		"""
		Pick and Place Environment: can be single or multi-object
		Args:
			goal_weight:
			terminal_eps:
			full_space_as_goal: Whether to add gripper position [:3] to the goal space
			feasible_hand:
			two_obj:
			first_in_place:
			stacking:
			fix_goal:
		"""
		
		if not two_obj:
			from .Fetch_Base.fetch_env_oneobj import FetchPickAndPlaceEnv
			env = FetchPickAndPlaceEnv()
		else:
			from .Fetch_Base.fetch_env_twoobj import FetchPickAndPlaceEnv
			env = FetchPickAndPlaceEnv(stacking=stacking, first_in_place=first_in_place,
									   target_in_the_air=target_in_the_air)
		
		env.unwrapped.spec = self
		self._env = env
		
		self._observation_space = env.observation_space.spaces['observation']
		self._action_space = env.action_space
		
		self._current_goal = None
		
		self.goal_weight = goal_weight
		self.terminal_eps = terminal_eps
		self.full_space_as_goal = full_space_as_goal
		self.two_obj = two_obj
		
		self.feasible_hand = feasible_hand  # if the position of the hand is always feasible to achieve
		
		self.fix_goal = fix_goal
		if fix_goal:
			self.fixed_goal = np.array([1.48673746, 0.69548325, 0.6])
		
		if two_obj:
			self.latent_dim = 3
		else:
			self.latent_dim = 3
	
	@property
	def observation_space(self):
		return self._observation_space
	
	@property
	def action_space(self):
		return self._action_space
	
	@property
	def current_goal(self):
		return self._current_goal
	
	def reset(self):
		d = self._env.reset()
		# Update the goal based on some checks
		self.update_goal(d=d)
		return self._transform_obs(d['observation'])
	
	def forced_reset(self, state_dict):
		d = self._env.forced_reset(state_dict)
		# Update the goal based on some checks
		self.update_goal(d=d)
		return self._transform_obs(d['observation'])
	
	def get_state_dict(self):
		state_dict = self._env.get_state_dict()
		# tf.print("PnPEnv: {}".format(state_dict['goal']))
		return state_dict
	
	def step(self, action):
		next_obs, reward, _, info = self._env.step(
			action)  # FetchPickAndPlaceEnv freezes done to False and stores termination response in info
		next_obs = self._transform_obs(next_obs['observation'])  # Remove unwanted portions of the observed state
		info['obs2goal'] = self.transform_to_goal_space(next_obs)
		info['distance'] = np.linalg.norm(self.current_goal - info['obs2goal'])
		if self.full_space_as_goal:
			info['block_distance'] = np.linalg.norm((self.current_goal - info['obs2goal'])[3:6])
			info['hand_distance'] = np.linalg.norm((self.current_goal - info['obs2goal'])[0:3])
		info['goal_reached'] = info['distance'] < self.terminal_eps
		done = info['goal_reached']
		return Step(next_obs, reward, done, **info)
	
	def get_current_obs(self):
		"""
		:return: current observation (state of the robot)
		"""
		return self._transform_obs(self._env._get_obs()['observation'])
	
	def transform_to_goal_space(self, obs):
		"""
			Transform the observation to the goal space by extracting the achieved goal from the observation
			For the PnP, it corresponds to obj. positions
		"""
		if not self.full_space_as_goal:
			ret = np.array(obs[3:6])
		else:
			ret = np.array(obs[:6])
		if self.two_obj:
			ret = np.concatenate([ret, obs[6:9]])
		return ret
	
	def render(self):
		self._env.render()
	
	def _transform_obs(self, obs):
		"""
			Extract the relevant information from the observation
		"""
		if self.two_obj:
			return obs[:16]
		else:
			return obs[:10]
	
	def sample_hand_pos(self, block_pos):
		if block_pos[2] == self._env.height_offset or not self.feasible_hand:
			xy = self._env.initial_gripper_xpos[:2] + np.random.uniform(-0.15, 0.15, size=2)
			z = np.random.uniform(self._env.height_offset, self._env.height_offset + 0.3)
			return np.concatenate([xy, [z]])
		else:
			return block_pos
	
	def update_goal(self, d=None):
		"""
			Set the goal for the env. in the current episode
		"""
		# Fix objects
		if self.get_current_obs()[5] < self._env.height_offset or \
				np.any(self.get_current_obs()[3:5] > self._env.initial_gripper_xpos[:2] + 0.15) or \
				np.any(self.get_current_obs()[3:5] < self._env.initial_gripper_xpos[:2] - 0.15):
			self._env._reset_sim()
		
		# Fix Goals
		if self.fix_goal:
			self._current_goal = self.fixed_goal
		else:
			
			if d is not None:
				self._current_goal = d['desired_goal']
			else:
				self._current_goal = self._env.goal = np.copy(self._env._sample_goal())
			
			if self.full_space_as_goal:
				self._current_goal = np.concatenate([self.sample_hand_pos(self._current_goal), self._current_goal])
	
	def set_feasible_hand(self, _bool):
		self.feasible_hand = _bool
	
	def get_init_latent_mode(self):
		init_latent_mode = SKILL_MAPPING['pick']
		# init_latent_mode = ACTION_TO_LATENT_MAPPING['pick:obj1']  # Randomly pick init latent mode for Dataset:1/2
		init_latent_mode = tf.one_hot(init_latent_mode, depth=self.latent_dim, dtype=tf.float32)
		init_latent_mode = tf.squeeze(init_latent_mode)
		return init_latent_mode


class MyPnPEnvWrapperForGoalGAIL(PnPEnv):
	def __init__(self, full_space_as_goal=False, **kwargs):
		"""
		GoalGAIL compatible Wrapper for PnP Env
		Args:
			full_space_as_goal:
			**kwargs:
		"""
		super(MyPnPEnvWrapperForGoalGAIL, self).__init__(full_space_as_goal, **kwargs)
	
	def reset(self, render=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		obs = super(MyPnPEnvWrapperForGoalGAIL, self).reset()
		if render:
			super(MyPnPEnvWrapperForGoalGAIL, self).render()
		achieved_goal = super(MyPnPEnvWrapperForGoalGAIL, self).transform_to_goal_space(obs)
		desired_goal = super(MyPnPEnvWrapperForGoalGAIL, self).current_goal
		return obs.astype(np.float32), achieved_goal.astype(np.float32), desired_goal.astype(np.float32)
	
	def step(self, action, render=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		if len(action.shape) > 1:
			action = action[0]
		obs, _, done, info = super(MyPnPEnvWrapperForGoalGAIL, self).step(action)  # ignore reward (re-computed in HER)
		
		if render:
			super(MyPnPEnvWrapperForGoalGAIL, self).render()
		
		achieved_goal = super(MyPnPEnvWrapperForGoalGAIL, self).transform_to_goal_space(obs)
		desired_goal = super(MyPnPEnvWrapperForGoalGAIL, self).current_goal
		success = int(done)
		
		return obs.astype(np.float32), achieved_goal.astype(np.float32), desired_goal.astype(np.float32), np.array(
			success, np.int32), info['distance'].astype(np.float32)
	
	def forced_reset(self, state_dict, render=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		obs = super(MyPnPEnvWrapperForGoalGAIL, self).forced_reset(state_dict)
		if render:
			super(MyPnPEnvWrapperForGoalGAIL, self).render()
		achieved_goal = super(MyPnPEnvWrapperForGoalGAIL, self).transform_to_goal_space(obs)
		desired_goal = super(MyPnPEnvWrapperForGoalGAIL, self).current_goal
		return obs.astype(np.float32), achieved_goal.astype(np.float32), desired_goal.astype(np.float32)
	
	def get_state_dict(self):
		state_dict = super(MyPnPEnvWrapperForGoalGAIL, self).get_state_dict()
		# tf.print("MyPnPEnvWrapperForGoalGAIL: {}".format(state_dict['goal']))
		return state_dict
	
	def reward_fn(self, ag_2, g, o, relative_goal=False, distance_metric='L1', only_feasible=False,
				  extend_dist_rew_weight=0.):
		"""
			Custom reward function with two components:
				(a) 0/1 for whether goal is reached (weighed by self.goal_weight),
				(b) relative distance between achieved goal and desired goal (weighed by extend_dist_rew_weight)
		"""
		
		if relative_goal:
			dif = o[:, -2:]
		else:
			dif = ag_2 - g
		
		if distance_metric == 'L1':
			goal_distance = np.linalg.norm(dif, ord=1, axis=-1)
		elif distance_metric == 'L2':
			goal_distance = np.linalg.norm(dif, ord=2, axis=-1)
		elif callable(distance_metric):
			goal_distance = distance_metric(ag_2, g)
		else:
			raise NotImplementedError('Unsupported distance metric type.')
		
		# if only_feasible:
		#     ret = np.logical_and(goal_distance < self.terminal_eps, [self.is_feasible(g_ind) for g_ind in
		#                                                              g]) * self.goal_weight - \
		#           extend_dist_rew_weight * goal_distance
		# else:
		ret = (goal_distance < self.terminal_eps) * self.goal_weight - extend_dist_rew_weight * goal_distance
		
		return ret


class PnPExpertOld:
	def __init__(self, num_skills=3):
		self.block0_picked = None
		self.step_size = 6
		self.num_skills = num_skills  # Should be same as the latent_dim of the PnPEnv
		self.time_step = 0
		self.start = None  # Initial position of the gripper
		
		self.reset()
	
	def act(self, state, env_goal, noise_eps=0., random_eps=0., compute_c=True, get_obj=False, **kwargs):
		if self.time_step == 0:
			self.start = state[0][:3]  # Gripper Position
		else:
			self.time_step += 1
		
		a, curr_goal, skill, obj = tf.numpy_function(func=self.transition, inp=[state[0], env_goal[0]],
													 Tout=[tf.float32, tf.float32, tf.int32, tf.int32])
		a = tf.numpy_function(func=self.add_noise_to_action, inp=[a, noise_eps, random_eps], Tout=tf.float32)
		a = tf.squeeze(a)
		
		skill = tf.one_hot(skill, depth=self.num_skills, dtype=tf.float32)
		skill = tf.squeeze(skill)
		
		# obj = tf.one_hot(obj, depth=2, dtype=tf.float32)
		# obj = tf.squeeze(obj)
		
		return curr_goal, skill, obj, a
	
	def transition(self, o, g) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		gripper_pos = o[:3]
		block_pos = o[3:6]
		env_goal_pos = g[-3:]
		block_rel_pos = o[6:9]  # block_pos - gripper_pos
		# print("---- #  |{}| -> |{}| # ----".format(np.linalg.norm(block_rel_pos), np.linalg.norm(env_goal_pos-block_pos)))
		
		if np.linalg.norm(block_rel_pos) > 0.05 and np.linalg.norm(block_rel_pos - np.array([0, 0, 0.03])) > 0.001:
			if self.block0_picked:
				self.reset()
			
			skill = 'pick'
			curr_goal = block_pos
			a = self.goto_block(gripper_pos, block_pos)
		
		elif np.linalg.norm(block_rel_pos) > 0.01 and not self.block0_picked:
			skill = 'grab'
			curr_goal = block_pos
			a = self.pickup_block(gripper_pos, block_pos)
		
		else:
			self.block0_picked = True
			skill = 'drop'
			curr_goal = env_goal_pos
			a = self.goto_goal(block_pos, env_goal_pos)
		
		obj = '0'
		return a, curr_goal, np.array([SKILL_MAPPING[skill]], dtype=np.int32), \
			   np.array([OBJ_MAPPING[obj]], dtype=np.int32)
	
	def reset(self):
		self.time_step = 0
		self.start = None
		#  # Hack: This flag helps bypass the irregular behaviour i.e. when block is picked and
		# gripper starts to move towards the goal, object's pos relative to gripper sometime inc. which then violates
		# move-to-goal condition 'np.linalg.norm(block_rel_pos) < 0.01'(see orig. implementation of GoalGAIL).
		# This behaviour is periodic making the gripper oscillate
		self.block0_picked = False
	
	def goto_block(self, cur_pos, block_pos, grip=1.0):
		target_pos = block_pos + np.array([0, 0, 0.03])
		a = clip_action((target_pos - cur_pos) * self.step_size)
		a = np.concatenate([a, np.array([grip])], dtype=np.float32)
		return a
	
	def pickup_block(self, cur_pos, block_pos, ):
		"""
		:param cur_pos: current gripper position
		:param block_pos: block position
		:return: action
		Logic: Move the gripper to the block position (w partially closed gripper = -0.005) and then close the gripper
		"""
		if np.linalg.norm(cur_pos - block_pos) < 0.01:  # and gripper_state > 0.025: # TODO: need to adjust
			a = np.array([0, 0, 0, -1.], dtype=np.float32)
		else:
			a = clip_action((block_pos - cur_pos) * self.step_size)
			a = np.concatenate([a, np.array([-0.005])], dtype=np.float32)
		return a
	
	def goto_goal(self, cur_pos, goal_pos, grip=-1.):
		"""
		:param cur_pos: current block position (overlapping with gripper)
		:param goal_pos: env goal position
		:param grip: gripper state (set to -1 to close the gripper)
		"""
		
		# We then move the gripper towards the goal -> Doing this after above action brings gripper vertically down
		if np.linalg.norm((goal_pos - cur_pos)) > 0.01:
			curr_goal = goal_pos  # = (x_goal, y_goal, z_goal)
			a = clip_action((goal_pos - cur_pos) * self.step_size)
			a = np.concatenate([a, np.array([grip])], dtype=np.float32)
		
		else:
			# ac = np.array([0, 0, 0, grip], dtype=np.float32)
			raise ValueError("Goal reached! Should not be in goto_goal!")
		
		return a, curr_goal
	
	def add_noise_to_action(self, a, noise_eps=0., random_eps=0., ):
		a = np.array([a])
		noise = noise_eps * np.random.randn(*a.shape)  # gaussian noise
		a += noise
		a = np.clip(a, -1, 1)
		a += np.random.binomial(1, random_eps, a.shape[0]).reshape(-1, 1) * (self._random_action(a.shape[0]) - a)
		return a
	
	@staticmethod
	def _random_action(n):
		return np.random.uniform(low=-1, high=1, size=(n, 4))


class PnPExpertTwoObjOld:
	def __init__(self, num_skills=3, expert_behaviour: str = '0'):
		self.block0_picked, self.block1_picked = None, None  # flags to check if block is picked
		self.block0_raised, self.block1_raised = None, None  # flags to check if block is raised after pickup
		# flags to check if block is dropped (Hack: after drop, dropped block's pos rel to its goal is within limits
		# for one step, then it gets out of limit for another step and then again within limits finally)
		self.block0_dropped, self.block1_dropped = None, None
		self.gripper_raised = None  # flag to check if gripper is raised after drop
		
		self.picking_order = None
		self.step_size = 6
		self.post_completion_step_size = 2
		self.num_skills = num_skills  # Should be same as the latent_dim of the PnPEnv
		self.time_step = 0
		self.end_at = None  # Initial position of the gripper
		
		# Thresholds
		self.sub_goal_height = 0.55  # Height to which block will be first taken before moving towards goal.
		# This should be greater than block's height
		self.dataset2_thresh = 0.7  # The probability of picking the obj0 first
		# Do switch with 5% prob.
		# For values like 0.1, 0.2, 0.5, we are always switching before any obj is being picked, thus dec. the value
		self.switch_prob = 0.05  # The probability of switching mid-demo
		self.num_do_switch: int = 0  # Number of times to switch before demo ends [DO NOT CHANGE HERE]
		
		self.expert_behaviour: str = expert_behaviour  # One of ['0', '1', '2']
		self.reset()
	
	def act(self, state, env_goal, noise_eps=0., random_eps=0., **kwargs):
		if self.time_step == 0:
			# self.end_at = state[0][:3]  # Initial Gripper Position
			self.end_at = env_goal[0][3:] + np.array([0, 0, self.sub_goal_height]) \
				if self.picking_order == 'zero_first' else env_goal[0][:3] + np.array([0, 0, self.sub_goal_height])
		
		a, curr_goal, skill, obj = tf.numpy_function(func=self.transition, inp=[state[0], env_goal[0]],
													 Tout=[tf.float32, tf.float32, tf.int32, tf.int32])
		a = tf.numpy_function(func=self.add_noise_to_action, inp=[a, noise_eps, random_eps], Tout=tf.float32)
		a = tf.squeeze(a)
		
		skill = tf.one_hot(skill, depth=self.num_skills, dtype=tf.float32)
		skill = tf.squeeze(skill)
		
		# obj = tf.one_hot(obj, depth=2, dtype=tf.float32)
		# obj = tf.squeeze(obj)
		self.time_step += 1
		return curr_goal, skill, obj, a
	
	def transition(self, o, g) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		time.sleep(0.01)
		
		gripper = o[:3]
		block0 = o[3:6]
		block1 = o[6:9]
		rel_block0 = o[9:12]
		rel_block1 = o[12:15]
		env_goal0 = g[-6:-3]
		env_goal1 = g[-3:]
		
		# To change expert's behaviour mid-trajectory
		if self.num_do_switch > 0:
			
			# Switch with some probability
			if np.random.uniform(0, 1) < self.switch_prob:
				
				# Case 1: When none of the blocks have been sent towards their goal
				if not self.block1_picked and not self.block0_picked:
					
					if self.picking_order == 'zero_first':
						self.picking_order = 'one_first'  # Switch 0 to 1
						self.end_at = env_goal0 + np.array([0, 0, self.sub_goal_height])
					else:
						self.picking_order = 'zero_first'  # Switch 1 to 0
						self.end_at = env_goal1 + np.array([0, 0, self.sub_goal_height])
				
				self.num_do_switch -= 1
				
			# Todo: Case 2:Deal with case one obj is picked (not delievered yet) and other is not
			# # Check if gripper is holding any block (we set block_picked after a block is gripped)
			# if np.linalg.norm(rel_block0) < 0.01:
			# 	# Action = RELEASE
			# 	a = np.array([gripper[0], gripper[1], 1., 1.], dtype=np.float32)
			# 	skill = 'pick'
			# 	obj = '1'
			# 	return a, np.array([SKILL_MAPPING[skill]], dtype=np.int32), \
			# 		   np.array([OBJ_MAPPING[obj]], dtype=np.int32)
			#
			# elif np.linalg.norm(rel_block1) < 0.01:
			# 	# Action = RELEASE
			# 	a = np.array([gripper[0], gripper[1], 1., 1.], dtype=np.float32)
			# 	skill = 'pick'
			# 	obj = '0'
			# 	return a, np.array([SKILL_MAPPING[skill]], dtype=np.int32), \
			# 		   np.array([OBJ_MAPPING[obj]], dtype=np.int32)
		
		# if the first and second block are placed in the right place
		# Earlier logic of this 'if' branch was only checking if block 1 was at its goal, this sometimes led to
		# misbehaviour in cases when block 1 was at its goal but not block 0.
		# Thus, we need to make sure that all the blocks are at their goals
		
		if self.picking_order == 'zero_first':
			
			# ################# Block0 and Block1 Delivered ################# #
			if (np.linalg.norm(block1 - env_goal1) <= 0.01 and np.linalg.norm(block0 - env_goal0) <= 0.01) or \
					(self.block0_dropped and self.block1_dropped):
				self.block1_dropped = True
				# Here do not stay at the goal, but go back to the start
				curr_goal = np.concatenate([env_goal1[:2], np.array([self.sub_goal_height])])
				# Slow down the gripper by taking small step action
				a = clip_action((curr_goal - gripper) * self.post_completion_step_size)
				a = np.append(a, 1.)
				skill = 'pick'
				obj = '-1'
			
			# ############# Block0 Delivered! Deal with Block1 ############## #
			elif np.linalg.norm(block0 - env_goal0) <= 0.01 or self.block0_dropped:
				self.block0_dropped = True
				obj = '1'
				
				# First gain some height after reaching the goal of previous block
				if not self.gripper_raised:
					sub_goal = np.concatenate([env_goal0[:2], np.array([self.sub_goal_height])])
					if np.linalg.norm((sub_goal - gripper)) < 0.05:  # Use distance thresh of FetchEnv
						# Sub-goal reached
						self.gripper_raised = True
						a, curr_goal, skill, self.block1_picked, self.block1_raised = self.deal_with_block(
							gripper, block1, rel_block1, env_goal1,
							self.block1_picked, self.block1_raised
						)
					else:
						# Reach sub-goal first
						a = clip_action((sub_goal - gripper) * self.step_size)
						a = np.concatenate([a, np.array([1.])], dtype=np.float32)
						curr_goal = sub_goal
						skill = 'pick'
						
				else:
					a, curr_goal, skill, self.block1_picked, self.block1_raised = self.deal_with_block(
						gripper, block1, rel_block1, env_goal1,
						self.block1_picked, self.block1_raised
					)
			
			# ###################### Deal with Block0 ###################### #
			else:
				obj = '0'
				a, curr_goal, skill, self.block0_picked, self.block0_raised = self.deal_with_block(
					gripper, block0, rel_block0, env_goal0,
					self.block0_picked, self.block0_raised
				)
		else:
			
			# ################# Block0 and Block1 Delivered ################# #
			if (np.linalg.norm(block1 - env_goal1) <= 0.01 and np.linalg.norm(block0 - env_goal0) <= 0.01) or \
					(self.block0_dropped and self.block1_dropped):
				self.block0_dropped = True
				# Here do not stay at the goal, but go back to the start
				curr_goal = np.concatenate([env_goal0[:2], np.array([self.sub_goal_height])])
				a = clip_action((curr_goal - gripper) * self.post_completion_step_size)
				a = np.append(a, 1.)
				skill = 'pick'
				obj = '-1'
			
			# ############# Block1 Delivered! Deal with Block0 ############## #
			elif np.linalg.norm(block1 - env_goal1) <= 0.01 or self.block1_dropped:
				self.block1_dropped = True
				obj = '0'
				
				# First gain some height after reaching the goal of previous block
				if not self.gripper_raised:
					sub_goal = np.concatenate([env_goal1[:2], np.array([self.sub_goal_height])])
					if np.linalg.norm((sub_goal - gripper)) < 0.05:  # Use distance thresh of FetchEnv
						# Sub-goal reached
						self.gripper_raised = True
						a, curr_goal, skill, self.block0_picked, self.block0_raised = self.deal_with_block(
							gripper, block0, rel_block0, env_goal0,
							self.block0_picked, self.block0_raised
						)
					else:
						# Reach sub-goal first
						a = clip_action((sub_goal - gripper) * self.step_size)
						a = np.concatenate([a, np.array([1.])], dtype=np.float32)
						curr_goal = sub_goal
						skill = 'pick'
				
				else:
					a, curr_goal, skill, self.block0_picked, self.block0_raised = self.deal_with_block(
						gripper, block0, rel_block0, env_goal0,
						self.block0_picked, self.block0_raised
					)
			# ###################### Deal with Block1 ###################### #
			else:
				obj = '1'
				a, curr_goal, skill, self.block1_picked, self.block1_raised = self.deal_with_block(
					gripper, block1, rel_block1, env_goal1,
					self.block1_picked, self.block1_raised
				)
		
		return a, curr_goal, \
			   np.array([SKILL_MAPPING[skill]], dtype=np.int32), np.array([OBJ_MAPPING[obj]], dtype=np.int32)
	
	def reset(self):
		self.time_step = 0
		self.end_at = None
		self.block0_picked, self.block1_picked = False, False
		self.block0_raised, self.block1_raised = False, False
		self.block0_dropped, self.block1_dropped = False, False
		self.gripper_raised = False
		
		# DATASET 1: For fixed picking order (in stacking the env is rendered with goal0 on table, thus use zero_first)
		if self.expert_behaviour == '0':
			self.picking_order: str = 'zero_first'
		
		else:
			
			# DATASET 2: For random picking order (70-30 Split)
			if np.random.uniform(0, 1) < self.dataset2_thresh:
				self.picking_order: str = 'zero_first'
			else:
				self.picking_order: str = 'one_first'
			
			# DATASET 3: When the picking order changes mid-demo
			# To change the behaviour mid-demo
			if self.expert_behaviour == '1':
				self.num_do_switch: int = 0
			elif self.expert_behaviour == '2':
				self.num_do_switch: int = 1
			else:
				logger.error("Invalid expert behaviour: {}".format(self.expert_behaviour))
				raise NotImplementedError
	
	def deal_with_block(self, gripper, block, rel_block, env_goal, block_picked: bool, tempGoal_reached: bool):
		
		if np.linalg.norm(rel_block) > 0.1 and np.linalg.norm(rel_block - np.array([0, 0, 0.08])) > 0.001:
			a = self.goto_block(gripper, block)
			curr_goal = block
			skill = 'pick'
		
		elif np.linalg.norm(rel_block) > 0.01 and not block_picked:
			a = self.pickup_block(gripper, block)
			curr_goal = block
			skill = 'grab'
		else:
			block_picked = True
			
			# Collision avoid: Move the block vertically up first
			if not tempGoal_reached:
				sub_goal = np.concatenate([block[:2], np.array([self.sub_goal_height])])
				if np.linalg.norm((sub_goal - block)) < 0.05:  # Use distance thresh of FetchEnv
					# Sub-goal reached
					tempGoal_reached = True
					a, curr_goal = self.goto_goal(block, env_goal)
				else:
					# Reach sub-goal first
					a = clip_action((sub_goal - block) * self.step_size)
					a = np.concatenate([a, np.array([-1])], dtype=np.float32)
					curr_goal = sub_goal
			
			else:
				a, curr_goal = self.goto_goal(block, env_goal)
			skill = 'drop'
		
		return a, curr_goal, skill, block_picked, tempGoal_reached
	
	def goto_block(self, cur_pos, block_pos, grip=1.):
		target_pos = block_pos + np.array([0, 0, 0.08])
		a = clip_action((target_pos - cur_pos) * self.step_size)
		a = np.concatenate([a, np.array([grip])], dtype=np.float32)
		return a
	
	def pickup_block(self, cur_pos, block_pos):
		"""
		:param cur_pos: current gripper position
		:param block_pos: block position
		:return: action
		Logic: Move the gripper to the block position (w partially closed gripper = -0.005) and then close the gripper
		"""
		if np.linalg.norm(cur_pos - block_pos) < 0.01:  # and gripper_state > 0.025: # TODO: need to adjust
			a = np.array([0, 0, 0, -1.], dtype=np.float32)
		else:
			a = clip_action((block_pos - cur_pos) * self.step_size)
			a = np.concatenate([a, np.array([-0.005])], dtype=np.float32)
		return a
	
	def goto_goal(self, cur_pos, goal_pos, grip=-1):
		"""
		:param cur_pos: current block position (overlapping with gripper)
		:param goal_pos: env goal position
		:param grip: gripper state (set to -1 to close the gripper)
		"""
		
		# Collision avoid: We first move the gripper towards (x, y) coordinates of goal maintaining the gripper height
		if np.linalg.norm((goal_pos[:2] - cur_pos[:2])) > 0.01:
			curr_goal = np.concatenate([goal_pos[:2], np.array([cur_pos[2]])])  # = (x_goal, y_goal, z_gripper)
			a_xy = clip_action((goal_pos[:2] - cur_pos[:2]) * self.step_size)
			a = np.concatenate([a_xy, np.array([0]), np.array([grip])], dtype=np.float32)
		
		# We then move the gripper towards the goal -> Doing this after above action brings gripper vertically down
		elif np.linalg.norm((goal_pos - cur_pos)) > 0.01:
			curr_goal = goal_pos  # = (x_goal, y_goal, z_goal)
			a = clip_action((goal_pos - cur_pos) * self.step_size)
			a = np.concatenate([a, np.array([grip])], dtype=np.float32)
		
		else:
			raise ValueError("Goal reached! Should not be in goto_goal!")
			# # Here the current goal to achieve should be the next object's position
			# curr_goal = None
			# a = np.array([0, 0, 0, grip], dtype=np.float32)
		
		return a, curr_goal
	
	def add_noise_to_action(self, a, noise_eps=0., random_eps=0., ):
		a = np.array([a])
		noise = noise_eps * np.random.randn(*a.shape)  # gaussian noise
		a += noise
		a = np.clip(a, -1, 1)
		a += np.random.binomial(1, random_eps, a.shape[0]).reshape(-1, 1) * (self._random_action(a.shape[0]) - a)
		return a
	
	@staticmethod
	def _random_action(n):
		return np.random.uniform(low=-1, high=1, size=(n, 4))


class PnPExpert:
	def __init__(self, num_skills=3):
		self.block0_picked = None
		self.step_size = 6
		self.num_skills = num_skills  # Should be same as the latent_dim of the PnPEnv
		self.time_step = 0
		self.start = None  # Initial position of the gripper
		
		self.reset()
	
	def act(self, state, env_goal, prev_goal, prev_skill, noise_eps=0., random_eps=0., **kwargs):
		if self.time_step == 0:
			self.start = state[0][:3]  # Gripper Position
		else:
			self.time_step += 1
		
		a, curr_goal, skill, obj = tf.numpy_function(func=self.transition, inp=[state[0], env_goal[0]],
													 Tout=[tf.float32, tf.float32, tf.int32, tf.int32])
		a = tf.numpy_function(func=self.add_noise_to_action, inp=[a, noise_eps, random_eps], Tout=tf.float32)
		a = tf.squeeze(a)
		
		skill = tf.one_hot(skill, depth=self.num_skills, dtype=tf.float32)
		skill = tf.squeeze(skill)
		
		# obj = tf.one_hot(obj, depth=2, dtype=tf.float32)
		# obj = tf.squeeze(obj)
		
		return curr_goal, skill, obj, a
	
	def transition(self, o, g) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		gripper_pos = o[:3]
		block_pos = o[3:6]
		env_goal_pos = g[-3:]
		block_rel_pos = o[6:9]  # block_pos - gripper_pos
		# print("---- #  |{}| -> |{}| # ----".format(np.linalg.norm(block_rel_pos), np.linalg.norm(env_goal_pos-block_pos)))
		
		if np.linalg.norm(block_rel_pos) > 0.05 and np.linalg.norm(block_rel_pos - np.array([0, 0, 0.03])) > 0.001:
			if self.block0_picked:
				self.reset()
			
			skill = 'pick'
			curr_goal = block_pos
			a = self.goto_block(gripper_pos, block_pos)
		
		elif np.linalg.norm(block_rel_pos) > 0.01 and not self.block0_picked:
			skill = 'grab'
			curr_goal = block_pos
			a = self.pickup_block(gripper_pos, block_pos)
		
		else:
			self.block0_picked = True
			skill = 'drop'
			curr_goal = env_goal_pos
			a = self.goto_goal(block_pos, env_goal_pos)
		
		obj = '0'
		return a, curr_goal, np.array([SKILL_MAPPING[skill]], dtype=np.int32), \
			   np.array([OBJ_MAPPING[obj]], dtype=np.int32)
	
	def reset(self):
		self.time_step = 0
		self.start = None
		#  # Hack: This flag helps bypass the irregular behaviour i.e. when block is picked and
		# gripper starts to move towards the goal, object's pos relative to gripper sometime inc. which then violates
		# move-to-goal condition 'np.linalg.norm(block_rel_pos) < 0.01'(see orig. implementation of GoalGAIL).
		# This behaviour is periodic making the gripper oscillate
		self.block0_picked = False
	
	def goto_block(self, cur_pos, block_pos, grip=1.0):
		target_pos = block_pos + np.array([0, 0, 0.03])
		a = clip_action((target_pos - cur_pos) * self.step_size)
		a = np.concatenate([a, np.array([grip])], dtype=np.float32)
		return a
	
	def pickup_block(self, cur_pos, block_pos, ):
		"""
		:param cur_pos: current gripper position
		:param block_pos: block position
		:return: action
		Logic: Move the gripper to the block position (w partially closed gripper = -0.005) and then close the gripper
		"""
		if np.linalg.norm(cur_pos - block_pos) < 0.01:  # and gripper_state > 0.025: # TODO: need to adjust
			a = np.array([0, 0, 0, -1.], dtype=np.float32)
		else:
			a = clip_action((block_pos - cur_pos) * self.step_size)
			a = np.concatenate([a, np.array([-0.005])], dtype=np.float32)
		return a
	
	def goto_goal(self, cur_pos, goal_pos, grip=-1.):
		"""
		:param cur_pos: current block position (overlapping with gripper)
		:param goal_pos: env goal position
		:param grip: gripper state (set to -1 to close the gripper)
		"""
		
		# We then move the gripper towards the goal -> Doing this after above action brings gripper vertically down
		if np.linalg.norm((goal_pos - cur_pos)) > 0.01:
			curr_goal = goal_pos  # = (x_goal, y_goal, z_goal)
			a = clip_action((goal_pos - cur_pos) * self.step_size)
			a = np.concatenate([a, np.array([grip])], dtype=np.float32)
		
		else:
			# ac = np.array([0, 0, 0, grip], dtype=np.float32)
			raise ValueError("Goal reached! Should not be in goto_goal!")
		
		return a, curr_goal
	
	def add_noise_to_action(self, a, noise_eps=0., random_eps=0., ):
		a = np.array([a])
		noise = noise_eps * np.random.randn(*a.shape)  # gaussian noise
		a += noise
		a = np.clip(a, -1, 1)
		a += np.random.binomial(1, random_eps, a.shape[0]).reshape(-1, 1) * (self._random_action(a.shape[0]) - a)
		return a
	
	@staticmethod
	def _random_action(n):
		return np.random.uniform(low=-1, high=1, size=(n, 4))


class PnPExpertTwoObj:
	def __init__(self, num_skills=3, expert_behaviour: str = '0'):
		self.block0_picked, self.block1_picked = None, None  # flags to check if block is picked
		self.block0_raised, self.block1_raised = None, None  # flags to check if block is raised after pickup
		# flags to check if block is dropped (Hack: after drop, dropped block's pos rel to its goal is within limits
		# for one step, then it gets out of limit for another step and then again within limits finally)
		self.block0_dropped, self.block1_dropped = None, None
		self.gripper_raised = None  # flag to check if gripper is raised after drop
		
		self.picking_order = None
		self.step_size = 6
		self.post_completion_step_size = 2
		self.num_skills = num_skills  # Should be same as the latent_dim of the PnPEnv
		self.time_step = 0
		self.end_at = None  # Initial position of the gripper
		
		# Thresholds
		self.sub_goal_height = 0.55  # Height to which block will be first taken before moving towards goal.
		# This should be greater than block's height
		self.dataset2_thresh = 0.7  # The probability of picking the obj0 first
		# Do switch with 5% prob.
		# For values like 0.1, 0.2, 0.5, we are always switching before any obj is being picked, thus dec. the value
		self.switch_prob = 0.05  # The probability of switching mid-demo
		self.num_do_switch: int = 0  # Number of times to switch before demo ends [DO NOT CHANGE HERE]
		
		self.expert_behaviour: str = expert_behaviour  # One of ['0', '1', '2']
		self.reset()
	
	def act(self, state, env_goal, prev_goal, prev_skill, noise_eps=0., random_eps=0., **kwargs):
		if self.time_step == 0:
			# self.end_at = state[0][:3]  # Initial Gripper Position
			self.end_at = env_goal[0][3:] + np.array([0, 0, self.sub_goal_height]) \
				if self.picking_order == 'zero_first' else env_goal[0][:3] + np.array([0, 0, self.sub_goal_height])
		
		a, curr_goal, curr_skill, obj = tf.numpy_function(func=self.transition, inp=[state[0], env_goal[0]],
														  Tout=[tf.float32, tf.float32, tf.int32, tf.int32])
		a = tf.numpy_function(func=self.add_noise_to_action, inp=[a, noise_eps, random_eps], Tout=tf.float32)
		a = tf.squeeze(a)
		
		curr_skill = tf.one_hot(curr_skill, depth=self.num_skills, dtype=tf.float32)
		curr_skill = tf.squeeze(curr_skill)
		
		# obj = tf.one_hot(obj, depth=2, dtype=tf.float32)
		# obj = tf.squeeze(obj)
		self.time_step += 1
		return curr_goal, curr_skill, a
	
	def transition(self, o, g) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		time.sleep(0.01)
		
		gripper = o[:3]
		block0 = o[3:6]
		block1 = o[6:9]
		rel_block0 = o[9:12]
		rel_block1 = o[12:15]
		env_goal0 = g[-6:-3]
		env_goal1 = g[-3:]
		
		# To change expert's behaviour mid-trajectory
		if self.num_do_switch > 0:
			
			# Switch with some probability
			if np.random.uniform(0, 1) < self.switch_prob:
				
				# Case 1: When none of the blocks have been sent towards their goal
				if not self.block1_picked and not self.block0_picked:
					print("Switching")
					if self.picking_order == 'zero_first':
						self.picking_order = 'one_first'  # Switch 0 to 1
						self.end_at = env_goal0 + np.array([0, 0, self.sub_goal_height])
					else:
						self.picking_order = 'zero_first'  # Switch 1 to 0
						self.end_at = env_goal1 + np.array([0, 0, self.sub_goal_height])
				
				self.num_do_switch -= 1
		
		# Todo: Case 2:Deal with case one obj is picked (not delievered yet) and other is not
		# # Check if gripper is holding any block (we set block_picked after a block is gripped)
		# if np.linalg.norm(rel_block0) < 0.01:
		# 	# Action = RELEASE
		# 	a = np.array([gripper[0], gripper[1], 1., 1.], dtype=np.float32)
		# 	skill = 'pick'
		# 	obj = '1'
		# 	return a, np.array([SKILL_MAPPING[skill]], dtype=np.int32), \
		# 		   np.array([OBJ_MAPPING[obj]], dtype=np.int32)
		#
		# elif np.linalg.norm(rel_block1) < 0.01:
		# 	# Action = RELEASE
		# 	a = np.array([gripper[0], gripper[1], 1., 1.], dtype=np.float32)
		# 	skill = 'pick'
		# 	obj = '0'
		# 	return a, np.array([SKILL_MAPPING[skill]], dtype=np.int32), \
		# 		   np.array([OBJ_MAPPING[obj]], dtype=np.int32)
		
		# if the first and second block are placed in the right place
		# Earlier logic of this 'if' branch was only checking if block 1 was at its goal, this sometimes led to
		# misbehaviour in cases when block 1 was at its goal but not block 0.
		# Thus, we need to make sure that all the blocks are at their goals
		
		if self.picking_order == 'zero_first':
			
			# ################# Block0 and Block1 Delivered ################# #
			if (np.linalg.norm(block1 - env_goal1) <= 0.01 and np.linalg.norm(block0 - env_goal0) <= 0.01) or \
					(self.block0_dropped and self.block1_dropped):
				self.block1_dropped = True
				# Here do not stay at the goal, but go back to the start
				curr_goal = np.concatenate([env_goal1[:2], np.array([self.sub_goal_height])])
				# Slow down the gripper by taking small step action
				a = clip_action((curr_goal - gripper) * self.post_completion_step_size)
				a = np.append(a, 1.)
				skill = 'pick'
				obj = '-1'
			
			# ############# Block0 Delivered! Deal with Block1 ############## #
			elif np.linalg.norm(block0 - env_goal0) <= 0.01 or self.block0_dropped:
				self.block0_dropped = True
				obj = '1'
				
				# First gain some height after reaching the goal of previous block
				if not self.gripper_raised:
					sub_goal = np.concatenate([env_goal0[:2], np.array([self.sub_goal_height])])
					if np.linalg.norm((sub_goal - gripper)) < 0.05:  # Use distance thresh of FetchEnv
						# Sub-goal reached
						self.gripper_raised = True
						a, curr_goal, skill, self.block1_picked, self.block1_raised = self.deal_with_block(
							gripper, block1, rel_block1, env_goal1,
							self.block1_picked, self.block1_raised
						)
					else:
						# Reach sub-goal first
						a = clip_action((sub_goal - gripper) * self.step_size)
						a = np.concatenate([a, np.array([1.])], dtype=np.float32)
						curr_goal = sub_goal
						skill = 'pick'
				
				else:
					a, curr_goal, skill, self.block1_picked, self.block1_raised = self.deal_with_block(
						gripper, block1, rel_block1, env_goal1,
						self.block1_picked, self.block1_raised
					)
			
			# ###################### Deal with Block0 ###################### #
			else:
				obj = '0'
				a, curr_goal, skill, self.block0_picked, self.block0_raised = self.deal_with_block(
					gripper, block0, rel_block0, env_goal0,
					self.block0_picked, self.block0_raised
				)
		else:
			
			# ################# Block0 and Block1 Delivered ################# #
			if (np.linalg.norm(block1 - env_goal1) <= 0.01 and np.linalg.norm(block0 - env_goal0) <= 0.01) or \
					(self.block0_dropped and self.block1_dropped):
				self.block0_dropped = True
				# Here do not stay at the goal, but go back to the start
				curr_goal = np.concatenate([env_goal0[:2], np.array([self.sub_goal_height])])
				a = clip_action((curr_goal - gripper) * self.post_completion_step_size)
				a = np.append(a, 1.)
				skill = 'pick'
				obj = '-1'
			
			# ############# Block1 Delivered! Deal with Block0 ############## #
			elif np.linalg.norm(block1 - env_goal1) <= 0.01 or self.block1_dropped:
				self.block1_dropped = True
				obj = '0'
				
				# First gain some height after reaching the goal of previous block
				if not self.gripper_raised:
					sub_goal = np.concatenate([env_goal1[:2], np.array([self.sub_goal_height])])
					if np.linalg.norm((sub_goal - gripper)) < 0.05:  # Use distance thresh of FetchEnv
						# Sub-goal reached
						self.gripper_raised = True
						a, curr_goal, skill, self.block0_picked, self.block0_raised = self.deal_with_block(
							gripper, block0, rel_block0, env_goal0,
							self.block0_picked, self.block0_raised
						)
					else:
						# Reach sub-goal first
						a = clip_action((sub_goal - gripper) * self.step_size)
						a = np.concatenate([a, np.array([1.])], dtype=np.float32)
						curr_goal = sub_goal
						skill = 'pick'
				
				else:
					a, curr_goal, skill, self.block0_picked, self.block0_raised = self.deal_with_block(
						gripper, block0, rel_block0, env_goal0,
						self.block0_picked, self.block0_raised
					)
			# ###################### Deal with Block1 ###################### #
			else:
				obj = '1'
				a, curr_goal, skill, self.block1_picked, self.block1_raised = self.deal_with_block(
					gripper, block1, rel_block1, env_goal1,
					self.block1_picked, self.block1_raised
				)
		
		return a, curr_goal, \
			   np.array([SKILL_MAPPING[skill]], dtype=np.int32), np.array([OBJ_MAPPING[obj]], dtype=np.int32)
	
	def reset(self):
		self.time_step = 0
		self.end_at = None
		self.block0_picked, self.block1_picked = False, False
		self.block0_raised, self.block1_raised = False, False
		self.block0_dropped, self.block1_dropped = False, False
		self.gripper_raised = False
		
		# DATASET 1: For fixed picking order (in stacking the env is rendered with goal0 on table, thus use zero_first)
		if self.expert_behaviour == '0':
			self.picking_order: str = 'zero_first'
		
		else:
			
			# DATASET 2: For random picking order (70-30 Split)
			if np.random.uniform(0, 1) < self.dataset2_thresh:
				self.picking_order: str = 'zero_first'
			else:
				self.picking_order: str = 'one_first'
			
			# DATASET 3: When the picking order changes mid-demo
			# To change the behaviour mid-demo
			if self.expert_behaviour == '1':
				self.num_do_switch: int = 0
			elif self.expert_behaviour == '2':
				self.num_do_switch: int = 1
			else:
				logger.error("Invalid expert behaviour: {}".format(self.expert_behaviour))
				raise NotImplementedError
	
	def deal_with_block(self, gripper, block, rel_block, env_goal, block_picked: bool, tempGoal_reached: bool):
		
		if np.linalg.norm(rel_block) > 0.1 and np.linalg.norm(rel_block - np.array([0, 0, 0.08])) > 0.001:
			a = self.goto_block(gripper, block)
			curr_goal = block
			skill = 'pick'
		
		elif np.linalg.norm(rel_block) > 0.01 and not block_picked:
			a = self.pickup_block(gripper, block)
			curr_goal = block
			skill = 'grab'
		else:
			block_picked = True
			
			# Collision avoid: Move the block vertically up first
			if not tempGoal_reached:
				sub_goal = np.concatenate([block[:2], np.array([self.sub_goal_height])])
				if np.linalg.norm((sub_goal - block)) < 0.05:  # Use distance thresh of FetchEnv
					# Sub-goal reached
					tempGoal_reached = True
					a, curr_goal = self.goto_goal(block, env_goal)
				else:
					# Reach sub-goal first
					a = clip_action((sub_goal - block) * self.step_size)
					a = np.concatenate([a, np.array([-1])], dtype=np.float32)
					curr_goal = sub_goal
			
			else:
				a, curr_goal = self.goto_goal(block, env_goal)
			skill = 'drop'
		
		return a, curr_goal, skill, block_picked, tempGoal_reached
	
	def goto_block(self, cur_pos, block_pos, grip=1.):
		target_pos = block_pos + np.array([0, 0, 0.08])
		a = clip_action((target_pos - cur_pos) * self.step_size)
		a = np.concatenate([a, np.array([grip])], dtype=np.float32)
		return a
	
	def pickup_block(self, cur_pos, block_pos):
		"""
		:param cur_pos: current gripper position
		:param block_pos: block position
		:return: action
		Logic: Move the gripper to the block position (w partially closed gripper = -0.005) and then close the gripper
		"""
		if np.linalg.norm(cur_pos - block_pos) < 0.01:  # and gripper_state > 0.025: # TODO: need to adjust
			a = np.array([0, 0, 0, -1.], dtype=np.float32)
		else:
			a = clip_action((block_pos - cur_pos) * self.step_size)
			a = np.concatenate([a, np.array([-0.005])], dtype=np.float32)
		return a
	
	def goto_goal(self, cur_pos, goal_pos, grip=-1):
		"""
		:param cur_pos: current block position (overlapping with gripper)
		:param goal_pos: env goal position
		:param grip: gripper state (set to -1 to close the gripper)
		"""
		
		# Collision avoid: We first move the gripper towards (x, y) coordinates of goal maintaining the gripper height
		if np.linalg.norm((goal_pos[:2] - cur_pos[:2])) > 0.01:
			curr_goal = np.concatenate([goal_pos[:2], np.array([cur_pos[2]])])  # = (x_goal, y_goal, z_gripper)
			a_xy = clip_action((goal_pos[:2] - cur_pos[:2]) * self.step_size)
			a = np.concatenate([a_xy, np.array([0]), np.array([grip])], dtype=np.float32)
		
		# We then move the gripper towards the goal -> Doing this after above action brings gripper vertically down
		elif np.linalg.norm((goal_pos - cur_pos)) > 0.01:
			curr_goal = goal_pos  # = (x_goal, y_goal, z_goal)
			a = clip_action((goal_pos - cur_pos) * self.step_size)
			a = np.concatenate([a, np.array([grip])], dtype=np.float32)
		
		else:
			raise ValueError("Goal reached! Should not be in goto_goal!")
		# # Here the current goal to achieve should be the next object's position
		# curr_goal = None
		# a = np.array([0, 0, 0, grip], dtype=np.float32)
		
		return a, curr_goal
	
	def add_noise_to_action(self, a, noise_eps=0., random_eps=0., ):
		a = np.array([a])
		noise = noise_eps * np.random.randn(*a.shape)  # gaussian noise
		a += noise
		a = np.clip(a, -1, 1)
		a += np.random.binomial(1, random_eps, a.shape[0]).reshape(-1, 1) * (self._random_action(a.shape[0]) - a)
		return a
	
	@staticmethod
	def _random_action(n):
		return np.random.uniform(low=-1, high=1, size=(n, 4))