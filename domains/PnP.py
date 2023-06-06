import collections
import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def Step(observation, reward, done, **kwargs):
	"""
	Convenience method creating a namedtuple with the results of the
	environment's step method.
	Put extra diagnostic info in the kwargs
	"""
	_Step = collections.namedtuple("Step", ["observation", "reward", "done", "info"])
	return _Step(observation, reward, done, kwargs)


class PnPEnv(object):
	def __init__(self, full_space_as_goal=False, goal_weight=1., distance_threshold=0.05, feasible_hand=True,
				 num_objs=1, first_in_place=False, stacking=False, fix_goal=False,
				 fix_object=False):
		"""
			Pick and Place Environment: can be single or multi-object
		"""
		
		if num_objs == 1:
			from .Fetch_Base.fetch_env_oneobj import FetchPickAndPlaceEnv
			env = FetchPickAndPlaceEnv(distance_threshold=distance_threshold, fix_object=fix_object, fix_goal=fix_goal)
		elif num_objs == 2:
			from .Fetch_Base.fetch_env_twoobj import FetchPickAndPlaceEnv
			env = FetchPickAndPlaceEnv(stacking=stacking, first_in_place=first_in_place, distance_threshold=distance_threshold)
		elif num_objs == 3:
			from .Fetch_Base.fetch_env_threeobj import FetchPickAndPlaceEnv
			env = FetchPickAndPlaceEnv(stacking=stacking, first_in_place=first_in_place, distance_threshold=distance_threshold)
		else:
			raise NotImplementedError("Number of objects not supported: Need to implement the environment xml file")
		
		env.unwrapped.spec = self
		self._env = env
		
		self._observation_space = env.observation_space.spaces['observation']
		self._action_space = env.action_space
		self._current_goal = None
		
		self.goal_weight = goal_weight
		self.distance_threshold = distance_threshold
		self.full_space_as_goal = full_space_as_goal
		self.num_objs = num_objs
		
		self.feasible_hand = feasible_hand  # if the position of the hand is always feasible to achieve
		
		# Effective latent dimension: Corresponding to pick, grab and drop
		self.latent_dim = 3
	
	def sample_hand_pos(self, block_pos):
		if block_pos[2] == self._env.height_offset or not self.feasible_hand:
			xy = self._env.initial_gripper_xpos[:2] + np.random.uniform(-0.15, 0.15, size=2)
			z = np.random.uniform(self._env.height_offset, self._env.height_offset + 0.3)
			return np.concatenate([xy, [z]])
		else:
			return block_pos
	
	def set_feasible_hand(self, _bool):
		self.feasible_hand = _bool
	
	def render(self):
		self._env.render()
	
	def reset(self):
		d = self._env.reset()
		
		# Reset if the goal is already achieved
		done = np.linalg.norm(d['achieved_goal'] - d['desired_goal']) < self.distance_threshold
		while done:
			d = self._env.reset()
			done = np.linalg.norm(d['achieved_goal'] - d['desired_goal']) < self.distance_threshold
		
		# Update the goal based on some checks
		self.update_goal(d=d)
		return self._transform_obs(d['observation'])
	
	def forced_reset(self, state_dict):
		d = self._env.forced_reset(state_dict)
		# Update the goal based on some checks
		self.update_goal(d=d)
		return self._transform_obs(d['observation'])
	
	def update_goal(self, d=None):
		"""
			Set the goal of the env. for the current episode
		"""
		# Based on some checks, reset the env. if needed
		if self.get_current_obs()[5] < self._env.height_offset or \
				np.any(self.get_current_obs()[3:5] > self._env.initial_gripper_xpos[:2] + 0.15) or \
				np.any(self.get_current_obs()[3:5] < self._env.initial_gripper_xpos[:2] - 0.15):
			self._env._reset_sim()
		
		# Set _current_goal that will be used by wrapper
		if d is not None:
			self._current_goal = d['desired_goal']
		else:
			self._current_goal = self._env.goal = np.copy(self._env._sample_goal())
		
		if self.full_space_as_goal:
			self._current_goal = np.concatenate([self.sample_hand_pos(self._current_goal), self._current_goal])
	
	def get_current_obs(self):
		"""
		:return: current observation (state of the robot)
		"""
		return self._transform_obs(self._env._get_obs()['observation'])
	
	def _transform_obs(self, obs):
		"""
			Extract the relevant information from the observation
		"""
		# Gripper pos + obj. pos(s) + obj. rel. pos(s) + gripper state
		rel_dims = 3 + 3 * self.num_objs + 3 * self.num_objs + 1
		return obs[:rel_dims]
	
	def transform_to_goal_space(self, obs):
		"""
			Transform the observation to the goal space by extracting the achieved goal from the observation
			For the PnP, it corresponds to obj. positions
		"""
		if self.full_space_as_goal:
			ret = np.array(obs[:3 + 3 * self.num_objs])
		else:
			ret = np.array(obs[3:3 + 3 * self.num_objs])
		return ret
	
	def step(self, action):
		next_obs, reward, _, info = self._env.step(
			action)  # FetchPickAndPlaceEnv freezes done to False and stores termination response in info
		next_obs = self._transform_obs(next_obs['observation'])  # Remove unwanted portions of the observed state
		info['obs2goal'] = self.transform_to_goal_space(next_obs)
		info['distance'] = np.linalg.norm(self.current_goal - info['obs2goal'])
		if self.full_space_as_goal:
			info['block_distance'] = np.linalg.norm((self.current_goal - info['obs2goal'])[3:6])
			info['hand_distance'] = np.linalg.norm((self.current_goal - info['obs2goal'])[0:3])
		info['goal_reached'] = info['distance'] < self.distance_threshold
		# print("PnPEnv: {}/{}".format(info['distance'], self.distance_threshold))
		done = info['goal_reached']
		return Step(next_obs, reward, done, **info)
	
	@property
	def observation_space(self):
		return self._observation_space
	
	@property
	def action_space(self):
		return self._action_space
	
	@property
	def current_goal(self):
		return self._current_goal
	
	def get_state_dict(self):
		state_dict = self._env.get_state_dict()
		# tf.print("PnPEnv: {}".format(state_dict['goal']))
		return state_dict


class MyPnPEnvWrapper(PnPEnv):
	def __init__(self, full_space_as_goal=False, **kwargs):
		"""
		GoalGAIL compatible Wrapper for PnP Env
		Args:
			full_space_as_goal:
			**kwargs:
		"""
		super(MyPnPEnvWrapper, self).__init__(full_space_as_goal, **kwargs)
	
	def reset(self, render=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		obs = super(MyPnPEnvWrapper, self).reset()
		if render:
			super(MyPnPEnvWrapper, self).render()
		achieved_goal = super(MyPnPEnvWrapper, self).transform_to_goal_space(obs)
		desired_goal = super(MyPnPEnvWrapper, self).current_goal
		return obs.astype(np.float32), achieved_goal.astype(np.float32), desired_goal.astype(np.float32)
	
	def step(self, action, render=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		if len(action.shape) > 1:
			action = action[0]
		obs, _, done, info = super(MyPnPEnvWrapper, self).step(action)  # ignore reward (re-computed in HER)
		
		if render:
			super(MyPnPEnvWrapper, self).render()
		
		achieved_goal = super(MyPnPEnvWrapper, self).transform_to_goal_space(obs)
		desired_goal = super(MyPnPEnvWrapper, self).current_goal
		success = int(done)
		
		return obs.astype(np.float32), achieved_goal.astype(np.float32), desired_goal.astype(np.float32), np.array(
			success, np.int32), info['distance'].astype(np.float32)
	
	def forced_reset(self, state_dict, render=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		obs = super(MyPnPEnvWrapper, self).forced_reset(state_dict)
		if render:
			super(MyPnPEnvWrapper, self).render()
		achieved_goal = super(MyPnPEnvWrapper, self).transform_to_goal_space(obs)
		desired_goal = super(MyPnPEnvWrapper, self).current_goal
		return obs.astype(np.float32), achieved_goal.astype(np.float32), desired_goal.astype(np.float32)
	
	def get_state_dict(self):
		state_dict = super(MyPnPEnvWrapper, self).get_state_dict()
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
		ret = (goal_distance < self.distance_threshold) * self.goal_weight - extend_dist_rew_weight * goal_distance
		
		return ret
	
	def _random_action(self, n):
		action_max = float(super(MyPnPEnvWrapper, self).action_space.high[0])
		return np.random.uniform(low=-action_max, high=action_max, size=(n, 4))
