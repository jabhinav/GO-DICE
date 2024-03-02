from typing import Tuple, Dict

import numpy as np
from gym import spaces

a_max = 1.0  # Maximum acceleration of the robot


def clip_action(a):
	return np.clip(a, -a_max, a_max)


class PointMassDropNReachEnv:
	def __init__(self, world_size: Tuple[int, int] = (10, 10), env_type: str = "fixed", penalise_steps: bool = False):
		# state space
		self.world_size = world_size
		assert self.world_size[0] == self.world_size[1], "World must be square"
		assert self.world_size[0] >= 3 and self.world_size[1] >= 3, "World must be at least 3x3"
		
		self.max_obstacle_occupancy = 0.4  # For each obstacle, its maximum occupancy in terms of % of the world size
		self.num_obstacles = 1  # Number of obstacles in the world
		assert self.max_obstacle_occupancy * self.num_obstacles < 1, "Obstacles cannot occupy more than 100% of the world"
		
		# # Current State: (x_agent, y_agent, x_obj, y_obj, x_obs, y_obs, h_obs, w_obs)
		self.empty_state = np.zeros((8,), dtype=np.float32)
		# # Goal State: (x_obj_goal, y_obj_goal, x_agent_final, y_agent_final)
		self.empty_goal = np.zeros((4,), dtype=np.float32)
		self.state = self.empty_state.copy()
		self._current_goal = self.empty_goal.copy()
		
		# # Define Rewards
		self.r_goal = 1  # For reaching the goal with the marker
		self.r_invalid = -1  # For reaching the goal without the marker or reaching the agent's goal w/o dropping marker
		self.r_step = 0.0 if not penalise_steps else -0.05  # For taking a step
		self.r_obstacle = -1  # For hitting an obstacle or the wall
		
		# # Environment Type
		self.env_type = env_type
		
		# Goal Completion Flags
		self.marker_dropped = False
		self.agent_goal_reached = False
		
		# Goal-Reach threshold
		self.goal_reach_threshold = 0.05
		
		self.action_space = spaces.Box(-a_max, a_max, shape=(2,), dtype="float32")
		self.observation_space = spaces.Dict(
			dict(
				desired_goal=spaces.Box(
					-np.inf, np.inf, shape=(4,), dtype="float32"
				),
				achieved_goal=spaces.Box(
					-np.inf, np.inf, shape=(4,), dtype="float32"
				),
				observation=spaces.Box(
					-np.inf, np.inf, shape=(8,), dtype="float32"
				),
			)
		)
	
	def random_world(self, obstacle_pos=None, robot_pos=None, marker_pos=None, goal_pos=None):
		"""
		This function defines a random world. Choices made are:
		1. Randomly place the robot
		2. Randomly place the marker
		3. Randomly place the obstacles
		4. Randomly place the goal
		:return: state, info
		"""
		
		if obstacle_pos is None:
			# 1. Randomly place the obstacles first
			# Choose starting x, y (0<=x,y<=world_size) and height, width (1<=h,w<=world_size)
			# such that h*w <= max_obstacle_occupancy*world_size**2
			obs_x = np.random.uniform(0, self.world_size[0])
			obs_y = np.random.uniform(0, self.world_size[0])
			obs_w = np.random.uniform(obs_x, self.world_size[0])  # w >= x so that it spans x to x + w
			obs_h = np.random.uniform(obs_y, self.world_size[0])  # h >= y so that it spans y to y + h
			while obs_h * obs_w > self.max_obstacle_occupancy * self.world_size[0] ** 2:
				obs_w = np.random.uniform(obs_x, self.world_size[0])
				obs_h = np.random.uniform(obs_y, self.world_size[0])
			
			obstacle_pos = (obs_x, obs_y, obs_w, obs_h)
		else:
			obs_x, obs_y, obs_w, obs_h = obstacle_pos
		
		# Now place all the rest of the objects so that they don't overlap and are not on top of obstacles
		# 2. Randomly place the robot
		if robot_pos is None:
			robot_x = np.random.uniform(0, self.world_size[0])
			robot_y = np.random.uniform(0, self.world_size[0])
			while obs_x <= robot_x <= obs_x + obs_w and obs_y <= robot_y <= obs_y + obs_h:
				robot_x = np.random.uniform(0, self.world_size[0])
				robot_y = np.random.uniform(0, self.world_size[0])
			
			robot_pos = (robot_x, robot_y)
		else:
			robot_x, robot_y = robot_pos
		
		# 3. Randomly place the marker
		if marker_pos is None:
			marker_x = np.random.uniform(0, self.world_size[0])
			marker_y = np.random.uniform(0, self.world_size[0])
			while obs_x <= marker_x <= obs_x + obs_w and obs_y <= marker_y <= obs_y + obs_h or \
					np.all([marker_x == robot_x, marker_y == robot_y]):
				marker_x = np.random.uniform(0, self.world_size[0])
				marker_y = np.random.uniform(0, self.world_size[0])
			
			marker_pos = (marker_x, marker_y)
		else:
			marker_x, marker_y = marker_pos
		
		# 4. Randomly place the goal for the object and the agent
		if goal_pos is None:
			# First define the goal for the object
			goal_marker_x = np.random.uniform(0, self.world_size[0])
			goal_marker_y = np.random.uniform(0, self.world_size[0])
			while obs_x <= goal_marker_x <= obs_x + obs_w and obs_y <= goal_marker_y <= obs_y + obs_h or \
					np.all([goal_marker_x == marker_x, goal_marker_y == marker_y]):
				goal_marker_x = np.random.uniform(0, self.world_size[0])
				goal_marker_y = np.random.uniform(0, self.world_size[0])
			
			# Now define the goal for the agent (can be the same as the goal for the object)
			goal_agent_x = np.random.uniform(0, self.world_size[0])
			goal_agent_y = np.random.uniform(0, self.world_size[0])
			while obs_x <= goal_agent_x <= obs_x + obs_w and obs_y <= goal_agent_y <= obs_y + obs_h or \
					np.all([goal_agent_x == marker_x, goal_agent_y == marker_y]):
				goal_agent_x = np.random.uniform(0, self.world_size[0])
				goal_agent_y = np.random.uniform(0, self.world_size[0])
			
			goal_pos = (goal_agent_x, goal_agent_y, goal_marker_x, goal_marker_y)
		
		else:
			goal_agent_x, goal_agent_y, goal_marker_x, goal_marker_y = goal_pos
		
		# 5. Create the state
		state = self.empty_state.copy()
		state[0] = robot_x
		state[1] = robot_y
		state[2] = marker_x
		state[3] = marker_y
		state[4] = obstacle_pos[0]
		state[5] = obstacle_pos[1]
		state[6] = obstacle_pos[2]
		state[7] = obstacle_pos[3]
		
		# 6. Create the goal
		goal = self.empty_goal.copy()
		goal[0] = goal_agent_x
		goal[1] = goal_agent_y
		goal[2] = goal_marker_x
		goal[3] = goal_marker_y
		
		# 7. Create the info
		info = {
			"robot_pos": robot_pos,
			"marker_pos": marker_pos,
			"obstacle_pos": obstacle_pos,
			"goal_pos": goal_pos
		}
		
		return state, goal, info
	
	def reset(self):
		"""
		Reset the world to a random state
		:return:
		"""
		
		# Reset Flags
		self.marker_dropped = False
		self.agent_goal_reached = False
		
		# Define elements of the world
		obstacle_pos, robot_pos, marker_pos, goal_pos = None, None, None, None
		
		# Fix Obstacles only
		if self.env_type == "fixed":
			# Determine the center of the world
			center = (self.world_size[0] // 2, self.world_size[1] // 2)
			h = self.world_size[0] // 4
			w = self.world_size[1] // 4
			obstacle_pos = (center[0] - w // 2, center[1] - h // 2, w, h)
		
		state, goal, info = self.random_world(obstacle_pos=obstacle_pos, robot_pos=robot_pos,
											  marker_pos=marker_pos, goal_pos=goal_pos)
		
		# # Check reachability of the goal (using Expert Policy) otherwise reset
		# expert_trajectory: List[int] = self.expert_policy.get_expert_trajectory(state)
		# while len(expert_trajectory) == 0:
		# 	print("[Info] Resetting the world as the marker + goal are not reachable")
		# 	state, info = self.random_world(obstacles_pos=obstacles_pos, robot_pos=robot_pos,
		# 									marker_pos=marker_pos, goal_pos=goal_pos)
		# 	expert_trajectory = self.expert_policy.get_expert_trajectory(state)
		
		self.state = state
		self._current_goal = goal
		return state, goal, info
	
	def step(self, action):
		"""
		How to update the state of the world based on the action performed by the robot
		- Update the robot position based on the action = (delta_x, delta_y) i.e. x + delta_x, y + delta_y
		- If updated position hits an obstacle or the wall, then the robot stays in the same position
		- If updated position has the marker, then the robot picks up the marker
		- If updated position is out of bounds, then the robot stays in the same position
		- If updated position is the goal position, then the robot reaches the goal

		:param action: The action to be performed by the robot. It can be one of the following:
		:return: The next state, reward and done flag
		"""
		# Declare Default Flags
		has_marker: bool = False
		picked_marker: bool = False
		reached_marker_goal: bool = False
		reached_agent_goal: bool = False
		
		curr_robot_pos = self.state[0:2]
		curr_marker_pos = self.state[2:4]
		obstacle_pos = self.state[4:8]
		goal_agent_pos = self._current_goal[0:2]
		goal_marker_pos = self._current_goal[2:4]
		
		# First update the robot position based on the action
		next_robot_pos = curr_robot_pos.copy() + action
		
		# # Check if the robot has hit the wall [use next_robot_pos]
		hit_wall: bool = False
		if next_robot_pos[0] < 0 or next_robot_pos[0] > self.world_size[0] or \
				next_robot_pos[1] < 0 or next_robot_pos[1] > self.world_size[1]:
			hit_wall = True
			next_robot_pos = curr_robot_pos.copy()  # Robot stays in the same position
		
		# Check if the robot has hit an obstacle [use next_robot_pos]
		hit_obstacle: bool = False
		if obstacle_pos[0] <= next_robot_pos[0] <= obstacle_pos[0] + obstacle_pos[2] and \
				obstacle_pos[1] <= next_robot_pos[1] <= obstacle_pos[1] + obstacle_pos[3]:
			hit_obstacle = True
			next_robot_pos = curr_robot_pos.copy()  # Robot stays in the same position
		
		if hit_obstacle or hit_wall:
			# Robot stays in the same position -> No change in the state
			next_state = self.state.copy()
			reward = self.r_obstacle
			done = True
		else:
			# Robot moves to the next position
			next_state = self.state.copy()
			next_state[0:2] = next_robot_pos
			
			# Check if the robot has the marker [use curr_robot_pos]
			has_marker: bool = np.linalg.norm(curr_robot_pos - curr_marker_pos) < self.goal_reach_threshold
			
			# Check if the robot has reached the marker position [use next_robot_pos]
			picked_marker: bool = np.linalg.norm(next_robot_pos - curr_marker_pos) < self.goal_reach_threshold \
								  and not has_marker
			
			# In either case of has_marker or picked_marker, update the marker position
			if (has_marker or picked_marker) and not self.marker_dropped:  # To avoid the marker being picked up again
				curr_marker_pos = next_robot_pos
				next_state[2:4] = curr_marker_pos  # Update the marker state
			
			# Check if the robot is has reached the goal position [use next_robot_pos]
			reached_marker_goal: bool = np.linalg.norm(curr_marker_pos - goal_marker_pos) < self.goal_reach_threshold
			self.marker_dropped = reached_marker_goal
			
			# Check if the robot is has reached the goal position [use next_robot_pos]
			reached_agent_goal: bool = np.linalg.norm(next_robot_pos - goal_agent_pos) < self.goal_reach_threshold
			self.agent_goal_reached = reached_agent_goal and self.marker_dropped
			
			# Generate the reward and done flag
			if reached_agent_goal:
				reward = self.r_goal if self.marker_dropped else self.r_invalid
			else:
				reward = self.r_step
		
		self.state = next_state
		
		info = {
			"has_marker": has_marker,
			"picked_marker": picked_marker,
			"hit_obstacle": hit_obstacle,
			"hit_wall": hit_wall,
			"marker_dropped": self.marker_dropped,
			"agent_goal_reached": self.agent_goal_reached,
			"distance": np.linalg.norm(self.state[:4] - self._current_goal),
		}
		done = info["distance"] < self.goal_reach_threshold
		
		return next_state, reward, done, info
	
	def render(self, state=None, scale=10.0):
		"""
		Show the state of the grid world.
		Use following legends for each entity:
		R: Robot
		M: Marker
		O: Obstacle
		G: Goal

		:return: Image of the grid world
		"""
		pass

	@staticmethod
	def transform_to_goal_space(obs):
		"""
		Transform the state space to goal space
		:param obs:
		:return:
		"""
		return obs[:4]
	
	@property
	def observation_space(self):
		return self._observation_space
	
	@property
	def action_space(self):
		return self._action_space
	
	@property
	def current_goal(self):
		return self._current_goal
	
	
	def forced_reset(self, state_dict):
		initial_state = state_dict['init_state']
		return initial_state
	
	def get_state_dict(self):
		state = self.state
		goal = self._current_goal
		return {
			'init_state': state,
			'goal': goal
		}
	
	@action_space.setter
	def action_space(self, value):
		self._action_space = value
	
	@observation_space.setter
	def observation_space(self, value):
		self._observation_space = value


class MyPointMassDropNReachEnvWrapper(PointMassDropNReachEnv):
	def __init__(self, full_space_as_goal=False, **kwargs):
		"""
		GoalGAIL compatible Wrapper for PnP Env
		Args:
			full_space_as_goal:
			**kwargs:
		"""
		super(MyPointMassDropNReachEnvWrapper, self).__init__(**kwargs)
	
	def reset(self, render=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		obs, goal, info = super(MyPointMassDropNReachEnvWrapper, self).reset()
		if render:
			super(MyPointMassDropNReachEnvWrapper, self).render()
		achieved_goal = super(MyPointMassDropNReachEnvWrapper, self).transform_to_goal_space(obs)
		desired_goal = super(MyPointMassDropNReachEnvWrapper, self).current_goal
		return obs.astype(np.float32), achieved_goal.astype(np.float32), desired_goal.astype(np.float32)
	
	def step(self, action, render=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		if len(action.shape) > 1:
			action = action[0]
		obs, reward, done, info = super(MyPointMassDropNReachEnvWrapper, self).step(action)
		
		if render:
			super(MyPointMassDropNReachEnvWrapper, self).render()
		
		achieved_goal = super(MyPointMassDropNReachEnvWrapper, self).transform_to_goal_space(obs)
		desired_goal = super(MyPointMassDropNReachEnvWrapper, self).current_goal
		success = int(done)
		
		return obs.astype(np.float32), achieved_goal.astype(np.float32), desired_goal.astype(np.float32), np.array(
			success, np.int32), info['distance'].astype(np.float32)
	
	def forced_reset(self, state_dict, render=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		obs = super(MyPointMassDropNReachEnvWrapper, self).forced_reset(state_dict)
		if render:
			super(MyPointMassDropNReachEnvWrapper, self).render()
		achieved_goal = super(MyPointMassDropNReachEnvWrapper, self).transform_to_goal_space(obs)
		desired_goal = super(MyPointMassDropNReachEnvWrapper, self).current_goal
		return obs.astype(np.float32), achieved_goal.astype(np.float32), desired_goal.astype(np.float32)
	
	def get_state_dict(self):
		state_dict = super(MyPointMassDropNReachEnvWrapper, self).get_state_dict()
		return state_dict
	
	def _random_action(self, n):
		action_max = float(a_max)
		return np.random.uniform(low=-action_max, high=action_max, size=(n, 2))
	
	def get_subgoal_info(self) -> Tuple[Dict[str, int], Dict[str, float]]:
		"""
		:return: a dictionary of subgoal info i.e. sub goal name and whether it is achieved along with distances in each
		"""
		subgoal_info = {'subgoals/MarkerDropped': int(self.marker_dropped),
						'subgoals/AgentReached': int(self.agent_goal_reached)}
		
		subgoal_distances = {'subgoals/MarkerDropped': np.linalg.norm(self.state[2:4] - self._current_goal[2:4]),
							 'subgoals/AgentReached': np.linalg.norm(self.state[:2] - self._current_goal[:2])}
		
		return subgoal_info, subgoal_distances
