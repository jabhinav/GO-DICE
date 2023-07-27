from collections import OrderedDict
from typing import List, Dict, Optional, Tuple, Union
import numpy as np


def get_expert_trajectory(state: np.ndarray, goal: np.ndarray, delta_d: float = 0.10) -> List[Tuple[float, float]]:
	a_max = 0.1

	curr_robot_pos = state[0:2]
	curr_marker_pos = state[2:4]
	obstacle_pos = state[4:8]
	goal_pos = goal
	
	# Get to the marker without hitting the obstacle or the wall
	trajectory: List[Tuple[float, float]] = []
	while not np.all(curr_robot_pos == curr_marker_pos):
		# Get the direction of the marker
		direction = curr_marker_pos - curr_robot_pos
		direction = direction / np.linalg.norm(direction)
		
		# Get the next robot position
		next_robot_pos = curr_robot_pos + a_max * direction
		
		# Check if the robot has hit an obstacle. If yes, then change the direction of the robot by delta_d degrees
		while obstacle_pos[0] <= next_robot_pos[0] <= obstacle_pos[0] + obstacle_pos[2] and \
				obstacle_pos[1] <= next_robot_pos[1] <= obstacle_pos[1] + obstacle_pos[3]:
			# Change the direction of the robot by delta_d degrees
			# Formula: x' = x*cos(delta_d) - y*sin(delta_d) and y' = x*sin(delta_d) + y*cos(delta_d)
			direction = np.array([direction[0] * np.cos(delta_d) - direction[1] * np.sin(delta_d),
								  direction[0] * np.sin(delta_d) + direction[1] * np.cos(delta_d)])
			direction = direction / np.linalg.norm(direction)
			next_robot_pos = curr_robot_pos + a_max * direction
			
		# Update the robot position
		curr_robot_pos = next_robot_pos
		trajectory.append(tuple(curr_robot_pos))
		
	# Get to the goal without hitting the obstacle
	while not np.all(curr_robot_pos == goal_pos):
		# Get the direction of the goal
		direction = goal_pos - curr_robot_pos
		direction = direction / np.linalg.norm(direction)
		
		# Get the next robot position
		next_robot_pos = curr_robot_pos + a_max * direction
		
		# Check if the robot has hit an obstacle. If yes, then change the direction of the robot by delta_d degrees
		while obstacle_pos[0] <= next_robot_pos[0] <= obstacle_pos[0] + obstacle_pos[2] and \
				obstacle_pos[1] <= next_robot_pos[1] <= obstacle_pos[1] + obstacle_pos[3]:
			# Change the direction of the robot by delta_d degrees
			# Formula: x' = x*cos(delta_d) - y*sin(delta_d) and y' = x*sin(delta_d) + y*cos(delta_d)
			direction = np.array([direction[0] * np.cos(delta_d) - direction[1] * np.sin(delta_d),
								  direction[0] * np.sin(delta_d) + direction[1] * np.cos(delta_d)])
			direction = direction / np.linalg.norm(direction)
			next_robot_pos = curr_robot_pos + a_max * direction
			
		# Update the robot position
		curr_robot_pos = next_robot_pos
		trajectory.append(tuple(curr_robot_pos))
		
	return trajectory


class PointMassPnP:
	def __init__(self, world_size: Tuple[int, int], env_type: str = "fixed", penalise_steps: bool = False):
		# state space
		self.world_size = world_size
		assert self.world_size[0] == self.world_size[1], "World must be square"
		assert self.world_size[0] >= 3 and self.world_size[1] >= 3, "World must be at least 3x3"
		
		self.max_obstacle_occupancy = 0.4  # For each obstacle, its maximum occupancy in terms of % of the world size
		self.num_obstacles = 1  # Number of obstacles in the world
		assert self.max_obstacle_occupancy * self.num_obstacles < 1, "Obstacles cannot occupy more than 100% of the world"
		
		# # Current State: (x_agent, y_agent, x_obj, y_obj, x_obs, y_obs, h_obs, w_obs)
		self.empty_state = np.zeros((8,), dtype=np.float32)
		self.empty_goal = np.zeros((2,), dtype=np.float32)
		self.state = self.empty_state.copy()
		self.goal = self.empty_goal.copy()
		
		# # Define Rewards
		self.r_goal = 1  # For reaching the goal with the marker
		self.r_invalid = -1  # For reaching the goal without the marker
		self.r_pickup = 0.5  # For picking up the marker. Should be less than r_goal
		self.r_step = 0.0 if not penalise_steps else -0.05  # For taking a step
		self.r_obstacle = -1  # For hitting an obstacle or the wall
		
		# # Environment Type
		self.env_type = env_type
		
		# Other Flags
		self.marker_picked = False
		
		# # Setup Expert Policy
		# self.expert_policy = ExpertPolicy(self)
	
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

		# 4. Randomly place the goal on the grid boundary
		if goal_pos is None:
			if np.random.uniform(0, 1) < 0.5:
				# Place on top or bottom boundary
				goal_x = np.random.uniform(0, self.world_size[0])
				if np.random.uniform(0, 1) < 0.5:
					goal_y = 0
				else:
					goal_y = self.world_size[0]
			else:
				# Place on left or right boundary
				goal_y = np.random.uniform(0, self.world_size[0])
				if np.random.uniform(0, 1) < 0.5:
					goal_x = 0
				else:
					goal_x = self.world_size[0]
			
			goal_pos = (goal_x, goal_y)
		else:
			goal_x, goal_y = goal_pos
			

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
		goal[0] = goal_x
		goal[1] = goal_y

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
		self.marker_picked = False

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
		self.goal = goal
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
		reached_goal: bool = False

		curr_robot_pos = self.state[0:2]
		curr_marker_pos = self.state[2:4]
		obstacle_pos = self.state[4:8]
		goal_pos = self.goal

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
			# Check if the robot has the marker [use curr_robot_pos]
			has_marker: bool = np.all(curr_robot_pos == curr_marker_pos)

			# Check if the robot has reached the marker position [use next_robot_pos]
			picked_marker: bool = np.all(next_robot_pos == curr_marker_pos) and not has_marker

			# Check if the robot is has reached the goal position [use next_robot_pos]
			reached_goal: bool = np.all(next_robot_pos == goal_pos)

			# Update the state i.e. robot position, marker position
			next_state = self.state.copy()
			next_state[0:2] = next_robot_pos
			if has_marker:
				next_state[2:4] = next_robot_pos

			# Update the reward and done flag
			if reached_goal:
				reward = self.r_goal if has_marker else self.r_invalid
				done = True
			elif picked_marker:
				reward = self.r_pickup
				done = False
			else:
				reward = self.r_step
				done = False

		info = {
			"has_marker": has_marker,
			"picked_marker": picked_marker,
			"hit_obstacle": hit_obstacle,
			"hit_wall": hit_wall,
			"reached_goal": reached_goal
		}

		return next_state, reward, done, info

	# def render(self, state=None, scale=10.0):
	# 	"""
	# 	Show the state of the grid world.
	# 	Use following legends for each entity:
	# 	R: Robot
	# 	M: Marker
	# 	O: Obstacle
	# 	G: Goal
	#
	# 	:return: Image of the grid world
	# 	"""
	# 	if state is None:
	# 		state = self.state
	#
	# 	# Get the positions of the entities (all entities exist in continuous space)
	# 	curr_robot_pos = self.state[0:2]
	# 	curr_marker_pos = self.state[2:4]
	# 	obstacle_pos = self.state[4:8]
	# 	goal_pos = self.goal
	#
	# 	# Create the image (need to discretize the continuous space)
	#
	#
	#
	# 	# image[robot_pos[0], robot_pos[1], :] = [1, 0, 0]  # Red
	# 	# image[marker_pos[0], marker_pos[1], :] = [0, 1, 0]  # Green
	# 	# for i in range(obstacles_pos.shape[0]):
	# 	# 	image[obstacles_pos[i, 0], obstacles_pos[i, 1], :] = [0, 0, 1]  # Blue
	# 	# image[goal_pos[0], goal_pos[1], :] = [1, 1, 0]  # Yellow
	#
	# 	return image