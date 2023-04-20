from typing import List

import numpy as np
import tensorflow as tf


def clip_action(ac):
	return np.clip(ac, -1, 1)


class PnPExpert:
	def __init__(self):
		self.num_objs = 1
		self.step_size = 6
		# Thresholds
		self.sub_goal_height = 0.55  # Height to which block will be first taken before moving towards goal.
		self.env_goal_thresh = 0.01  # Distance threshold for to consider some goal reached
		
		self.num_skills = 3  # 3 skills: pick, grab, drop
		
		self.goal_pool = []
		self.ordered_goal_to_achieve_pool = []
		
		self.subgoal_first_obj = None
		self.firstBlockPicked = None
		self.firstBlockRaised = None
		self.firstBlockDropped = None
		
	def policy_pick_skill(self, state, goal_pos, grip=1.):
		"""
		:param state: current state
		:param goal_pos: goal position gripper should reach (object position)
		:param grip: gripper state (set to 1 to open the gripper)
		"""
		curr_pos = state[:3]
		
		def _pick():
			target_pos = goal_pos + np.array([0, 0, 0.08])
			_a = clip_action((target_pos - curr_pos) * self.step_size)
			_a = np.concatenate([_a, np.array([grip])], dtype=np.float32)
			return _a
		
		a = _pick()
		
		return a
	
	def policy_grab_skill(self, state, goal_pos, grip=-1.):
		"""
		:param state: current state
		:param goal_pos: goal position gripper should reach (object position)
		:param grip: gripper state (set to -1 to close the gripper)
		Logic: Move the gripper to the block position (w partially closed gripper = -0.005) and then close the gripper
		"""
		curr_pos = state[:3]
		a = clip_action((goal_pos - curr_pos) * self.step_size)
		if np.linalg.norm(curr_pos - goal_pos) < self.env_goal_thresh:
			a = np.concatenate([a, np.array([grip])], dtype=np.float32)
		else:
			a = np.concatenate([a, np.array([-0.005])], dtype=np.float32)
		return a
	
	def policy_drop_skill(self, state, goal_pos, grip=-1.):
		"""
		:param state: current state
		:param goal_pos: env goal position
		:param grip: gripper state (set to -1 to close the gripper)
		This logic is fine, however sometimes the gripper is not able to satisfy the env goal threshold as the current
		position is of the gripper and not the block
		"""
		curr_pos = state[:3]
		
		def _drop():
			# Collision avoid: We first move the gripper towards (x, y) coordinates of goal maintaining the gripper height
			if np.linalg.norm((goal_pos[:2] - curr_pos[:2])) > self.env_goal_thresh:
				_a_xy = clip_action((goal_pos[:2] - curr_pos[:2]) * self.step_size)
				_a = np.concatenate([_a_xy, np.array([0]), np.array([grip])], dtype=np.float32)
			else:
				_a = clip_action((goal_pos - curr_pos) * self.step_size)
				_a = np.concatenate([_a, np.array([grip])], dtype=np.float32)
			return _a
		
		# First raise the current block to sub_goal_height
		if self.firstBlockPicked and not self.firstBlockRaised:
			if np.linalg.norm((self.subgoal_first_obj - curr_pos)) > self.env_goal_thresh:
				a = clip_action((self.subgoal_first_obj - curr_pos) * self.step_size)
				a = np.concatenate([a, np.array([grip])], dtype=np.float32)
			else:
				self.firstBlockRaised = True
				a = _drop()
		else:
			a = _drop()
		return a
	
	def sample_curr_goal(self, state, env_goal, prev_goal, for_expert=True):
		"""
		:param state: current state of the environment
		:param env_goal: goal position of the environment
		(env_goal is used to construct the goal pool, so it is used to sample goals)
		:param prev_goal: previous goal achieved
		:param for_expert: if True, sample goals for expert, else sample goals for agent
		:return: goal to achieve
		"""
		gripper_pos = state[:3]
		# Check whether the prev_goal is achieved or not
		if np.linalg.norm(gripper_pos - prev_goal) <= self.env_goal_thresh:
			try:
				# Extract the index of array from the list of arrays
				if for_expert:
					# Do a hard lookup for expert goals
					prev_goal_idx = np.where((self.ordered_goal_to_achieve_pool == prev_goal).all(axis=1))[0][0]
				else:
					# Do a soft lookup for agent goals by computing the distance between the list of goals and prev_goal
					# If the distance is less than env_goal_thresh, then we extract the index else index = -1
					prev_goal_idx = -1  # If not found, agent must achieve the ordered_goal_to_achieve_pool[0]
					for i, goal in enumerate(self.ordered_goal_to_achieve_pool):
						if np.linalg.norm(goal - prev_goal) <= self.env_goal_thresh:
							prev_goal_idx = i
							break
			except IndexError:
				prev_goal_idx = -1
			
			if prev_goal_idx == -1:
				curr_goal = self.ordered_goal_to_achieve_pool[0]
			elif prev_goal_idx == 0:
				# First block is picked
				self.firstBlockPicked = True
				curr_goal = self.ordered_goal_to_achieve_pool[1]
			elif prev_goal_idx == 1:
				# First block is dropped
				self.firstBlockDropped = True
				curr_goal = prev_goal
			else:
				raise ValueError("Invalid prev_goal_idx!")
		else:
			curr_goal = prev_goal
		
		return curr_goal
	
	def sample_curr_skill(self, prev_goal, prev_skill, state, env_goal, curr_goal):
		# Do argmax on prev_skill and obtain the string name of the skill
		prev_skill = self.skill_idx_to_name(prev_skill)
		
		gripper_pos = state[:3]
		
		# Determine if prev_skill is achieved or not
		if prev_skill == "pick":
			# Check the termination condition for the pick skill
			rel_pos = gripper_pos - curr_goal
			if np.linalg.norm(rel_pos) <= 0.1 or np.linalg.norm(
					rel_pos - np.array([0, 0, 0.08])) <= self.env_goal_thresh:
				# Pick skill should be terminated -> Transition to grab skill
				curr_skill = "grab"
			else:
				curr_skill = "pick"
		
		elif prev_skill == "grab":
			# Check the termination condition for the grab skill
			prev_rel_pos = gripper_pos - prev_goal
			if np.linalg.norm(prev_rel_pos) <= self.env_goal_thresh:
				# Grab skill should be terminated -> Transition to drop skill
				curr_skill = "drop"
			else:
				curr_skill = "grab"
		
		elif prev_skill == "drop":
			prev_rel_pos = gripper_pos - prev_goal
			if np.linalg.norm(prev_rel_pos) <= self.env_goal_thresh:
				if np.all(prev_goal == curr_goal):
					# All goals are achieved
					curr_skill = "drop"
				else:
					# Drop skill should be terminated -> Transition to pick skill
					curr_skill = "pick"
			else:
				if np.all(prev_goal == curr_goal):
					# Previous goal not achieved, so current goal is the same as previous goal
					curr_skill = "drop"
				else:
					# Previous goal not achieved, however new goal set
					raise ValueError("Previous goal not achieved, however new goal set! "
									 "Should take object to new goal ? or drop it ?")
		else:
			raise ValueError("Invalid skill!")
		
		return self.skill_name_to_idx(curr_skill)
	
	def sample_action(self, state, env_goal, curr_goal, curr_skill):
		# Execute the current skill
		curr_skill_name = self.skill_idx_to_name(curr_skill)
		if curr_skill_name == "pick":
			a = self.policy_pick_skill(state, curr_goal)
		elif curr_skill_name == "grab":
			a = self.policy_grab_skill(state, curr_goal)
		elif curr_skill_name == "drop":
			a = self.policy_drop_skill(state, curr_goal)
		else:
			raise ValueError("Invalid skill!")
		return a
	
	def act(self, state, env_goal, prev_goal, prev_skill, **kwargs):
		state, env_goal, prev_goal, prev_skill = state[0], env_goal[0], prev_goal[0], prev_skill[0]
		state, env_goal, prev_goal, prev_skill = state.numpy(), env_goal.numpy(), prev_goal.numpy(), prev_skill.numpy()
		
		# Determine the current goal to achieve
		curr_goal = self.sample_curr_goal(state, env_goal, prev_goal)
		
		# Determine the current skill to execute
		curr_skill = self.sample_curr_skill(prev_goal, prev_skill, state, env_goal, curr_goal)
		
		# Execute the current skill
		curr_action = self.sample_action(state, env_goal, curr_goal, curr_skill)
		
		# Convert np arrays to tensorflow tensor with shape (1,-1)
		curr_goal = tf.convert_to_tensor(curr_goal.reshape(1, -1), dtype=tf.float32)
		curr_skill = tf.convert_to_tensor(curr_skill.reshape(1, -1), dtype=tf.int32)
		curr_action = tf.convert_to_tensor(curr_action.reshape(1, -1), dtype=tf.float32)
		return curr_goal, curr_skill, curr_action
	
	def reset(self, init_state, env_goal):
		"""
		param init_state: Initial state of the environment
		param env_goal: Goal position of the environment
		"""
		init_state = init_state.numpy() if isinstance(init_state, tf.Tensor) else init_state
		env_goal = env_goal.numpy() if isinstance(env_goal, tf.Tensor) else env_goal
		
		# Populate goal pool with positions of all objects and the goal position based on number of objects
		self.goal_pool = []
		for i in range(self.num_objs):
			self.goal_pool.append((init_state[3 * (i + 1): 3 * (i + 2)], env_goal[3 * i: 3 * (i + 1)]))
		# Define the default order of goals to achieve from the goal pool
		self.ordered_goal_to_achieve_pool = [_goal for obj_goal in self.goal_pool for _goal in obj_goal]
		
		self.subgoal_first_obj = np.concatenate([self.goal_pool[0][0][:2], np.array([self.sub_goal_height])])
		
		self.firstBlockPicked = False
		self.firstBlockRaised = False
		self.firstBlockDropped = False
	
	@staticmethod
	def skill_idx_to_name(skill):
		"""
		:param skill: skill index
		:return: skill name
		"""
		skill = np.argmax(skill, axis=-1)
		skill = skill.item()
		if skill == 0:
			return 'pick'
		elif skill == 1:
			return 'grab'
		elif skill == 2:
			return 'drop'
		else:
			raise ValueError("Invalid skill index")
	
	@staticmethod
	def skill_name_to_idx(skill):
		"""
		:param skill: skill name
		:return: skill index
		"""
		if skill == 'pick':
			# Numpy one hot encoding
			return np.array([1, 0, 0])
		elif skill == 'grab':
			return np.array([0, 1, 0])
		elif skill == 'drop':
			return np.array([0, 0, 1])
		else:
			raise ValueError("Invalid skill name")
	
	@staticmethod
	def unwrap_skill(skill):
		"""
		:param skill: skill executed of shape (self.num_skills, )
		:return: Effective skill executed
		"""
		return skill
	
	@staticmethod
	def wrap_skill(skill):
		"""
		:param skill: effective skill executed of shape (3, )
		:return: model consumable skill of shape (self.num_skills, )
		"""
		return skill
		
	@staticmethod
	def get_init_skill():
		# Will always start with pick skill
		skill = tf.one_hot(np.array([0]), depth=3, dtype=tf.float32)
		skill = tf.reshape(skill, shape=(1, -1))
		return skill
	
	@staticmethod
	def get_init_goal(init_state, g_env):
		curr_goal = init_state[:3]  # g_-1 = s_0 (gripper pos in the state)
		curr_goal = tf.reshape(curr_goal, shape=(1, -1))
		return curr_goal


class PnPExpertTwoObj:
	def __init__(self, expert_behaviour: str = '0', wrap_skill_id: str = '0'):
		self.num_objs = 2
		self.step_size = 6
		# Thresholds
		self.sub_goal_height = 0.55  # Height to which block will be first taken before moving towards goal.
		self.env_goal_thresh = 0.01  # Distance threshold for to consider some goal reached
		
		# How to wrap the effective skill to model-consumable skill
		# 0: pick, grab, drop
		# 1: pick:0, grab:0, drop:0, pick:1, grab:1, drop:1
		# 2: obj:0, obj:1, pick, grab, drop
		self.num_skills = 5 if wrap_skill_id == '2' else 6 if wrap_skill_id == '1' else 3
		
		self.goal_pool = []
		self.ordered_goal_to_achieve_pool = []
		
		self.subgoal_gripper_post_drop = None
		self.subgoal_first_obj = None
		self.subgoal_second_obj = None
		self.firstBlockPicked, self.secondBlockPicked = None, None
		self.firstBlockRaised, self.secondBlockRaised = None, None
		self.firstBlockDropped, self.secondBlockDropped = None, None
		self.gripperRaised = None
		self.obj_order = None
		
		self.expert_behaviour: str = expert_behaviour  # One of ['0', '1', '2']
		
	def policy_pick_skill(self, state, goal_pos, grip=1.):
		"""
		:param state: current state
		:param goal_pos: goal position gripper should reach (object position)
		:param grip: gripper state (set to 1 to open the gripper)
		"""
		curr_pos = state[:3]
		
		def _pick():
			target_pos = goal_pos + np.array([0, 0, 0.08])
			_a = clip_action((target_pos - curr_pos) * self.step_size)
			_a = np.concatenate([_a, np.array([grip])], dtype=np.float32)
			return _a
		
		if self.firstBlockDropped and not self.gripperRaised:
			
			if np.linalg.norm((self.subgoal_gripper_post_drop - curr_pos)) > 0.05:
				a = clip_action((self.subgoal_gripper_post_drop - curr_pos) * self.step_size)
				a = np.concatenate([a, np.array([grip])], dtype=np.float32)
			
			else:
				self.gripperRaised = True
				a = _pick()
		
		else:
			a = _pick()
		
		return a

	def policy_grab_skill(self, state, goal_pos, grip=-1.):
		"""
		:param state: current state
		:param goal_pos: goal position gripper should reach (object position)
		:param grip: gripper state (set to -1 to close the gripper)
		Logic: Move the gripper to the block position (w partially closed gripper = -0.005) and then close the gripper
		"""
		curr_pos = state[:3]
		a = clip_action((goal_pos - curr_pos) * self.step_size)
		if np.linalg.norm(curr_pos - goal_pos) < self.env_goal_thresh:
			a = np.concatenate([a, np.array([grip])], dtype=np.float32)
		else:
			a = np.concatenate([a, np.array([-0.005])], dtype=np.float32)
		return a
	
	def policy_drop_skill(self, state, goal_pos, grip=-1.):
		"""
		:param state: current state
		:param goal_pos: env goal position
		:param grip: gripper state (set to -1 to close the gripper)
		This logic is fine, however sometimes the gripper is not able to satisfy the env goal threshold as the current
		position is of the gripper and not the block
		"""
		curr_pos = state[:3]
		
		def _drop():
			# Collision avoid: We first move the gripper towards (x, y) coordinates of goal maintaining the gripper height
			if np.linalg.norm((goal_pos[:2] - curr_pos[:2])) > self.env_goal_thresh:
				_a_xy = clip_action((goal_pos[:2] - curr_pos[:2]) * self.step_size)
				_a = np.concatenate([_a_xy, np.array([0]), np.array([grip])], dtype=np.float32)
			else:
				_a = clip_action((goal_pos - curr_pos) * self.step_size)
				_a = np.concatenate([_a, np.array([grip])], dtype=np.float32)
			return _a
		
		# First raise the current block to sub_goal_height
		if self.firstBlockPicked and not self.firstBlockRaised:
			if np.linalg.norm((self.subgoal_first_obj - curr_pos)) > self.env_goal_thresh:
				a = clip_action((self.subgoal_first_obj - curr_pos) * self.step_size)
				a = np.concatenate([a, np.array([grip])], dtype=np.float32)
			else:
				self.firstBlockRaised = True
				a = _drop()
		elif self.secondBlockPicked and not self.secondBlockRaised:
			if np.linalg.norm((self.subgoal_second_obj - curr_pos)) > self.env_goal_thresh:
				a = clip_action((self.subgoal_second_obj - curr_pos) * self.step_size)
				a = np.concatenate([a, np.array([grip])], dtype=np.float32)
			else:
				self.secondBlockRaised = True
				a = _drop()
		else:
			a = _drop()
		return a
	
	def sample_curr_goal(self, state, prev_goal, for_expert=True):
		"""
		:param state: current state of the environment
		(env_goal is used to construct the goal pool, so it is used to sample goals)
		:param prev_goal: previous goal achieved
		:param for_expert: if True, sample goals for expert, else sample goals for agent
		:return: goal to achieve
		"""
		gripper_pos = state[:3]
		# Check whether the prev_goal is achieved or not
		if np.linalg.norm(gripper_pos - prev_goal) <= self.env_goal_thresh:
			try:
				# Extract the index of array from the list of arrays
				if for_expert:
					# Do a hard lookup for expert goals
					prev_goal_idx = np.where((self.ordered_goal_to_achieve_pool == prev_goal).all(axis=1))[0][0]
				else:
					# Do a soft lookup for agent goals by computing the distance between the list of goals and prev_goal
					# If the distance is less than env_goal_thresh, then we extract the index else index = -1
					prev_goal_idx = -1  # If not found, agent must achieve the ordered_goal_to_achieve_pool[0]
					for i, goal in enumerate(self.ordered_goal_to_achieve_pool):
						if np.linalg.norm(goal - prev_goal) <= self.env_goal_thresh:
							prev_goal_idx = i
							break
			except IndexError:
				prev_goal_idx = -1
			
			if prev_goal_idx == -1:
				curr_goal = self.ordered_goal_to_achieve_pool[0]
			elif prev_goal_idx == 0:
				# First block is picked
				self.firstBlockPicked = True
				curr_goal = self.ordered_goal_to_achieve_pool[1]
			elif prev_goal_idx == 1:
				# First block is dropped
				self.firstBlockDropped = True
				curr_goal = self.ordered_goal_to_achieve_pool[2]
			elif prev_goal_idx == 2:
				# Second block is picked
				self.secondBlockPicked = True
				curr_goal = self.ordered_goal_to_achieve_pool[3]
			elif prev_goal_idx == 3:
				# Second block is dropped
				self.secondBlockDropped = True
				# All goals achieved
				curr_goal = prev_goal
			else:
				raise ValueError("Invalid prev_goal_idx!")
		else:
			curr_goal = prev_goal
			
		return curr_goal
	
	def sample_curr_skill(self, prev_goal, prev_skill, state, curr_goal):
		# Do argmax on prev_skill and obtain the string name of the skill
		prev_skill = self.skill_idx_to_name(prev_skill)
		
		gripper_pos = state[:3]
		
		# Determine if prev_skill is achieved or not
		if prev_skill == "pick":
			# Check the termination condition for the pick skill
			rel_pos = gripper_pos - curr_goal
			if np.linalg.norm(rel_pos) <= 0.1 or np.linalg.norm(rel_pos - np.array([0, 0, 0.08])) <= self.env_goal_thresh:
				# Pick skill should be terminated -> Transition to grab skill
				curr_skill = "grab"
			else:
				curr_skill = "pick"
				
		elif prev_skill == "grab":
			# Check the termination condition for the grab skill
			prev_rel_pos = gripper_pos - prev_goal
			if np.linalg.norm(prev_rel_pos) <= self.env_goal_thresh:
				# Grab skill should be terminated -> Transition to drop skill
				curr_skill = "drop"
			else:
				curr_skill = "grab"
				
		elif prev_skill == "drop":
			prev_rel_pos = gripper_pos - prev_goal
			if np.linalg.norm(prev_rel_pos) <= self.env_goal_thresh:
				if np.all(prev_goal == curr_goal):  # TODO: Make it a soft lookup
					# All goals are achieved -> stay at drop skill
					curr_skill = "drop"
				else:
					# Drop skill should be terminated -> Transition to pick skill
					curr_skill = "pick"
			else:
				if np.all(prev_goal == curr_goal):  # TODO: Make it a soft lookup
					# Previous goal not achieved, so current goal is the same as previous goal -> stay at drop skill
					curr_skill = "drop"
				else:
					# Previous goal not achieved, however new goal set
					raise ValueError("Previous goal not achieved, however new goal set! "
									 "Should take object to new goal ? or drop it ?")
		else:
			raise ValueError("Invalid skill!")
		
		return self.skill_name_to_idx(curr_skill)
	
	def sample_action(self, state, curr_goal, curr_skill):
		# Execute the current skill
		curr_skill_name = self.skill_idx_to_name(curr_skill)
		if curr_skill_name == "pick":
			a = self.policy_pick_skill(state, curr_goal)
		elif curr_skill_name == "grab":
			a = self.policy_grab_skill(state, curr_goal)
		elif curr_skill_name == "drop":
			a = self.policy_drop_skill(state, curr_goal)
		else:
			raise ValueError("Invalid skill!")
		return a
		
	def act(self, state, env_goal, prev_goal, prev_skill, **kwargs):
		state, env_goal, prev_goal, prev_skill = state[0], env_goal[0], prev_goal[0], prev_skill[0]
		state, env_goal, prev_goal, prev_skill = state.numpy(), env_goal.numpy(), prev_goal.numpy(), prev_skill.numpy()
		
		# Unwrap the previous skill
		prev_skill = self.unwrap_skill(prev_skill)
		
		# Determine the current goal to achieve
		curr_goal = self.sample_curr_goal(state, prev_goal)
		
		# Determine the current skill to execute
		curr_skill = self.sample_curr_skill(prev_goal, prev_skill, state, curr_goal)
		
		# Execute the current skill
		curr_action = self.sample_action(state, curr_goal, curr_skill)
		
		# Wrap the current skill
		curr_skill = self.wrap_skill(curr_skill)

		# Convert np arrays to tensorflow tensor with shape (1,-1)
		curr_goal = tf.convert_to_tensor(curr_goal.reshape(1, -1), dtype=tf.float32)
		curr_skill = tf.convert_to_tensor(curr_skill.reshape(1, -1), dtype=tf.int32)
		curr_action = tf.convert_to_tensor(curr_action.reshape(1, -1), dtype=tf.float32)
		return curr_goal, curr_skill, curr_action
	
	def reset(self, init_state, env_goal):
		"""
		param init_state: Initial state of the environment
		param env_goal: Goal position of the environment
		"""
		init_state = init_state.numpy() if isinstance(init_state, tf.Tensor) else init_state
		env_goal = env_goal.numpy() if isinstance(env_goal, tf.Tensor) else env_goal
		
		# Populate goal pool with positions of all objects and the goal position based on number of objects
		
		# TODO: Make it mistake-proof i.e. update the ordered_goal_to_achieve_pool at each step. Sometimes, you end up
		#  hitting an uncollected object and displace from its initial pos which was the future goal to achieve.
		#  So, you need to update the ordered_goal_to_achieve_pool at each step to account for updated positions.
		self.goal_pool = []
		for i in range(self.num_objs):
			self.goal_pool.append((init_state[3*(i+1): 3*(i+2)],
								   env_goal[3*i: 3*(i+1)]))
		
		# Define the default order of goals to achieve from the goal pool
		self.ordered_goal_to_achieve_pool = [_goal for obj_goal in self.goal_pool for _goal in obj_goal]
	
		self.subgoal_first_obj = np.concatenate([self.goal_pool[0][0][:2], np.array([self.sub_goal_height])])
		self.subgoal_second_obj = np.concatenate([self.goal_pool[1][0][:2], np.array([self.sub_goal_height])])
		self.subgoal_gripper_post_drop = np.concatenate(
			[self.goal_pool[0][1][:2], np.array([self.sub_goal_height])])
		
		self.firstBlockPicked, self.secondBlockPicked = False, False
		self.firstBlockDropped, self.secondBlockDropped = False, False
		self.firstBlockRaised, self.secondBlockRaised = False, False
		self.gripperRaised = False
		self.obj_order = [0, 1]
		
		if self.expert_behaviour == '0':
			pass
		elif self.expert_behaviour == '1':
			if np.random.uniform() > 0.5:
				self.permute_goals([1, 0])
		else:
			raise NotImplementedError("Expert behaviour not implemented!: {}".format(self.expert_behaviour))
	
	def update_goal_pool(self, state, env_goal):
		state = state.numpy() if isinstance(state, tf.Tensor) else state
		env_goal = env_goal.numpy() if isinstance(env_goal, tf.Tensor) else env_goal
		
		self.goal_pool = []
		for i in range(self.num_objs):
			self.goal_pool.append((state[3 * (i + 1): 3 * (i + 2)],
								   env_goal[3 * i: 3 * (i + 1)]))
		
		# Define the default order of goals to achieve from the goal pool
		self.ordered_goal_to_achieve_pool = [_goal for obj_goal in self.goal_pool for _goal in obj_goal]
	
	def permute_goals(self, order: List[int]):
		"""
		:param order: Order of object indices to be picked up
		"""

		# Reset the flags based on their current state
		if self.firstBlockDropped and self.secondBlockDropped:
			# Do nothing as both blocks are already dropped
			pass
		elif self.firstBlockDropped and not self.secondBlockDropped:
			# Do nothing as we can only drop the second block
			pass
		else:
			self.obj_order = order
			# Permute the goal pool based on the order of object indices. The passed obj indices then define
			# 0 -> first object, 1 -> second object otherwise 1 -> first object, 0 -> second object
			self.goal_pool = [self.goal_pool[i] for i in order]
			self.ordered_goal_to_achieve_pool = [_goal for obj_goal in self.goal_pool for _goal in obj_goal]
			# Reset the sub_goals
			self.subgoal_first_obj = np.concatenate(
				[self.goal_pool[order[0]][0][:2], np.array([self.sub_goal_height])])
			self.subgoal_second_obj = np.concatenate(
				[self.goal_pool[order[1]][0][:2], np.array([self.sub_goal_height])])
			self.subgoal_gripper_post_drop = np.concatenate(
				[self.goal_pool[order[0]][1][:2], np.array([self.sub_goal_height])])
				
			# Reset the flags
			self.firstBlockRaised = False
			self.firstBlockPicked = False

	@staticmethod
	def skill_idx_to_name(skill):
		"""
		:param skill: skill index
		:return: skill name
		"""
		skill = np.argmax(skill, axis=-1)
		skill = skill.item()
		if skill == 0:
			return 'pick'
		elif skill == 1:
			return 'grab'
		elif skill == 2:
			return 'drop'
		else:
			raise ValueError("Invalid skill index")
	
	@staticmethod
	def skill_name_to_idx(skill):
		"""
		:param skill: skill name
		:return: skill index
		"""
		if skill == 'pick':
			# Numpy one hot encoding
			return np.array([1, 0, 0])
		elif skill == 'grab':
			return np.array([0, 1, 0])
		elif skill == 'drop':
			return np.array([0, 0, 1])
		else:
			raise ValueError("Invalid skill name")
		
	def unwrap_skill(self, skill):
		"""
		:param skill: skill executed of shape (self.num_skills, )
		:return: Effective skill executed
		"""
		if self.num_skills == 3:
			# Leave the skill unchanged
			return skill
		elif self.num_skills == 6:
			# Extract the effective skill from the skill executed
			# [pick:0, grab:0, drop:0, pick:1, grab:1, drop:1] -> [pick, grab, drop]
			return np.eye(3, dtype=np.int64)[int(skill.argmax() % 3)]
		elif self.num_skills == 5:
			# Extract the effective skill from the skill executed
			# [obj:0, obj:1, pick, grab, drop] -> [pick, grab, drop]
			return skill[2:]
		else:
			raise ValueError("Invalid number of skills! Check initialization of the expert")
		
	def wrap_skill(self, skill):
		"""
		:param skill: effective skill executed of shape (3, )
		:return: model consumable skill of shape (self.num_skills, )
		"""
		if self.num_skills == 3:
			# Leave the skill unchanged
			return skill
		else:
			# Get the current object index
			if not self.firstBlockDropped and not self.secondBlockDropped:
				obj_idx: int = self.obj_order[0]
			else:
				obj_idx: int = self.obj_order[1]
				
			if self.num_skills == 6:
				# [pick, grab, drop] -> [pick:0, grab:0, drop:0, pick:1, grab:1, drop:1]
				return np.eye(6, dtype=np.int64)[obj_idx * 3 + skill.argmax()]
			
			elif self.num_skills == 5:
				# [pick, grab, drop] -> [obj:0, obj:1, pick, grab, drop]
				return np.concatenate([np.eye(2, dtype=np.int64)[obj_idx], skill], axis=-1)
			else:
				raise ValueError("Invalid number of skills! Check initialization of the expert")
			
	def get_init_skill(self):
		skill = self.wrap_skill(np.array([1, 0, 0], dtype=np.int64))
		# Convert to tf tensor
		skill = tf.convert_to_tensor(skill, dtype=tf.float32)
		skill = tf.reshape(skill, shape=(1, -1))
		return skill
	
	@staticmethod
	def get_init_goal(init_state, g_env):
		curr_goal = init_state[:3]  # g_-1 = s_0 (gripper pos in the state)
		curr_goal = tf.reshape(curr_goal, shape=(1, -1))
		return curr_goal


class PnPExpertTwoObjImitator:
	def __init__(self, wrap_skill_id: str = '1'):
		
		self.wrap_skill_id = wrap_skill_id
		self.step_size = 6
		# self.sub_goal_height = 0.55  # Height to which block will be first taken before moving towards goal.
		# self.env_goal_thresh = 0.005  # Distance threshold for to consider some goal reached
		
		self.use_expert_pick = False
		self.use_expert_grab = False
		self.use_expert_drop = True
		
		self.gripper_raise_iters: int = 8
	
	def policy_pick(self, state, env_goal, curr_skill, grip=0.05):  # My open grip is 1.0
		gripper_pos = state[:3]
		if np.argmax(curr_skill) == 0:
			# Pick the first block
			objectRelPos = state[9:12]
		elif np.argmax(curr_skill) == 3:
			# Pick the second block
			objectRelPos = state[12:15]
		else:
			raise ValueError("Invalid skill index for pick! {}".format(np.argmax(curr_skill)))
		
		if np.argmax(curr_skill) == 3:
			# We need to raise the gripper first before moving towards the goal
			gripperRaised = True if self.gripper_raise_iters == 0 else False
			if not gripperRaised:
				# First open the gripper slightly
				if self.gripper_raise_iters > 4:
					a = np.array([0, 0, 0])
				else:
					# Then raise the gripper
					a = np.array([0, 0, 1.])
				a = np.concatenate([a, np.array([0.005])], dtype=np.float32)
				self.gripper_raise_iters -= 1
				return a
		
		object_oriented_goal = objectRelPos + np.array([0, 0, 0.03])  # My threshold is 0.08
		a = clip_action(object_oriented_goal * self.step_size)
		a = np.concatenate([a, np.array([grip])], dtype=np.float32)
		return a
	
	def policy_grab(self, state, env_goal, curr_skill, grip=-0.005):
		if np.argmax(curr_skill) == 1:
			# Grab the first block
			objectRelPos = state[9:12]
		elif np.argmax(curr_skill) == 4:
			# Grab the second block
			objectRelPos = state[12:15]
		else:
			raise ValueError("Invalid skill index for drop! {}".format(np.argmax(curr_skill)))
		
		a = clip_action(objectRelPos * self.step_size)
		a = np.concatenate([a, np.array([grip])], dtype=np.float32)
		return a
		
	def policy_drop(self, state, env_goal, curr_skill, grip=-0.005):
		"""
		:param state: current state
		:param env_goal: env goal position
		:param curr_skill: current skill
		:param grip: gripper state (set to -1 to close the gripper)
		This logic is fine, however sometimes the gripper is not able to satisfy the env goal threshold as the current
		position is of the gripper and not the block
		"""
		# gripper_pos = state[:3]
		if np.argmax(curr_skill) == 2:
			# Drop the first block
			object_pos = state[3:6]
			goal = env_goal[:3]
			delta_obj = state[9:12]
			delta_goal = goal - state[3:6]
		elif np.argmax(curr_skill) == 5:
			# Drop the second block
			object_pos = state[6:9]
			goal = env_goal[3:]
			delta_obj = state[12:15]
			delta_goal = goal - state[6:9]
		else:
			raise ValueError("Invalid skill index for drop! {}".format(np.argmax(curr_skill)))
		
		# Move Diagonally above the goal
		BlockAboveGoal = True if np.linalg.norm(delta_goal[:2]) <= 0.01 else False
		if not BlockAboveGoal:
			a = clip_action((goal + np.array([0, 0, 0.1]) - object_pos) * self.step_size)
			a = np.concatenate([a, np.array([grip])], dtype=np.float32)
		else:
			_a = clip_action((goal - object_pos) * self.step_size)
			a = np.concatenate([_a, np.array([grip])], dtype=np.float32)
			
		return a
	
	def sample_action(self, state, env_goal, curr_skill, model_action=None):
		assert self.wrap_skill_id == '1'
		
		if np.argmax(curr_skill) == 0 or np.argmax(curr_skill) == 3:
			# Pick the block
			a = self.policy_pick(state, env_goal, curr_skill) if self.use_expert_pick else model_action
			
		elif np.argmax(curr_skill) == 1 or np.argmax(curr_skill) == 4:
			# Grab the block
			a = self.policy_grab(state, env_goal, curr_skill) if self.use_expert_grab else model_action
		else:
			# Drop the block
			a = self.policy_drop(state, env_goal, curr_skill) if self.use_expert_drop else model_action
			
		return a

	
	def sample_curr_skill(self, state, env_goal, prev_skill):
		
		assert self.wrap_skill_id == '1'
		
		delta_obj1 = state[9:12]
		delta_obj2 = state[12:15]
		delta_goal1 = env_goal[:3] - state[3:6]
		delta_goal2 = env_goal[3:] - state[6:9]
		
		# If previous skill was pick object 1
		if np.argmax(prev_skill) == 0:
			# If gripper is not close to object 1, then keep picking object 1
			# print(np.linalg.norm(delta_obj1 + np.array([0, 0, 0.03])))
			if np.linalg.norm(delta_obj1 + np.array([0, 0, 0.03])) < 0.01:
				# Transition to grab object 1
				curr_skill = np.array([0, 1, 0, 0, 0, 0])
			else:
				# Keep picking object 1
				curr_skill = np.array([1, 0, 0, 0, 0, 0])
		
		# If previous skill was grab object 1
		elif np.argmax(prev_skill) == 1:
			# If gripper is not close to object 1, then keep grabbing object 1
			if np.linalg.norm(delta_obj1) < 0.005:
				# Grab skill should be terminated -> Transition to drop skill
				curr_skill = np.array([0, 0, 1, 0, 0, 0])
			else:
				curr_skill = np.array([0, 1, 0, 0, 0, 0])
		
		# If previous skill was drop object 1
		elif np.argmax(prev_skill) == 2:
			# If gripper is not close to goal 1, then keep dropping object 1
			if np.linalg.norm(delta_goal1) < 0.01:
				# Drop skill should be terminated -> Transition to pick skill of object 2
				curr_skill = np.array([0, 0, 0, 1, 0, 0])
			else:
				curr_skill = np.array([0, 0, 1, 0, 0, 0])
		
		# If previous skill was pick object 2
		elif np.argmax(prev_skill) == 3:
			# If gripper is not close to object 2, then keep picking object 2
			if np.linalg.norm(delta_obj2 + np.array([0, 0, 0.03])) <= 0.005:
				# Transition to grab object 2
				curr_skill = np.array([0, 0, 0, 0, 1, 0])
			else:
				# Keep picking object 2
				curr_skill = np.array([0, 0, 0, 1, 0, 0])
		
		# If previous skill was grab object 2
		elif np.argmax(prev_skill) == 4:
			# If gripper is not close to object 2, then keep grabbing object 2
			if np.linalg.norm(delta_obj2) <= 0.005:
				# Grab skill should be terminated -> Transition to drop skill
				curr_skill = np.array([0, 0, 0, 0, 0, 1])
			else:
				curr_skill = np.array([0, 0, 0, 0, 1, 0])
		
		# If previous skill was drop object 2
		else:
			# Stay in drop skill
			curr_skill = np.array([0, 0, 0, 0, 0, 1])
		
		return np.cast[np.float32](curr_skill)
	
	def act(self, state, env_goal, prev_goal, prev_skill, **kwargs):
		state, env_goal, prev_goal, prev_skill = state[0], env_goal[0], prev_goal[0], prev_skill[0]
		state, env_goal, prev_goal, prev_skill = state.numpy(), env_goal.numpy(), prev_goal.numpy(), prev_skill.numpy()
		
		# Determine the current goal to achieve
		curr_goal = env_goal
		
		# Determine the current skill to execute
		curr_skill = self.sample_curr_skill(state, env_goal, prev_skill)
		
		# Execute the current skill
		curr_action = self.sample_action(state, env_goal, curr_skill)
		
		# Convert np arrays to tensorflow tensor with shape (1,-1)
		curr_goal = tf.convert_to_tensor(curr_goal.reshape(1, -1), dtype=tf.float32)
		curr_skill = tf.convert_to_tensor(curr_skill.reshape(1, -1), dtype=tf.int32)
		curr_action = tf.convert_to_tensor(curr_action.reshape(1, -1), dtype=tf.float32)
		return curr_goal, curr_skill, curr_action
	
	def sample_curr_goal(self, state, prev_goal, for_expert=True):
		raise NotImplementedError
	
	def reset(self, init_state, env_goal):
		self.gripper_raise_iters: int = 8
	
	@staticmethod
	def get_init_skill():
		skill = np.array([1, 0, 0, 0, 0, 0], dtype=np.int64)
		# Convert to tf tensor
		skill = tf.convert_to_tensor(skill, dtype=tf.float32)
		skill = tf.reshape(skill, shape=(1, -1))
		return skill
	
	@staticmethod
	def get_init_goal(init_state, g_env):
		curr_goal = tf.reshape(g_env, shape=(1, -1))
		return curr_goal
