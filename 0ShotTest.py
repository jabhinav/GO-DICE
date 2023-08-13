import json
import os
from abc import ABC
from argparse import Namespace
from collections import OrderedDict

import logging
import numpy as np
import tensorflow as tf

from configs.DICE import get_DICE_args
from her.replay_buffer import ReplayBufferTf
from models.Base import AgentBase
from networks.general import Actor
from networks.general import SkilledActors
from utils.env import get_expert


class skilledDemoDICE(tf.keras.Model, ABC):
	def __init__(self, args: Namespace):
		super(skilledDemoDICE, self).__init__()
		self.args = args
		
		self.args.EPS = np.finfo(np.float32).eps  # Small value = 1.192e-07 to avoid division by zero in grad penalty
		self.args.EPS2 = 1e-3
		
		# Define effective state and goal dimensions
		self.args.s_eff_dim = 10
		self.args.c_eff_dim = 3
		self.args.g_eff_dim = 3
		
		# Define Networks
		self.skilled_actors = SkilledActors(args.a_dim, args.c_dim)
		# Override the number of actors
		self.skilled_actors.actors = [Actor(self.args.a_dim) for _ in range(self.args.c_eff_dim)]
		self.skilled_actors.target_actors = [Actor(self.args.a_dim) for _ in range(self.args.c_eff_dim)]
		
		self.build_model()
		
		self.act_w_expert_skill = False
		self.act_w_expert_action = False
		self.expert = get_expert(args.num_objs, args)
	
	@tf.function(experimental_relax_shapes=True)  # This is needed to avoid shape errors
	def act(self, state, env_goal, prev_goal, prev_skill, epsilon, stddev):
		state = tf.clip_by_value(state, -self.args.clip_obs, self.args.clip_obs)
		env_goal = tf.clip_by_value(env_goal, -self.args.clip_obs, self.args.clip_obs)
		prev_goal = tf.clip_by_value(prev_goal, -self.args.clip_obs, self.args.clip_obs)
		
		# ###################################### Current Goal ####################################### #
		curr_goal = env_goal
		
		# ###################################### Current Skill ###################################### #
		if self.act_w_expert_skill:
			curr_skill = tf.numpy_function(self.expert.sample_curr_skill, [state[0], env_goal[0], prev_skill[0]],
										   tf.float32)
			curr_skill = tf.expand_dims(curr_skill, axis=0)
		else:
			# Get the director corresponding to the previous skill and obtain the current skill
			_, curr_skill, _ = self.skilled_actors.call_director(tf.concat([state, curr_goal], axis=1), prev_skill)
		
		# ########################################## Action ######################################### #
		# Explore
		if tf.random.uniform(()) < epsilon:
			action = tf.random.uniform((1, self.args.a_dim), -self.args.action_max, self.args.action_max)
		
		# Exploit
		else:
			
			eff_state, eff_curr_goal, eff_curr_skill = self.wrap_state_and_goal(state, curr_goal, curr_skill)
			
			# Get the actor corresponding to the current skill and obtain the action
			action_mu, _, _ = self.skilled_actors.call_actor(tf.concat([eff_state, eff_curr_goal], axis=1), eff_curr_skill)
			# Add noise to action
			action_dev = tf.random.normal(tf.shape(action_mu), mean=0.0, stddev=stddev)
			action = action_mu + action_dev
			
			if self.act_w_expert_action:
				action = tf.numpy_function(
					self.expert.sample_action,
					[state[0], env_goal[0], curr_skill[0], action[0]],
					tf.float32
				)
				action = tf.expand_dims(action, axis=0)
			
			action = tf.clip_by_value(action, -self.args.action_max, self.args.action_max)
		
		# Safety check for action, should not be nan or inf
		has_nan = tf.math.reduce_any(tf.math.is_nan(action))
		has_inf = tf.math.reduce_any(tf.math.is_inf(action))
		if has_nan or has_inf:
			tf.print('Action has nan or inf. Setting action to zero. Action: {}'.format(action))
			action = tf.zeros_like(action)
		
		return curr_goal, curr_skill, action
	
	def wrap_state_and_goal(self, state, goal, curr_skill):
		_id = tf.argmax(curr_skill, axis=-1) // self.args.c_eff_dim  # Get the object id
		_id = tf.cast(_id, tf.int32)
		_id = tf.squeeze(_id)
		
		# Wrap state: [gripper_x_y_z, obj0_x_y_z, ..., objN_x_y_z, obj0_rel_x_y_z, ..., objN_rel_x_y_z, gripper_state]
		# 				-> [gripper_x_y_z, obj_id_x_y_z, obj_id_rel_x_y_z, gripper_state]
		# Wrap goal: [obj0_x_y_z, ..., objN_x_y_z]
		# 				-> [obj_id_x_y_z]
		# Wrap curr_skill: [0, ..., 3*N]
		# 				-> [0, ..., N]
		state = tf.concat([state[:, :3],
						   state[:, 3 + _id * 3: 3 + (_id + 1) * 3],
						   state[:, 3 + 3 * self.args.num_objs + _id * 3:
									3 + 3 * self.args.num_objs + (_id + 1) * 3],
						   state[:, -1:]],
						  axis=1)
		goal = goal[:, _id * 3:
					   (_id + 1) * 3]
		curr_skill = tf.argmax(curr_skill, axis=-1) % self.args.c_eff_dim
		curr_skill = tf.one_hot(curr_skill, self.args.c_eff_dim, dtype=tf.float32)
		
		return state, goal, curr_skill
	
	def get_init_skill(self):
		"""
		One-Obj: Pick Object 0 i.e. [1, 0, 0]
		Two-Obj: Pick Object 0 i.e. [1, 0, 0, 0, 0, 0] if expert_behaviour = 0
		Two-Obj: Pick Object 0 i.e. [1, 0, 0, 0, 0, 0] or [0, 0, 0, 1, 0, 0] if expert_behaviour = 1
		Three-Obj with stacking: Pick Object 0 i.e. [1, 0, 0, 0, 0, 0, 0, 0, 0]
		"""
		if self.args.num_objs == 1:
			skill = tf.one_hot(0, self.args.c_dim, dtype=tf.float32)
		
		elif self.args.num_objs == 2:
			if self.args.expert_behaviour == '0':
				skill = tf.one_hot(0, self.args.c_dim, dtype=tf.float32)  # Pick Object 0 first (default)
			else:
				raise ValueError("Invalid expert behaviour to determine init skill in two-object environment: " + str(
					self.args.expert_behaviour))
		
		elif self.args.num_objs == 3:
			# For three object-stacking, the order is fixed by the environment i.e. 0, 1, 2
			skill = tf.one_hot(0, self.args.c_dim, dtype=tf.float32)
		
		else:
			raise ValueError("Invalid number of objects: " + str(self.args.num_objs))
		
		skill = tf.reshape(skill, shape=(1, -1))
		return skill
	
	@staticmethod
	def get_init_goal(init_state, g_env):
		return g_env
	
	def load_(self, one_obj_dir, multi_obj_dir):
		
		# Load Directors of multi-object environment
		for i in range(len(self.skilled_actors.directors)):
			self.skilled_actors.directors[i].load_weights(multi_obj_dir + "/director_" + str(i) + ".h5")
		
		# Load Actors of one-object environment
		for i in range(len(self.skilled_actors.actors)):
			self.skilled_actors.actors[i].load_weights(one_obj_dir + "/policy_" + str(i) + ".h5")
	
	def build_model(self):
		
		# Define the Multi-Object Directors (for each prev skill, there is a director to determine the current skill)
		for director in self.skilled_actors.directors:
			_ = director(tf.concat([tf.ones([1, self.args.s_dim]), tf.ones([1, self.args.g_dim])], 1))
		
		# Define the One-Object Actors (for each current skill, there is an actor to determine the action)
		for actor in self.skilled_actors.actors:
			_ = actor(tf.concat([tf.ones([1, self.args.s_eff_dim]), tf.ones([1, self.args.g_eff_dim])], 1))
			
	def change_training_mode(self, training_mode: bool):
		self.skilled_actors.change_training_mode(training_mode)


class Agent(AgentBase):
	def __init__(self, args,
				 expert_buffer: ReplayBufferTf = None,
				 offline_buffer: ReplayBufferTf = None):
		super().__init__(args, skilledDemoDICE(args), 'skilledDemoDICE', expert_buffer, offline_buffer)


def main():
	tf.config.run_functions_eagerly(True)
	
	log_dir = os.path.join('./logging', '0Shot')
	if not os.path.exists(log_dir):
		os.makedirs(log_dir, exist_ok=True)
	
	logging.basicConfig(filename=os.path.join(log_dir, 'logs.txt'), filemode='w',
						format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S',
						level=logging.INFO)
	logger = logging.getLogger(__name__)
	
	logger.info("# ################# Verifying ################# #")
	
	args = get_DICE_args(log_dir, debug=True)
	args.algo = 'SkilledDemoDICE'
	args.log_dir = log_dir
	args.log_wandb = False
	args.visualise_test = True
	
	logger.info("---------------------------------------------------------------------------------------------")
	config: dict = vars(args)
	config = {key: str(value) for key, value in config.items()}
	config = OrderedDict(sorted(config.items()))
	logger.info(json.dumps(config, indent=4))
	
	# List Model Directories
	one_obj_model_dirs = []
	for root, dirs, files in os.walk('./logging/offlineILPnPOneExp/SkilledDemoDICE_full'):
		for name in dirs:
			if 'run' in name:
				one_obj_model_dirs.append(os.path.join(root, name, 'models'))
	
	two_obj_model_dirs = []
	for root, dirs, files in os.walk('./logging/offlineILPnPTwoExp/SkilledDemoDICE_full_6'):
		for name in dirs:
			if 'run' in name:
				two_obj_model_dirs.append(os.path.join(root, name, 'models'))
	
	# Zip the model directories
	model_dirs = list(zip(one_obj_model_dirs, two_obj_model_dirs))
	for one_obj_dir, two_obj_dir in model_dirs:
		logger.info("---------------------------------------------------------------------------------------------")
		logger.info("One-Obj Model Dir: " + one_obj_dir)
		logger.info("Two-Obj Model Dir: " + two_obj_dir)
		agent = Agent(args)
		agent.model.load_(one_obj_dir, two_obj_dir)
		
		agent.visualise(
			use_expert_skill=True,
			use_expert_action=False,
			resume_states=None,
			num_episodes=1,
		)
		


if __name__ == "__main__":
	main()
