import datetime
import logging
import os
import sys
import time
from abc import ABC
from argparse import Namespace
from typing import Dict, Union

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from domains.PnP import MyPnPEnvWrapper
from domains.PnPExpert import PnPExpert, PnPExpertTwoObj
from her.replay_buffer import ReplayBufferTf
from her.transitions import sample_non_her_transitions
from networks.general import Actor, GoalPredictor, SkillPredictor, SkillTerminationPredictor
from utils.buffer import get_buffer_shape
from utils.env import get_PnP_env

logger = logging.getLogger(__name__)


class GoalGuidedOptionBC(tf.keras.Model, ABC):
	def __init__(self, args: Namespace,
				 use_expert_policy: bool = False,
				 use_expert_goal: bool = False,
				 use_expert_skill: bool = False):
		super(GoalGuidedOptionBC, self).__init__()
		self.args = args
		
		# Declare Policies corresponding to each option
		for i in range(args.num_skills):
			setattr(self, "policy_{}".format(i), Actor(args.a_dim))
			setattr(self, "optimP_{}".format(i), tf.keras.optimizers.Adam(self.args.vae_lr))
		
		# Declare the goal predictor
		self.goal_pred = GoalPredictor(args.ag_dim)
		self.optimG = tf.keras.optimizers.Adam(self.args.vae_lr)
		
		# Declare Skill Terminator: for num_skills
		for i in range(args.num_skills):
			setattr(self, "skill_term_{}".format(i), SkillTerminationPredictor())
			setattr(self, "optimST_{}".format(i), tf.keras.optimizers.Adam(self.args.vae_lr))
		
		# Declare the skill predictor
		self.skill_pred = SkillPredictor(args.c_dim)
		self.optimS = tf.keras.optimizers.Adam(self.args.vae_lr)
		
		# Alternate: Predict both goal and skill simultaneously
		# # Declare Encoder Network: Sill and Goal Predictor
		# self.goal_skill_pred = SGCEncoder(args.ag_dim, args.c_dim)
		# # Declare Optimisers
		# self.optim_gc = tf.keras.optimizers.Adam(self.args.vae_lr)
		
		# Build Model
		self.build_model()
		
		# Get the expert policy [HACK: This is only for the validating gBC without latent mode prediction]
		exp_env = get_PnP_env(args)
		num_skills = exp_env.latent_dim
		if args.two_object:
			self.expert_guide = PnPExpertTwoObj(num_skills, expert_behaviour=args.expert_behaviour)
		else:
			self.expert_guide = PnPExpert(num_skills)
		self.use_expert_policy = use_expert_policy
		self.use_expert_goal = use_expert_goal
		self.use_expert_skill = use_expert_skill
		
		self.thresh_g = self.expert_guide.env_goal_thresh
		self.thresh_c = 0.5
		
		# Weight for binary cross entropy loss for skill termination.
		# `pos_weight > 1` dec. the false -ve count, inc. recall. (dec. termination classified as non-termination)
		# `pos_weight < 1` dec. the false +ve count, inc. precision. (dec. non-termination classified as termination)
		num_term_samples = 6*args.expert_demos
		num_non_term_samples = (args.horizon-6)*args.expert_demos  # >>> num_term_samples
		weight_st = num_non_term_samples / num_term_samples
		self.weight_st = tf.constant(weight_st, dtype=tf.float32)  # will be multiplied against +ve samples
		
	# 	# Create a TF Variable to store the gumbel softmax temperature
	# 	self.temp = tf.Variable(self.args.temp_min, trainable=False, dtype=tf.float32)
	#
	# def set_temp(self, temp):
	# 	self.temp.assign(temp)
	
	def compute_loss_skill_policy(self, data: dict, skill: int):
		# Compute the policy loss
		policy = getattr(self, "policy_{}".format(skill))
		actions_mu = policy(data['states'], data['env_goals'], data['curr_goals'])
		loss_p = tf.reduce_sum(tf.math.squared_difference(data['actions'], actions_mu), axis=-1)
		loss = tf.reduce_mean(loss_p)
		return loss
	
	def compute_loss_curr_goal_pred(self, data: dict):
		delta_g = self.goal_pred(data['prev_goals'], data['states'], data['env_goals'])
		g_pred = data['prev_goals'] + delta_g
		g_pred = tf.clip_by_value(g_pred, -self.args.clip_obs, self.args.clip_obs)
		loss_g = tf.reduce_sum(tf.math.squared_difference(data['curr_goals'], g_pred), axis=-1)
		loss = tf.reduce_mean(loss_g)
		return loss
	
	def compute_loss_prev_skill_termination(self, data: dict, skill: int):
		# Compute the skill termination loss [discrete]
		skill_term_net = getattr(self, "skill_term_{}".format(skill))
		skill_term_logit = skill_term_net(data['prev_goals'], data['states'], data['env_goals'], data['curr_goals'])
		# Compute cross entropy with probability of termination
		loss_st = tf.nn.weighted_cross_entropy_with_logits(labels=data['terminate_skills'], logits=skill_term_logit,
														   pos_weight=self.weight_st)
		loss = tf.reduce_mean(loss_st)
		return loss
	
	def compute_loss_curr_skill_pred(self, data: dict):
		# Compute the skill prediction loss [discrete]
		c_logits = self.skill_pred(data['prev_skills'], data['states'], data['env_goals'], data['curr_goals'])
		loss_c = tf.nn.softmax_cross_entropy_with_logits(labels=data['curr_skills'], logits=c_logits)
		loss = tf.reduce_mean(loss_c)
		return loss
	
	@tf.function(experimental_relax_shapes=True)
	def train_skill_policy(self, data: dict, do_teacher_forcing: bool = True):
		loss_skills = {}
		for skill in range(self.args.num_skills):
			
			# If not teacher forcing, then use the predicted goal and stop the gradient
			if not do_teacher_forcing:
				data = data.copy()
				# Compute the predicted goal
				delta_g = self.goal_pred(data[skill]['prev_goals'], data[skill]['states'], data[skill]['env_goals'])
				g_pred = data['prev_goals'] + delta_g
				g_pred = tf.clip_by_value(g_pred, -self.args.clip_obs, self.args.clip_obs)
				data[skill]['curr_goals'] = tf.stop_gradient(g_pred)
			
			policy = getattr(self, "policy_{}".format(skill))
			optimiser = getattr(self, "optimP_{}".format(skill))
			with tf.GradientTape() as tape:
				loss = self.compute_loss_skill_policy(data[skill], skill)
			grads = tape.gradient(loss, policy.trainable_variables)
			optimiser.apply_gradients(zip(grads, policy.trainable_variables))
			loss_skills[f'{skill}'] = loss
		return loss_skills
	
	@tf.function(experimental_relax_shapes=True)
	def train_curr_goal_pred(self, data: dict):
		with tf.GradientTape() as tape:
			loss = self.compute_loss_curr_goal_pred(data)
		grads = tape.gradient(loss, self.goal_pred.trainable_variables)
		self.optimG.apply_gradients(zip(grads, self.goal_pred.trainable_variables))
		return loss
	
	@tf.function(experimental_relax_shapes=True)
	def train_prev_skill_termination(self, data: dict, do_teacher_forcing: bool = True):
		loss_skills = {}
		for skill in range(self.args.num_skills):
			
			# If not teacher forcing, then use the predicted goal and stop the gradient
			if not do_teacher_forcing:
				data = data.copy()
				# Compute the predicted goal
				delta_g = self.goal_pred(data[skill]['prev_goals'], data[skill]['states'], data[skill]['env_goals'])
				g_pred = data['prev_goals'] + delta_g
				g_pred = tf.clip_by_value(g_pred, -self.args.clip_obs, self.args.clip_obs)
				data[skill]['curr_goals'] = tf.stop_gradient(g_pred)
				
			skill_term_net = getattr(self, "skill_term_{}".format(skill))
			optimiser = getattr(self, "optimST_{}".format(skill))
			with tf.GradientTape() as tape:
				loss = self.compute_loss_prev_skill_termination(data[skill], skill)
			grads = tape.gradient(loss, skill_term_net.trainable_variables)
			optimiser.apply_gradients(zip(grads, skill_term_net.trainable_variables))
			loss_skills[f'{skill}'] = loss
		return loss_skills
	
	@tf.function(experimental_relax_shapes=True)
	def train_curr_skill_pred(self, data: dict, do_teacher_forcing: bool = True):
		
		# If not teacher forcing, then use the predicted goal to predict the current skill
		if not do_teacher_forcing:
			data = data.copy()
			# Compute the predicted goal
			delta_g = self.goal_pred(data['prev_goals'], data['states'], data['env_goals'])
			g_pred = data['prev_goals'] + delta_g
			g_pred = tf.clip_by_value(g_pred, -self.args.clip_obs, self.args.clip_obs)
			data['curr_goals'] = tf.stop_gradient(g_pred)
		
		with tf.GradientTape() as tape:
			loss = self.compute_loss_curr_skill_pred(data)
		grads = tape.gradient(loss, self.skill_pred.trainable_variables)
		self.optimS.apply_gradients(zip(grads, self.skill_pred.trainable_variables))
		return loss
		
	def act(self, state, env_goal, prev_goal, prev_skill):
		state = tf.clip_by_value(state, -self.args.clip_obs, self.args.clip_obs)
		env_goal = tf.clip_by_value(env_goal, -self.args.clip_obs, self.args.clip_obs)
		prev_goal = tf.clip_by_value(prev_goal, -self.args.clip_obs, self.args.clip_obs)
		
		
		if self.use_expert_goal:
			curr_goal = self.expert_guide.sample_curr_goal(state[0], env_goal[0], prev_goal[0], for_expert=False)
			curr_goal = tf.expand_dims(curr_goal, axis=0)
		else:
			# Predict GOAL
			# First check the termination condition for goal i.e. distance between prev_goal and gripper < eps
			# If the termination condition is met, then sample a curr_goal
			# If the termination condition is not met, then use the prev_goal as the curr_goal
			gripper_pos = state[:, :3]
			dist = tf.norm(gripper_pos - prev_goal, axis=-1)
			terminate_goal = tf.cast(dist < self.thresh_g, tf.float32)
			curr_goal = prev_goal + self.goal_pred(prev_goal, state, env_goal)
			curr_goal = tf.clip_by_value(curr_goal, -self.args.clip_obs, self.args.clip_obs)
			curr_goal = (1 - terminate_goal) * prev_goal + terminate_goal * curr_goal
			curr_goal = tf.clip_by_value(curr_goal, -self.args.clip_obs, self.args.clip_obs)
		
		if self.use_expert_skill:
			curr_skill = self.expert_guide.sample_curr_skill(prev_goal[0], prev_skill[0], state[0], env_goal[0], curr_goal[0])
			curr_skill = tf.expand_dims(curr_skill, axis=0)
		else:
			# Predict SKILL
			# Now check the termination condition for skill using prev_skill's termination network
			prev_c = tf.argmax(prev_skill, axis=-1).numpy().item()
			skill_term_net = getattr(self, "skill_term_{}".format(prev_c))
			skill_term_logit = skill_term_net(prev_goal, state, env_goal, curr_goal)
			terminate_skill = tf.cast(tf.nn.sigmoid(skill_term_logit) > self.thresh_c, tf.float32)
			curr_skill = tf.nn.softmax(self.skill_pred(prev_skill, state, env_goal, curr_goal))
			# print("c_prob: {}".format(curr_skill.numpy()[0]))
			curr_skill = tf.argmax(curr_skill, axis=-1)
			curr_skill = tf.one_hot(curr_skill, self.args.c_dim)
			curr_skill = (1 - terminate_skill) * prev_skill + terminate_skill * curr_skill
		
		# Predict ACTION
		skill_idx = tf.argmax(curr_skill, axis=-1)
		skill_idx = skill_idx.numpy().item()
		if self.use_expert_policy:
			action = self.expert_guide.sample_action(state[0], env_goal[0], curr_goal[0], curr_skill[0])
			action = tf.expand_dims(action, axis=0)
		else:
			policy = getattr(self, "policy_{}".format(skill_idx))
			action_mu = policy(state, env_goal, curr_goal)  # a_t = mu(s_t, g_t)
			action = tf.clip_by_value(action_mu, -self.args.action_max, self.args.action_max)
		
		return curr_goal, curr_skill, action
	
	def build_model(self):
		# a_t <- f_c(s_t, g_t) for each skill
		for i in range(self.args.num_skills):
			_ = getattr(self, "policy_{}".format(i))(
				np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim]), np.ones([1, self.args.ag_dim])
			)
		
		# g_t <- f(g_{t-1}, s_t)
		_ = self.goal_pred(np.ones([1, self.args.ag_dim]), np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim]))
		
		# 0/1 <- f(g_{t-1}, s_t, g_t) for each skill
		for i in range(self.args.num_skills):
			_ = getattr(self, "skill_term_{}".format(i))(
				np.ones([1, self.args.ag_dim]),
				np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim]), np.ones([1, self.args.ag_dim])
			)
		
		# c_t <- f(c_{t-1}, s_t, g_t)
		_ = self.skill_pred(
			np.ones([1, self.args.c_dim]),
			np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim]), np.ones([1, self.args.ag_dim])
		)
		
		# _, _ = self.goal_skill_pred(
		# 	np.ones([1, self.args.ag_dim]), np.ones([1, self.args.c_dim]), np.ones([1, self.args.s_dim]),
		# 	np.ones([1, self.args.g_dim])
		# )
	
	def save_policy(self, dir_param, skill):
		getattr(self, "policy_{}".format(skill)).save_weights(dir_param + "/policy_{}.h5".format(skill))

	def save_goal_pred(self, dir_param):
		self.goal_pred.save_weights(dir_param + "/goal_pred.h5")

	def save_skill_term(self, dir_param, skill):
		getattr(self, "skill_term_{}".format(skill)).save_weights(dir_param + "/skill_term_{}.h5".format(skill))
			
	def save_skill_pred(self, dir_param):
		self.skill_pred.save_weights(dir_param + "/skill_pred.h5")
		
	def load_all_skill_policy(self, dir_param):
		for i in range(self.args.num_skills):
			getattr(self, "policy_{}".format(i)).load_weights(dir_param + "/policy_{}.h5".format(i))
			
	def load_goal_pred(self, dir_param):
		self.goal_pred.load_weights(dir_param + "/goal_pred.h5")
		
	def load_all_skill_term(self, dir_param):
		for i in range(self.args.num_skills):
			getattr(self, "skill_term_{}".format(i)).load_weights(dir_param + "/skill_term_{}.h5".format(i))
	
	def load_skill_pred(self, dir_param):
		self.skill_pred.load_weights(dir_param + "/skill_pred.h5")


class Agent(object):
	def __init__(self, args,
				 expert_buffer_seg: ReplayBufferTf = None,
				 expert_buffer_unseg: ReplayBufferTf = None,
				 val_buffer_seg: ReplayBufferTf = None,
				 val_buffer_unseg: ReplayBufferTf = None,):
		
		self.args = args
		
		# Define the Buffers
		self.expert_buffer_seg = expert_buffer_seg
		self.expert_buffer_unseg = expert_buffer_unseg
		self.val_buffer_seg = val_buffer_seg
		self.val_buffer_unseg = val_buffer_unseg
		
		# Define Tensorboard for logging Losses and Other Metrics
		if not os.path.exists(args.dir_summary):
			os.makedirs(args.dir_summary)
		
		if not os.path.exists(args.dir_plot):
			os.makedirs(args.dir_plot)
		self.summary_writer = tf.summary.create_file_writer(args.dir_summary)
		
		# Declare Model
		self.model = GoalGuidedOptionBC(args)
		
		# Evaluation
		self.eval_env: MyPnPEnvWrapper = get_PnP_env(args)
	
	def preprocess_in_state_space(self, item):
		item = tf.clip_by_value(item, -self.args.clip_obs, self.args.clip_obs)
		return item
		
	def save_model(self, dir_param, model_name: str):
		if not os.path.exists(dir_param):
			os.makedirs(dir_param)
			
		if model_name.startswith("Policy"):
			self.model.save_policy(dir_param, skill=int(model_name.split("_")[-1]))
		elif model_name.startswith("Goal_Pred"):
			self.model.save_goal_pred(dir_param)
		elif model_name.startswith("Skill_Term"):
			self.model.save_skill_term(dir_param, skill=int(model_name.split("_")[-1]))
		elif model_name.startswith("Skill_Pred"):
			self.model.save_skill_pred(dir_param)
		else:
			raise ValueError("Invalid Model Name: {}".format(model_name))
			
	def load_model(self, dir_param):
		self.model.load_all_skill_policy(dir_param)
		self.model.load_goal_pred(dir_param)
		self.model.load_all_skill_term(dir_param)
		self.model.load_skill_pred(dir_param)
	
	@tf.function
	def process_data(self, transitions, expert=False, is_supervised=False):
		
		trans = transitions.copy()
		
		# Process the states and goals
		trans['states'] = self.preprocess_in_state_space(trans['states'])
		trans['states_2'] = self.preprocess_in_state_space(trans['states_2'])
		if 'init_states' in trans.keys():
			trans['init_states'] = self.preprocess_in_state_space(trans['init_states'])
		trans['env_goals'] = self.preprocess_in_state_space(trans['env_goals'])
		if 'inter_goals' in trans.keys():
			trans['inter_goals'] = self.preprocess_in_state_space(trans['inter_goals'])
		
		# Define if the transitions are from expert or not/are supervised or not
		trans['is_demo'] = tf.cast(expert, dtype=tf.int32) * tf.ones_like(trans['successes'], dtype=tf.int32)
		trans['is_sup'] = tf.cast(is_supervised, dtype=tf.int32) * tf.ones_like(trans['successes'], dtype=tf.int32)
		
		# Compute terminate skills i.e. if prev_skill != curr_skill then terminate_skill = 1 else 0
		trans['terminate_skills'] = tf.cast(tf.not_equal(tf.argmax(trans['prev_skills'], axis=-1),
														 tf.argmax(trans['curr_skills'], axis=-1)),
											dtype=tf.int32)
		# reshape the terminate_skills to be of shape (batch_size, 1)
		trans['terminate_skills'] = tf.reshape(trans['terminate_skills'], shape=(-1, 1))
		
		# Make sure the data is of type tf.float32
		for key in trans.keys():
			trans[key] = tf.cast(trans[key], dtype=tf.float32)
		
		return trans
	
	@tf.function
	def sample_data(self, buffer, batch_size):
		
		# Sample Transitions
		transitions: Union[Dict[int, dict], dict] = buffer.sample_transitions(batch_size)
		
		# Process the transitions
		if all(isinstance(v, dict) for v in transitions.values()):
			for skill in transitions.keys():
				transitions[skill] = self.process_data(
					transitions[skill], tf.constant(True, dtype=tf.bool), tf.constant(True, dtype=tf.bool)
				)
		elif isinstance(transitions, dict):
			transitions = self.process_data(
				transitions, tf.constant(True, dtype=tf.bool), tf.constant(True, dtype=tf.bool)
			)
		else:
			raise ValueError("Invalid type of transitions")
		
		return transitions
	
	@tf.function
	def train(self):
		# Get Data
		seg_data = self.sample_data(self.expert_buffer_seg,
									batch_size=tf.constant(self.args.expert_batch_size, dtype=tf.int32))
		unseg_data = self.sample_data(self.expert_buffer_unseg,
									  batch_size=tf.constant(self.args.expert_batch_size, dtype=tf.int32))
		
		train_loss_dict = {}
		
		# Randomly decide if to use teacher forcing or not
		teacher_forcing = tf.random.uniform(shape=()) < self.args.teacher_forcing_ratio
		
		# Train Goal Prediction
		loss = self.model.train_curr_goal_pred(unseg_data)
		train_loss_dict['Loss/Train/Goal_Pred'] = loss
		
		# Train Skill Termination
		loss: dict = self.model.train_prev_skill_termination(seg_data)
		for skill in range(self.args.num_skills):
			train_loss_dict[f'Loss/Train/Skill_Term_{skill}'] = loss[f'{skill}']
		
		# Train Skill Prediction
		# if not teacher_forcing:
		# 	tf.print("Not Teacher Forcing Skill Prediction, will use predicted goal")
		loss = self.model.train_curr_skill_pred(unseg_data, do_teacher_forcing=teacher_forcing)
		train_loss_dict['Loss/Train/Skill_Pred'] = loss
		
		# Train Skill Policies
		loss: dict = self.model.train_skill_policy(seg_data)
		for skill in range(self.args.num_skills):
			train_loss_dict[f'Loss/Train/Policy_{skill}'] = loss[f'{skill}']
			
		return train_loss_dict
	
	def learn(self):
		args = self.args
		global_step = 0

		# Define the metrics to monitor
		monitor_metrics = {}
		for skill in range(args.num_skills):
			monitor_metrics[f'Policy_{skill}'] = np.inf
			monitor_metrics[f'Skill_Term_{skill}'] = np.inf
		monitor_metrics['Goal_Pred'] = np.inf
		monitor_metrics['Skill_Pred'] = np.inf
		
		# Pre-load validation data
		val_unseg_data = self.sample_data(self.val_buffer_unseg,
										  batch_size=tf.constant(args.val_demos, dtype=tf.int32))
		
		val_seg_data = self.sample_data(self.val_buffer_seg,
										batch_size=tf.constant(args.val_demos, dtype=tf.int32))
		with tqdm(total=args.num_epochs, position=0, leave=True, desc='Training: ') as pbar:
			for epoch in range(args.num_epochs):
				
				# ##################################################################### #
				# ############################### Train ############################### #
				# ##################################################################### #
				
				# # Decay the Gumbel softmax temperature from self.args.temp_max to self.args.temp_min
				# temp = self.args.temp_min + (self.args.temp_max - self.args.temp_min) * np.exp(
				# 	-1. * global_step * self.args.temp_decay)
				# self.model.set_temp(temp)
				
				for _ in range(args.n_batches):
						
					train_loss_dict = self.train()
					
					with self.summary_writer.as_default():
						for key, value in train_loss_dict.items():
							tf.summary.scalar(key, value, step=global_step)
						
					# for key, value in train_loss_dict.items():
					# 	train_loss_dict[key] = value.numpy() if isinstance(value, tf.Tensor) else value
					# pbar.set_postfix(train_loss_dict, refresh=True)
					
					global_step += 1
				
				pbar.update(1)
				
				# ##################################################################### #
				# ################################ Val ################################ #
				# ##################################################################### #
												
				# # Set Gumbel softmax temperature to min before evaluation for deterministic skill selection
				# self.model.set_temp(self.args.temp_min)
				
				# Evaluate the policy and compute all the losses
				val_loss_dict = {}
				for skill in range(args.num_skills):
					loss = self.model.compute_loss_skill_policy(val_seg_data[skill], skill)
					val_loss_dict[f'Loss/Val/Policy_{skill}'] = loss
					
				loss = self.model.compute_loss_curr_goal_pred(val_unseg_data)
				val_loss_dict['Loss/Val/Goal_Pred'] = loss
				
				for skill in range(args.num_skills):
					loss = self.model.compute_loss_prev_skill_termination(val_seg_data[skill], skill)
					val_loss_dict[f'Loss/Val/Skill_Term_{skill}'] = loss
					
				loss = self.model.compute_loss_curr_skill_pred(val_unseg_data)
				val_loss_dict['Loss/Val/Skill_Pred'] = loss
				
				with self.summary_writer.as_default():
					for key, value in val_loss_dict.items():
						tf.summary.scalar(key, value, step=global_step)
				
				for key, value in val_loss_dict.items():
					val_loss_dict[key] = value.numpy() if isinstance(value, tf.Tensor) else value
					
				# Save the best models by comparing monitor_metrics and val_loss_dict
				for key, value in val_loss_dict.items():
					metric = key.split('/')[-1]
					if value < monitor_metrics[metric]:
						monitor_metrics[metric] = value
						logger.info(f"Saving best model for {metric} with value {value} at epoch {epoch}")
						self.save_model(args.dir_param + '_best', metric)


def run(args):
	args.pretrained_dir_param = './pretrained_models/models_bestPolicy_{}'.format(f'two_obj_{args.expert_behaviour}'
																				  if args.two_object else 'one_obj')
	
	# ######################################################################################################## #
	# ############################################# DATA LOADING ############################################# #
	# ######################################################################################################## #
	# Load Buffer to store expert data
	expert_buffer_seg = ReplayBufferTf(
		get_buffer_shape(args), args.buffer_size, args.horizon, sample_non_her_transitions('random_segmented')
	)
	expert_buffer_unseg = ReplayBufferTf(
		get_buffer_shape(args), args.buffer_size, args.horizon, sample_non_her_transitions('random_unsegmented')
	)
	val_buffer_seg = ReplayBufferTf(
		get_buffer_shape(args), args.buffer_size, args.horizon, sample_non_her_transitions('all_segmented')
	)
	val_buffer_unseg = ReplayBufferTf(
		get_buffer_shape(args), args.buffer_size, args.horizon, sample_non_her_transitions('all_unsegmented')
	)
	
	train_data_path = os.path.join(args.dir_data, '{}_train.pkl'.format(
		'two_obj_{}'.format(args.expert_behaviour) if args.two_object else 'single_obj'))
	
	if not os.path.exists(train_data_path):
		logger.error("Train data not found at {}. Please run the validation data generation script first.".format(
			train_data_path))
		sys.exit(-1)
	else:
		logger.info("Loading Expert Demos from {} into TrainBuffer for training.".format(train_data_path))
		expert_buffer_seg.load_data_into_buffer(train_data_path)
		expert_buffer_unseg.load_data_into_buffer(train_data_path)
	
	val_data_path = os.path.join(args.dir_data, '{}_val.pkl'.format(
		'two_obj_{}'.format(args.expert_behaviour) if args.two_object else 'single_obj'))
	
	if not os.path.exists(val_data_path):
		logger.error("Validation data not found at {}. Please run the validation data generation script first.".format(
			val_data_path))
		sys.exit(-1)
	elif os.path.exists(val_data_path):
		logger.info("Loading Expert Demos from {} into ValBuffer for validation.".format(val_data_path))
		val_buffer_seg.load_data_into_buffer(val_data_path)
		val_buffer_unseg.load_data_into_buffer(val_data_path)
	
	# ########################################################################################################### #
	# ############################################# TRAINING #################################################### #
	# ########################################################################################################### #
	if args.do_train:
		start = time.time()
		agent = Agent(args, expert_buffer_seg, expert_buffer_unseg, val_buffer_seg, val_buffer_unseg)
		
		logger.info("Training .......")
		agent.learn()
		logger.info("Done Training in {}".format(str(datetime.timedelta(seconds=time.time() - start))))
