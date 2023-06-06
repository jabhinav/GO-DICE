import datetime
import logging
import os
import sys
import time
from abc import ABC
from typing import Dict, Union, Optional

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from argparse import Namespace

from domains.PnP import MyPnPEnvWrapperForGoalGAIL
from domains.PnPExpert import PnPExpert, PnPExpertTwoObj
from her.replay_buffer import ReplayBufferTf
from her.transitions import sample_goal_oriented_transitions
from networks.general import Policy, SGCEncoder, Discriminator
from utils.buffer import get_buffer_shape
from utils.env import get_PnP_env
from utils.sample import gumbel_softmax_tf

logger = logging.getLogger(__name__)


class optionBC(tf.keras.Model, ABC):
	def __init__(self, args: Namespace):
		super(optionBC, self).__init__()
		self.args = args
		
		# Declare Decoder Network: num_options number of policies
		for i in range(args.num_options):
			setattr(self, "policy_{}".format(i), Policy(args.a_dim, args.action_max))
			setattr(self, "optimiser_{}".format(i), tf.keras.optimizers.Adam(self.args.vae_lr))
		# Declare Encoder Network: Sill and Goal Predictor
		self.goal_skill_pred = SGCEncoder(args.ag_dim, args.c_dim)
		# Declare Discriminator Network: Goal Discriminator
		self.goal_disc = Discriminator()
		
		# Declare Optimisers
		self.optim_ae = tf.keras.optimizers.Adam(self.args.vae_lr)
		self.optim_disc = tf.keras.optimizers.Adam(self.args.vae_lr)
		self.optim_gen = tf.keras.optimizers.Adam(self.args.vae_lr)
		
		# Declare Losses
		self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
		
		# Build Model
		self.build_model()
		
		# Get the expert policy [HACK: This is only for the validating gBC without latent mode prediction]
		exp_env = get_PnP_env(args)
		latent_dim = exp_env.latent_dim
		if args.two_object:
			self.expert_policy = PnPExpertTwoObj(latent_dim, args.full_space_as_goal,
												 expert_behaviour=args.expert_behaviour)
		else:
			self.expert_policy = PnPExpert(latent_dim, args.full_space_as_goal)
		
		# Create a TF Variable to store the gumbel softmax temperature
		self.temp = tf.Variable(self.args.temp_min, trainable=False, dtype=tf.float32)
		
		self.pick_gt_c = False
		self.curr_c = None
		self.curr_c_idx = None
	
	def set_temp(self, temp):
		self.temp.assign(temp)
	
	@tf.function
	def compute_ae_loss(self, data):
		delta_g, c_logits = self.goal_skill_pred(data['achieved_goals'], data['states'], data['goals'])
		
		# Compute the KL divergence between c_pred and c_prior (c_prior is a uniform distribution)
		c_prob = tf.nn.softmax(c_logits, axis=-1)
		c_prior = tf.ones_like(c_prob) / self.args.num_options
		kl_div = tf.reduce_sum(c_prob * tf.math.log(1e-10 + c_prob / c_prior), axis=-1)
		kl_div = tf.reduce_mean(kl_div)
		
		# Compute the goal prediction
		g_pred = data['achieved_goals'] + delta_g
		g_pred = tf.clip_by_value(g_pred, -self.args.clip_obs, self.args.clip_obs)
		
		# Sample skill from c_pred
		c_pred = gumbel_softmax_tf(c_logits, temperature=self.temp)
		# tf.print("c_pred", tf.argmax(c_pred, axis=-1))
		# tf.print("c_prob", c_prob)
		
		# Use the predicted goal to compute the policy loss [weighing each option by the sampled skill]
		pred_actions = []
		for i in range(self.args.num_options):
			policy = getattr(self, "policy_{}".format(i))
			actions = policy(data['states'], data['goals'], g_pred)
			pred_actions.append(actions)
		
		pred_actions = tf.stack(pred_actions, axis=1)
		c_pred = tf.expand_dims(c_pred, axis=-1)
		pred_actions = tf.reduce_sum(pred_actions * c_pred, axis=1)
		policy_loss = tf.reduce_sum(tf.math.squared_difference(data['actions'], pred_actions), axis=-1)
		policy_loss = tf.reduce_mean(policy_loss)
		
		loss = policy_loss + self.args.kl_coeff * kl_div
		
		# # Use the predicted goal to compute the policy loss [taking the minimum over all options]
		# policy_losses = []
		# for i in range(self.args.num_options):
		# 	policy = getattr(self, "policy_{}".format(i))
		# 	actions = policy(data['states'], data['goals'], g_pred)
		# 	policy_loss = tf.reduce_sum(tf.math.squared_difference(data['actions'], actions), axis=-1)
		# 	policy_losses.append(policy_loss)
		#
		# # Take the minimum of all the policy losses
		# policy_losses = tf.stack(policy_losses, axis=1)
		# min_policy_loss = tf.reduce_min(policy_losses, axis=1)
		# loss = tf.reduce_mean(min_policy_loss)
		return self.args.ae_loss_weight * loss, policy_loss, kl_div
	
	@tf.function
	def compute_disc_loss(self, data):
		# Compute the goal prediction
		delta_g, c_logits = self.goal_skill_pred(data['achieved_goals'], data['states'], data['goals'])
		g_pred = data['achieved_goals'] + delta_g
		g_pred = tf.clip_by_value(g_pred, -self.args.clip_obs, self.args.clip_obs)
		
		# Compute the goal discriminator loss
		disc_real = self.goal_disc(data['inter_goals'], data['states'], data['goals'])
		disc_fake = self.goal_disc(g_pred, data['states'], data['goals'])
		disc_loss = self.cross_entropy(tf.ones_like(disc_real), disc_real) + \
					self.cross_entropy(tf.zeros_like(disc_fake), disc_fake)
		return self.args.disc_loss_weight * disc_loss
	
	@tf.function
	def compute_gen_loss(self, data):
		# Compute the goal prediction
		delta_g, c_logits = self.goal_skill_pred(data['achieved_goals'], data['states'], data['goals'])
		g_pred = data['achieved_goals'] + delta_g
		g_pred = tf.clip_by_value(g_pred, -self.args.clip_obs, self.args.clip_obs)
		
		# Compute the goal discriminator loss
		disc_fake = self.goal_disc(g_pred, data['states'], data['goals'])
		gen_loss = self.cross_entropy(tf.ones_like(disc_fake), disc_fake)
		return self.args.gen_loss_weight * gen_loss
	
	@tf.function(experimental_relax_shapes=True)
	def train_ae(self, data, do_grad_clip=False):
		with tf.GradientTape() as tape:
			loss, policy_loss, kl_div = self.compute_ae_loss(data)
		
		gradients = tape.gradient(loss, self.goal_skill_pred.trainable_variables)
		if do_grad_clip:
			# # Compute the gradient norm
			# grad_norm = tf.linalg.global_norm(gradients)
			# tf.print("\ngrad_norm: ", grad_norm)
			# Clip the gradients
			gradients, _ = tf.clip_by_global_norm(gradients, self.args.grad_norm_clip)
		self.optim_ae.apply_gradients(zip(gradients, self.goal_skill_pred.trainable_variables))
		return loss, policy_loss, kl_div
	
	@tf.function
	def train_disc(self, data, do_grad_clip=False):
		with tf.GradientTape() as tape:
			# Compute the loss
			loss = self.compute_disc_loss(data)
		
		gradients = tape.gradient(loss, self.goal_disc.trainable_variables)
		if do_grad_clip:
			# # Compute the gradient norm
			# grad_norm = tf.linalg.global_norm(gradients)
			# tf.print("\ngrad_norm: ", grad_norm)
			# Clip the gradients
			gradients, _ = tf.clip_by_global_norm(gradients, self.args.grad_norm_clip)
		self.optim_disc.apply_gradients(zip(gradients, self.goal_disc.trainable_variables))
		return loss
	
	@tf.function
	def train_gen(self, data, do_grad_clip=False):
		with tf.GradientTape() as tape:
			# Compute the loss
			loss = self.compute_gen_loss(data)
		
		# TODO: Get the trainable variables only for the goal prediction branch of the goal_skill_pred network
		gradients = tape.gradient(loss, self.goal_skill_pred.trainable_variables)
		if do_grad_clip:
			# # Compute the gradient norm
			# grad_norm = tf.linalg.global_norm(gradients)
			# tf.print("\ngrad_norm: ", grad_norm)
			# Clip the gradients
			gradients, _ = tf.clip_by_global_norm(gradients, self.args.grad_norm_clip)
		self.optim_gen.apply_gradients(zip(gradients, self.goal_skill_pred.trainable_variables))
		return loss
	
	def compute_policy_loss(self, data: dict, option: int):
		# Compute the policy loss
		policy = getattr(self, "policy_{}".format(option))
		actions_mu = policy(data['states'], data['goals'], data['inter_goals'])
		loss_p = tf.reduce_sum(tf.math.squared_difference(data['actions'], actions_mu), axis=-1)
		loss_p = tf.reduce_mean(loss_p)
		return loss_p
	
	@tf.function(experimental_relax_shapes=True)
	def pre_train_policy(self, options_data):
		options_loss = {}
		for option in range(self.args.num_options):
			policy = getattr(self, "policy_{}".format(option))
			optimiser = getattr(self, "optimiser_{}".format(option))
			data = options_data[option]
			with tf.GradientTape() as tape:
				loss = self.compute_policy_loss(data, option)
			gradients = tape.gradient(loss, policy.trainable_variables)
			optimiser.apply_gradients(zip(gradients, policy.trainable_variables))
			options_loss['LossOption_' + str(option)] = loss
		return options_loss
	
	def build_model(self):
		for i in range(self.args.num_options):
			_ = getattr(self, "policy_{}".format(i))(
				np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim]), np.ones([1, self.args.ag_dim])
			)
		_, _ = self.goal_skill_pred(
			np.ones([1, self.args.ag_dim]), np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim])
		)
		
		_ = self.goal_disc(np.ones([1, self.args.ag_dim]), np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim]))
	
	def act(self, state, achieved_goal, goal, compute_Q=False, compute_c=False, **kwargs):
		state = tf.clip_by_value(state, -self.args.clip_obs, self.args.clip_obs)
		achieved_goal = tf.clip_by_value(achieved_goal, -self.args.clip_obs, self.args.clip_obs)
		goal = tf.clip_by_value(goal, -self.args.clip_obs, self.args.clip_obs)
		
		delta_g, c_logits = self.goal_skill_pred(achieved_goal, state, goal)
		g_pred = achieved_goal + delta_g
		g_pred = tf.clip_by_value(g_pred, -self.args.clip_obs, self.args.clip_obs)
		
		c_prob = tf.nn.softmax(c_logits)
		print("c_prob: {}".format(c_prob.numpy()[0]))
		c_pred = tf.argmax(c_prob, axis=-1)
		c = c_pred.numpy().item()
		
		# If Expert c are provide, then use them [Just for Testing Option Policies]
		if self.pick_gt_c and self.curr_c is not None and self.curr_c_idx is not None:
			c = self.curr_c.__getitem__(self.curr_c_idx)
			self.curr_c_idx += 1
		
		policy = getattr(self, "policy_{}".format(c))
		action_mu = policy(state, goal, g_pred)
		action = tf.clip_by_value(action_mu, -self.args.action_max, self.args.action_max)
		
		if compute_Q and compute_c:
			return action, delta_g, tf.one_hot(c_pred, self.args.num_options)
		elif compute_Q:
			return action, delta_g
		elif compute_c:
			return action, tf.one_hot(c_pred, self.args.num_options)
		else:
			return action
	
	def save_policy(self, dir_param, option):
		policy = getattr(self, "policy_{}".format(option))
		policy.save_weights(os.path.join(dir_param, 'policyOptionBC_{}.h5'.format(option)), overwrite=True)
	
	def load_policies(self, dir_param):
		for i in range(self.args.num_options):
			logger.info("Loading policy_{}".format(i))
			policy_model_path = os.path.join(dir_param, "policyOptionBC_{}.h5".format(i))
			if not os.path.exists(policy_model_path):
				logger.info("Options Policy Weights Not Found at {}. Exiting!".format(policy_model_path))
				sys.exit(-1)
			policy = getattr(self, "policy_{}".format(i))
			policy.load_weights(policy_model_path)
			logger.info("Policy Weights Loaded from {}".format(policy_model_path))
	
	def save_goalPred(self, dir_param):
		self.goal_skill_pred.save_weights(os.path.join(dir_param, 'goalPredOptionBC.h5'), overwrite=True)
	
	def load_goalPred(self, dir_param):
		goal_pred_model_path = os.path.join(dir_param, "goalPredOptionBC.h5")
		if not os.path.exists(goal_pred_model_path):
			logger.info("Goal Predictor Weights Not Found at {}. Exiting!".format(goal_pred_model_path))
			sys.exit(-1)
		self.goal_skill_pred.load_weights(goal_pred_model_path)
		logger.info("Loaded Goal Predictor Weights from {}".format(goal_pred_model_path))
	
	def save_goalDisc(self, dir_param):
		self.goal_disc.save_weights(os.path.join(dir_param, 'goalDiscOptionBC.h5'), overwrite=True)
	
	def load_goalDisc(self, dir_param):
		goal_disc_model_path = os.path.join(dir_param, "goalDiscOptionBC.h5")
		if not os.path.exists(goal_disc_model_path):
			logger.info("Goal Discriminator Weights Not Found at {}. Exiting!".format(goal_disc_model_path))
			sys.exit(-1)
		self.goal_disc.load_weights(goal_disc_model_path)
		logger.info("Loaded Goal Discriminator Weights from {}".format(goal_disc_model_path))


class Agent(object):
	def __init__(self, args, expert_buffer: ReplayBufferTf, val_buffer: ReplayBufferTf):
		
		self.args = args
		
		# Define the Buffers
		self.expert_buffer = expert_buffer
		self.val_buffer = val_buffer
		
		# Define Tensorboard for logging Losses and Other Metrics
		if not os.path.exists(args.dir_summary):
			os.makedirs(args.dir_summary)
		
		if not os.path.exists(args.dir_plot):
			os.makedirs(args.dir_plot)
		self.summary_writer = tf.summary.create_file_writer(args.dir_summary)
		
		# Declare Model
		self.model = optionBC(args)
		
		# Evaluation
		self.eval_env: MyPnPEnvWrapperForGoalGAIL = get_PnP_env(args)
	
	def preprocess_og(self, states, achieved_goals, goals):
		states = tf.clip_by_value(states, -self.args.clip_obs, self.args.clip_obs)
		achieved_goals = tf.clip_by_value(achieved_goals, -self.args.clip_obs, self.args.clip_obs)
		goals = tf.clip_by_value(goals, -self.args.clip_obs, self.args.clip_obs)
		return states, achieved_goals, goals
	
	def save_option(self, dir_param, pretrain: Optional[int] = None):
		if not os.path.exists(dir_param):
			os.makedirs(dir_param)
		
		# Save the option policy if pretrain is not None
		if pretrain is not None:
			self.model.save_policy(dir_param, option=pretrain)
	
	def load_options(self, dir_param):
		self.model.load_policies(dir_param)
	
	def save_model(self, dir_param):
		if not os.path.exists(dir_param):
			os.makedirs(dir_param)
		self.model.save_goalPred(dir_param)
	
	def load_model(self, dir_param):
		self.model.load_goalPred(dir_param)
	
	@tf.function
	def process_data(self, transitions, expert=False, is_supervised=False):
		
		trans = transitions.copy()
		s, ag, g = trans['states'], trans['achieved_goals'], trans['goals']
		
		# Process the states and goals
		s, ag, g = self.preprocess_og(s, ag, g)
		
		trans['states'] = s
		trans['achieved_goals'] = ag
		trans['goals'] = g
		
		if 'inter_goals' in trans:
			iag = trans['inter_goals']
			_, iag, _ = self.preprocess_og(s, iag, g)
			trans['inter_goals'] = iag
		
		# Define if the transitions are from expert or not/are supervised or not
		trans['is_demo'] = tf.cast(expert, dtype=tf.int32) * tf.ones_like(trans['successes'], dtype=tf.int32)
		trans['is_sup'] = tf.cast(is_supervised, dtype=tf.int32) * tf.ones_like(trans['successes'], dtype=tf.int32)
		
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
			for option in transitions:
				transitions[option] = self.process_data(
					transitions[option], tf.constant(True, dtype=tf.bool), tf.constant(True, dtype=tf.bool)
				)
		elif isinstance(transitions, dict):
			transitions = self.process_data(
				transitions, tf.constant(True, dtype=tf.bool), tf.constant(True, dtype=tf.bool)
			)
		else:
			raise ValueError("Invalid type of transitions")
		
		return transitions
	
	def learn_options(self):
		args = self.args
		global_step = 0
		monitor_loss = {}
		for option in range(args.num_options):
			monitor_loss[option] = np.inf
		
		with tqdm(total=args.num_epochs, position=0, leave=True, desc='Training: ') as pbar:
			for epoch in range(args.num_epochs):
				
				# ##################################################################### #
				# ############################### Train ############################### #
				# ##################################################################### #
				
				for _ in range(args.n_batches):
					data = self.sample_data(self.expert_buffer,
											batch_size=tf.constant(args.expert_batch_size, dtype=tf.int32))
					
					options_loss = self.model.pre_train_policy(data)
					
					# Write to Tensorboard
					with self.summary_writer.as_default():
						for option in range(args.num_options):
							tf.summary.scalar(
								'Loss/Train/Policy/Option_' + str(option),
								options_loss['LossOption_' + str(option)],
								step=global_step
							)
					
					options_loss = {
						key: value.numpy() if isinstance(value, tf.Tensor) else value
						for key, value in options_loss.items()
					}
					
					# Set pbar postfix with all the option losses
					pbar.set_postfix(options_loss, refresh=True)
					global_step += 1
				
				pbar.update(1)
				
				# ##################################################################### #
				# ################################ Val ################################ #
				# ##################################################################### #
				# Do Evaluation
				val_data = self.sample_data(
					self.val_buffer, batch_size=tf.constant(args.val_batch_size, dtype=tf.int32)
				)
				val_options_loss = {}
				for option in range(self.args.num_options):
					data = val_data[option]
					val_options_loss['LossOption_' + str(option)] = self.model.compute_policy_loss(data, option)
				
				# Log Validation Losses
				with self.summary_writer.as_default():
					for option in range(args.num_options):
						tf.summary.scalar(
							'Loss/Val/Policy/Option_' + str(option),
							val_options_loss['LossOption_' + str(option)],
							step=epoch
						)
				
				val_options_loss = {
					key: value.numpy() if isinstance(value, tf.Tensor) else value
					for key, value in val_options_loss.items()
				}
				
				# Save Best (post-epoch)
				for option in range(args.num_options):
					if val_options_loss['LossOption_' + str(option)] < monitor_loss[option]:
						monitor_loss[option] = val_options_loss['LossOption_' + str(option)]
						logger.info(
							"[POLICY_{}] Saving the best model (best loss: {}) "
							"after epoch: {}".format(option, monitor_loss[option], epoch)
						)
						self.save_option(args.dir_param + '_bestPolicy', pretrain=option)
				
				# Save Last (post-epoch)
				for option in range(args.num_options):
					self.save_option(args.dir_param + '_lastPolicy', pretrain=option)
	
	def learn(self):
		args = self.args
		global_step = 0
		monitor_metric = np.inf
		val_data = None
		with tqdm(total=args.num_epochs, position=0, leave=True, desc='Training: ') as pbar:
			for epoch in range(args.num_epochs):
				
				# ##################################################################### #
				# ############################### Train ############################### #
				# ##################################################################### #
				
				# Decay the Gumbel softmax temperature from self.args.temp_max to self.args.temp_min
				temp = self.args.temp_min + (self.args.temp_max - self.args.temp_min) * np.exp(
					-1. * global_step * self.args.temp_decay)
				self.model.set_temp(temp)
				
				for _ in range(args.n_batches):
					data = self.sample_data(self.expert_buffer,
											batch_size=tf.constant(args.expert_batch_size, dtype=tf.int32))
					loss, policy_loss, kl_div = self.model.train_ae(data)
					
					# Adversarial Training
					if self.args.do_adversarial_train:
						# Train Discriminator
						d_loss = self.model.train_disc(data)
						
						# Train Generator
						g_loss = self.model.train_gen(data)
					
					# Write to Tensorboard
					with self.summary_writer.as_default():
						tf.summary.scalar('Loss/Train/AE/Total', loss, step=global_step)
						tf.summary.scalar('Loss/Train/AE/Policy', policy_loss, step=global_step)
						tf.summary.scalar('Loss/Train/AE/KL_Div', kl_div, step=global_step)
						tf.summary.scalar('GumbelControl/Temperature', temp, step=global_step)
						
						if self.args.do_adversarial_train:
							tf.summary.scalar('Loss/Train/Discriminator', d_loss, step=global_step)
							tf.summary.scalar('Loss/Train/Generator', g_loss, step=global_step)
					
					loss = loss.numpy() if isinstance(loss, tf.Tensor) else loss
					policy_loss = policy_loss.numpy() if isinstance(policy_loss, tf.Tensor) else policy_loss
					kl_div = kl_div.numpy() if isinstance(kl_div, tf.Tensor) else kl_div
					if self.args.do_adversarial_train:
						d_loss = d_loss.numpy() if isinstance(d_loss, tf.Tensor) else d_loss
						g_loss = g_loss.numpy() if isinstance(g_loss, tf.Tensor) else g_loss
					
					# Set pbar postfix
					pbar.set_postfix({
						'Loss/Total': loss, 'Loss/Policy': policy_loss, 'Loss/KL': kl_div, 'temp': temp
					}, refresh=True)
					global_step += 1
				
				pbar.update(1)
				
				# ##################################################################### #
				# ################################ Val ################################ #
				# ##################################################################### #
				if val_data is None:
					val_data = self.sample_data(self.val_buffer,
												batch_size=tf.constant(args.val_batch_size, dtype=tf.int32))
				
				# Set Gumbel softmax temperature to min before evaluation for deterministic skill selection
				self.model.set_temp(self.args.temp_min)
				
				val_loss, val_policy, val_kl = self.model.compute_ae_loss(val_data)
				
				# Log Validation Losses
				with self.summary_writer.as_default():
					tf.summary.scalar('Loss/Val/AE/Total', val_loss, step=epoch)
					tf.summary.scalar('Loss/Val/AE/Policy', val_policy, step=epoch)
					tf.summary.scalar('Loss/Val/AE/KL_Div', val_kl, step=epoch)
				
				val_loss = val_loss.numpy() if isinstance(val_loss, tf.Tensor) else val_loss
				
				# Save Best (post-epoch)
				if val_loss < monitor_metric:
					monitor_metric = val_loss
					logger.info(
						"Saving the best model (best action_loss: {}) at epoch: {}".format(monitor_metric, epoch)
					)
					self.save_model(args.dir_param + '_bestGoal')
				
				# Save Last (post-epoch)
				self.save_model(args.dir_param + '_lastGoal')


def run(args):
	args.pretrained_dir_param = './pretrained_models/models_bestPolicy_{}'.format(f'two_obj_{args.expert_behaviour}'
																				  if args.two_object else 'one_obj')
	
	# ######################################################################################################## #
	# ############################################# DATA LOADING ############################################# #
	# ######################################################################################################## #
	# Load Buffer to store expert data
	expert_buffer = ReplayBufferTf(
		get_buffer_shape(args), args.buffer_size, args.horizon,
		sample_goal_oriented_transitions(args.train_trans_style, args.future_gamma)
	)
	
	val_buffer = ReplayBufferTf(
		get_buffer_shape(args), args.buffer_size, args.horizon,
		sample_goal_oriented_transitions(args.val_trans_style, args.future_gamma)
	)
	
	train_data_path = os.path.join(args.dir_data, '{}_train.pkl'.format(
		'two_obj_{}'.format(args.expert_behaviour) if args.two_object else 'single_obj'))
	
	if not os.path.exists(train_data_path):
		logger.error("Train data not found at {}. Please run the validation data generation script first.".format(
			train_data_path))
		sys.exit(-1)
	else:
		logger.info("Loading Expert Demos from {} into TrainBuffer for training.".format(train_data_path))
		expert_buffer.load_data_into_buffer(train_data_path)
	
	val_data_path = os.path.join(args.dir_data, '{}_val.pkl'.format(
		'two_obj_{}'.format(args.expert_behaviour) if args.two_object else 'single_obj'))
	
	if not os.path.exists(val_data_path):
		logger.error("Validation data not found at {}. Please run the validation data generation script first.".format(
			val_data_path))
		sys.exit(-1)
	elif os.path.exists(val_data_path):
		logger.info("Loading Expert Demos from {} into ValBuffer for validation.".format(val_data_path))
		val_buffer.load_data_into_buffer(val_data_path)
	
	# ########################################################################################################### #
	# ############################################# TRAINING #################################################### #
	# ########################################################################################################### #
	if args.do_pretrain:
		start = time.time()
		agent = Agent(args, expert_buffer, val_buffer)
		
		logger.info("Pre-Training .......")
		agent.learn_options()
		logger.info("Done Pre-Training in {}".format(str(datetime.timedelta(seconds=time.time() - start))))
	
	if args.do_train:
		start = time.time()
		agent = Agent(args, expert_buffer, val_buffer)
		
		# Load the option policies
		agent.load_options(args.pretrained_dir_param)
		
		logger.info("Training .......")
		agent.learn()
		logger.info("Done Training in {}".format(str(datetime.timedelta(seconds=time.time() - start))))
