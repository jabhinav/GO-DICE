import datetime
import logging
import os
import sys
import time
from abc import ABC
from argparse import Namespace
from typing import Dict, Union

import wandb
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from domains.PnP import MyPnPEnvWrapper
from domains.PnPExpert import PnPExpert, PnPExpertTwoObj
from her.replay_buffer import ReplayBufferTf
from her.transitions import sample_transitions
from her.rollout import RolloutWorker
from networks.general import Actor
from utils.buffer import get_buffer_shape
from utils.env import get_PnP_env
from utils.custom import evaluate_worker

logger = logging.getLogger(__name__)


def orthogonal_regularization(model, reg_coef=1e-4):
	"""Orthogonal regularization v2.
		See equation (3) in https://arxiv.org/abs/1809.11096.
		Rβ(W) = β∥W⊤W ⊙ (1 − I)∥2F, where ⊙ is the Hadamard product.
		Args:
		  model: A keras model to apply regularization for.
		  reg_coef: Orthogonal regularization coefficient. Don't change this value.
		Returns:
		  A regularization loss term.
	"""
	reg = 0
	for layer in model.layers:
		if isinstance(layer, tf.keras.layers.Dense):
			prod = tf.matmul(tf.transpose(layer.kernel), layer.kernel)
			reg += tf.reduce_sum(tf.math.square(prod * (1 - tf.eye(prod.shape[0]))))
	return reg * reg_coef


class BC(tf.keras.Model, ABC):
	def __init__(self, args: Namespace):
		super(BC, self).__init__()
		self.args = args
		
		# Declare Policy Network and Optimiser
		self.actor = Actor(args.a_dim)
		self.actor_optimizer = tf.keras.optimizers.Adam(1e-3)
		
		# Build Model
		self.build_model()
		
		# Get the expert policy [HACK: This is only for the validating gBC without latent mode prediction]
		exp_env = get_PnP_env(args)
		num_skills = exp_env.latent_dim
		if args.two_object:
			self.expert_guide = PnPExpertTwoObj(expert_behaviour=args.expert_behaviour, wrap_skill_id=args.wrap_skill_id)
		else:
			self.expert_guide = PnPExpert()
	
	def compute_loss_policy(self, data: dict):
		# Compute the policy loss
		actions_mu, _, _ = self.actor(tf.concat([data['states'], data['env_goals']], axis=1))
		loss_p = tf.reduce_sum(tf.math.squared_difference(data['actions'], actions_mu), axis=-1)
		loss = tf.reduce_mean(loss_p)
		loss += orthogonal_regularization(self.actor.base)
		return loss
	
	@tf.function(experimental_relax_shapes=True)
	def train_policy(self, data: dict):
		with tf.GradientTape() as tape:
			loss = self.compute_loss_policy(data)
		grads = tape.gradient(loss, self.actor.trainable_variables)
		self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
		return loss
	
	def act(self, state, env_goal, prev_goal, prev_skill, epsilon, stddev):
		state = tf.clip_by_value(state, -self.args.clip_obs, self.args.clip_obs)
		env_goal = tf.clip_by_value(env_goal, -self.args.clip_obs, self.args.clip_obs)
		prev_goal = tf.clip_by_value(prev_goal, -self.args.clip_obs, self.args.clip_obs)
		
		# Current Goal and Skill
		curr_goal = prev_goal
		curr_skill = prev_skill
		
		# # Action
		# Explore
		if np.random.rand() < epsilon:
			action = tf.random.uniform((1, self.args.a_dim), -self.args.action_max, self.args.action_max)
		# Exploit
		else:
			action_mu, _, _ = self.actor(tf.concat([state, env_goal], axis=1))  # a_t = mu(s_t, g_t)
			action_dev = tf.random.normal(action_mu.shape, mean=0.0, stddev=stddev)
			action = action_mu + action_dev
			action = tf.clip_by_value(action, -self.args.action_max, self.args.action_max)
		
		return curr_goal, curr_skill, action
	
	def build_model(self):
		# a_t <- f(s_t) for each skill
		_ = self.actor(tf.concat([np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim])], 1))

	def save_policy(self, dir_param):
		self.actor.save_weights(dir_param + "/policy.h5")
	
	def load_policy(self, dir_param):
		self.actor.load_weights(dir_param + "/policy.h5")


class Agent(object):
	def __init__(self, args,
				 expert_buffer_unseg: ReplayBufferTf = None,
				 policy_buffer_unseg: ReplayBufferTf = None,
				 log_wandb: bool = True):
		
		self.args = args
		self.log_wandb = log_wandb
		
		# Define the Buffers
		self.expert_buffer_unseg = expert_buffer_unseg
		self.policy_buffer_unseg = policy_buffer_unseg
		
		# Define Tensorboard for logging Losses and Other Metrics
		if not os.path.exists(args.dir_summary):
			os.makedirs(args.dir_summary)
		
		if not os.path.exists(args.dir_plot):
			os.makedirs(args.dir_plot)
		self.summary_writer = tf.summary.create_file_writer(args.dir_summary)
		
		# Declare Model
		self.model = BC(args)
		
		# Evaluation
		self.env: MyPnPEnvWrapper = get_PnP_env(args)
		self.eval_env: MyPnPEnvWrapper = get_PnP_env(args)
		
		# Define the Rollout Workers
		self.policy_worker = RolloutWorker(
			self.env, self.model, T=args.horizon, rollout_terminate=False, render=False,
			is_expert_worker=False
		)
		
		self.eval_worker = RolloutWorker(
			self.eval_env, self.model, T=args.horizon, rollout_terminate=True, render=False,
			is_expert_worker=False
		)
		
		self.visualise_worker = RolloutWorker(
			self.eval_env, self.model, T=args.horizon, rollout_terminate=True, render=True,
			is_expert_worker=False
		)
		
		# Define wandb logging
		if self.log_wandb:
			current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
			self.wandb_logger = wandb.init(project='DICE', config=vars(args), id='BC_{}'.format(current_time))
		
		
	def store_offline_data(self, num_traj=100):
		# Store the data from the expert
		for i in range(num_traj):
			# Randomly pick epsilon
			epsilon = np.random.uniform(0, 1)
			# Randomly pick stddev (noise) for the action
			stddev = np.random.uniform(0, 0.1)
			episode, stats = self.policy_worker.generate_rollout(epsilon=epsilon, stddev=stddev)
			tf.print(f"{i+1}/{num_traj} Episode Success: ", stats["ep_success"])
			
			self.policy_buffer_unseg.store_episode(episode)
		
		self.policy_buffer_unseg.save_buffer_data(os.path.join(self.args.dir_root_log, f'BC_{num_traj}_noisyRollouts.pkl'))
	
	def preprocess_in_state_space(self, item):
		item = tf.clip_by_value(item, -self.args.clip_obs, self.args.clip_obs)
		return item
	
	def save_model(self, dir_param):
		if not os.path.exists(dir_param):
			os.makedirs(dir_param)
		self.model.save_policy(dir_param)
		
	def load_model(self, dir_param):
		self.model.load_policy(dir_param)
	
	@tf.function
	def process_data(self, transitions, expert=False, is_supervised=False):
		
		trans = transitions.copy()
		
		# Process the states and goals
		trans['states'] = self.preprocess_in_state_space(trans['states'])
		trans['states_2'] = self.preprocess_in_state_space(trans['states_2'])
		trans['env_goals'] = self.preprocess_in_state_space(trans['env_goals'])
		
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
		avg_loss_dict = {}
		
		# Get Data
		unseg_data = self.sample_data(self.expert_buffer_unseg, self.args.batch_size)
		# Train Skill Policies
		loss = self.model.train_policy(unseg_data)
		avg_loss_dict['loss/pi'] = loss
		
		return avg_loss_dict
	
	def learn(self):
		args = self.args
		
		total_timesteps = 0
		log_step = 0
		
		avg_return, avg_time, avg_goal_dist = evaluate_worker(self.eval_worker, args.eval_demos)
		max_return = avg_return
		# Log the data
		if self.log_wandb:
			self.wandb_logger.log({
				'stats/eval_max_return': max_return,
				'stats/eval_avg_time': avg_time,
				'stats/eval_avg_goal_dist': avg_goal_dist,
			}, step=log_step)
		tf.print(f"Eval Return: {max_return}, Eval Avg Time: {avg_time}, Eval Avg Goal Dist: {avg_goal_dist}")

		with tqdm(total=args.max_time_steps, desc='') as pbar:
			
			while total_timesteps <= args.max_time_steps:
				
				# Evaluate the policy
				if log_step % args.eval_interval == 0:
					avg_return, avg_time, avg_goal_dist = evaluate_worker(self.eval_worker, args.eval_demos)
					if avg_return > max_return:
						max_return = avg_return
						self.save_model(os.path.join(args.dir_param, 'best_model'))
					# Log the data
					if self.log_wandb:
						self.wandb_logger.log({
							'stats/eval_max_return': max_return,
							'stats/eval_avg_time': avg_time,
							'stats/eval_avg_goal_dist': avg_goal_dist,
						}, step=log_step)
					tf.print(f"Eval Return: {avg_return}, Eval Avg Time: {avg_time}, Eval Avg Goal Dist: {avg_goal_dist}")
					
				total_timesteps += args.horizon
				
				train_loss_dict = self.train()
				
				if self.log_wandb:
					self.wandb_logger.log(train_loss_dict, step=log_step)
				
				# Update
				pbar.update(args.horizon)
				log_step += 1
				
		# Save the model
		self.save_model(args.dir_param)
		
		# Store BC generated data (with some noise)
		self.store_offline_data()
		
		# Set to eager mode
		tf.config.run_functions_eagerly(True)
		op = evaluate_worker(self.visualise_worker, num_episodes=args.test_demos)
		
		
def run(args):
	# For Debugging
	if args.fix_goal and args.fix_object:
		data_prefix = 'fOfG_'
	elif args.fix_goal and not args.fix_object:
		data_prefix = 'dOfG_'
	elif args.fix_object and not args.fix_goal:
		data_prefix = 'fOdG_'
	else:
		data_prefix = 'dOdG_'
	
	# ######################################################################################################## #
	# ############################################# DATA LOADING ############################################# #
	# ######################################################################################################## #
	# Load Buffer to store expert data
	expert_buffer_unseg = ReplayBufferTf(
		get_buffer_shape(args), args.buffer_size, args.horizon, sample_transitions('random_unsegmented')
	)
	
	# Keep a buffer to store BC generated data
	policy_buffer_unseg = ReplayBufferTf(
		get_buffer_shape(args), args.buffer_size, args.horizon, sample_transitions('random_unsegmented')
	)
	
	train_data_path = os.path.join(args.dir_data, '{}{}_train.pkl'.format(data_prefix,
		'two_obj_{}'.format(args.expert_behaviour) if args.two_object else 'single_obj'))
	
	if not os.path.exists(train_data_path):
		logger.error("Train data not found at {}. Please run the validation data generation script first.".format(
			train_data_path))
		sys.exit(-1)
	else:
		logger.info("Loading Expert Demos from {} into TrainBuffer for training.".format(train_data_path))
		expert_buffer_unseg.load_data_into_buffer(train_data_path, num_demos_to_load=args.expert_demos)
	
	# ########################################################################################################### #
	# ############################################# TRAINING #################################################### #
	# ########################################################################################################### #
	if args.do_train:
		start = time.time()
		agent = Agent(args, expert_buffer_unseg, policy_buffer_unseg)
		
		logger.info("Training .......")
		agent.learn()
		logger.info("Done Training in {}".format(str(datetime.timedelta(seconds=time.time() - start))))
