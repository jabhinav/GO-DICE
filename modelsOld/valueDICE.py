import datetime
import logging
import os
import sys
import time
from abc import ABC
from argparse import Namespace
from typing import Dict, Union

import numpy as np
import wandb
import tensorflow as tf
from tqdm import tqdm

from domains.PnP import MyPnPEnvWrapper
from her.replay_buffer import ReplayBufferTf
from her.transitions import sample_transitions
from her.rollout import RolloutWorker
from networks.general import preTanhActor, Critic
from utils.buffer import get_buffer_shape
from utils.env import get_PnP_env
from evaluation.eval import evaluate_worker

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


def weighted_softmax(x, weights, axis=0):
	x = x - tf.reduce_max(x, axis=axis)
	return weights * tf.exp(x) / tf.reduce_sum(weights * tf.exp(x), axis=axis, keepdims=True)


def _update_pbar_msg(args, pbar, total_timesteps):
	"""Update the progress bar with the current training phase."""
	if total_timesteps < args.start_training_timesteps:
		msg = 'Not Training'
	else:
		msg = 'Training'
	if total_timesteps < args.num_random_actions:
		msg += ' - Exploration'
	else:
		msg += ' - Exploitation'
	if pbar.desc != msg:
		pbar.set_description(msg)



class ValueDICE(tf.keras.Model, ABC):
	def __init__(self, args: Namespace):
		super(ValueDICE, self).__init__()
		self.args = args
		
		self.args.EPS = np.finfo(np.float32).eps  # Small value = 1.192e-07 to avoid division by zero in grad penalty
		
		# Define Networks
		self.actor = preTanhActor(args.a_dim)
		self.critic = Critic()
		
		# Define Optimizers
		self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=args.actor_lr)
		self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=args.critic_lr)
		
		self.build_model()
		
		
	@tf.function(experimental_relax_shapes=True)
	def train_policy(self, data_exp, data_rb):
		with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
			tape.watch(self.actor.variables)
			tape.watch(self.critic.variables)
			
			# Get a_0 ~ pi(s_0)
			_, act_init_exp_state, _ = self.actor(tf.concat([data_exp['states'], data_exp['env_goals']], axis=1))
			# Get a'_E ~ pi(s'_E)
			_, act_next_exp_state, _ = self.actor(tf.concat([data_exp['states_2'], data_exp['env_goals']], axis=1))
			# Get a'_R ~ pi(s'_R)
			_, act_next_rb_state, _ = self.actor(tf.concat([data_rb['states_2'], data_rb['env_goals']], axis=1))
			
			# Form (s_0, a_0)
			expert_init_inputs = tf.concat([data_exp['states'], data_exp['env_goals'], act_init_exp_state], 1)
			# Form (s_E, a_E)
			expert_inputs = tf.concat([data_exp['states'], data_exp['env_goals'], data_exp['actions']], 1)
			# Form (s'_E, a'_E)
			expert_next_inputs = tf.concat([data_exp['states_2'], data_exp['env_goals'], act_next_exp_state], 1)
			# Form (s_R, a_R)
			rb_inputs = tf.concat([data_rb['states'], data_rb['env_goals'], data_rb['actions']], 1)
			# Form (s'_R, a'_R)
			rb_next_inputs = tf.concat([data_rb['states_2'], data_rb['env_goals'], act_next_rb_state], 1)
			
			# Compute the value function
			expert_nu_0 = self.critic(expert_init_inputs)
			expert_nu = self.critic(expert_inputs)
			expert_nu_next = self.critic(expert_next_inputs)
			rb_nu = self.critic(rb_inputs)
			rb_nu_next = self.critic(rb_next_inputs)
			
			# Compute the value function difference (Bellman Error)
			expert_diff = expert_nu - self.args.discount * expert_nu_next
			rb_diff = rb_nu - self.args.discount * rb_nu_next
			
			# Linear Loss Components (on replay buffer)
			linear_loss_expert = tf.reduce_mean((1 - self.args.discount) * expert_nu_0)  # (1-gamma) * V(s_0, a_0)
			linear_loss_rb = tf.reduce_mean(rb_diff)  # V(s_R, a_R) - gamma * V(s'_R, a'_R)
			linear_loss = (1 - self.args.replay_regularization) * linear_loss_expert \
						  + self.args.replay_regularization * linear_loss_rb  # (1-alpha) * linear_loss_expert + alpha * linear_loss_rb
			
			# Log Loss Components (on expert data)
			expert_rb_diff = tf.concat([expert_diff, rb_diff], 0)
			expert_rb_weights = tf.concat([
				tf.ones(expert_diff.shape) * (1 - self.args.replay_regularization),
				tf.ones(rb_diff.shape) * self.args.replay_regularization
			], 0)
			expert_rb_weights /= tf.reduce_sum(expert_rb_weights)  # Do alpha/B and (1-alpha)/B
			non_linear_loss = tf.reduce_sum(
				tf.stop_gradient(weighted_softmax(expert_rb_diff, expert_rb_weights, axis=0))  # No grads on weights
				* expert_rb_diff
			)  # This is equivalent to the log loss
			
			loss = non_linear_loss - linear_loss
			
			# Compute gradient penalty for nu
			alpha = tf.random.uniform(shape=(expert_inputs.shape[0], 1))
			nu_inter = alpha * expert_inputs + (1 - alpha) * rb_inputs
			nu_next_inter = alpha * expert_next_inputs + (1 - alpha) * rb_next_inputs
			nu_input = tf.concat([nu_inter, nu_next_inter], 0)
			with tf.GradientTape(watch_accessed_variables=False) as tape2:
				tape2.watch(nu_input)
				nu_output = self.critic(nu_input)
			nu_grad = tape2.gradient(nu_output, [nu_input])[0] + self.args.EPS
			nu_grad_penalty = tf.reduce_mean(tf.square(tf.norm(nu_grad, axis=-1, keepdims=True) - 1))
			
			# Compute Regularization for pi
			reg = orthogonal_regularization(self.actor.base)
			
			# Compute the losses (max_pi min_nu loss)
			nu_loss = loss + nu_grad_penalty * self.args.nu_grad_penalty_coeff
			pi_loss = -loss + reg
		
		nu_grads = tape.gradient(nu_loss, self.critic.variables)
		pi_grads = tape.gradient(pi_loss, self.actor.variables)
		
		self.critic_optimizer.apply_gradients(zip(nu_grads, self.critic.variables))
		self.actor_optimizer.apply_gradients(zip(pi_grads, self.actor.variables))
		
		return {
			'loss/log': non_linear_loss,
			'loss/linear': linear_loss,
			'loss/log-linear': loss,
			'penalty/nu_grad_penalty': nu_grad_penalty * self.args.nu_grad_penalty_coeff,
			'penalty/orthogonal': reg,
			'loss/nu': nu_loss,
			'loss/pi': pi_loss,
			'avg_nu/expert': tf.reduce_mean(expert_nu),
			'avg_nu/rb': tf.reduce_mean(rb_nu),
		}
	
	def act(self, state, env_goal, prev_goal, prev_skill, epsilon, stddev):
		state = tf.clip_by_value(state, -self.args.clip_obs, self.args.clip_obs)
		env_goal = tf.clip_by_value(env_goal, -self.args.clip_obs, self.args.clip_obs)
		
		# Current Goal and Skill
		curr_goal = prev_goal
		curr_skill = prev_skill
		
		# # Action
		# Explore
		if tf.random.uniform(()) < epsilon:
			action = tf.random.uniform((1, self.args.a_dim), -self.args.action_max, self.args.action_max)
		# Exploit
		else:
			action_mu, _, _ = self.actor(tf.concat([state, env_goal], axis=1))  # a_t = mu(s_t, g_t)
			action_dev = tf.random.normal(action_mu.shape, mean=0.0, stddev=stddev)
			action = action_mu + action_dev  # Add noise to action
			action = tf.clip_by_value(action, -self.args.action_max, self.args.action_max)
		
		return curr_goal, curr_skill, action
	
	def build_model(self):
		# Actor
		_ = self.actor(tf.concat([np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim])], 1))
		_ = self.critic(np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim]), np.ones([1, self.args.a_dim]))
	
	def save_(self, dir_param):
		self.actor.save_weights(dir_param + "/policy.h5")
		self.critic.save_weights(dir_param + "/nu_net.h5")
	
	def load_(self, dir_param):
		self.actor.load_weights(dir_param + "/policy.h5")
		self.critic.load_weights(dir_param + "/nu_net.h5")


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
		self.model = ValueDICE(args)
		
		# Environments
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
			self.wandb_logger = wandb.init(project='DICE', config=vars(args), id='valueDICE_{}'.format(current_time))
			# Clear tensorflow graph and cache
			tf.keras.backend.clear_session()
			tf.compat.v1.reset_default_graph()
	
	def preprocess_in_state_space(self, item):
		item = tf.clip_by_value(item, -self.args.clip_obs, self.args.clip_obs)
		return item
	
	def save_model(self, dir_param):
		if not os.path.exists(dir_param):
			os.makedirs(dir_param)
		self.model.save_(dir_param)
		
	def load_actor(self, dir_param):
		self.model.actor.load_weights(dir_param + "/policy.h5")
	
	def load_model(self, dir_param):
		self.model.load_(dir_param)
	
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
		
		avg_loss_dict = {}
		
		for _ in range(self.args.updates_per_step):
			data_expert = self.sample_data(self.expert_buffer_unseg, self.args.batch_size)
			data_policy = self.sample_data(self.policy_buffer_unseg, self.args.batch_size)
			loss_dict = self.model.train_policy(data_expert, data_policy)
			for key in loss_dict.keys():
				if key not in avg_loss_dict.keys():
					avg_loss_dict[key] = []
				avg_loss_dict[key].append(loss_dict[key])
		
		# Average the losses
		for key in avg_loss_dict.keys():
			avg_loss_dict[key] = tf.reduce_mean(avg_loss_dict[key])
		
		return avg_loss_dict
	
	
	def learn(self):
		args = self.args
		
		total_time_steps = 0
		log_step = 0
		success_rate = []
		with tqdm(total=args.max_time_steps, desc='') as pbar:
			
			while total_time_steps < args.max_time_steps:
				_update_pbar_msg(args, pbar, total_time_steps)
				
				# Evaluate the policy
				if total_time_steps % args.eval_interval == 0:
					avg_return, avg_time, avg_goal_dist = evaluate_worker(self.eval_worker, args.eval_demos)
					
					# Log the data
					if self.log_wandb:
						self.wandb_logger.log({
							'stats/eval_avg_return': avg_return,
							'stats/eval_avg_time': avg_time,
							'stats/eval_avg_goal_dist': avg_goal_dist,
						}, step=log_step)
					tf.print(f"Eval Avg Return: {avg_return}, Eval Avg Time: {avg_time}")
				
				# Collect data from the policy
				if total_time_steps < args.num_random_actions:
					# Do exploration
					episode, stats = self.policy_worker.generate_rollout(epsilon=1.0, stddev=0.0)
				else:
					# Do exploitation
					episode, stats = self.policy_worker.generate_rollout(epsilon=0, stddev=0.1)

				success_rate.append(stats['ep_success'])
				
				# Add the episode to the buffer
				# self.expert_buffer_unseg.store_episode(episode)  # TODO: Add successful episodes to the expert buffer?
				self.policy_buffer_unseg.store_episode(episode)
				
				total_time_steps += args.horizon
				if total_time_steps > args.start_training_timesteps:
					
					avg_loss_dict = self.train()
					
					for key in avg_loss_dict.keys():
						avg_loss_dict[key] = avg_loss_dict[key].numpy().item()
						
					# Log the data
					if self.log_wandb:
						self.wandb_logger.log(avg_loss_dict, step=log_step)
					
				# Log
				if self.log_wandb:
					self.wandb_logger.log({
						'policy_buffer_size': self.policy_buffer_unseg.get_current_size_trans(),
						'expert_buffer_size': self.expert_buffer_unseg.get_current_size_trans(),
						'stats/train_ep_success': stats['ep_success'],
						'stats/train_success_rate': np.mean(success_rate),
					}, step=log_step)
				
				
				# Update
				pbar.update(args.horizon)
				log_step += 1
				
		# Save the model
		self.save_model(args.dir_param)
		
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
		data_prefix = ''
	
	# Clear tensorflow graph and cache
	tf.keras.backend.clear_session()
	tf.compat.v1.reset_default_graph()

	# ######################################################################################################## #
	# ############################################# DATA LOADING ############################################# #
	# ######################################################################################################## #
	# Load Buffer to store expert data
	expert_buffer_unseg = ReplayBufferTf(
		get_buffer_shape(args), args.buffer_size, args.horizon, sample_transitions('random_unsegmented')
	)
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
		expert_buffer_unseg.load_data_into_buffer(train_data_path)
	
	# ########################################################################################################### #
	# ############################################# TRAINING #################################################### #
	# ########################################################################################################### #
	if args.do_train:
		start = time.time()
		agent = Agent(args, expert_buffer_unseg, policy_buffer_unseg)
		
		# logger.info("Load Actor Policy from {}".format(args.dir_pre))
		# agent.load_actor(dir_param=args.dir_pre)
		# print("Actor Loaded")
		
		logger.info("Training .......")
		agent.learn()
		logger.info("Done Training in {}".format(str(datetime.timedelta(seconds=time.time() - start))))
