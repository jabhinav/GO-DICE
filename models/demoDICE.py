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
from networks.general import Actor, Critic, Discriminator
from utils.buffer import get_buffer_shape
from utils.env import get_PnP_env
from utils.custom import evaluate_worker, state_to_goal
from tensorflow_gan.python.losses import losses_impl as tfgan_losses
from domains.PnPExpert import PnPExpert, PnPExpertTwoObj

logger = logging.getLogger(__name__)


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


class DemoDICE(tf.keras.Model, ABC):
	def __init__(self, args: Namespace):
		super(DemoDICE, self).__init__()
		self.args = args
		
		self.args.EPS = np.finfo(np.float32).eps  # Small value = 1.192e-07 to avoid division by zero in grad penalty
		self.args.EPS2 = 1e-3
		
		# Define Networks
		self.actor = Actor(args.a_dim)
		self.critic = Critic()
		self.disc = Discriminator()
		
		# Define Optimizers
		self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=args.actor_lr)
		self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=args.critic_lr)
		self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=args.disc_lr)
		
		self.build_model()
		
		# For expert Assistance
		exp_env = get_PnP_env(args)
		num_skills = exp_env.latent_dim
		self.use_expert_goal = False
		if args.two_object:
			self.expert_guide = PnPExpertTwoObj(num_skills, expert_behaviour=args.expert_behaviour)
		else:
			self.expert_guide = PnPExpert(num_skills)
		
		# For HER
		self.use_her = True
		logger.info('[[[ Using HER ? ]]]: {}'.format(self.use_her))
	
	@tf.function(experimental_relax_shapes=True)
	def train_policy(self, data_exp, data_rb):
		with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
			tape.watch(self.actor.variables)
			tape.watch(self.critic.variables)
			tape.watch(self.disc.variables)
			
			init_rb_ip = tf.concat([data_rb['init_states'], data_rb['goals']], 1)
			curr_exp_ip = tf.concat([data_exp['states'], data_exp['goals']], 1)
			next_exp_ip = tf.concat([data_exp['states_2'], data_exp['goals']], 1)
			curr_rb_ip = tf.concat([data_rb['states'], data_rb['goals']], 1)
			next_rb_ip = tf.concat([data_rb['states_2'], data_rb['goals']], 1)
			
			disc_expert_inputs = tf.concat([curr_exp_ip, data_exp['actions']], 1)
			disc_rb_inputs = tf.concat([curr_rb_ip, data_rb['actions']], 1)
			
			# Compute cost of (s_E, a_E)
			cost_expert = self.disc(disc_expert_inputs)
			# Compute cost of (s_R, a_R)
			cost_rb = self.disc(disc_rb_inputs)
			# Get Reward from Discriminator = log-dist ratio between expert and rb.
			# This is the reward = - log (1/c(s,a) - 1)
			reward = - tf.math.log(1 / (tf.nn.sigmoid(cost_rb) + self.args.EPS2) - 1 + self.args.EPS2)
			
			# Compute Discriminator loss
			cost_loss = tfgan_losses.modified_discriminator_loss(cost_expert, cost_rb, label_smoothing=0.)
			
			# Compute gradient penalty for Discriminator
			alpha = tf.random.uniform(shape=(disc_expert_inputs.shape[0], 1))
			interpolates_1 = alpha * disc_expert_inputs + (1 - alpha) * disc_rb_inputs
			interpolates_2 = alpha * tf.random.shuffle(disc_rb_inputs) + (1 - alpha) * disc_rb_inputs
			interpolates = tf.concat([interpolates_1, interpolates_2], axis=0)
			with tf.GradientTape() as tape2:
				tape2.watch(interpolates)
				cost_interpolates = self.disc(interpolates)
				cost_interpolates = tf.math.log(1 / (tf.nn.sigmoid(cost_interpolates) + self.args.EPS2) - 1 + self.args.EPS2)
			cost_grads = tape2.gradient(cost_interpolates, [interpolates])[0] + self.args.EPS
			cost_grad_penalty = tf.reduce_mean(tf.square(tf.norm(cost_grads, axis=1, keepdims=True) - 1))
			cost_loss_w_pen = cost_loss + self.args.cost_grad_penalty_coeff * cost_grad_penalty
			
			# Compute the value function
			init_nu = self.critic(init_rb_ip)
			expert_nu = self.critic(curr_exp_ip)  # not used in loss calc
			rb_nu = self.critic(curr_rb_ip)
			rb_nu_next = self.critic(next_rb_ip)
			
			# Compute the Advantage function (on replay buffer)
			rb_adv = tf.stop_gradient(reward) + self.args.discount * rb_nu_next - rb_nu
			
			# Linear Loss = (1 - gamma) * E[init_nu]
			linear_loss = (1 - self.args.discount) * tf.reduce_mean(init_nu)
			# Non-Linear Loss = (1 + alpha) * E[exp(Adv_nu / (1 + alpha))]
			non_linear_loss = (1 + self.args.replay_regularization) * tf.reduce_logsumexp\
				(rb_adv / (1 + self.args.replay_regularization))
			nu_loss = linear_loss + non_linear_loss
			
			# Compute gradient penalty for nu
			beta = tf.random.uniform(shape=(disc_expert_inputs.shape[0], 1))
			nu_inter = beta * curr_exp_ip + (1 - beta) * curr_rb_ip
			nu_next_inter = beta * next_exp_ip + (1 - beta) * next_rb_ip
			nu_input = tf.concat([curr_exp_ip, nu_inter, nu_next_inter], 0)
			with tf.GradientTape(watch_accessed_variables=False) as tape3:
				tape3.watch(nu_input)
				nu_output = self.critic(nu_input)
			nu_grad = tape3.gradient(nu_output, [nu_input])[0] + self.args.EPS
			nu_grad_penalty = tf.reduce_mean(tf.square(tf.norm(nu_grad, axis=-1, keepdims=True) - 1))
			nu_loss_w_pen = nu_loss + self.args.nu_grad_penalty_coeff * nu_grad_penalty
			
			# Compute Policy Loss : Weighted BC Loss with the Advantage function
			weight = tf.expand_dims(tf.math.exp(rb_adv / (1 + self.args.replay_regularization)), 1)
			weight = weight / tf.reduce_mean(weight)  # Normalise weight using self-normalised importance sampling
			pi_loss = - tf.reduce_mean(tf.stop_gradient(weight) * self.actor.get_log_prob(curr_rb_ip, data_rb['actions']))
			
			# # Check if pi_loss is NaN
			# if tf.math.is_nan(pi_loss):
			# 	x = self.actor.get_log_prob(tf.concat([data_rb['states'], data_rb['env_goals']], axis=1), data_rb['actions'])
			# 	print('pi_loss is NaN')
			# 	sys.exit(-1)
		
		nu_grads = tape.gradient(nu_loss_w_pen, self.critic.variables)
		pi_grads = tape.gradient(pi_loss, self.actor.variables)
		cost_grads = tape.gradient(cost_loss_w_pen, self.disc.variables)
		
		self.critic_optimizer.apply_gradients(zip(nu_grads, self.critic.variables))
		self.actor_optimizer.apply_gradients(zip(pi_grads, self.actor.variables))
		self.disc_optimizer.apply_gradients(zip(cost_grads, self.disc.variables))
		
		return {
			'loss/cost': cost_loss,
			'loss/linear': linear_loss,
			'loss/non-linear': non_linear_loss,
			'loss/nu': nu_loss,
			'loss/pi': pi_loss,
			
			'penalty/cost_grad_penalty': self.args.cost_grad_penalty_coeff * cost_grad_penalty,
			'penalty/nu_grad_penalty': self.args.nu_grad_penalty_coeff * nu_grad_penalty,
			
			'avg_nu/expert': tf.reduce_mean(expert_nu),
			'avg_nu/rb': tf.reduce_mean(rb_nu),
			'avg_nu/init': tf.reduce_mean(init_nu),
			'avg/rb_adv': tf.reduce_mean(rb_adv),
		}
	
	def act(self, state, env_goal, prev_goal, prev_skill, epsilon, stddev):
		state = tf.clip_by_value(state, -self.args.clip_obs, self.args.clip_obs)
		env_goal = tf.clip_by_value(env_goal, -self.args.clip_obs, self.args.clip_obs)
		prev_goal = tf.clip_by_value(prev_goal, -self.args.clip_obs, self.args.clip_obs)
		
		# Current Goal and Skill
		if self.use_expert_goal:
			curr_goal = tf.numpy_function(self.expert_guide.sample_curr_goal,
										  [state[0], env_goal[0], prev_goal[0], False], tf.float32)
			curr_goal = tf.expand_dims(curr_goal, axis=0)
		else:
			curr_goal = env_goal
		
		curr_skill = prev_skill  # Not used in this implementation
		
		# # Action
		# Explore
		if tf.random.uniform(()) < epsilon:
			action = tf.random.uniform((1, self.args.a_dim), -self.args.action_max, self.args.action_max)
		# Exploit
		else:
			action_mu, _, _ = self.actor(tf.concat([state, curr_goal], axis=1))  # a_t = mu(s_t, g_t)
			action_dev = tf.random.normal(action_mu.shape, mean=0.0, stddev=stddev)
			action = action_mu + action_dev  # Add noise to action
			action = tf.clip_by_value(action, -self.args.action_max, self.args.action_max)
		
		return curr_goal, curr_skill, action
	
	def build_model(self):
		# Actor
		_ = self.actor(tf.concat([np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim])], 1))
		_ = self.critic(np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim]))
		_ = self.disc(np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim]), np.ones([1, self.args.a_dim]))
	
	def save_(self, dir_param):
		self.actor.save_weights(dir_param + "/policy.h5")
		self.critic.save_weights(dir_param + "/nu_net.h5")
		self.disc.save_weights(dir_param + "/cost_net.h5")
	
	def load_(self, dir_param):
		self.actor.load_weights(dir_param + "/policy.h5")
		self.critic.load_weights(dir_param + "/nu_net.h5")
		self.disc.load_weights(dir_param + "/cost_net.h5")


class Agent(object):
	def __init__(self, args,
				 expert_buffer_unseg: ReplayBufferTf = None,
				 policy_buffer_unseg: ReplayBufferTf = None,
				 log_wandb: bool = True	):
		
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
		self.model = DemoDICE(args)
		
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
			self.wandb_logger = wandb.init(project='demoDICE', config=vars(args), id='demoDICE_{}'.format(current_time))
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
		trans['env_goals'] = self.preprocess_in_state_space(trans['env_goals'])
		trans['init_states'] = self.preprocess_in_state_space(trans['init_states'])
		trans['her_goals'] = self.preprocess_in_state_space(trans['her_goals'])
		
		# # Make sure her goals has the same shape as env_goals
		# try:
		# 	assert trans['env_goals'].shape == trans['her_goals'].shape
		# except AssertionError:
		# 	tf.print("Shapes of env goals {} and her goals {} are not the same.".format(trans['env_goals'].shape,
		# 																				trans['her_goals'].shape))
		# 	raise AssertionError
		
		if self.model.use_her:
			trans['goals'] = trans['her_goals']
		else:
			trans['goals'] = trans['env_goals']

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
		
		# Evaluate the policy
		max_return, max_return_with_exp_assist = self.evaluate(log_step=log_step)
		
		with tqdm(total=args.max_time_steps, desc='') as pbar:
			
			while total_time_steps < args.max_time_steps:
				_update_pbar_msg(args, pbar, total_time_steps)
				
				# Evaluate the policy
				if log_step % args.eval_interval == 0:
					max_return, max_return_with_exp_assist = self.evaluate(max_return=max_return,
																		   max_return_with_exp_assist=max_return_with_exp_assist,
																		   log_step=log_step)
				
				# Train the policy
				avg_loss_dict = self.train()
				for key in avg_loss_dict.keys():
					avg_loss_dict[key] = avg_loss_dict[key].numpy().item()
				
				# Log
				if self.log_wandb:
					self.wandb_logger.log(avg_loss_dict, step=log_step)
					self.wandb_logger.log({
						'policy_buffer_size': self.policy_buffer_unseg.get_current_size_trans(),
						'expert_buffer_size': self.expert_buffer_unseg.get_current_size_trans(),
					}, step=log_step)
				
				# Update
				pbar.update(args.horizon)
				log_step += 1
				total_time_steps += args.horizon  # We will use horizon to update the number of time steps

		
		# Save the model
		self.save_model(args.dir_param)
		
		# # Set to eager mode
		tf.config.run_functions_eagerly(True)
		self.model.use_expert_goal = False
		_ = evaluate_worker(self.visualise_worker, num_episodes=args.test_demos)
		self.model.use_expert_goal = True
		_ = evaluate_worker(self.visualise_worker, num_episodes=args.test_demos, expert_assist=True)
		
	def evaluate(self, max_return=None, max_return_with_exp_assist=None, log_step=None):
		
		self.model.use_expert_goal = False
		avg_return, avg_time, avg_goal_dist = evaluate_worker(self.eval_worker, self.args.eval_demos)
		if max_return is None:
			max_return = avg_return
		elif avg_return > max_return:
			max_return = avg_return
			self.save_model(os.path.join(self.args.dir_param, 'best_model'))
		
		# Log the data
		if self.log_wandb:
			self.wandb_logger.log({
				'stats/eval_max_return': max_return,
				'stats/eval_avg_time': avg_time,
				'stats/eval_avg_goal_dist': avg_goal_dist,
			}, step=log_step)
		tf.print(f"Eval Return: {avg_return}, "
				 f"Eval Avg Time: {avg_time}, "
				 f"Eval Avg Goal Dist: {avg_goal_dist}")
		
		self.model.use_expert_goal = True
		avg_return, avg_time, avg_goal_dist = evaluate_worker(self.eval_worker, self.args.eval_demos, expert_assist=True)
		if max_return_with_exp_assist is None:
			max_return_with_exp_assist = avg_return
		elif avg_return > max_return_with_exp_assist:
			max_return_with_exp_assist = avg_return
			self.save_model(os.path.join(self.args.dir_param, 'best_model_with_exp_assist'))
		
		# Log the data
		if self.log_wandb:
			self.wandb_logger.log({
				'stats/eval_max_return_exp_assist': max_return_with_exp_assist,
				'stats/eval_avg_time_exp_assist': avg_time,
				'stats/eval_avg_goal_dist_exp_assist': avg_goal_dist,
			}, step=log_step)
		tf.print(f"Eval Return (Exp Assist): {avg_return}, "
				 f"Eval Avg Time (Exp Assist): {avg_time}, "
				 f"Eval Avg Goal Dist (Exp Assist): {avg_goal_dist}")
		
		return max_return, max_return_with_exp_assist
		


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
	
	# Clear tensorflow graph and cache
	tf.keras.backend.clear_session()
	tf.compat.v1.reset_default_graph()
	
	# ######################################################################################################## #
	# ############################################# DATA LOADING ############################################# #
	# ######################################################################################################## #
	# Load Buffer to store expert data
	n_objs = 2 if args.two_object else 1
	expert_buffer_unseg = ReplayBufferTf(
		get_buffer_shape(args), args.buffer_size, args.horizon,
		sample_transitions('random_unsegmented', state_to_goal=state_to_goal(n_objs))
	)
	policy_buffer_unseg = ReplayBufferTf(
		get_buffer_shape(args), args.buffer_size, args.horizon,
		sample_transitions('random_unsegmented', state_to_goal=state_to_goal(n_objs))
	)
	
	train_data_path = os.path.join(args.dir_data, '{}{}_train.pkl'.format(data_prefix,
																		  'two_obj_{}'.format(
																			  args.expert_behaviour) if args.two_object else 'single_obj'))
	
	if not os.path.exists(train_data_path):
		logger.error("Train data not found at {}. Please run the validation data generation script first.".format(
			train_data_path))
		sys.exit(-1)
	else:
		logger.info("Loading Expert Demos from {} into TrainBuffer for training.".format(train_data_path))
		# Store the expert data in the expert buffer -> D_E
		expert_buffer_unseg.load_data_into_buffer(train_data_path,
												  num_demos_to_load=args.expert_demos)
		
		# Store the expert data in the policy buffer for DemoDICE -> D_U = D_E + D_I
		policy_buffer_unseg.load_data_into_buffer(train_data_path,
												  num_demos_to_load=args.expert_demos)
		# # BC offline data
		policy_buffer_unseg.load_data_into_buffer(f'./pnp_data/BC_{data_prefix}offline_data.pkl',
												  num_demos_to_load=args.imperfect_demos,
												  clear_buffer=False)
	
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
