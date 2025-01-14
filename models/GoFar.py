import logging
import sys
from abc import ABC
from argparse import Namespace

import numpy as np
import tensorflow as tf
from tensorflow_gan.python.losses import losses_impl as tfgan_losses
from tqdm import tqdm

from utils.env import get_expert
from her.replay_buffer import ReplayBufferTf
from networks.general import Actor, Critic, Discriminator
from .Base import AgentBase

logger = logging.getLogger(__name__)


class GoFar(tf.keras.Model, ABC):
	def __init__(self, args: Namespace, f_div='chi'):
		super(GoFar, self).__init__()
		self.args = args
		self.f_div = f_div  # Options: 'chi', 'kl'
		
		self.args.EPS = np.finfo(np.float32).eps  # Small value = 1.192e-07 to avoid division by zero in grad penalty
		self.args.EPS2 = 1e-3
		
		# Define Networks
		self.actor = Actor(args.a_dim)
		self.critic = Critic()
		self.disc = Discriminator()
		
		# Define Target Networks
		self.actor_target = Actor(args.a_dim)
		self.critic_target = Critic()
		self.actor_target.trainable = False
		self.critic_target.trainable = False
		
		# Define Optimizers
		self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=args.actor_lr)
		self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=args.critic_lr)
		self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=args.disc_lr)
		
		self.build_model()
		
		# For expert Assistance
		self.use_expert_goal = False
		self.expert = get_expert(args.num_objs, args)
		
		# For HER
		self.use_her = False
		logger.info('[[[ Using HER ? ]]]: {}'.format(self.use_her))
	
	@tf.function(experimental_relax_shapes=True)
	def train(self,  data_exp, data_rb):
		with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
			tape.watch(self.actor.variables)
			tape.watch(self.critic.variables)
			tape.watch(self.disc.variables)
			
			init_rb_ip = tf.concat([data_rb['init_states'], data_rb['goals']], 1)
			curr_rb_ip = tf.concat([data_rb['states'], data_rb['goals']], 1)
			next_rb_ip = tf.concat([data_rb['states_2'], data_rb['goals']], 1)
			
			disc_pos_pairs = tf.concat([data_rb['goals'], data_rb['goals']], 1)
			disc_neg_pairs = tf.concat([data_rb['achieved_goals'], data_rb['goals']], 1)
			
			cost_expert = self.disc(disc_pos_pairs)
			cost_policy = self.disc(disc_neg_pairs)
			# Get Reward from Discriminator = log-dist ratio between expert and rb.
			# This is the reward = - log (1/c(ag,g) - 1)
			reward = - tf.math.log(1 / (tf.nn.sigmoid(cost_policy) + self.args.EPS2) - 1 + self.args.EPS2)
			
			# Compute Discriminator loss
			cost_loss = tfgan_losses.modified_discriminator_loss(cost_expert, cost_policy, label_smoothing=0.)
			
			# Compute gradient penalty for Discriminator
			alpha = tf.random.uniform(shape=(disc_pos_pairs.shape[0], 1))
			interpolates_1 = alpha * disc_pos_pairs + (1 - alpha) * disc_neg_pairs
			interpolates_2 = alpha * tf.random.shuffle(disc_neg_pairs) + (1 - alpha) * disc_neg_pairs
			interpolates = tf.concat([interpolates_1, interpolates_2], axis=0)
			with tf.GradientTape() as tape2:
				tape2.watch(interpolates)
				cost_interpolates = self.disc(interpolates)
				cost_interpolates = tf.math.log(
					1 / (tf.nn.sigmoid(cost_interpolates) + self.args.EPS2) - 1 + self.args.EPS2)
			cost_grads = tape2.gradient(cost_interpolates, [interpolates])[0] + self.args.EPS
			cost_grad_penalty = tf.reduce_mean(tf.square(tf.norm(cost_grads, axis=1, keepdims=True) - 1))
			cost_loss_w_pen = cost_loss + self.args.cost_grad_penalty_coeff * cost_grad_penalty
			
			# Compute the value function
			init_nu = self.critic(init_rb_ip)
			rb_nu = self.critic(curr_rb_ip)
			rb_nu_next = self.critic(next_rb_ip)  # choose to use main/target critic
			
			# Compute the Advantage function (on replay buffer)
			rb_adv = tf.stop_gradient(reward) + self.args.discount * rb_nu_next - rb_nu
			
			# Linear Loss = (1 - gamma) * E[init_nu]
			linear_loss = (1 - self.args.discount) * tf.reduce_mean(init_nu)
			# Non-Linear Loss = Div_f (rb_adv)
			if self.f_div == 'chi':
				# non_linear_loss = Mean[ (rb_adv + 1)^2 ]
				non_linear_loss = tf.reduce_mean(tf.math.square(rb_adv + 1))
			elif self.f_div == 'kl':
				# non_linear_loss = Log [ Mean[ exp(rb_adv) ] ]
				non_linear_loss = tf.math.log(tf.reduce_mean(tf.math.exp(rb_adv)) + self.args.EPS2)
			
			nu_loss = linear_loss + non_linear_loss
			
			# Compute Policy Loss : Weighted BC Loss with the Advantage function
			if self.f_div == 'chi':
				# weight = ReLU(rb_adv + 1)
				weight = tf.nn.relu(rb_adv + 1)
			elif self.f_div == 'kl':
				# weight = Clamp (exp(rb_adv), 0, 10)
				weight = tf.clip_by_value(tf.math.exp(rb_adv), 0, 10)
			
			weight = tf.reshape(weight, (-1, 1))
			weight = tf.expand_dims(weight, -1)
			weight = weight / tf.reduce_mean(weight)  # Normalise weight using self-normalised importance sampling
			pi_loss = - tf.reduce_mean(
				tf.stop_gradient(weight) * self.actor.get_log_prob(curr_rb_ip, data_rb['actions']))
			
		# Check if the loss is NaN
		if tf.math.is_nan(cost_loss_w_pen) or tf.math.is_nan(nu_loss) or tf.math.is_nan(pi_loss):
			tf.print('NaN Loss encountered.')
		
		nu_grads = tape.gradient(nu_loss, self.critic.variables)
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
			
			'avg_nu/rb': tf.reduce_mean(rb_nu),
			'avg_nu/init': tf.reduce_mean(init_nu),
			'avg/rb_adv': tf.reduce_mean(rb_adv),
		}
	
	def act(self, state, env_goal, prev_goal, prev_skill, epsilon, stddev):
		state = tf.clip_by_value(state, -self.args.clip_obs, self.args.clip_obs)
		env_goal = tf.clip_by_value(env_goal, -self.args.clip_obs, self.args.clip_obs)
		prev_goal = tf.clip_by_value(prev_goal, -self.args.clip_obs, self.args.clip_obs)
		
		# ###################################### Current Goal ####################################### #
		curr_goal = env_goal
		
		# ###################################### Current Skill ###################################### #
		curr_skill = prev_skill  # Not used in this implementation
		
		# ########################################## Action ######################################### #
		# Explore
		if tf.random.uniform(()) < epsilon:
			action = tf.random.uniform((1, self.args.a_dim), -self.args.action_max, self.args.action_max)
		# Exploit
		else:
			action_mu, _, _ = self.actor(tf.concat([state, curr_goal], axis=1))  # a_t = mu(s_t, g_t)
			action_dev = tf.random.normal(action_mu.shape, mean=0.0, stddev=stddev)
			action = action_mu + action_dev  # Add noise to action
			action = tf.clip_by_value(action, -self.args.action_max, self.args.action_max)
			
		# Safety check for action, should not be nan or inf
		has_nan = tf.math.reduce_any(tf.math.is_nan(action))
		has_inf = tf.math.reduce_any(tf.math.is_inf(action))
		if has_nan or has_inf:
			logger.warning('Action has nan or inf. Setting action to zero. Action: {}'.format(action))
			action = tf.zeros_like(action)
		
		return curr_goal, curr_skill, action
	
	def get_init_skill(self):
		"""
				demoDICE does not use skills. Use this function to return a dummy skill of dimension (1, c_dim)
		"""
		skill = tf.zeros((1, self.args.c_dim))
		return skill
	
	@staticmethod
	def get_init_goal(init_state, g_env):
		return g_env
	
	def build_model(self):
		# Networks
		_ = self.actor(tf.concat([np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim])], 1))
		_ = self.critic(np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim]))
		_ = self.disc(np.ones([1, self.args.g_dim]), np.ones([1, self.args.g_dim]))
		
		# Target Networks
		_ = self.actor_target(tf.concat([np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim])], 1))
		_ = self.critic_target(np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim]))
	
	def save_(self, dir_param):
		self.actor.save_weights(dir_param + "/policy.h5")
		self.critic.save_weights(dir_param + "/nu_net.h5")
		self.disc.save_weights(dir_param + "/cost_net.h5")
	
	def load_(self, dir_param):
		self.actor.load_weights(dir_param + "/policy.h5")
		self.critic.load_weights(dir_param + "/nu_net.h5")
		self.disc.load_weights(dir_param + "/cost_net.h5")
	
	def change_training_mode(self, training_mode: bool):
		pass
	
	def update_target_networks(self):
		# Update target networks
		main_actor_weights = self.actor.get_weights()
		target_actor_weights = self.actor_target.get_weights()
		for i in range(len(target_actor_weights)):
			target_actor_weights[i] = (1 - self.args.actor_polyak) * main_actor_weights[i] + \
									  self.args.actor_polyak * target_actor_weights[i]
		self.actor_target.set_weights(target_actor_weights)
		
		main_critic_weights = self.critic.get_weights()
		target_critic_weights = self.critic_target.get_weights()
		for i in range(len(target_critic_weights)):
			target_critic_weights[i] = (1 - self.args.critic_polyak) * main_critic_weights[i] + \
									   self.args.critic_polyak * target_critic_weights[i]
		self.critic_target.set_weights(target_critic_weights)


class Agent(AgentBase):
	def __init__(self, args,
				 expert_buffer: ReplayBufferTf = None,
				 offline_buffer: ReplayBufferTf = None):
		
		super().__init__(args, GoFar(args), 'GoFar', expert_buffer, offline_buffer)
	
	def load_actor(self, dir_param):
		self.model.actor.load_weights(dir_param + "/policy.h5")
	
	def learn(self):
		args = self.args
		
		# Evaluate the policy
		max_return, max_return_with_exp_assist = None, None
		log_step = 0

		# [Update] Load the expert data into the expert buffer, expert data and offline data into the offline buffer
		data_exp = self.expert_buffer.sample_episodes()
		data_off = self.offline_buffer.sample_episodes()
		self.expert_buffer.load_data_into_buffer(buffered_data=data_exp, clear_buffer=True)
		self.offline_buffer.load_data_into_buffer(buffered_data=data_exp, clear_buffer=True)
		self.offline_buffer.load_data_into_buffer(buffered_data=data_off, clear_buffer=False)
		
		with tqdm(total=args.max_time_steps, leave=False) as pbar:
			for curr_t in range(0, args.max_time_steps):
				
				# Evaluate the policy
				if curr_t % args.eval_interval == 0 and self.args.eval_demos > 0:
					pbar.set_description('Evaluating')
					max_return, max_return_with_exp_assist = self.evaluate(
						max_return=max_return,
						max_return_with_exp_assist=max_return_with_exp_assist,
						log_step=log_step
					)
				
				if curr_t % args.update_target_interval == 0:
					tf.print("Updating the target actors and critic at train step {}".format(curr_t))
					self.model.update_target_networks()
				
				# Train the policy
				pbar.set_description('Training')
				avg_loss_dict = self.train()
				
				# Log
				for key in avg_loss_dict.keys():
					avg_loss_dict[key] = avg_loss_dict[key].numpy().item()
				if self.args.log_wandb:
					self.wandb_logger.log(avg_loss_dict, step=log_step)
					self.wandb_logger.log({
						'policy_buffer_size': self.offline_buffer.get_current_size_trans(),
						'expert_buffer_size': self.expert_buffer.get_current_size_trans(),
					}, step=log_step)
				
				# Update
				pbar.update(1)
				log_step += 1
		
		# Save the model
		self.save_model(args.dir_param)
		
		if args.test_demos > 0:
			self.visualise(use_expert_options=False)
