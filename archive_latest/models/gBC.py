import os
import pickle
import sys
import time
import logging
import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from typing import List, Dict, Tuple, Union, Optional
from utils.env import get_PnP_env
from utils.normalise import Normalizer
from utils.buffer import get_buffer_shape
from domains.PnP import MyPnPEnvWrapperForGoalGAIL
from domains.PnPExpert import PnPExpert, PnPExpertTwoObj
from her.rollout import RolloutWorker
from her.replay_buffer import ReplayBufferTf
from her.transitions import sample_goal_oriented_transitions
from networks.general import Policy, GoalPred
from utils.plot import plot_metric, plot_metrics

logger = logging.getLogger(__name__)


class gBC(tf.keras.Model):
	def __init__(self, args):
		super(gBC, self).__init__()
		self.args = args
		self.policy = Policy(args.a_dim, args.action_max)
		self.goal_pred = GoalPred(args.g_dim)
		self.optimiser = tf.keras.optimizers.Adam(self.args.vae_lr)
		
		self.build_model()
		
		# Get the expert policy [HACK: This is only for the validating gBC without latent mode prediction]
		exp_env = get_PnP_env(args)
		latent_dim = exp_env.latent_dim
		if args.two_object:
			self.expert_policy = PnPExpertTwoObj(latent_dim, args.full_space_as_goal, expert_behaviour=args.expert_behaviour)
		else:
			self.expert_policy = PnPExpert(latent_dim, args.full_space_as_goal)
	
	@tf.function
	def compute_loss(self, data):
		
		delta_g = self.goal_pred(data['achieved_goals'], data['states'], data['goals'])
		g_pred = data['achieved_goals'] + delta_g
		g_pred = tf.clip_by_value(g_pred, -self.args.clip_obs, self.args.clip_obs)
		
		actions_mu = self.policy(data['states'], data['goals'], g_pred)
		# actions_mu = self.policy(g_pred, data['states'])
		loss = tf.reduce_sum(tf.math.squared_difference(data['actions'], actions_mu), axis=-1)
		
		return tf.reduce_mean(loss)
	
	@tf.function(experimental_relax_shapes=True)
	def train(self, data):
		with tf.GradientTape() as tape:
			loss = self.compute_loss(data)
		
		gradients = tape.gradient(loss, self.policy.trainable_variables)
		self.optimiser.apply_gradients(zip(gradients, self.policy.trainable_variables))
		return loss
	
	def act(self, state, achieved_goal, goal, compute_Q=False, **kwargs):
		# Pre-process the state and goals
		state = tf.clip_by_value(state, -self.args.clip_obs, self.args.clip_obs)
		achieved_goal = tf.clip_by_value(achieved_goal, -self.args.clip_obs, self.args.clip_obs)
		goal = tf.clip_by_value(goal, -self.args.clip_obs, self.args.clip_obs)
		
		delta_g = self.goal_pred(achieved_goal, state, goal)
		g_pred = achieved_goal + delta_g
		g_pred = tf.clip_by_value(g_pred, -self.args.clip_obs, self.args.clip_obs)
		action_mu = self.policy(state, goal, g_pred)
		action = tf.clip_by_value(action_mu, -self.args.action_max, self.args.action_max)

		if compute_Q:
			return action, delta_g
		else:
			return action
		
	def build_model(self):
		# BUILD First
		_ = self.goal_pred(np.ones([1, self.args.g_dim]), np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim]))
		_ = self.policy(np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim]), np.ones([1, self.args.g_dim]))
	
	def preload_goalPred(self, path):
		
		if not os.path.exists(path):
			logger.info("Goal Predictor Weights Not Found at {}. Exiting!".format(path))
			sys.exit(-1)
		self.goal_pred.load_weights(path)
		logger.info("Goal prediction Model Pre-Loaded from {}".format(path))
	
	def load_model(self, dir_param):
		
		goal_pred_model_path = os.path.join(dir_param, "goalPred_gBC.h5")
		if not os.path.exists(goal_pred_model_path):
			logger.info("Goal Predictor Weights Not Found at {}. Exiting!".format(goal_pred_model_path))
			sys.exit(-1)
		
		policy_model_path = os.path.join(dir_param, "policy_gBC.h5")
		if not os.path.exists(policy_model_path):
			logger.info("Policy Weights Not Found at {}. Exiting!".format(policy_model_path))
			sys.exit(-1)
		
		# Load Models
		self.goal_pred.load_weights(goal_pred_model_path)
		self.policy.load_weights(policy_model_path)
		
		logger.info("Models Loaded from {} and {}".format(goal_pred_model_path, policy_model_path))
	
	def save_model(self, dir_param):
		self.goal_pred.save_weights(os.path.join(dir_param, "goalPred_gBC.h5"), overwrite=True)
		self.policy.save_weights(os.path.join(dir_param, "policy_gBC.h5"), overwrite=True)


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
		self.model = gBC(args)
		
		# Evaluation
		self.eval_env: MyPnPEnvWrapperForGoalGAIL = get_PnP_env(args)
		# Here compute_Q is an alias for computing delta_G
		self.eval_rollout_worker = RolloutWorker(
			self.eval_env, self.model, T=args.horizon, rollout_terminate=args.rollout_terminate, exploit=True,
			noise_eps=0., random_eps=0., compute_Q=True, use_target_net=False, render=False
		)
	
	def preprocess_og(self, states, achieved_goals, goals):
		states = tf.clip_by_value(states, -self.args.clip_obs, self.args.clip_obs)
		achieved_goals = tf.clip_by_value(achieved_goals, -self.args.clip_obs, self.args.clip_obs)
		goals = tf.clip_by_value(goals, -self.args.clip_obs, self.args.clip_obs)
		return states, achieved_goals, goals
		
	def preload_goalPred(self, path):
		self.model.preload_goalPred(path)
	
	def load_model(self, dir_param):
		
		# Load Models
		self.model.load_model(dir_param)
	
	def save_model(self, dir_param):
		
		if not os.path.exists(dir_param):
			os.makedirs(dir_param)
		
		# Save weights
		self.model.save_model(dir_param)
	
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
				transitions[option] = self.process_data(transitions[option], tf.constant(True, dtype=tf.bool),
														tf.constant(True, dtype=tf.bool))
		elif isinstance(transitions, dict):
			transitions = self.process_data(transitions, tf.constant(True, dtype=tf.bool),
											tf.constant(True, dtype=tf.bool))
		else:
			raise ValueError("Invalid type of transitions")
		
		return transitions
	
	def learn(self):
		args = self.args
		global_step = 0
		
		monitor_policy = np.inf
		val_data = None
		
		with tqdm(total=args.num_epochs, position=0, leave=True, desc='Training: ') as pbar:
			
			for epoch in range(args.num_epochs):
				
				for _ in range(args.n_batches):
					data = self.sample_data(self.expert_buffer, batch_size=tf.constant(args.expert_batch_size, dtype=tf.int32))
					
					# Train Policy on each batch
					loss = self.model.train(data)
					
					# Log the VAE Losses
					with self.summary_writer.as_default():
						tf.summary.scalar('train/loss', loss, step=global_step)
					
					loss = loss.numpy() if isinstance(loss, tf.Tensor) else loss
					
					pbar.set_postfix({'PolicyLoss': loss}, refresh=True)
					global_step += 1
				
				pbar.update(1)
				
				# Compute Validation Losses
				if val_data is None:
					val_data = self.sample_data(self.val_buffer, batch_size=tf.constant(args.val_batch_size, dtype=tf.int32))
				
				val_loss = self.model.compute_loss(val_data)
				
				# Log Validation Losses
				with self.summary_writer.as_default():
					tf.summary.scalar('val/loss', val_loss, step=epoch)
				
				val_loss = val_loss.numpy() if isinstance(val_loss, tf.Tensor) else val_loss
				
				# Save Best Model (post-epoch)
				if val_loss < monitor_policy:
					monitor_policy = val_loss
					logger.info(
						"[POLICY] Saving the best model (best policy_loss: {}) after epoch: {}".format(monitor_policy,
																									   epoch))
					self.save_model(args.dir_param + '_bestPolicy')
				
				# Save Last Model (post-epoch)
				self.save_model(args.dir_param)
				
				if (epoch + 1) % args.log_interval == 0 and args.log_interval > 0:
					
					# # Clear the worker's history to avoid retention of poor perf. during early training
					# self.eval_rollout_worker.clear_history()
					
					# Do rollouts using VAE Policy
					for n in range(args.eval_demos):
						
						# Reset the policy to reset its variables [Hack: to get correct G.T. latent modes]
						self.model.expert_policy.reset()
						
						episode, eval_stats = self.eval_rollout_worker.generate_rollout(
							slice_goal=(3, 6) if args.full_space_as_goal else None)
						
						# Monitor the metrics during each episode
						delta_AG = np.linalg.norm(episode['quality'].numpy()[0], axis=-1)
						fig_path = os.path.join(args.dir_plot, 'Distances_{}_{}.png'.format(epoch + 1, n))
						plot_metrics(
							metrics=[episode['distances'].numpy()[0], delta_AG],
							labels=['|G_env - AG_curr|', '|G_pred - AG_curr|'],
							fig_path=fig_path, y_label='Distances', x_label='Steps'
						)
					
					# Plot avg. success rate for each epoch
					with self.summary_writer.as_default():
						tf.summary.scalar('stats/Eval/Success_rate', eval_stats['success_rate'], step=epoch)
				

def run(args):
	args.train_demos = int(args.expert_demos * args.perc_train)
	args.val_demos = args.expert_demos - args.train_demos
	args.goal_pred_path = './pretrained_models/models_bestGoal/goalPredOptionBC.h5'
	args.train_trans_style = 'random'
	args.val_trans_style = 'all'
	args.val_batch_size = args.val_demos
	
	# ############################################# DATA LOADING ############################################# #
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
	
	# ############################################# TRAINING #################################################### #
	start = time.time()
	agent = Agent(args, expert_buffer, val_buffer)
	
	# Load the option policies
	agent.preload_goalPred(args.goal_pred_path)
	
	logger.info("Training .......")
	agent.learn()
	logger.info("Done Training in {}".format(str(datetime.timedelta(seconds=time.time() - start))))
