import os
import pickle
import sys
import time
import logging
import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm
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
	def __init__(self, args, norm_s: Normalizer, norm_g: Normalizer):
		super(gBC, self).__init__()
		self.args = args
		self.policy = Policy(args.a_dim, args.action_max)
		
		self.goal_pred = GoalPred(3)
		self.optimiser = tf.keras.optimizers.Adam(self.args.vae_lr)
		
		self.norm_s = norm_s
		self.norm_g = norm_g
		
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
		loss_g = tf.reduce_sum(tf.math.squared_difference(g_pred, data['inter_goals']), axis=-1)
		
		actions_mu = self.policy(g_pred, data['states'], data['goals'])
		# actions_mu = self.policy(g_pred, data['states'])
		loss_p = tf.reduce_sum(tf.math.squared_difference(data['actions'], actions_mu), axis=-1)
		total_loss = tf.reduce_mean(self.args.g_coeff * loss_g + loss_p)
		
		return total_loss, tf.reduce_mean(loss_g), tf.reduce_mean(loss_p)
	
	@tf.function(experimental_relax_shapes=True)
	def train(self, data):
		with tf.GradientTape() as tape:
			total_loss, loss_g, loss_p = self.compute_loss(data)
		
		gradients = tape.gradient(total_loss, self.trainable_variables)
		self.optimiser.apply_gradients(zip(gradients, self.trainable_variables))
		return total_loss, loss_g, loss_p
	
	def act(self, state, achieved_goal, goal, compute_Q=False, compute_c=False, **kwargs):
		# Pre-process the state and goals
		state = tf.clip_by_value(state, -self.args.clip_obs, self.args.clip_obs)
		achieved_goal = tf.clip_by_value(achieved_goal, -self.args.clip_obs, self.args.clip_obs)
		goal = tf.clip_by_value(goal, -self.args.clip_obs, self.args.clip_obs)
		
		# ----------------------------------------- Latent Mode Prediction -----------------------------------------
		# TODO: Extract the relevant component of the achieved goal based on the latent mode or skill being executed
		# Current Hack: The current latent mode decides which part of the achieved goal to use as input for the goal
		# predictor network. This is fine when an object is being picked and dropped. Once it has been dropped,
		# curr_latent_mode changes to that of next object's pick. Ideally at this time we would want to keep the
		# achieved position of previous object as the achieved goal and predict the difference between it and the next
		# object's pos. However, using expert's latent mode trans would give us next object's position as the achieved
		# goal. This is incorrect but without a latent mode prediction network, we cannot do anything about it
		
		# obj_index = latent_mode // 2
		obj_index = self.expert_policy.get_current_skill(state, goal)
		achieved_goal = achieved_goal[:, obj_index * 3: (obj_index + 1) * 3]
		
		# Normalise (if running stats of Normaliser are updated, we get normalised data else un-normalised data)
		if self.args.use_norm:
			state = self.norm_s.normalize(state)
			achieved_goal = self.norm_g.normalize(achieved_goal)
			goal = self.norm_g.normalize(goal)
		
		delta_g = self.goal_pred(achieved_goal, state, goal)
		g_pred = achieved_goal + delta_g
		g_pred = tf.clip_by_value(g_pred, -self.args.clip_obs, self.args.clip_obs)
		action_mu = self.policy(g_pred, state, goal)
		action = tf.clip_by_value(action_mu, -self.args.action_max, self.args.action_max)

		if compute_Q and compute_c:
			return action, delta_g, tf.expand_dims(obj_index, axis=0)
		elif compute_Q:
			return action, delta_g
		elif compute_c:
			return action, tf.expand_dims(obj_index, axis=0)
		else:
			return action
	
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
		
		# Setup Normalisers
		self.norm_s = Normalizer(args.s_dim, args.eps_norm, args.clip_norm)
		self.norm_g = Normalizer(args.g_dim, args.eps_norm, args.clip_norm)
		if args.use_norm:
			self.setup_normalisers()
		
		# Declare Model
		self.model = gBC(args, self.norm_s, self.norm_g)
		
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
	
	def setup_normalisers(self):
		episode_batch = self.expert_buffer.sample_episodes()
		# Sample number of transitions in the batch
		# num_normalizing_transitions = episode_batch['actions'].shape[0] * episode_batch['actions'].shape[1]
		transitions = sample_goal_oriented_transitions('all')(episode_batch)
		# Preprocess the states and goals
		states, achieved_goals, goals = transitions['states'], transitions['achieved_goals'], transitions['goals']
		states, achieved_goals, goals = self.preprocess_og(states, achieved_goals, goals)
		# Update the normalisation stats: Updated Only after an episode (or more) has been rolled out
		self.norm_s.update(states)
		self.norm_g.update(goals)
		self.norm_s.recompute_stats()
		self.norm_g.recompute_stats()
	
	def set_learning_rate(self, lr=None):
		"""Update learning rate."""
		pass
	
	def load_model(self, dir_param):
		
		# BUILD First
		_ = self.model.goal_pred(np.ones([1, 3]), np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim]))
		_ = self.model.policy(np.ones([1, 3]), np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim]))
		# _ = self.model.policy(np.ones([1, self.args.g_dim]), np.ones([1, self.args.s_dim]))
		
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
		s, ag, iag, g = trans['states'], trans['achieved_goals'], trans['inter_goals'], trans['goals']
		
		# Process the states and goals
		s, ag, g = self.preprocess_og(s, ag, g)
		_, iag, _ = self.preprocess_og(s, iag, g)
		
		if self.args.use_norm:
			# Normalise before the actor-critic networks are exposed to data
			trans['states'] = self.norm_s.normalize(s)
			trans['achieved_goals'] = self.norm_g.normalize(ag)
			trans['inter_goals'] = self.norm_g.normalize(iag)
			trans['goals'] = self.norm_g.normalize(g)
		else:
			trans['states'] = s
			trans['achieved_goals'] = ag
			trans['inter_goals'] = iag
			trans['goals'] = g
		
		# Define if the transitions are from expert or not/are supervised or not
		trans['is_demo'] = tf.cast(expert, dtype=tf.int32) * tf.ones_like(trans['successes'], dtype=tf.int32)
		trans['is_sup'] = tf.cast(is_supervised, dtype=tf.int32) * tf.ones_like(trans['successes'], dtype=tf.int32)
		
		# Make sure the data is of type tf.float32
		for key in trans.keys():
			trans[key] = tf.cast(trans[key], dtype=tf.float32)
		
		return trans
	
	@tf.function
	def sample_data(self, expert_batch_size, batch_size=None):
		
		# Sample Expert Transitions
		exp_trans = self.expert_buffer.sample_transitions(expert_batch_size)
		exp_trans = self.process_data(exp_trans, tf.constant(True, dtype=tf.bool), tf.constant(True, dtype=tf.bool))
		return exp_trans
	
	def learn(self):
		args = self.args
		global_step = 0
		
		monitor_policy, monitor_goal, monitor_total = np.inf, np.inf, np.inf
		val_trans = None
		
		with tqdm(total=args.num_epochs, position=0, leave=True, desc='Training: ') as pbar:
			
			for epoch in range(args.num_epochs):
				
				for _ in range(args.n_batches):
					data = self.sample_data(expert_batch_size=tf.constant(args.expert_batch_size, dtype=tf.int32))
					
					# Train Policy on each batch
					total_loss, loss_g, loss_p = self.model.train(data)
					
					# Log the VAE Losses
					with self.summary_writer.as_default():
						tf.summary.scalar('train/loss/total', total_loss, step=global_step)
						tf.summary.scalar('train/loss/loss_g', loss_g, step=global_step)
						tf.summary.scalar('train/loss/loss_p', loss_p, step=global_step)
					
					pbar.set_postfix(Loss=total_loss.numpy(), refresh=True)
					global_step += 1
				
				pbar.update(1)
				
				# Save Last Model (post-epoch)
				self.save_model(args.dir_param)
				
				if args.do_eval:
					
					# Compute Validation Losses
					if val_trans is None:
						val_episodic_data = self.val_buffer.sample_episodes()
						# Transition Fn for val buffer giving error while tracing, so calling it out in eager mode
						val_trans = self.val_buffer.transition_fn(val_episodic_data, tf.constant(0, dtype=tf.int32))
						s, ag, ig, g = val_trans['states'], val_trans['achieved_goals'], val_trans['inter_goals'], val_trans['goals']
						s, ag, g = self.preprocess_og(s, ag, g)
						_, ig, _ = self.preprocess_og(s, ig, g)
						if self.args.use_norm:
							# Normalise before the actor-critic networks are exposed to data
							val_trans['states'] = self.norm_s.normalize(s)
							val_trans['achieved_goals'] = self.norm_g.normalize(ag)
							val_trans['inter_goals'] = self.norm_g.normalize(ig)
							val_trans['goals'] = self.norm_g.normalize(g)
						else:
							val_trans['states'] = s
							val_trans['achieved_goals'] = ag
							val_trans['inter_goals'] = ig
							val_trans['goals'] = g
						
						# Make sure the data is of type tf.float32
						for key in val_trans.keys():
							val_trans[key] = tf.cast(val_trans[key], dtype=tf.float32)
					
					val_total_loss, val_loss_g, val_loss_p = self.model.compute_loss(val_trans)
					
					# Log Validation Losses
					with self.summary_writer.as_default():
						tf.summary.scalar('val/loss/total', val_total_loss, step=epoch)
						tf.summary.scalar('val/loss/loss_g', val_loss_g, step=epoch)
						tf.summary.scalar('val/loss/loss_p', val_loss_p, step=epoch)
						
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
							# delta_G = np.linalg.norm(episode['goals'].numpy()[0] - (
							# 			episode['achieved_goals'].numpy()[0, :args.horizon] +
							# 			episode['quality'].numpy()[0]), axis=-1)  # |G_pred - G_env|
							fig_path = os.path.join(args.dir_plot, 'Distances_{}_{}.png'.format(epoch + 1, n))
							plot_metrics(
								metrics=[episode['distances'].numpy()[0], delta_AG],
								labels=['|G_env - AG_curr|', '|G_pred - AG_curr|'],
								fig_path=fig_path, y_label='Distances', x_label='Steps'
							)
						
						# Plot avg. success rate for each epoch
						with self.summary_writer.as_default():
							tf.summary.scalar('stats/Eval/Success_rate', eval_stats['success_rate'], step=epoch)
					
					# Monitor Loss for saving the best model
					if val_loss_p.numpy() < monitor_policy:
						monitor_policy = val_loss_p.numpy()
						logger.info("[POLICY] Saving the best model (best policy_loss: {}) at epoch: {}".format(monitor_policy, epoch + 1))
						self.save_model(args.dir_param + '_bestPolicy')
						
					if val_loss_g.numpy() < monitor_goal:
						monitor_goal = val_loss_g.numpy()
						logger.info("[GOAL] Saving the best model (best goal_pred_loss: {}) at epoch: {}".format(monitor_goal, epoch + 1))
						self.save_model(args.dir_param + '_bestGoal')
						
					if val_total_loss.numpy() < monitor_total:
						monitor_total = val_total_loss.numpy()
						logger.info("[TOTAL] Saving the best model (best total_loss: {}) at epoch: {}".format(monitor_total, epoch + 1))
						self.save_model(args.dir_param + '_bestTotal')


def run(args):
	args.train_demos = int(args.expert_demos * args.perc_train)
	args.val_demos = args.expert_demos - args.train_demos
	
	# ############################################# DATA LOADING ############################################# #
	# Load Buffer to store expert data
	expert_buffer = ReplayBufferTf(
		get_buffer_shape(args), args.buffer_size, args.horizon,
		sample_goal_oriented_transitions('random', args.future_gamma)
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
	
	val_buffer = ReplayBufferTf(
		get_buffer_shape(args), args.buffer_size, args.horizon,
		sample_goal_oriented_transitions('all', args.future_gamma)
	)
	
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
	if args.do_train:
		start = time.time()
		agent = Agent(args, expert_buffer, val_buffer)
		
		logger.info("Training .......")
		agent.learn()
		logger.info("Done Training in {}".format(str(datetime.timedelta(seconds=time.time() - start))))
	
	# ############################################# TESTING #################################################### #
	if args.do_verify:
		tf.config.run_functions_eagerly(True)  # To render, must run eagerly
		test_env = get_PnP_env(args)
		agent_test = Agent(args, expert_buffer, val_buffer)
		
		for testModel in ['models_bestPolicy', 'models_bestGoal']:
			print("\n------------- Verifying {} -------------".format(testModel))
			if not args.dir_test:
				dir_test = os.path.join(args.dir_root_log, testModel)
			else:
				dir_test = os.path.join(args.dir_test, testModel)
			logger.info("Loading Model Weights from {}".format(dir_test))
			agent_test.load_model(dir_param=dir_test)
			
			# Here compute_Q is an alias for computing delta_G
			test_rollout_worker = RolloutWorker(
				test_env, agent_test.model, T=args.horizon, rollout_terminate=args.rollout_terminate,
				exploit=True, noise_eps=0., random_eps=0., compute_Q=True, compute_c=True,
				use_target_net=False, render=True
			)
			
			env_state_dir = os.path.join(
				args.dir_data, '{}_env_states_train'.format(
					'two_obj_{}'.format(args.expert_behaviour) if args.two_object else 'single_obj'))
			env_state_paths = [os.path.join(env_state_dir, 'env_{}.pkl'.format(n)) for n in range(args.train_demos)]
			
			for n in range(args.test_demos):
				# if n != 1:
				# 	continue
				print('Demo: ', n)

				# Reset the policy to reset its variables [Hack: to get correct G.T. latent modes]
				agent_test.model.expert_policy.reset()
				
				with open(env_state_paths[n], 'rb') as handle:
					init_state_dict = pickle.load(handle)
					init_state_dict['goal'] = init_state_dict['goal'].numpy() if tf.is_tensor(init_state_dict['goal']) else \
						init_state_dict['goal']
				
				_episode, _ = test_rollout_worker.generate_rollout(
					slice_goal=(3, 6) if args.full_space_as_goal else None, init_state_dict=init_state_dict
				)
				print('Episode Obj Indexes: ', _episode['latent_modes'].numpy())
				
				# Monitor the metrics during each episode
				delta_AG = np.linalg.norm(_episode['quality'].numpy()[0], axis=-1)
				
				# delta_G = np.linalg.norm(_episode['goals'].numpy()[0] - (
				# 			_episode['achieved_goals'].numpy()[0, :args.horizon] + _episode['quality'].numpy()[0]), axis=-1)
				
				fig_path = os.path.join(args.dir_plot, 'Test_{}_{}.png'.format(testModel, n))
				plot_metrics(
					metrics=[_episode['distances'].numpy()[0], delta_AG, ],
					labels=['|G_env - AG_curr|', '|G_pred - AG_curr|',],
					fig_path=fig_path, y_label='Metrics', x_label='Steps'
				)

