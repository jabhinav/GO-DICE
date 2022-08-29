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
from domains.PnP import MyPnPEnvWrapperForGoalGAIL, PnPExpert, PnPExpertTwoObj
from her.rollout import RolloutWorker
from her.replay_buffer import ReplayBufferTf
from her.transitions import sample_no_her_all_transitions, sample_her_fwd_bkd_transitions
from utils.debug import debug
from networks.ccvae import Encoder, Decoder, Policy, Classifier, Conditional_Prior
from utils.vae import kl_divergence_gaussian, kl_divergence_unit_gaussian
from utils.plot import plot_metric, plot_metrics

logger = logging.getLogger(__name__)


class CCVAE(tf.keras.Model):
	def __init__(self, args, norm_s: Normalizer, norm_g: Normalizer):
		super(CCVAE, self).__init__()
		self.args = args
		self.encoder = Encoder(args.z_dim)
		self.decoder = Decoder(args.g_dim)
		# Classifier to classify encoded input, z into one of the labels, y
		self.classifier = Classifier(args.c_dim)
		# Conditional Prior to obtain Prior probability of z given y (Originally it was N(0, I))
		self.cond_prior = Conditional_Prior(args.z_dim)
		self.policy = Policy(args.a_dim, args.action_max)
		self.optimiser = tf.keras.optimizers.Adam(self.args.vae_lr)
		
		self.p_Y = tf.Variable(1 / args.c_dim, dtype=tf.float32, trainable=False)
		
		self.norm_s = norm_s
		self.norm_g = norm_g
	
	@staticmethod
	def sample_normal(mu, std, latent_dim):
		epsilon = tf.random.normal(tf.shape(std), mean=0.0, stddev=1.0, )
		z = mu + tf.math.multiply(std, epsilon)
		return tf.reshape(z, shape=[-1, latent_dim])
	
	def multi_sample_normal(self, mu, std, latent_dim, k=100):
		samples = []
		for _ in range(k):
			z = self.sample_normal(mu, std, latent_dim)
			samples.append(z)
		return samples
	
	def classifier_loss(self, data, k=100):
		[post_locs, post_scales] = self.encoder(data['states'], data['goals'])
		# Draw k samples from q(z|x) and compute log(q(y_curr|z_k)) = log(q(y|z_k))*y_curr for each.
		qy_z_k = [(lambda _z: tf.reduce_sum(tf.math.multiply(self.classifier(_z), data['latent_modes']), axis=-1))(_z) for _z in self.multi_sample_normal(post_locs, post_scales, self.args.z_dim, k)]
		qy_z_k = tf.concat(values=[tf.expand_dims(qy_z, axis=0) for qy_z in qy_z_k], axis=0)
		lqy_x = tf.math.log(tf.reduce_sum(qy_z_k, axis=0) + self.args.underflow_eps) - tf.cast(tf.math.log(float(k)), dtype=tf.float32)
		return tf.reshape(lqy_x, shape=[tf.shape(data['states'])[0], ])
	
	@tf.function
	def compute_sup_loss(self, data):
		
		# ######################################## Goal Transitioning Network ########################################
		# INFERENCE: Compute the approximate posterior q(z|s_t, G)
		[post_locs, post_scales] = self.encoder(data['states'], data['goals'])
		z_g = self.sample_normal(post_locs, post_scales, self.args.z_dim)
		
		# INFERENCE: Compute the classification prob q(y|z)
		qy_z = self.classifier(z_g)
		log_qy_z = tf.reduce_sum(tf.math.log(qy_z + self.args.underflow_eps) * data['latent_modes'], axis=-1)
		
		# INFERENCE: Compute label classification q(y|x) <- Sum_z(q(y|z)*q(z|x)) ~ 1/k * Sum(q(y|z_k))
		log_qy_x = self.classifier_loss(data)
		
		# GENERATION: Compute the Prior p(y)
		log_py = tf.reduce_sum(tf.math.log(self.p_Y + self.args.underflow_eps) * data['latent_modes'], axis=-1)
		
		# GENERATION: Compute the Conditional prior p(z|y)
		[prior_locs, prior_scales, _] = self.cond_prior(data['latent_modes'])
		
		# GENERATION: Compute the log-likelihood of actions i.e. p(delta_g|z)
		delta_g = self.decoder(z_g)
		
		# We only want gradients wrt to params of qyz, so stop them propagating to qzx! Why? Ref. Appendix C.3.1
		# In short, to reduce the variance in the gradients of classifier param! To a certain extent these gradients can
		# be viewed as redundant, as there is already gradients to update the predictive distribution due to the
		# log q(y|x) term anyway
		
		# Note: PYTORCH Detach stops the tensor from being tracked in the subsequent operations involving the tensor:
		# The original implementation is detaching the tensor z
		# log_qy_z_ = tf.stop_gradient(tf.reduce_sum(tf.log(self.classifier([z]) + self.eps) * ln_curr_encode_y_ip,
		#                                            axis=-1))
		# Compute weighted ratio
		# w = tf.exp(log_qy_z_ - log_qy_x)
		w = tf.exp(log_qy_z - log_qy_x)
		
		ll = -tf.reduce_sum(tf.math.squared_difference(data['delta_g'], delta_g), axis=-1) * 100.
		kl = kl_divergence_gaussian(mu1=post_locs, log_sigma_sq1=tf.math.log(post_scales + self.args.underflow_eps),
									mu2=prior_locs, log_sigma_sq2=tf.math.log(prior_scales + self.args.underflow_eps),
									mean_batch=False)
		
		elbo = tf.math.multiply(w, ll - kl - log_qy_z) + log_py + self.args.alpha * log_qy_x
		loss_g = -elbo
		
		# ############################################## Policy Network ###############################################
		pred_future_ag = data['achieved_goals'] + delta_g
		actions_mu = self.policy(data['states'], pred_future_ag)
		loss_p = tf.reduce_sum(tf.math.squared_difference(data['actions'], actions_mu), axis=-1)
		
		total_loss = tf.reduce_mean(loss_g + loss_p)
		
		return total_loss, tf.reduce_mean(loss_g), tf.reduce_mean(-ll), tf.reduce_mean(kl), tf.reduce_mean(-log_qy_x), \
			   tf.reduce_mean(loss_p)
	
	@tf.function
	def do_eval(self, data):
		# INFERENCE: Compute the approximate posterior q(z|s_t, G)
		[post_locs, post_scales] = self.encoder(data['states'], data['goals'])
		z_g = self.sample_normal(post_locs, post_scales, self.args.z_dim)
		
		# INFERENCE: Compute label classification q(y|x) <- Sum_z(q(y|z)*q(z|x)) ~ 1/k * Sum(q(y|z_k))
		log_qy_x = self.classifier_loss(data)
		
		# GENERATION: Compute the log-likelihood of actions i.e. p(delta_g|z)
		delta_g = self.decoder(z_g)
		
		pred_future_ag = data['achieved_goals'] + delta_g
		actions_mu = self.policy(data['states'], pred_future_ag)
		loss_p = tf.reduce_sum(tf.math.squared_difference(data['actions'], actions_mu), axis=-1)
		return tf.reduce_mean(-log_qy_x), tf.reduce_mean(loss_p)
	
	@tf.function(experimental_relax_shapes=True)
	def train(self, data):
		with tf.GradientTape() as tape:
			total_loss, vae_loss, kl_div, delta_loss, class_loss, policy_loss = self.compute_sup_loss(data)
		
		gradients = tape.gradient(total_loss, self.trainable_variables)
		self.optimiser.apply_gradients(zip(gradients, self.trainable_variables))
		return total_loss, vae_loss, kl_div, delta_loss, class_loss, policy_loss
	
	def act(self, state, achieved_goal, goal, compute_Q=False, **kwargs):
		# Pre-process the state and goals
		state = tf.clip_by_value(state, -self.args.clip_obs, self.args.clip_obs)
		achieved_goal = tf.clip_by_value(achieved_goal, -self.args.clip_obs, self.args.clip_obs)
		goal = tf.clip_by_value(goal, -self.args.clip_obs, self.args.clip_obs)
		
		# Normalise (if running stats of Normaliser are updated, we get normalised data else un-normalised data)
		if self.args.use_norm:
			state = self.norm_s.normalize(state)
			achieved_goal = self.norm_g.normalize(achieved_goal)
			goal = self.norm_g.normalize(goal)
		
		[post_locs, post_scales] = self.encoder(state, goal)
		z_g = self.sample_normal(post_locs, post_scales, self.args.z_dim)
		
		delta_g = self.decoder(z_g)
		
		# Add layer - Skip Connection
		pred_goal = achieved_goal + delta_g
		
		action_mu = self.policy(state, pred_goal)
		action = tf.clip_by_value(action_mu, -self.args.action_max, self.args.action_max)
		if compute_Q:
			return action, delta_g
		return action
	
	def load_model(self, dir_param):
		encoder_model_path = os.path.join(dir_param, "encoder_ccVAE.h5")
		if not os.path.exists(encoder_model_path):
			logger.info("Encoder Weights Not Found at {}. Exiting!".format(encoder_model_path))
			sys.exit(-1)
		
		decoder_model_path = os.path.join(dir_param, "decoder_ccVAE.h5")
		if not os.path.exists(decoder_model_path):
			logger.info("Decoder Weights Not Found at {}. Exiting!".format(decoder_model_path))
			sys.exit(-1)
		
		policy_model_path = os.path.join(dir_param, "policy_ccVAE.h5")
		if not os.path.exists(policy_model_path):
			logger.info("Policy Weights Not Found at {}. Exiting!".format(policy_model_path))
			sys.exit(-1)
		
		classifier_model_path = os.path.join(dir_param, "classifier_ccVAE.h5")
		if not os.path.exists(classifier_model_path):
			logger.info("Classifier Weights Not Found at {}. Exiting!".format(classifier_model_path))
			sys.exit(-1)
		
		condPrior_model_path = os.path.join(dir_param, "condPrior_ccVAE.h5")
		if not os.path.exists(condPrior_model_path):
			logger.info("CondPrior Weights Not Found at {}. Exiting!".format(condPrior_model_path))
			sys.exit(-1)
		
		# Load Models
		self.encoder.load_weights(encoder_model_path)
		self.decoder.load_weights(decoder_model_path)
		self.policy.load_weights(policy_model_path)
		self.classifier.load_weights(classifier_model_path)
		self.cond_prior.load_weights(condPrior_model_path)
		
		logger.info("Encoder Weights Loaded from {}.".format(encoder_model_path))
		logger.info("Decoder Weights Loaded from {}.".format(decoder_model_path))
		logger.info("Policy Weights Loaded from {}.".format(policy_model_path))
		logger.info("Classifier Weights Loaded from {}.".format(classifier_model_path))
		logger.info("CondPrior Weights Loaded from {}.".format(condPrior_model_path))
	
	def save_model(self, dir_param):
		self.encoder.save_weights(os.path.join(dir_param, "encoder_ccVAE.h5"), overwrite=True)
		self.decoder.save_weights(os.path.join(dir_param, "decoder_ccVAE.h5"), overwrite=True)
		self.policy.save_weights(os.path.join(dir_param, "policy_ccVAE.h5"), overwrite=True)
		self.classifier.save_weights(os.path.join(dir_param, "classifier_ccVAE.h5"), overwrite=True)
		self.cond_prior.save_weights(os.path.join(dir_param, "condPrior_ccVAE.h5"), overwrite=True)


class Agent(object):
	def __init__(self, args, expert_buffer: ReplayBufferTf, val_buffer: ReplayBufferTf):
		
		self.args = args
		
		# Define the Buffers
		self.transition_fn = sample_her_fwd_bkd_transitions
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
		self.model = CCVAE(args, self.norm_s, self.norm_g)
		
		# Evaluation
		self.eval_env: MyPnPEnvWrapperForGoalGAIL = get_PnP_env(args)
		self.eval_rollout_worker = RolloutWorker(self.eval_env, self.model, T=args.horizon, rollout_terminate=args.rollout_terminate, exploit=True, noise_eps=0., random_eps=0., compute_Q=True, use_target_net=False, render=False)  # Here, Q=deltaG
	
	def preprocess_og(self, states, achieved_goals, goals):
		states = tf.clip_by_value(states, -self.args.clip_obs, self.args.clip_obs)
		achieved_goals = tf.clip_by_value(achieved_goals, -self.args.clip_obs, self.args.clip_obs)
		goals = tf.clip_by_value(goals, -self.args.clip_obs, self.args.clip_obs)
		return states, achieved_goals, goals
	
	def setup_normalisers(self):
		episode_batch = self.expert_buffer.sample_episodes()
		# Sample number of transitions in the batch
		num_normalizing_transitions = episode_batch['actions'].shape[0] * episode_batch['actions'].shape[1]
		transitions = self.transition_fn(episode_batch, tf.constant(num_normalizing_transitions, dtype=tf.int32))
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
		_ = self.model.encoder(np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim]))
		_ = self.model.decoder(np.ones([1, self.args.z_dim]))
		_ = self.model.policy(np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim]))
		_ = self.model.classifier(np.ones([1, self.args.z_dim]))
		_ = self.model.cond_prior(np.ones([1, self.args.c_dim]))
		
		# Load Models
		self.model.load_model(dir_param)
	
	def save_model(self, dir_param):
		
		if not os.path.exists(dir_param):
			os.makedirs(dir_param)
		
		# Save weights
		self.model.save_model(dir_param)
	
	@tf.function
	def process_data(self, transitions, expert=False, is_supervised=False):
		
		transitions_copy = transitions.copy()
		states, achieved_goals, goals = transitions_copy['states'], transitions_copy['achieved_goals'], transitions_copy['goals']
		inter_goals = transitions_copy['inter_goals']
		
		# Process the states and goals
		states, achieved_goals, goals = self.preprocess_og(states, achieved_goals, goals)
		_, inter_goals, _ = self.preprocess_og(states, inter_goals, goals)
		
		if self.args.use_norm:
			# Normalise before the actor-critic networks are exposed to data
			transitions_copy['states'] = self.norm_s.normalize(states)
			transitions_copy['achieved_goals'] = self.norm_g.normalize(achieved_goals)
			transitions_copy['inter_goals'] = self.norm_g.normalize(inter_goals)
			transitions_copy['goals'] = self.norm_g.normalize(goals)
		else:
			transitions_copy['states'] = states
			transitions_copy['achieved_goals'] = achieved_goals
			transitions_copy['inter_goals'] = inter_goals
			transitions_copy['goals'] = goals
		
		# Define if the transitions are from expert or not/are supervised or not
		transitions_copy['is_demo'] = tf.cast(expert, dtype=tf.int32) * tf.ones_like(transitions_copy['successes'], dtype=tf.int32)
		transitions_copy['is_sup'] = tf.cast(is_supervised, dtype=tf.int32) * tf.ones_like(
			transitions_copy['successes'],
			dtype=tf.int32)
		
		# Compute the difference between current achieved goal and a future goal to achieve
		transitions_copy['delta_g'] = transitions_copy['inter_goals'] - transitions_copy['achieved_goals']
		
		# Make sure the data is of type tf.float32
		for key in transitions_copy.keys():
			transitions_copy[key] = tf.cast(transitions_copy[key], dtype=tf.float32)
		
		return transitions_copy
	
	@tf.function
	def sample_data(self, expert_batch_size, batch_size=None):
		
		# Sample Expert Transitions
		expert_transitions = self.expert_buffer.sample_transitions(expert_batch_size)
		processed_transitions = self.process_data(expert_transitions, expert=tf.constant(True, dtype=tf.bool), is_supervised=tf.constant(True, dtype=tf.bool))
		return processed_transitions
	
	def learn(self):
		args = self.args
		global_step = 0
		
		monitor_metric = np.inf
		val_trans = None
		
		with tqdm(total=args.num_epochs, position=0, leave=True, desc='Training: ') as pbar:
			
			for epoch in range(args.num_epochs):
				
				for _ in range(args.n_batches):
					data = self.sample_data(expert_batch_size=tf.constant(args.expert_batch_size, dtype=tf.int32))
					
					# Train Policy on each batch
					train_losses = self.model.train(data)
					total_loss, vae_loss, kl_div, delta_loss, class_loss, policy_loss = train_losses
					
					# Log the VAE Losses
					with self.summary_writer.as_default():
						tf.summary.scalar('train/loss/total', total_loss, step=global_step)
						tf.summary.scalar('train/loss/vae', vae_loss, step=global_step)
						tf.summary.scalar('train/loss/vae/kl_div', kl_div, step=global_step)
						tf.summary.scalar('train/loss/vae/delta_ag', delta_loss, step=global_step)
						tf.summary.scalar('train/loss/vae/classifier', class_loss, step=global_step)
						tf.summary.scalar('train/loss/policy', policy_loss, step=global_step)
					
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
						s, ag, g = val_trans['states'], val_trans['achieved_goals'], val_trans['goals']
						s, ag, g = self.preprocess_og(s, ag, g)
						if self.args.use_norm:
							# Normalise before the actor-critic networks are exposed to data
							val_trans['states'] = self.norm_s.normalize(s)
							val_trans['achieved_goals'] = self.norm_g.normalize(ag)
							val_trans['goals'] = self.norm_g.normalize(g)
						else:
							val_trans['states'] = s
							val_trans['achieved_goals'] = ag
							val_trans['goals'] = g
						
						# Make sure the data is of type tf.float32
						for key in val_trans.keys():
							val_trans[key] = tf.cast(val_trans[key], dtype=tf.float32)
					
					val_class_loss, val_policy_loss = self.model.do_eval(val_trans)
					
					# Log Validation Losses
					with self.summary_writer.as_default():
						tf.summary.scalar('val/loss/vae/classifier', val_class_loss, step=epoch)
						tf.summary.scalar('val/loss/policy', val_policy_loss, step=epoch)
					
					if (epoch + 1) % args.log_interval == 0:
						
						# # Clear the worker's history to avoid retention of poor perf. during early training
						# self.eval_rollout_worker.clear_history()
						
						# Do rollouts using learned Policy
						for n in range(args.eval_demos):
							episode, eval_stats = self.eval_rollout_worker.generate_rollout(
								slice_goal=(3, 6) if args.full_space_as_goal else None)
							
							# Monitor the metrics during each episode
							delta_AG = np.linalg.norm(episode['quality'].numpy()[0], axis=-1)
							delta_G = np.linalg.norm(episode['goals'].numpy()[0] - (episode['achieved_goals'].numpy()[0, :args.horizon] + episode['quality'].numpy()[0]), axis=-1)
							fig_path = os.path.join(args.dir_plot, 'Distances_{}_{}.png'.format(epoch + 1, n))
							plot_metrics(metrics=[episode['distances'].numpy()[0], delta_AG, delta_G], labels=['|G_env - AG_curr|', '|G_pred - AG_curr|', '|G_pred - G_env|'], fig_path=fig_path, y_label='Distances', x_label='Steps')
						
						# Plot avg. success rate for each epoch
						with self.summary_writer.as_default():
							tf.summary.scalar('stats/Eval/Success_rate', eval_stats['success_rate'], step=epoch)
					
					# Monitor Loss for saving the best model
					if val_policy_loss.numpy() < monitor_metric:
						monitor_metric = val_policy_loss.numpy()
						logger.info("Saving the best model (best policy_loss: {}) at epoch: {}".format(monitor_metric, epoch + 1))
						self.save_model(args.dir_param + '_best')


def run(args):
	# Two Object Env: target_in_the_air activates only for 2-object case
	exp_env = get_PnP_env(args)
	
	# ############################################# EXPERT POLICY ############################################# #
	if args.two_object:
		expert_policy = PnPExpertTwoObj(exp_env, args.full_space_as_goal, expert_behaviour=args.expert_behaviour)
	else:
		expert_policy = PnPExpert(exp_env, args.full_space_as_goal)
	
	# Initiate a worker to generate expert rollouts
	expert_worker = RolloutWorker(exp_env, expert_policy, T=args.horizon, rollout_terminate=args.rollout_terminate, compute_c=True, exploit=True, noise_eps=0., random_eps=0., compute_Q=False, use_target_net=False, render=False)
	
	# ############################################# DATA TRAINING ############################################# #
	# Load Buffer to store expert data
	expert_buffer = ReplayBufferTf(get_buffer_shape(args), args.buffer_size, args.horizon, sample_her_fwd_bkd_transitions)
	num_train_episodes = int(args.expert_demos * args.perc_train)
	
	train_data_path = os.path.join(args.dir_data, '{}_train.pkl'.format('two_obj_{}'.format(args.expert_behaviour) if args.two_object else 'single_obj'))
	env_state_dir = os.path.join(args.dir_data, '{}_env_states_train'.format('two_obj_{}'.format(args.expert_behaviour) if args.two_object else 'single_obj'))
	if not os.path.exists(env_state_dir):
		os.makedirs(env_state_dir)
	
	if not os.path.exists(train_data_path):
		exp_stats = {'success_rate': 0.}
		logger.info("Generating {} Expert Demos for training.".format(num_train_episodes))
		
		# Generate and store expert training data
		for i in range(num_train_episodes):
			# Expert Policy Needs to be reset everytime
			expert_worker.policy.reset()
			_episode, exp_stats = expert_worker.generate_rollout(slice_goal=(3, 6) if args.full_space_as_goal else None)
			expert_buffer.store_episode(_episode)

			path_to_init_state_dict = tf.constant(os.path.join(env_state_dir, 'env_{}.pkl'.format(i)))
			with open(path_to_init_state_dict.numpy(), 'wb') as handle:
				# init_state_dict = expert_worker.env.get_state_dict()  # Call it from outside since when run in
				# non-eager mode, exp_stats store the init_state_dict generated during tracing.
				# Else generate data in execution mode
				pickle.dump(exp_stats['init_state_dict'], handle, protocol=pickle.HIGHEST_PROTOCOL)
			
		expert_buffer.save_buffer_data(train_data_path)
		logger.info("Saved Expert Demos at {} training.".format(train_data_path))
		
		plot_metric(_episode['distances'].numpy()[0], fig_path=os.path.join(args.dir_root_log, 'Sample(Expert).png'), y_label='Distances', x_label='Steps')
		logger.info("Expert Policy Success Rate: {}".format(exp_stats['success_rate']))
	
	else:
		logger.info("Loading Expert Demos from {} into TrainBuffer for training.".format(train_data_path))
		expert_buffer.load_data_into_buffer(train_data_path)
	
	# ############################################# DATA VALIDATION ############################################# #
	val_buffer = ReplayBufferTf(get_buffer_shape(args), args.buffer_size, args.horizon, sample_no_her_all_transitions)
	num_val_episodes = args.expert_demos - num_train_episodes
	
	val_data_path = os.path.join(args.dir_data, '{}_val.pkl'.format('two_obj_{}'.format(args.expert_behaviour) if args.two_object else 'single_obj'))
	env_state_dir = os.path.join(args.dir_data, '{}_env_states_val'.format('two_obj_{}'.format(args.expert_behaviour) if args.two_object else 'single_obj'))
	if not os.path.exists(env_state_dir):
		os.makedirs(env_state_dir)
	
	if not os.path.exists(val_data_path) and num_val_episodes:
		logger.info("Generating {} Expert Demos for validation.".format(num_val_episodes))
		
		# Generate and store expert validation data
		for i in range(num_val_episodes):
			# Expert Policy Needs to be reset everytime
			expert_worker.policy.reset()
			_episode, exp_stats = expert_worker.generate_rollout(slice_goal=(3, 6) if args.full_space_as_goal else None)
			val_buffer.store_episode(_episode)
			
			path_to_init_state_dict = tf.constant(os.path.join(env_state_dir, 'env_{}.pkl'.format(i)))
			with open(path_to_init_state_dict.numpy(), 'wb') as handle:
				pickle.dump(exp_stats['init_state_dict'], handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		val_buffer.save_buffer_data(val_data_path)
		logger.info("Saved Expert Demos at {} for validation.".format(val_data_path))
	
	elif os.path.exists(val_data_path) and num_val_episodes:
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
		
		if not args.dir_test:
			args.dir_test = os.path.join(args.dir_root_log, 'models_best')
		logger.info("Loading Model Weights from {}".format(args.dir_test))
		agent_test.load_model(dir_param=args.dir_test)
		
		test_rollout_worker = RolloutWorker(test_env, agent_test.model, T=args.horizon, rollout_terminate=args.rollout_terminate, exploit=True, noise_eps=0., random_eps=0., compute_Q=True, use_target_net=False, render=True)
		
		# To test on the initial environment states, model saw during training
		env_state_dir = os.path.join(args.dir_data, '{}_env_states_train'.format('two_obj_{}'.format(args.expert_behaviour) if args.two_object else 'single_obj'))
		env_state_paths = [os.path.join(env_state_dir, 'env_{}.pkl'.format(n)) for n in range(args.test_demos)]
		
		for n in range(args.test_demos):
			
			with open(env_state_paths[n], 'rb') as handle:
				init_state_dict = pickle.load(handle)
				init_state_dict['goal'] = init_state_dict['goal'].numpy() if tf.is_tensor(init_state_dict['goal']) else init_state_dict['goal']
			
			_episode, _ = test_rollout_worker.generate_rollout(slice_goal=(3, 6) if args.full_space_as_goal else None, init_state_dict=init_state_dict)
			
			# Compute the cosine similarity metric
			indices = tf.stack((tf.zeros(50, dtype=tf.int32), tf.range(50, dtype=tf.int32)), axis=-1)
			g = tf.gather_nd(_episode['goals'], indices)
			ag = tf.gather_nd(_episode['achieved_goals'], indices)
			deltag = g - ag
			deltag_norm = tf.math.l2_normalize(deltag, axis=-1)
			
			deltag_pred = tf.gather_nd(_episode['quality'], indices)
			deltag_pred_norm = tf.math.l2_normalize(deltag_pred, axis=-1)
			cosine_sim = tf.reduce_sum(tf.multiply(deltag_norm, deltag_pred_norm), axis=-1)
			
			# Monitor the metrics during each episode
			delta_AG = np.linalg.norm(_episode['quality'].numpy()[0], axis=-1)
			
			delta_G = np.linalg.norm(_episode['goals'].numpy()[0] - (_episode['achieved_goals'].numpy()[0, :args.horizon] + _episode['quality'].numpy()[0]), axis=-1)
			
			fig_path = os.path.join(args.dir_plot, 'TestMetrics_{}.png'.format(n))
			plot_metrics(metrics=[_episode['distances'].numpy()[0], delta_AG, delta_G, cosine_sim.numpy()], labels=['|G_env - AG_curr|', '|G_pred - AG_curr|', '|G_pred - G_env|', 'cos(G_pred - AG_curr, G_env - AG_curr)'], fig_path=fig_path, y_label='Metrics', x_label='Steps')
	
	sys.exit(-1)
