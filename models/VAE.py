import os
import sys
import time
import json
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
from her.transitions import sample_random_consecutive_transitions, sample_all_consecutive_transitions
from utils.debug import debug
from networks.vae import Encoder, Decoder
from utils.vae import kl_divergence_gaussian, kl_divergence_unit_gaussian
from utils.plot import plot_metric, plot_metrics

logger = logging.getLogger(__name__)


class ClassicVAE(tf.keras.Model):
    def __init__(self, args, norm_s: Normalizer, norm_g: Normalizer):
        super(ClassicVAE, self).__init__()
        self.args = args
        self.encoder = Encoder(args.g_dim)
        self.decoder = Decoder(args.a_dim, args.action_max)
        self.optimiser = tf.keras.optimizers.Adam(self.args.vae_lr)

        self.norm_s = norm_s
        self.norm_g = norm_g

    @staticmethod
    def sample_normal(mu, std, latent_dim):
        epsilon = tf.random.normal(tf.shape(std), mean=0.0, stddev=1.0, )
        z = mu + tf.math.multiply(std, epsilon)
        return tf.reshape(z, shape=[-1, latent_dim])

    @tf.function
    def compute_loss(self, data):
        # INFERENCE: Compute the approximate posterior q(z|x)
        [post_locs, post_scales] = self.encoder(data['achieved_goals'], data['states'], data['goals'])
        delta_g = self.sample_normal(post_locs, post_scales, self.args.g_dim)
    
        # Add layer - Skip Connection
        pred_goals = data['achieved_goals'] + delta_g
    
        # GENERATION: Compute the log-likelihood of actions i.e. p(x|z)
        action_mus = self.decoder(data['states'], pred_goals)
    
        # Obj: max (ELBO = Log_Likelihood-KL_Divergence)
        ll = -tf.reduce_sum(tf.math.squared_difference(action_mus, data['actions']), axis=-1)
        kl = kl_divergence_unit_gaussian(mu=post_locs, log_sigma_sq=tf.math.log(post_scales + self.args.underflow_eps),
                                         mean_batch=False)
        elbo = ll - kl
        loss = tf.reduce_mean(-elbo)
        return loss, tf.reduce_mean(-ll), tf.reduce_mean(kl)

    @tf.function(experimental_relax_shapes=True)
    def train(self, data):
        with tf.GradientTape() as tape:
            loss, act_loss, kl_div = self.compute_loss(data)
    
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, act_loss, kl_div
    
    def act(self, state, achieved_goal, goal, compute_Q=False, **kwargs):
        # Pre-process the state and goals
        state = tf.clip_by_value(state, -self.args.clip_obs, self.args.clip_obs)
        achieved_goal = tf.clip_by_value(achieved_goal, -self.args.clip_obs, self.args.clip_obs)
        goal = tf.clip_by_value(goal, -self.args.clip_obs, self.args.clip_obs)
    
        # Normalise (if running stats of Normaliser are updated, we get normalised data else un-normalised data)
        state = self.norm_s.normalize(state)
        achieved_goal = self.norm_g.normalize(achieved_goal)
        goal = self.norm_g.normalize(goal)

        [post_locs, post_scales] = self.encoder(achieved_goal, state, goal)
        delta_g = self.sample_normal(post_locs, post_scales, self.args.g_dim)

        # Add layer - Skip Connection
        pred_goals = achieved_goal + delta_g

        action_mu = self.decoder(state, pred_goals)
        action = tf.clip_by_value(action_mu, -self.args.action_max, self.args.action_max)
        if compute_Q:
            return action, delta_g
        return action
    
    def load_model(self, param_dir):
        encoder_model_path = os.path.join(param_dir, "encoder_classicVAE.h5")
        if not os.path.exists(encoder_model_path):
            logger.info("Encoder Weights Not Found at {}. Exiting!".format(encoder_model_path))
            sys.exit(-1)
    
        decoder_model_path = os.path.join(param_dir, "decoder_classicVAE.h5")
        if not os.path.exists(decoder_model_path):
            logger.info("Decoder Weights Not Found at {}. Exiting!".format(decoder_model_path))
            sys.exit(-1)
    
        # Load Models
        self.encoder.load_weights(encoder_model_path)
        self.decoder.load_weights(decoder_model_path)
    
        logger.info("Encoder Weights Loaded from {}.".format(encoder_model_path))
        logger.info("Decoder Weights Loaded from {}.".format(decoder_model_path))
    
    def save_model(self, param_dir):
        self.encoder.save_weights(os.path.join(param_dir, "encoder_classicVAE.h5"), overwrite=True)
        self.decoder.save_weights(os.path.join(param_dir, "decoder_classicVAE.h5"), overwrite=True)
        

class Agent(object):
    def __init__(self, args, expert_buffer: ReplayBufferTf, val_buffer: ReplayBufferTf):

        self.args = args

        # Define the Buffers
        self.transition_fn = sample_random_consecutive_transitions
        self.expert_buffer = expert_buffer
        self.val_buffer = val_buffer

        # Define Tensorboard for logging Losses and Other Metrics
        if not os.path.exists(args.summary_dir):
            os.makedirs(args.summary_dir)
            
        if not os.path.exists(args.plot_dir):
            os.makedirs(args.plot_dir)
        self.summary_writer = tf.summary.create_file_writer(args.summary_dir)
        
        # Setup Normalisers
        self.norm_s = Normalizer(args.s_dim, args.eps_norm, args.clip_norm)
        self.norm_g = Normalizer(args.g_dim, args.eps_norm, args.clip_norm)
        self.setup_normalisers()
        
        # Declare Model
        self.model = ClassicVAE(args, self.norm_s, self.norm_g)

        # Evaluation
        self.eval_env: MyPnPEnvWrapperForGoalGAIL = get_PnP_env(args)
        self.eval_rollout_worker = RolloutWorker(self.eval_env, self.model, T=args.horizon,
                                                 rollout_terminate=args.rollout_terminate,
                                                 exploit=True, noise_eps=0., random_eps=0.,
                                                 compute_Q=True, use_target_net=False, render=False)  # Here, Q=deltaG
        
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

    def load_model(self, param_dir):

        # BUILD First
        _ = self.model.encoder(np.ones([1, self.args.g_dim]), np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim]))
        _ = self.model.decoder(np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim]))

        # Load Models
        self.model.load_model(param_dir)

    def save_model(self, param_dir):

        if not os.path.exists(param_dir):
            os.makedirs(param_dir)

        # Save weights
        self.model.save_model(param_dir)

    @tf.function
    def process_data(self, transitions, expert=False, is_supervised=False, normalise=True):
        
        transitions_copy = transitions.copy()
        states, achieved_goals, goals = transitions_copy['states'], transitions_copy['achieved_goals'], transitions_copy['goals']
        
        # Process the states and goals
        states, achieved_goals, goals = self.preprocess_og(states, achieved_goals, goals)
    
        if normalise:
            # Normalise before the actor-critic networks are exposed to data
            transitions_copy['states'] = self.norm_s.normalize(states)
            transitions_copy['achieved_goals'] = self.norm_g.normalize(achieved_goals)
            transitions_copy['goals'] = self.norm_g.normalize(goals)
        else:
            transitions_copy['states'] = states
            transitions_copy['achieved_goals'] = achieved_goals
            transitions_copy['goals'] = goals
    
        # Define if the transitions are from expert or not/are supervised or not
        transitions_copy['is_demo'] = tf.cast(expert, dtype=tf.int32) * tf.ones_like(transitions_copy['successes'],
                                                                                     dtype=tf.int32)
        transitions_copy['is_sup'] = tf.cast(is_supervised, dtype=tf.int32) * tf.ones_like(transitions_copy['successes'],
                                                                                           dtype=tf.int32)
    
        # Make sure the data is of type tf.float32
        for key in transitions_copy.keys():
            transitions_copy[key] = tf.cast(transitions_copy[key], dtype=tf.float32)
    
        return transitions_copy

    @tf.function
    def sample_data(self, expert_batch_size, batch_size=None):
    
        # Sample Expert Transitions
        expert_transitions = self.expert_buffer.sample_transitions(expert_batch_size)
        processed_transitions = self.process_data(expert_transitions, expert=tf.constant(True, dtype=tf.bool),
                                                  is_supervised=tf.constant(True, dtype=tf.bool))
        return processed_transitions

    def learn(self):
        args = self.args
        global_step = 0
        
        monitor_metric = np.inf
        val_transitions = None
        
        with tqdm(total=args.num_epochs, position=0, leave=True, desc='Training: ') as pbar:

            for epoch in range(args.num_epochs):
                
                for _ in range(args.n_batches):
                    data = self.sample_data(expert_batch_size=tf.constant(args.expert_batch_size, dtype=tf.int32))

                    # Train Policy on each batch
                    loss, act_loss, kl_div = self.model.train(data)

                    # Log the VAE Losses
                    with self.summary_writer.as_default():
                        tf.summary.scalar('loss/train/total', loss, step=global_step)
                        tf.summary.scalar('loss/train/act_loss', act_loss, step=global_step)
                        tf.summary.scalar('loss/train/kl_div', kl_div, step=global_step)

                    pbar.set_postfix(Loss=loss.numpy(), refresh=True)
                    global_step += 1
                
                pbar.update(1)

                # Save Last Model (post-epoch)
                self.save_model(args.param_dir)

                if args.do_eval:
                    
                    # Compute Validation Losses
                    if val_transitions is None:
                        val_episodic_data = self.val_buffer.sample_episodes()
                        # Transition Fn for val buffer giving error while tracing, so calling it out in eager mode
                        val_transitions = self.val_buffer.transition_fn(val_episodic_data, tf.constant(0, dtype=tf.int32))
                        val_transitions = self.process_data(val_transitions, expert=tf.constant(True, dtype=tf.bool),
                                                            is_supervised=tf.constant(True, dtype=tf.bool))
                    val_loss, act_loss, kl_div = self.model.compute_loss(val_transitions)
                    
                    # Log Validation Losses
                    with self.summary_writer.as_default():
                        tf.summary.scalar('loss/val/total', val_loss, step=epoch)
                        tf.summary.scalar('loss/val/act_loss', act_loss, step=epoch)
                        tf.summary.scalar('loss/val/kl_div', kl_div, step=epoch)

                    if (epoch + 1) % args.log_interval == 0:
                        
                        # Do rollouts using VAE Policy
                        for n in range(args.eval_demos):
                            episode, eval_stats = self.eval_rollout_worker.generate_rollout(
                                slice_goal=(3, 6) if args.full_space_as_goal else None)
        
                            fig_path = os.path.join(args.plot_dir, 'Distances_{}_{}.png'.format(epoch + 1, n))
                            delta_ag = np.linalg.norm(episode['quality'].numpy()[0], axis=-1)
                            plot_metrics(metrics=[episode['distances'].numpy()[0], delta_ag],
                                         labels=['|AG-G|', 'delta_AG'], fig_path=fig_path,
                                         y_label='Distances', x_label='Steps')
                            with self.summary_writer.as_default():
                                tf.summary.scalar('stats/Eval/Success_rate', eval_stats['success_rate'], step=epoch)
                        
                    # Monitor Loss for saving the best model
                    if act_loss.numpy() < monitor_metric:
                        monitor_metric = act_loss.numpy()
                        logger.info("Saving the best model (best act_loss: {}) at epoch: {}".format(monitor_metric,
                                                                                                    epoch+1))
                        self.save_model(args.param_dir + '_best')


def run(args, store_data_path=None):
    
    # Two Object Env: target_in_the_air activates only for 2-object case
    exp_env = get_PnP_env(args)

    # Load Expert Policy
    if args.two_object:
        expert_policy = PnPExpertTwoObj(exp_env, args.full_space_as_goal, expert_behaviour=args.expert_behaviour)
    else:
        expert_policy = PnPExpert(exp_env, args.full_space_as_goal)
    
    # Load Buffer to store expert data
    expert_buffer = ReplayBufferTf(get_buffer_shape(args), args.buffer_size, args.horizon, sample_random_consecutive_transitions)
    val_buffer = ReplayBufferTf(get_buffer_shape(args), args.buffer_size, args.horizon, sample_all_consecutive_transitions)
    
    # Initiate a worker to generate expert rollouts
    expert_worker = RolloutWorker(exp_env, expert_policy, T=args.horizon, rollout_terminate=args.rollout_terminate,
                                  exploit=True, noise_eps=0., random_eps=0., compute_Q=False, use_target_net=False,
                                  render=False)
    
    if args.do_train:
        start = time.time()
        exp_stats = {'success_rate': 0.}
        logger.info("Generating {} Expert Demos.".format(args.expert_demos))
        
        num_train_episodes = int(args.expert_demos * args.perc_train)
        # Generate and store expert training data
        for i in range(num_train_episodes):
            expert_worker.policy.reset()
            _episode, exp_stats = expert_worker.generate_rollout(slice_goal=(3, 6) if args.full_space_as_goal else None)
            expert_buffer.store_episode(_episode)

        # Generate and store expert validation data
        for i in range(args.expert_demos - num_train_episodes):
            expert_worker.policy.reset()
            _episode, exp_stats = expert_worker.generate_rollout(slice_goal=(3, 6) if args.full_space_as_goal else None)
            val_buffer.store_episode(_episode)

        plot_metric(_episode['distances'].numpy()[0], fig_path=os.path.join(args.root_log_dir, 'Sample(Expert).png'),
                    y_label='Distances', x_label='Steps')
        logger.info("Expert Demos generated in {}.".format(str(datetime.timedelta(seconds=time.time() - start))))
        logger.info("Expert Policy Success Rate: {}".format(exp_stats['success_rate']))

        if store_data_path:
            expert_buffer.save_buffer_data(path=store_data_path)

        start = time.time()
        agent = Agent(args, expert_buffer, val_buffer)

        logger.info("Training .......")
        agent.learn()
        logger.info("Done Training in {}".format(str(datetime.timedelta(seconds=time.time() - start))))

    sys.exit(-1)
