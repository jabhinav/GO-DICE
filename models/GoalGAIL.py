import os
import sys
import time
import json
import logging
import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from typing import Dict, Tuple

from networks.actor_critic import Actor, Critic
from networks.discriminator import Discriminator
from utils.env import get_PnP_env
from utils.normalise import Normalizer
from utils.buffer import get_buffer_shape
import tensorflow_probability as tfp
from domains.PnP import MyPnPEnvWrapperForGoalGAIL, PnPExpert, PnPExpertTwoObj
from her.rollout import RolloutWorker
from her.replay_buffer import ReplayBufferTf
from her.transitions import make_sample_her_transitions_tf
from utils.debug import debug

logger = logging.getLogger(__name__)


class DDPG(object):
    def __init__(self, args, transition_fn):
        self.args = args
        self.a_dim: int = args.a_dim
        self.s_dim: int = args.s_dim
        self.g_dim: int = args.g_dim

        # Declare Networks
        self.actor_main = Actor(self.a_dim, args.action_max)
        self.actor_target = Actor(self.a_dim, args.action_max)
        self.critic_main = Critic(args.action_max)
        self.critic_target = Critic(args.action_max)

        # Define Optimisers
        self.a_lr = tf.Variable(args.a_lr, trainable=False)
        self.a_opt = tf.keras.optimizers.Adam(args.a_lr)
        self.c_lr = tf.Variable(args.c_lr, trainable=False)
        self.c_opt = tf.keras.optimizers.Adam(args.c_lr)

        # Build Models
        self.build_model()

        # Compile the target networks so that no errors are thrown while copying weights
        self.actor_target.compile(optimizer=self.a_opt)
        self.critic_target.compile(optimizer=self.c_opt)

        # # Setup Normaliser
        self.norm_s = Normalizer(args.s_dim, args.eps_norm, args.clip_norm)
        self.norm_g = Normalizer(args.g_dim, args.eps_norm, args.clip_norm)

        # # Setup Policy's Buffer
        self.transition_fn = transition_fn
        self.buffer = ReplayBufferTf(get_buffer_shape(args), args.buffer_size, args.horizon, transition_fn)

    def set_learning_rate(self, actor_learning_rate=None, critic_learning_rate=None):
        """Update learning rate."""
        if actor_learning_rate:
            self.a_lr.assign(actor_learning_rate)
        if critic_learning_rate:
            self.c_lr.assign(critic_learning_rate)

    def compute_Qsga(self, state, goal, action_mu, use_target_network=False):
        """
            This will be called after action is computed and state, goal are processed
        """
        if use_target_network:
            Q = self.critic_target(state, goal, action_mu)
        else:
            Q = self.critic_main(state, goal, action_mu)
        return Q

    def act(self, state, achieved_goal, goal, noise_eps=0., random_eps=0.,
            use_target_net=False, compute_Q=False, **kwargs):

        # Pre-process the state and goal
        state, goal = self.preprocess_og(state, achieved_goal, goal)

        # Normalise (if running stats of Normaliser are updated, we get normalised data else un-normalised data)
        state = self.norm_s.normalize(state)
        goal = self.norm_g.normalize(goal)

        # Predict action
        if use_target_net:
            action_mu = self.actor_target(state, goal)
        else:
            action_mu = self.actor_main(state, goal)

        # # Action Post-Processing
        # First add gaussian noise and clip to get the predicted action
        noise = noise_eps * self.args.action_max * tf.experimental.numpy.random.randn(*action_mu.shape)
        action = action_mu + tf.cast(noise, tf.float32)
        action = tf.clip_by_value(action, -self.args.action_max, self.args.action_max)

        # Then take epsilon greedy act i.e. act = take_random_act_coeff*random_act + (1-take_random_act_coeff)*pred_act
        random_action = self._random_action(action.shape)
        random_action_prob = tfp.distributions.Binomial(1, probs=random_eps)
        take_random_act_coeff = random_action_prob.sample(sample_shape=(action.shape[0], 1))
        action += take_random_act_coeff * (random_action - action)

        if compute_Q:
            Q = self.compute_Qsga(state, goal, action_mu, use_target_net)
            return action, Q
        return action

    def _random_action(self, shape):
        return tf.random.uniform(shape, -self.args.action_max, self.args.action_max)

    def update_targets(self, tau=None):
        if tau is None:
            tau = self.args.polyak_tau

        # Update target networks using Polyak averaging.
        new_actor_weights = [tau * trg + (1 - tau) * main for trg, main in zip(self.actor_target.get_weights(),
                                                                               self.actor_main.get_weights())]
        self.actor_target.set_weights(new_actor_weights)
        new_critic_weights = [tau * trg + (1 - tau) * main for trg, main in zip(self.critic_target.get_weights(),
                                                                                self.critic_main.get_weights())]
        self.critic_target.set_weights(new_critic_weights)

    def compute_loss(self, data):

        # ################### Critic Loss: (Q(s, g, a) - (r + gamma * Q(s', g, a') * (1-done)))^2 ################### #
        target_pi = self.args.action_max * self.actor_target(data['states_2'], data['goals_2'])  # a' <- Pi(s', g')
        target_Q_pi = self.critic_target(data['states_2'], data['goals_2'], target_pi / self.args.action_max)
        target_Q_pi = tf.squeeze(target_Q_pi, axis=1)

        target_values = data['rewards'] + self.args.gamma * target_Q_pi * (1 - data['successes'])

        if self.args.rew_type == 'negative':
            # Since our Disc. rewards are <= 0 and completion rew. is either 0 or 1.
            # The max. return with terminate_bootstrapping we can get is 1. while least could be -inf.'
            target_values = tf.clip_by_value(target_values, -np.inf, 1.)
        elif self.args.rew_type == 'gail' or self.args.rew_type == 'normalized':
            target_values = tf.clip_by_value(target_values, 0., np.inf)
        else:
            target_values = tf.clip_by_value(target_values, -np.inf, np.inf)

        main_Q = self.critic_main(data['states'], data['goals'], data['actions'] / self.args.action_max)
        main_Q = tf.squeeze(main_Q, axis=1)
        critic_loss = tf.keras.losses.MSE(tf.stop_gradient(target_values), main_Q)

        # ########################################## Actor Loss: -Q(s,g,a) ######################################### #
        main_pi = self.args.action_max * self.actor_main(data['states'], data['goals'])
        main_Q_pi = self.critic_main(data['states'], data['goals'], main_pi / self.args.action_max)
        main_Q_pi = tf.squeeze(main_Q_pi, axis=1)
        actor_loss = -tf.reduce_mean(main_Q_pi) * self.args.Actor_Loss_coeff

        # Add other components to actor loss
        l2_action_penalty = self.args.l2_action_penalty * tf.reduce_mean(tf.square(main_pi / self.args.action_max))

        # # To Use Expert Demos (1): add BC Loss
        # bc_loss = data['is_demo'] * data['annealing_factor'] * tf.reduce_sum(tf.square(main_pi - data['actions']),
        #                                                                      axis=-1)
        # bc_loss = self.args.BC_Loss_coeff * tf.reduce_mean(bc_loss)

        # To Use Expert Demos (2): add BC Loss with Q-Filter i.e. I(Q > Q_pi) or (1 - I(Q_pi > Q))
        # bc_loss = data['is_demo'] * data['annealing_factor'] * tf.reduce_sum(tf.square(main_pi - data['actions']),
        #                                                                      axis=-1) * (
        #                       1 - self.args.anneal_coeff_BC * tf.cast(tf.greater_equal(main_Q_pi, main_Q),
        #                                                               dtype=tf.float32))
        bc_loss_q_filter = data['is_demo'] \
                           * data['annealing_factor'] \
                           * tf.reduce_sum(tf.square(main_pi - data['actions']), axis=-1) \
                           * (tf.cast(tf.greater_equal(main_Q, main_Q_pi), dtype=tf.float32))
        bc_loss_q_filter = self.args.BC_Loss_coeff * tf.reduce_mean(bc_loss_q_filter)

        actor_loss = actor_loss + l2_action_penalty + bc_loss_q_filter

        return bc_loss_q_filter, actor_loss, critic_loss

    @tf.function(experimental_relax_shapes=True)
    def train(self, data: Dict):
        debug("train_ddpg")

        # Compute DDPG losses
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            bc_loss, actor_loss, critic_loss = self.compute_loss(data)

        gradients_actor = actor_tape.gradient(actor_loss, self.actor_main.trainable_variables)
        gradients_critic = critic_tape.gradient(critic_loss, self.critic_main.trainable_variables)
        self.a_opt.apply_gradients(zip(gradients_actor, self.actor_main.trainable_variables))
        self.c_opt.apply_gradients(zip(gradients_critic, self.critic_main.trainable_variables))
        return bc_loss, actor_loss, critic_loss

    def preprocess_og(self, states, achieved_goals, goals):
        if self.args.relative_goals:
            goals = goals - achieved_goals
        states = tf.clip_by_value(states, -self.args.clip_obs, self.args.clip_obs)
        goals = tf.clip_by_value(goals, -self.args.clip_obs, self.args.clip_obs)
        return states, goals

    def store_episode(self, episode_batch, update_stats=True):
        """
            episode_batch: array of batch_size x (T or T+1) x dim_key
                           'states/achieved_goals' is of size T+1, others are of size T
        """
        self.buffer.store_episode(episode_batch=episode_batch)
        if update_stats:
            # Sample number of transitions in the batch
            num_normalizing_transitions = episode_batch['actions'].shape[0] * episode_batch['actions'].shape[1]
            transitions = self.transition_fn(episode_batch, tf.constant(num_normalizing_transitions, dtype=tf.int32))
            # Preprocess the states and goals
            states, achieved_goals, goals = transitions['states'], transitions['achieved_goals'], transitions['goals']
            states, goals = self.preprocess_og(states, achieved_goals, goals)
            # Update the normalisation stats: Updated Only after an episode (or more) has been rolled out
            self.norm_s.update(states)
            self.norm_g.update(goals)
            self.norm_s.recompute_stats()
            self.norm_g.recompute_stats()

    def build_model(self):
        # BUILD First
        _ = self.actor_main(np.ones([1, self.s_dim]), np.ones([1, self.g_dim]))
        _ = self.actor_target(np.ones([1, self.s_dim]), np.ones([1, self.g_dim]))
        _ = self.critic_main(np.ones([1, self.s_dim]), np.ones([1, self.g_dim]), np.ones([1, self.a_dim]))
        _ = self.critic_target(np.ones([1, self.s_dim]), np.ones([1, self.g_dim]), np.ones([1, self.a_dim]))

    def load_model(self, param_dir):

        actor_model_path = os.path.join(param_dir, "actor_main.h5")
        if not os.path.exists(actor_model_path):
            logger.info("Actor Weights Not Found at {}. Exiting!".format(actor_model_path))
            sys.exit(-1)

        critic_model_path = os.path.join(param_dir, "critic_main.h5")
        if not os.path.exists(critic_model_path):
            logger.info("Critic Weights Not Found at {}. Exiting!".format(critic_model_path))
            sys.exit(-1)

        # Load Models
        self.actor_main.load_weights(actor_model_path)
        self.critic_main.load_weights(critic_model_path)
        self.actor_target.load_weights(actor_model_path)
        self.critic_target.load_weights(critic_model_path)

        logger.info("Actor Weights Loaded from {}.".format(actor_model_path))
        logger.info("Critic Weights Loaded from {}.".format(critic_model_path))

    def save_model(self, param_dir):
        # Save weights
        self.actor_main.save_weights(os.path.join(param_dir, "actor_main.h5"), overwrite=True)
        self.critic_main.save_weights(os.path.join(param_dir, "critic_main.h5"), overwrite=True)


class Agent(object):
    def __init__(self,
                 args,
                 expert_buffer: ReplayBufferTf = None,
                 gail_discriminator: Discriminator = None):

        self.args = args

        # Declare Environment for Policy
        self.train_env: MyPnPEnvWrapperForGoalGAIL = get_PnP_env(args)
        self.eval_env: MyPnPEnvWrapperForGoalGAIL = get_PnP_env(args)

        # Declare Discriminator
        self.discriminator = gail_discriminator
        self.d_lr = tf.Variable(args.d_lr, trainable=False)
        self.d_opt = tf.keras.optimizers.Adam(args.d_lr)

        # Define the Transition function to map episodic data to transitional data (passed env will be used)
        sample_her_transitions_pol = make_sample_her_transitions_tf(args.replay_strategy, args.replay_k,
                                                                    reward_fun=self.train_env.reward_fn,
                                                                    goal_weight=self.train_env.goal_weight,
                                                                    discriminator=gail_discriminator,
                                                                    gail_weight=args.gail_weight,
                                                                    terminal_eps=self.train_env.terminal_eps,
                                                                    two_rs=args.two_rs and args.anneal_disc,
                                                                    with_termination=args.rollout_terminate)

        # # Define the Buffers
        self.init_state = None
        # This On-policy buffer is for Discriminator Training
        self.on_policy_buffer = ReplayBufferTf(get_buffer_shape(args), args.buffer_size, args.horizon,
                                               sample_her_transitions_pol)
        self.expert_buffer = expert_buffer

        # Define Policy
        self.policy = DDPG(args, transition_fn=sample_her_transitions_pol)

        # ROLLOUT WORKERS
        self.policy_rollout_worker = RolloutWorker(self.train_env, self.policy, T=args.horizon,
                                                   rollout_terminate=args.rollout_terminate,
                                                   exploit=False, noise_eps=args.noise_eps, random_eps=args.random_eps,
                                                   compute_Q=True, use_target_net=False, render=False)
        self.eval_rollout_worker = RolloutWorker(self.eval_env, self.policy, T=args.horizon,
                                                 rollout_terminate=args.rollout_terminate,
                                                 exploit=True, noise_eps=0., random_eps=0.,
                                                 compute_Q=True, use_target_net=False, render=False)

        # # Define Losses
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # Define Tensorboard for logging Losses and Other Metrics
        if not os.path.exists(args.summary_dir):
            os.makedirs(args.summary_dir)
        self.summary_writer = tf.summary.create_file_writer(args.summary_dir)

    def set_learning_rate(self, disc_learning_rate=None, actor_learning_rate=None, critic_learning_rate=None):
        """Update learning rate."""
        if disc_learning_rate:
            self.d_lr.assign(disc_learning_rate)
        self.policy.set_learning_rate(actor_learning_rate, critic_learning_rate)

    def load_model(self, param_dir):
        _ = self.discriminator(np.ones([1, self.args.s_dim]),
                               np.ones([1, self.args.g_dim]),
                               np.ones([1, self.args.a_dim]))

        # Load Models
        if self.args.use_disc:
            disc_path = os.path.join(param_dir, "discriminator.h5")
            if not os.path.exists(disc_path):
                logger.info("Discriminator Weights Not Found at {}. Exiting!".format(disc_path))
                sys.exit(-1)
            self.discriminator.load_weights(os.path.join(param_dir, "discriminator.h5"))
            logger.info("Discriminator Weights Loaded from {}.".format(disc_path))
        self.policy.load_model(param_dir)

    def save_model(self, param_dir):

        if not os.path.exists(param_dir):
            os.makedirs(param_dir)

        # Save weights
        if self.args.use_disc:
            self.discriminator.save_weights(os.path.join(param_dir, "discriminator.h5"), overwrite=True)
        self.policy.save_model(param_dir)

    def sample_data_disc(self):
        """
        For Discriminator, we sample equal number of transitions from expert and on_policy_buffer
        """

        def _process(_data):
            # Add gaussian noise to actions
            noise = self.args.noise_eps * self.args.action_max * tf.experimental.numpy.random.randn(
                *_data['actions'].shape)
            noise = tf.cast(noise, dtype=tf.float32)
            _data['actions'] = tf.clip_by_value(_data['actions'] + noise, -self.args.action_max, self.args.action_max)

            for key in _data.keys():
                _data[key] = tf.cast(_data[key], dtype=tf.float32)

            _data = [_data['states'], _data['goals'], _data['actions']]
            return _data

        sampled_data = self.on_policy_buffer.sample_transitions(tf.constant(self.args.disc_batch_size, dtype=tf.int32))
        sampled_data = _process(sampled_data)
        expert_data = self.expert_buffer.sample_transitions(tf.constant(self.args.disc_batch_size, dtype=tf.int32))
        expert_data = _process(expert_data)
        return sampled_data, expert_data

    def process_data(self, transitions, expert=False, annealing_factor=1., w_q2=1.):

        states, achieved_goals, goals = transitions['states'], transitions['achieved_goals'], transitions['goals']
        states_2, achieved_goals_2 = transitions['states_2'], transitions['achieved_goals_2']

        # Process the states and goals
        transitions['states'], transitions['goals'] = self.policy.preprocess_og(states, achieved_goals, goals)
        transitions['states_2'], transitions['goals_2'] = self.policy.preprocess_og(states_2, achieved_goals_2, goals)

        # Normalise before the actor-critic networks are exposed to data
        transitions['states'] = self.policy.norm_s.normalize(transitions['states'])
        transitions['states_2'] = self.policy.norm_s.normalize(transitions['states_2'])
        transitions['goals'] = self.policy.norm_g.normalize(transitions['goals'])
        transitions['goals_2'] = self.policy.norm_g.normalize(transitions['goals_2'])

        # Define if the transitions are from expert or not
        transitions['is_demo'] = tf.cast(expert, dtype=tf.int32) * tf.ones_like(transitions['rewards'], dtype=tf.int32)

        # Define the annealing factor for the transitions
        transitions['annealing_factor'] = annealing_factor * tf.ones_like(transitions['rewards'])

        # Compute the total rewards (in case multiple-components of it exists)
        if self.args.two_rs:
            if self.args.use_disc:
                if self.args.anneal_disc:
                    transitions['rewards'] = transitions['rewards'] + w_q2 * transitions['rewards_disc']
                else:
                    transitions['rewards'] = transitions['rewards'] + transitions['rewards_disc']

        # Make sure the data is of type tf.float32
        for key in transitions.keys():
            transitions[key] = tf.cast(transitions[key], dtype=tf.float32)

        return transitions

    @tf.function  # Make sure all the passed parameters are tensor flow constants to avoid retracing
    def sample_data_pol(self, batch_size, expert_batch_size, annealing_factor=1., w_q2=1.):
        """
        For training DDPG, we sample transitions from policy's off-policy buffer?
        WHY? Because we do want to overfit to bad data and DDPG is off-policy
        """
        debug("Sample_Policy_Train_Data")

        # Sample Policy Transitions
        policy_transitions = self.policy.buffer.sample_transitions(batch_size)
        policy_transitions = self.process_data(policy_transitions, expert=tf.constant(False, dtype=tf.bool), w_q2=w_q2)

        # Sample Expert Transitions
        expert_transitions = self.expert_buffer.sample_transitions(expert_batch_size)
        expert_transitions = self.process_data(expert_transitions, tf.constant(True, dtype=tf.bool),
                                               annealing_factor=annealing_factor, w_q2=w_q2)

        # Combine the data: We are actually training actor and critic with policy and expert data
        combined_data = {}
        for key in policy_transitions.keys():
            combined_data[key] = tf.concat((policy_transitions[key], expert_transitions[key]), axis=0)
        return combined_data

    def compute_disc_loss(self, sampled_data, expert_data):

        with tf.GradientTape() as disc_tape:
            # Probability that sampled data is from expert
            fake_output = self.discriminator(*sampled_data)
            fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)  # Prob=0: sampled data from expert

            # Prob that expert data is from expert
            real_output = self.discriminator(*expert_data)
            real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)  # Prob=1: expert data from expert

            gan_loss = fake_loss + real_loss

            # Compute gradient penalty (GP replaces weight clipping with a constraint on the gradient norm of the critic
            # to enforce Lipschitz continuity) <- Inspired by WGAN-GP paper
            alpha = tf.random.uniform(shape=(sampled_data[0].shape[0], 1), minval=0, maxval=1)
            inter_data = [alpha * sampled_i + (1 - alpha) * expert_i
                          for sampled_i, expert_i in zip(sampled_data, expert_data)]
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(inter_data)
                inter_output = self.discriminator(*inter_data)
            grads = gp_tape.gradient(inter_output, [inter_data])[0]  # This will grads w.r.t to 3 components of ip s,g,a
            grads = tf.concat(grads, axis=-1)
            grad_penalty = self.args.lambd * tf.reduce_mean(tf.pow(tf.norm(grads, axis=-1) - 1, 2))

            # # # Total loss # # #
            d_loss = gan_loss + grad_penalty

        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return d_loss, gan_loss, grad_penalty

    @tf.function(experimental_relax_shapes=True)
    def train_disc(self):
        debug("train_disc")

        # For Book-keeping
        avg_d_loss = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        avg_gan_loss = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        avg_grad_penalty = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        avg_disc_reward_pol = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        avg_disc_reward_exp = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        for n in range(self.args.n_batches_disc):
            sampled_data, expert_data = self.sample_data_disc()
            d_loss, gan_loss, grad_penalty = self.compute_disc_loss(sampled_data, expert_data)

            # For Book-keeping
            batched_disc_reward_pol = tf.reduce_mean(self.discriminator.get_reward(*sampled_data))
            batched_disc_reward_exp = tf.reduce_mean(self.discriminator.get_reward(*expert_data))
            avg_d_loss = avg_d_loss.write(n, d_loss)
            avg_gan_loss = avg_gan_loss.write(n, gan_loss)
            avg_grad_penalty = avg_grad_penalty.write(n, grad_penalty)
            avg_disc_reward_pol = avg_disc_reward_pol.write(n, batched_disc_reward_pol)
            avg_disc_reward_exp = avg_disc_reward_exp.write(n, batched_disc_reward_exp)

        avg_d_loss = tf.reduce_mean(avg_d_loss.stack())
        avg_gan_loss = tf.reduce_mean(avg_gan_loss.stack())
        avg_grad_penalty = tf.reduce_mean(avg_grad_penalty.stack())
        avg_disc_reward_pol = tf.reduce_mean(avg_disc_reward_pol.stack())
        avg_disc_reward_exp = tf.reduce_mean(avg_disc_reward_exp.stack())

        return avg_d_loss, avg_gan_loss, avg_grad_penalty, avg_disc_reward_pol, avg_disc_reward_exp

    def learn(self):
        args = self.args
        global_step = 0

        with tqdm(total=(args.outer_iters - 1) * args.num_epochs, position=0, leave=True) as pbar:

            for outer_iter in range(1, args.outer_iters):

                logger.info("Outer iteration: {}/{}".format(outer_iter, args.outer_iters - 1))
                annealing_factor = args.annealing_coeff ** outer_iter
                q_annealing = args.q_annealing ** (outer_iter - 1)

                for epoch in range(args.num_epochs):

                    self.policy_rollout_worker.clear_history()  # This resets Q_history and success_rate_history to 0
                    self.eval_rollout_worker.clear_history()

                    for cycle in range(args.num_cycles):

                        # ###################################### Collect Data ###################################### #
                        for ep_num in range(args.rollout_batch_size):
                            episode, pol_stats = self.policy_rollout_worker.generate_rollout(
                                slice_goal=(3, 6) if args.full_space_as_goal else None)
                            self.policy.store_episode(episode)  # Off-Policy Buffer
                            self.on_policy_buffer.store_episode(episode)  # On-Policy Buffer for Disc

                        # ###################################### Train Policy ###################################### #
                        for _ in range(args.n_batches):
                            data = self.sample_data_pol(batch_size=tf.constant(args.batch_size, dtype=tf.int32),
                                                        expert_batch_size=tf.constant(args.expert_batch_size, dtype=tf.int32),
                                                        annealing_factor=tf.constant(annealing_factor, dtype=tf.float32),
                                                        w_q2=tf.constant(q_annealing, dtype=tf.float32))

                            # Train Policy on each batch
                            bc_loss, a_loss, c_loss = self.policy.train(data)

                        # Update Policy Target Network
                        self.policy.update_targets()

                        # Log the DDPG Losses
                        with self.summary_writer.as_default():
                            tf.summary.scalar('loss/Actor', a_loss, step=global_step)
                            tf.summary.scalar('loss/Actor/BC', bc_loss, step=global_step)
                            tf.summary.scalar('loss/Critic', c_loss, step=global_step)

                        # ################### Train Discriminator (per cycle of Policy Training) ################### #
                        if self.discriminator and args.train_dis_per_rollout and args.n_batches_disc > 0 and \
                                not (epoch == args.num_epochs - 1 and cycle == args.num_cycles - 1):
                            d_loss, gan_loss, grad_penalty, disc_reward_pol, disc_reward_exp = self.train_disc()

                            # Empty the on policy buffer
                            self.on_policy_buffer.clear_buffer()

                            # Log the Discriminator Loss and Rewards
                            with self.summary_writer.as_default():
                                tf.summary.scalar('loss/Discriminator', d_loss, step=global_step)
                                tf.summary.scalar('loss/Discriminator/GAN', gan_loss, step=global_step)
                                tf.summary.scalar('loss/Discriminator/Grad_penalty', grad_penalty, step=global_step)
                                tf.summary.scalar('reward/Disc/Policy', disc_reward_pol, step=global_step)
                                tf.summary.scalar('reward/Disc/Expert', disc_reward_exp, step=global_step)

                        global_step += 1
                    pbar.update(1)

                    # ##################### Train Discriminator (per epoch of Policy Training) ##################### #
                    if self.discriminator and not args.train_dis_per_rollout and args.n_batches_disc > 0 and \
                            epoch != args.num_epochs - 1:
                        d_loss, gan_loss, grad_penalty, disc_reward_pol, disc_reward_exp = self.train_disc()

                        # Empty the on policy buffer
                        self.on_policy_buffer.clear_buffer()

                        # Log the Discriminator Loss and Rewards
                        with self.summary_writer.as_default():
                            tf.summary.scalar('loss/Discriminator', d_loss, step=global_step)
                            tf.summary.scalar('loss/Discriminator/GAN', gan_loss, step=global_step)
                            tf.summary.scalar('loss/Discriminator/Grad_penalty', grad_penalty, step=global_step)
                            tf.summary.scalar('reward/Disc/Policy', disc_reward_pol, step=global_step)
                            tf.summary.scalar('reward/Disc/Expert', disc_reward_exp, step=global_step)

                    # Log the Losses and Rewards (pol_stats carried over here carries the upto date stats)
                    curr_epoch = (outer_iter - 1) * args.num_epochs + epoch
                    with self.summary_writer.as_default():
                        tf.summary.scalar('stats/Train/Mean_Q', pol_stats['mean_Q'], step=curr_epoch)
                        tf.summary.scalar('stats/Train/Success_rate', pol_stats['success_rate'], step=curr_epoch)

                    # Save Last Model (post-epoch)
                    self.save_model(args.param_dir)

                    if args.do_eval:
                        for _ in range(args.eval_demos):
                            _, eval_stats = self.eval_rollout_worker.generate_rollout(
                                slice_goal=(3, 6) if args.full_space_as_goal else None)
                        with self.summary_writer.as_default():
                            tf.summary.scalar('stats/Eval/Mean_Q', eval_stats['mean_Q'], step=curr_epoch)
                            tf.summary.scalar('stats/Eval/Success_rate', eval_stats['success_rate'], step=curr_epoch)


def run(args, store_data_path=None):

    # Two Object Env: target_in_the_air activates only for 2-object case
    exp_env = get_PnP_env(args)

    # Define and Load the Discriminator Network - GAIL
    if args.use_disc:
        gail_discriminator = Discriminator(rew_type=args.rew_type)
    else:
        gail_discriminator = None

    # Define the Transition function to map episodic data to transitional data
    sample_her_transitions_exp = make_sample_her_transitions_tf(args.replay_strategy, args.replay_k,
                                                                reward_fun=exp_env.reward_fn,
                                                                goal_weight=exp_env.goal_weight,
                                                                discriminator=gail_discriminator,
                                                                gail_weight=args.gail_weight,
                                                                terminal_eps=exp_env.terminal_eps,
                                                                two_rs=args.two_rs and args.anneal_disc,
                                                                with_termination=args.rollout_terminate)

    # Load Expert Policy
    if args.two_object:
        expert_policy = PnPExpertTwoObj(exp_env, args.full_space_as_goal, expert_behaviour=args.expert_behaviour)
    else:
        expert_policy = PnPExpert(exp_env, args.full_space_as_goal)
    # Load Buffer to store expert data
    expert_buffer = ReplayBufferTf(get_buffer_shape(args), args.buffer_size, args.horizon, sample_her_transitions_exp)
    # Initiate a worker to generate expert rollouts
    expert_worker = RolloutWorker(exp_env, expert_policy, T=args.horizon, rollout_terminate=args.rollout_terminate,
                                  exploit=True, noise_eps=0., random_eps=0., compute_Q=False, use_target_net=False,
                                  render=False)
    # ################################################### Train Model ############################################## #
    # ###################################################### Expert ################################################# #
    if args.do_train:
        start = time.time()
        exp_stats = {'success_rate': 0.}
        logger.info("Generating {} Expert Demos.".format(args.expert_demos))
        # Generate and store expert data
        for i in range(args.expert_demos):
            # print("\nGenerating demo:", i + 1)
            expert_worker.policy.reset()
            _episode, exp_stats = expert_worker.generate_rollout(slice_goal=(3, 6) if args.full_space_as_goal else None)
            expert_buffer.store_episode(_episode)

        logger.info("Expert Demos generated in {}.".format(str(datetime.timedelta(seconds=time.time() - start))))
        logger.info("Expert Policy Success Rate: {}".format(exp_stats['success_rate']))

        # ################################################### Store Data ############################################# #
        if store_data_path:
            expert_buffer.save_buffer_data(path=store_data_path)

        start = time.time()
        agent = Agent(args, expert_buffer, gail_discriminator)

        logger.info("Training .......")
        agent.learn()
        logger.info("Done Training in {}".format(str(datetime.timedelta(seconds=time.time() - start))))

    if args.do_test:

        agent = Agent(args, expert_buffer, gail_discriminator)
        logger.info("Testing .......")

        # TODO: Save and Load the Normaliser stats
        agent.load_model(args.test_param_dir)

        test_env = get_PnP_env(args)
        # While testing, we do not want DDPG policy to explore since it is a deterministic policy. Thus exploit=True
        test_worker = RolloutWorker(test_env, agent.policy, T=args.horizon, rollout_terminate=args.rollout_terminate,
                                    exploit=True, noise_eps=0., random_eps=0., compute_Q=True,
                                    use_target_net=False, render=True)

        # HACK: Generate and store expert data to compute normalisation stats for Policy
        for i in range(args.expert_demos):
            # print("\nGenerating demo:", i + 1)
            expert_worker.policy.reset()
            _episode, exp_stats = expert_worker.generate_rollout(slice_goal=(3, 6) if args.full_space_as_goal else None)
            agent.policy.store_episode(_episode)

        # To show policy
        for i in range(args.test_demos):
            print("\nShowing demo:", i + 1)
            ep, pol_stats = test_worker.generate_rollout(slice_goal=(3, 6) if args.full_space_as_goal else None)

    sys.exit(-1)
