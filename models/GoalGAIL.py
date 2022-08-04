import os
import sys
import time
import json
import logging
import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from typing import Dict, List, Tuple, Union
from utils.env import get_PnP_env
from utils.mpi import Normalizer
from keras.layers import Dense, Flatten, Add, Concatenate, LeakyReLU, BatchNormalization
import tensorflow_probability as tfp
from domains.PnP import PnPEnv, MyPnPEnvWrapperForGoalGAIL, PnPExpert, PnPExpertTwoObj
from her.rollout import RolloutWorker
from her.replay_buffer import ReplayBuffer
from her.transitions import make_sample_her_transitions

logger = logging.getLogger(__name__)


class Actor(tf.keras.Model):
    def __init__(self, a_dim, actions_max):
        super(Actor, self).__init__()

        self.max_actions = actions_max
        self.fc1 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.fc2 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.a_out = Dense(units=a_dim, activation=tf.nn.tanh, kernel_initializer=tf.keras.initializers.GlorotUniform())

    def call(self, curr_state, goal_state):
        ip = tf.concat([curr_state, goal_state], axis=1)
        h = self.fc1(ip)
        h = self.fc2(h)
        actions = self.a_out(h) * self.max_actions
        return actions


class Critic(tf.keras.Model):
    def __init__(self, actions_max):
        super(Critic, self).__init__()

        self.max_actions = actions_max
        self.fc1 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.fc2 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.q_out = Dense(units=1, activation=None, kernel_initializer=tf.keras.initializers.GlorotUniform())

    def call(self, state, goal, actions):
        ip = tf.concat([state, goal, actions / self.max_actions], axis=1)
        h = self.fc1(ip)
        h = self.fc2(h)
        q = self.q_out(h)
        return q


class Discriminator(tf.keras.Model):
    def __init__(self, rew_type: str = 'negative'):
        super(Discriminator, self).__init__()
        kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)
        self.concat = Concatenate()
        # self.normalise = BatchNormalization(axis=0)
        self.fc1 = Dense(units=256, activation=tf.nn.tanh, kernel_initializer=kernel_init)
        self.fc2 = Dense(units=256, activation=tf.nn.tanh, kernel_initializer=kernel_init)
        self.d_out = Dense(units=1, kernel_initializer=kernel_init)

        self.rew_type: str = rew_type

    def call(self, state, goal, action):
        ip = self.concat([state, goal, action])
        # ip = self.normalise(ip)

        h = self.fc1(ip)
        h = self.fc2(h)
        d_out = self.d_out(h)
        return d_out

    def get_reward(self, state, goal, action):
        # Compute the Discriminator Output
        ip = self.concat([state, goal, action])
        # ip = self.normalise(ip)

        h = self.fc1(ip)
        h = self.fc2(h)
        d_out = self.d_out(h)

        # Convert the output into reward
        if self.rew_type == 'airl':
            return d_out
        elif self.rew_type == 'gail':
            return -tf.math.log(1 - tf.nn.sigmoid(d_out) + 1e-8)
        elif self.rew_type == 'normalized':
            return tf.nn.sigmoid(d_out)
        elif self.rew_type == 'negative':
            return tf.math.log(tf.nn.sigmoid(d_out) + 1e-8)
        else:
            print("Specify the correct reward type")
            raise NotImplementedError


class DDPG(object):
    def __init__(self, env: MyPnPEnvWrapperForGoalGAIL, args):
        self.env = env
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

        # Setup Normaliser
        self.norm_s = Normalizer(self.s_dim, self.args.eps_norm, self.args.clip_norm)
        self.norm_g = Normalizer(self.g_dim, self.args.eps_norm, self.args.clip_norm)

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
        if self.args.relative_goals:
            goal = goal - achieved_goal
        state = tf.clip_by_value(state, -self.args.clip_obs, self.args.clip_obs)
        goal = tf.clip_by_value(goal, -self.args.clip_obs, self.args.clip_obs)

        # Predict action
        if use_target_net:
            action_mu = self.actor_target(state, goal)
        else:
            action_mu = self.actor_main(state, goal)

        # # Action Post-Processing
        # First add gaussian noise and clip
        noise = noise_eps * self.args.action_max * tf.experimental.numpy.random.rand(*action_mu.shape)
        action = action_mu + tf.cast(noise, tf.float32)
        action = tf.clip_by_value(action, -self.args.action_max, self.args.action_max)

        # Take epsilon greedy action
        random_action = self._random_action(action.shape)

        action_sampler = tfp.distributions.Binomial(1, random_eps)
        choose = action_sampler.sample(sample_shape=(action.shape[0], 1))
        action += choose * (random_action - action)

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
        actor_loss = -tf.reduce_mean(main_Q_pi)

        # Add other components to actor loss
        actor_loss += self.args.l2_action_penalty * tf.reduce_mean(tf.square(main_pi / self.args.action_max))

        # # To Use Expert Demos (1): add BC Loss
        # bc_loss = data['is_demo'] * data['annealing_factor'] * tf.reduce_sum(tf.square(main_pi - data['actions']),
        #                                                                      axis=-1)
        # bc_loss = self.args.BC_Loss_coeff * tf.reduce_mean(bc_loss)

        # To Use Expert Demos (2): add BC Loss with Q-Filter i.e. I(Q > Q_pi) or (1 - I(Q_pi > Q))
        # bc_loss = data['is_demo'] * data['annealing_factor'] * tf.reduce_sum(tf.square(main_pi - data['actions']),
        #                                                                      axis=-1) * (
        #                       1 - self.args.anneal_coeff_BC * tf.cast(tf.greater_equal(main_Q_pi, main_Q),
        #                                                               dtype=tf.float32))
        bc_loss = data['is_demo'] * data['annealing_factor'] * tf.reduce_sum(tf.square(main_pi - data['actions']),
                                                                             axis=-1) * \
                  (tf.cast(tf.greater_equal(main_Q, main_Q_pi), dtype=tf.float32))
        bc_loss = self.args.BC_Loss_coeff * tf.reduce_mean(bc_loss)

        actor_loss += bc_loss

        return bc_loss, actor_loss, critic_loss

    @tf.function
    def train(self, data: Dict, normalise_obs_data=False):

        # # Normalise -> We update the mean and std of Normaliser.
        # # No need to use 'states_2'/'goals_2' to compute mean stats since their data overlaps with 'states'/'goals'
        if normalise_obs_data:
            self.norm_s.update(data['states'])
            self.norm_g.update(data['goals'])
            self.norm_s.recompute_stats()
            self.norm_g.recompute_stats()
            data['states'], data['states_2'] = self.norm_s.normalize(data['states']), self.norm_s.normalize(data['states_2'])
            data['goals'], data['goals_2'] = self.norm_s.normalize(data['goals']), self.norm_s.normalize(data['goals_2'])

        # Compute DDPG losses
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            bc_loss, actor_loss, critic_loss = self.compute_loss(data)

        gradients_actor = actor_tape.gradient(actor_loss, self.actor_main.trainable_variables)
        gradients_critic = critic_tape.gradient(critic_loss, self.critic_main.trainable_variables)
        self.a_opt.apply_gradients(zip(gradients_actor, self.actor_main.trainable_variables))
        self.c_opt.apply_gradients(zip(gradients_critic, self.critic_main.trainable_variables))
        return bc_loss, actor_loss, critic_loss

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
                 buffer_shape: Dict[str, Tuple],
                 expert_buffer: ReplayBuffer = None,
                 gail_discriminator: Discriminator = None):

        self.args = args
        self.buffer_shape = buffer_shape

        # Declare Environment for Policy
        env = get_PnP_env(args)
        self.env: MyPnPEnvWrapperForGoalGAIL = env

        # Define Policy
        self.policy = DDPG(env, args)

        # Declare Discriminator
        self.discriminator = gail_discriminator
        self.d_lr = tf.Variable(args.d_lr, trainable=False)
        self.d_opt = tf.keras.optimizers.Adam(args.d_lr)

        # Define the Transition function to map episodic data to transitional data (passed env will be used)
        sample_her_transitions_pol = make_sample_her_transitions(args.replay_strategy, args.replay_k, env.reward_fn,
                                                                 env,
                                                                 discriminator=gail_discriminator,
                                                                 gail_weight=args.gail_weight,
                                                                 two_rs=args.two_rs and args.anneal_disc,
                                                                 with_termination=args.rollout_terminate)

        # Define the Buffers
        self.init_state = None
        self.on_policy_buffer = ReplayBuffer(buffer_shape, args.buffer_size, args.horizon, sample_her_transitions_pol)
        self.expert_buffer = expert_buffer

        # ROLLOUT WORKER: It is important that we set use_target_net=False for rolling out trajectories
        # because we want actions taken using the current policu
        self.policy_rollout_worker = RolloutWorker(self.env, self.policy, T=args.horizon,
                                                   rollout_terminate=args.rollout_terminate,
                                                   exploit=False, noise_eps=args.noise_eps, random_eps=args.random_eps,
                                                   compute_Q=True, use_target_net=False, render=False)

        # # Define Losses
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # Define Tensorboard for logging Losses and Other Metrics
        if not os.path.exists(args.summary_dir):
            os.makedirs(args.summary_dir)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(args.summary_dir, current_time, 'train')
        test_log_dir = os.path.join(args.summary_dir, current_time, 'test')
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

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
        self.discriminator.save_weights(os.path.join(param_dir, "discriminator.h5"), overwrite=True)
        self.policy.save_model(param_dir)

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

            # Total loss
            d_loss = gan_loss + grad_penalty

        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return d_loss, gan_loss, grad_penalty

    def get_disc_data(self):

        def _process(_data):
            # Add noise to actions
            noise_policy = self.args.noise_eps * self.args.action_max * \
                           tf.experimental.numpy.random.rand(*_data['actions'].shape)
            noise_policy = tf.cast(noise_policy, dtype=tf.float32)
            _data['actions'] = tf.clip_by_value(_data['actions'] + noise_policy, -self.args.action_max,
                                                self.args.action_max)

            for key in _data.keys():
                _data[key] = tf.cast(_data[key], dtype=tf.float32)

            _data = [_data['states'], _data['goals'], _data['actions']]
            return _data

        sampled_data = self.on_policy_buffer.sample(self.args.disc_batch_size)
        sampled_data = _process(sampled_data)
        expert_data = self.expert_buffer.sample(self.args.disc_batch_size)
        expert_data = _process(expert_data)
        return sampled_data, expert_data

    @tf.function
    def disc_train(self):

        avg_d_loss = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        avg_gan_loss = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        avg_grad_penalty = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        avg_disc_reward_pol = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        avg_disc_reward_exp = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        for n in range(self.args.n_batches_disc):
            sampled_data, expert_data = self.get_disc_data()
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

    def _preprocess_og(self, states, achieved_goals, goals):
        if self.args.relative_goals:
            goals = goals - achieved_goals
        states = tf.clip_by_value(states, -self.args.clip_obs, self.args.clip_obs)
        goals = tf.clip_by_value(goals, -self.args.clip_obs, self.args.clip_obs)
        return states, goals

    def process_transitions(self, transitions, expert=False, annealing_factor=1., w_q2=1.):

        states, achieved_goals, goals = transitions['states'], transitions['achieved_goals'], transitions['goals']
        states_2, achieved_goals_2 = transitions['states_2'], transitions['achieved_goals_2']

        transitions['states'], transitions['goals'] = self._preprocess_og(states, achieved_goals, goals)
        transitions['states_2'], transitions['goals_2'] = self._preprocess_og(states_2, achieved_goals_2, goals)
        transitions['is_demo'] = int(expert) * tf.ones_like(transitions['rewards'])
        transitions['annealing_factor'] = annealing_factor * tf.ones_like(transitions['rewards'])
        if self.args.anneal_disc:
            transitions['rewards'] = transitions['rewards'] + w_q2 * transitions['rewards_disc']

        # Make sure the data is of type tf.float32
        for key in transitions.keys():
            transitions[key] = tf.cast(transitions[key], dtype=tf.float32)

        return transitions

    @tf.function
    def sample_data(self, batch_size, expert_batch_size, annealing_factor=1., w_q2=1.) -> Tuple[Dict, Dict]:

        policy_transitions, expert_transitions = None, None

        # Sample Policy Transitions
        if batch_size > 0:
            policy_transitions = self.on_policy_buffer.sample(batch_size)
            policy_transitions = self.process_transitions(policy_transitions, w_q2=w_q2)

        # Sample Expert Transitions
        if self.expert_buffer and expert_batch_size > 0:
            expert_transitions = self.expert_buffer.sample(batch_size)
            expert_transitions = self.process_transitions(expert_transitions, expert=True,
                                                          annealing_factor=annealing_factor, w_q2=w_q2)

        return policy_transitions, expert_transitions

    def train(self):
        args = self.args
        global_step = 0

        with tqdm(total=(args.outer_iters - 1) * args.num_epochs, position=0, leave=True) as pbar:

            for outer_iter in range(1, args.outer_iters):
                logger.info("Outer iteration: {}/{}".format(outer_iter, args.outer_iters - 1))
                annealing_factor = args.annealing_coeff ** outer_iter
                q_annealing = args.q_annealing ** (outer_iter - 1)

                for epoch in range(args.num_epochs):

                    self.policy_rollout_worker.clear_history()  # This resets Q_history and success_rate_history to 0
                    for cycle in range(args.num_cycles):

                        # ###################################### Collect Data ###################################### #
                        for ep_num in range(args.rollout_batch_size):
                            episode, pol_stats = self.policy_rollout_worker.generate_rollout(
                                slice_goal=(3, 6) if args.full_space_as_goal else None)
                            self.on_policy_buffer.store_episode(episode)

                        # ###################################### Train Policy ###################################### #
                        for _ in range(args.n_batches):
                            data_pol, data_exp = self.sample_data(batch_size=args.batch_size,
                                                                  expert_batch_size=args.expert_batch_size,
                                                                  annealing_factor=annealing_factor, w_q2=q_annealing)

                            # Combine the data: We are actually training actor and critic with policy and expert data
                            combined_data = {}
                            if data_pol is not None and data_exp is not None:
                                for key in data_pol.keys():
                                    combined_data[key] = tf.concat((data_pol[key], data_exp[key]), axis=0)
                            elif data_exp is not None:
                                combined_data = data_exp
                            else:
                                combined_data = data_pol

                            # Train Policy on each batch
                            bc_loss, a_loss, c_loss = self.policy.train(combined_data)

                        # Update Policy Target Network
                        self.policy.update_targets()

                        # Log the DDPG Losses
                        with self.train_summary_writer.as_default():
                            tf.summary.scalar('loss/Actor', a_loss, step=global_step)
                            tf.summary.scalar('loss/Actor/BC', bc_loss, step=global_step)
                            tf.summary.scalar('loss/Critic', c_loss, step=global_step)

                        # ################### Train Discriminator (per cycle of Policy Training) ################### #
                        if args.train_dis_per_rollout and args.n_batches_disc > 0 and not (
                                epoch == args.num_epochs - 1 and cycle == args.num_cycles - 1):
                            d_loss, gan_loss, grad_penalty, disc_reward_pol, disc_reward_exp = self.disc_train()

                            # Empty the on policy buffer
                            self.on_policy_buffer.clear_buffer()

                            # Log the Discriminator Loss and Rewards
                            with self.train_summary_writer.as_default():
                                tf.summary.scalar('loss/Discriminator', d_loss, step=global_step)
                                tf.summary.scalar('loss/Discriminator/GAN', gan_loss, step=global_step)
                                tf.summary.scalar('loss/Discriminator/Grad_penalty', grad_penalty, step=global_step)
                                tf.summary.scalar('reward/Disc/Policy', disc_reward_pol, step=global_step)
                                tf.summary.scalar('reward/Disc/Expert', disc_reward_exp, step=global_step)

                        global_step += 1
                    pbar.update(1)

                    # ##################### Train Discriminator (per epoch of Policy Training) ##################### #
                    if not args.train_dis_per_rollout and args.n_batches_disc > 0 and epoch != args.num_epochs - 1:
                        d_loss, gan_loss, grad_penalty, disc_reward_pol, disc_reward_exp = self.disc_train()

                        # Empty the on policy buffer
                        self.on_policy_buffer.clear_buffer()

                        # Log the Discriminator Loss and Rewards
                        with self.train_summary_writer.as_default():
                            tf.summary.scalar('loss/Discriminator', d_loss, step=global_step)
                            tf.summary.scalar('loss/Discriminator/GAN', gan_loss, step=global_step)
                            tf.summary.scalar('loss/Discriminator/Grad_penalty', grad_penalty, step=global_step)
                            tf.summary.scalar('reward/Disc/Policy', disc_reward_pol, step=global_step)
                            tf.summary.scalar('reward/Disc/Expert', disc_reward_exp, step=global_step)

                    # Log the Losses and Rewards (pol_stats carried over here carries the upto date stats)
                    curr_epoch = (outer_iter - 1)*args.num_epochs + epoch
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('stats/Mean_Q', pol_stats['mean_Q'], step=curr_epoch)
                        tf.summary.scalar('stats/Success_rate', pol_stats['success_rate'], step=curr_epoch)

                    # Save Last Model
                    self.save_model(args.param_dir)


def run(args, store_data_path=None):
    logger.info("\n\n---------------------------------------------------------------------------------------------")
    logger.info(json.dumps(vars(args), indent=4))

    # Two Object Env: target_in_the_air activates only for 2-object case
    exp_env = get_PnP_env(args)

    buffer_shape = {
        'states': (args.horizon + 1, args.s_dim),
        'achieved_goals': (args.horizon + 1, args.g_dim),
        'goals': (args.horizon, args.g_dim),
        'actions': (args.horizon, args.a_dim),
        'successes': (args.horizon,)
    }

    # Define and Load the Discriminator Network - GAIL
    gail_discriminator = Discriminator(rew_type=args.rew_type)

    # Define the Transition function to map episodic data to transitional data
    sample_her_transitions_exp = make_sample_her_transitions(args.replay_strategy, args.replay_k,
                                                             exp_env.reward_fn, exp_env,
                                                             discriminator=gail_discriminator,
                                                             gail_weight=args.gail_weight,
                                                             two_rs=args.two_rs and args.anneal_disc,
                                                             with_termination=args.rollout_terminate)

    # ###################################################### Expert ################################################# #
    start = time.time()
    logger.info("Generating {} Expert Demos.".format(args.num_demos))
    # Load Expert Policy
    if args.two_object:
        expert_policy = PnPExpertTwoObj(exp_env, args.full_space_as_goal, expert_behaviour=args.expert_behaviour)
    else:
        expert_policy = PnPExpert(exp_env, args.full_space_as_goal)
    # Load Buffer to store expert data
    expert_buffer = ReplayBuffer(buffer_shape, args.buffer_size, args.horizon, sample_her_transitions_exp)
    # Initiate a worker to generate expert rollouts
    expert_worker = RolloutWorker(exp_env, expert_policy, T=args.horizon, rollout_terminate=args.rollout_terminate,
                                  exploit=True, noise_eps=0., random_eps=0., compute_Q=False, use_target_net=False,
                                  render=False)
    # Generate and store expert data
    for i in range(args.num_demos):
        # print("\nGenerating demo:", i + 1)
        expert_worker.policy.reset()
        _episode, exp_stats = expert_worker.generate_rollout(slice_goal=(3, 6) if args.full_space_as_goal else None)
        expert_buffer.store_episode(_episode)
    logger.info("Expert Demos generated in {}.".format(str(datetime.timedelta(seconds=time.time()-start))))
    # ################################################### Store Data ############################################## #
    if store_data_path:
        expert_buffer.save_buffer_data(path=store_data_path)

    # ################################################### Train Model ############################################## #
    start = time.time()
    agent = Agent(args, buffer_shape, expert_buffer, gail_discriminator)

    if args.do_train:
        logger.info("Training .......")
        agent.train()
        logger.info("Done Training in {}".format(str(datetime.timedelta(seconds=time.time()-start))))

    if args.do_test:
        logger.info("Testing .......")
        agent.load_model(args.param_dir)

        test_env = get_PnP_env(args)
        # While testing, we do not want DDPG policy to explore since it is a deterministic policy. Thus exploit=True
        test_worker = RolloutWorker(test_env, agent.policy, T=args.horizon, rollout_terminate=args.rollout_terminate,
                                    exploit=True, noise_eps=0., random_eps=0., compute_Q=True,
                                    use_target_net=False, render=True)

        # To show policy
        for i in range(args.test_demos):
            print("\nShowing demo:", i + 1)
            _, pol_stats = test_worker.generate_rollout(slice_goal=(3, 6) if args.full_space_as_goal else None)

    sys.exit(-1)
