import logging
import os
from abc import ABC
from argparse import Namespace

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from her.replay_buffer import ReplayBufferTf
from networks.general import Actor
from utils.env import get_expert
from .Base import AgentBase

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
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=args.actor_lr)

        # Build Model
        self.build_model()

        # For expert Assistance
        self.use_expert_goal = False
        self.expert = get_expert(args.num_objs, args)

        # For HER
        self.use_her = False
        logger.info('[[[ Using HER ? ]]]: {}'.format(self.use_her))

        # Beta
        self.beta = self.args.BC_beta

    @tf.function(experimental_relax_shapes=True)
    def train(self, data_exp, data_rb):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(self.actor.variables)

            # On Offline Data
            actions_mu, _, _ = self.actor(tf.concat([data_rb['states'], data_rb['goals']], axis=1))
            pi_loss_offline = tf.reduce_sum(tf.math.squared_difference(data_rb['actions'], actions_mu), axis=-1)
            pi_loss_offline = tf.reduce_mean(pi_loss_offline)

            # On Expert Data
            actions_mu, _, _ = self.actor(tf.concat([data_exp['states'], data_exp['goals']], axis=1))
            pi_loss_expert = tf.reduce_sum(tf.math.squared_difference(data_exp['actions'], actions_mu), axis=-1)
            pi_loss_expert = tf.reduce_mean(pi_loss_expert)

            penalty = orthogonal_regularization(self.actor.base)
            pi_loss_w_penalty = self.beta * pi_loss_offline + (1 - self.beta) * pi_loss_expert + penalty

        grads = tape.gradient(pi_loss_w_penalty, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        return {
            'loss/pi': pi_loss_offline,
            'penalty/pi_ortho_penalty': penalty,
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
        # a_t <- f(s_t) for each skill
        _ = self.actor(tf.concat([np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim])], 1))

    def save_(self, dir_param):
        self.actor.save_weights(dir_param + "/policy.h5")

    def load_(self, dir_param):
        self.actor.load_weights(dir_param + "/policy.h5")

    def change_training_mode(self, training_mode: bool):
        pass

    def update_target_networks(self):
        pass


class Agent(AgentBase):
    def __init__(self, args,
                 expert_buffer: ReplayBufferTf = None,
                 offline_buffer: ReplayBufferTf = None):

        super(Agent, self).__init__(args, BC(args), 'BC', expert_buffer, offline_buffer)

    def store_offline_data(self, num_traj=100):
        # Store the data from the expert
        for i in range(num_traj):
            # Randomly pick epsilon
            epsilon = np.random.uniform(0, 1)
            # Randomly pick stddev (noise) for the action
            stddev = np.random.uniform(0, 0.1)
            episode, stats = self.policy_worker.generate_rollout(epsilon=epsilon, stddev=stddev)
            tf.print(f"{i + 1}/{num_traj} Episode Success: ", stats["ep_success"])

            self.offline_buffer.store_episode(episode)

        self.offline_buffer.save_buffer_data(os.path.join(self.args.dir_root_log, f'BC_{num_traj}_noisyRollouts.pkl'))

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

                # Update the reference actors and directors using polyak averaging
                if curr_t % args.update_target_interval == 0:
                    tf.print("Updating the target actors and critics at train step {}".format(curr_t))
                    self.model.update_target_networks()

                # Train the policy
                pbar.set_description('Training')
                avg_loss_dict = self.train()
                for key in avg_loss_dict.keys():
                    avg_loss_dict[key] = avg_loss_dict[key].numpy().item()

                # Log
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
            self.visualise(use_expert_skill=False)
