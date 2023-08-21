import logging
from abc import ABC
from argparse import Namespace

import tensorflow as tf
from tqdm import tqdm

from utils.env import get_expert
from her.replay_buffer import ReplayBufferTf
from .Base import AgentBase

logger = logging.getLogger(__name__)


class Expert(tf.keras.Model, ABC):
    def __init__(self, args: Namespace):
        super(Expert, self).__init__()
        self.args = args

        # For expert
        self.expert = get_expert(args.num_objs, args)

        # For HER
        self.use_her = False
        logger.info('[[[ Using HER ? ]]]: {}'.format(self.use_her))

    def act(self, state, env_goal, prev_goal, prev_skill, epsilon, stddev):
        state = tf.clip_by_value(state, -self.args.clip_obs, self.args.clip_obs)
        env_goal = tf.clip_by_value(env_goal, -self.args.clip_obs, self.args.clip_obs)
        prev_goal = tf.clip_by_value(prev_goal, -self.args.clip_obs, self.args.clip_obs)

        # ###################################### Current Goal ####################################### #
        curr_goal = env_goal

        # ###################################### Current Skill ###################################### #
        curr_skill = tf.numpy_function(
            self.expert.sample_curr_skill,
            [state[0], env_goal[0], prev_skill[0]],
            tf.float32
        )
        curr_skill = tf.expand_dims(curr_skill, axis=0)

        # ########################################## Action ######################################### #
        # Explore
        if tf.random.uniform(()) < epsilon:
            action = tf.random.uniform((1, self.args.a_dim), -self.args.action_max, self.args.action_max)
        # Exploit
        else:
            action = tf.numpy_function(
                self.expert.sample_action,
                [state[0], env_goal[0], curr_skill[0]],
                tf.float32
            )
            action = tf.expand_dims(action, axis=0)

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

    def save_(self, dir_param):
        pass

    def load_(self, dir_param):
        pass

    def change_training_mode(self, training_mode: bool):
        pass

    def update_target_networks(self):
        pass


class Agent(AgentBase):
    def __init__(self, args,
                 expert_buffer: ReplayBufferTf = None,
                 offline_buffer: ReplayBufferTf = None):

        super(Agent, self).__init__(args, Expert(args), 'Expert', expert_buffer, offline_buffer)

    def load_actor(self, dir_param):
        pass

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

                # Train the policy
                pbar.set_description('Expert Training.')

                # Log
                if self.args.log_wandb:
                    self.wandb_logger.log({
                        'policy_buffer_size': self.offline_buffer.get_current_size_trans(),
                        'expert_buffer_size': self.expert_buffer.get_current_size_trans(),
                    }, step=log_step)

                # Update
                pbar.update(1)
                log_step += 1

        if args.test_demos > 0:
            self.visualise(use_expert_skill=False)
