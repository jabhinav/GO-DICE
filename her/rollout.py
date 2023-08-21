import logging
import os
from collections import deque

import tensorflow as tf
# from mujoco_py import MujocoException

from utils.env import save_env_img
from domains.PnP import MyPnPEnvWrapper

logger = logging.getLogger(__name__)


class RolloutWorker:
    def __init__(self, env: MyPnPEnvWrapper, policy, T, rollout_terminate=False,
                 compute_Q=False, use_target_net=False, render=False, history_len=100, is_expert_worker=False):
        """
        Rollout worker generates experience by interacting with policy.
        Args:
            env: List of environments.
            policy: the policy that is used to act
            T: Horizon Length
            rollout_terminate: If true, rollout is terminated when a goal is reached. Otherwise, it continues
            # exploit: whether to explore (random action sampling) or exploit (greedy)
            # noise_eps: scale of the additive Gaussian noise
            # random_eps: probability of selecting a completely random action
            compute_Q: Whether to compute Q Values
            use_target_net: whether to use the target net for action selection
            history_len (int): length of history for statistics smoothing
        """
        self.env = env
        self.policy = policy
        self.horizon = T
        self.rollout_terminate: bool = rollout_terminate
        # self.dims = dims

        self.n_episodes = 0
        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.compute_Q = compute_Q
        self.use_target_net = use_target_net

        self.render = render
        self.resume_state = None

        self.is_expert_worker = is_expert_worker

    @tf.function
    def reset_rollout(self):
        """
            Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
            and `g` arrays accordingly.
        """
        curr_state, curr_ag, curr_g = tf.numpy_function(func=self.env.reset, inp=[self.render],
                                                        Tout=(tf.float32, tf.float32, tf.float32))

        return curr_state, curr_ag, curr_g

    def force_reset_rollout(self, resume_state_dict):
        # curr_state, curr_ag, curr_g = tf.numpy_function(func=self.env.forced_reset, inp=[init_state_dict, self.render],
        #                                                 Tout=(tf.float32, tf.float32, tf.float32))
        curr_state, curr_ag, curr_g = self.env.forced_reset(resume_state_dict, self.render)
        return curr_state, curr_ag, curr_g

    @tf.function
    def generate_rollout(self, reset=True, resume_state_dict=None, epsilon=0.0, stddev=0.0):
        # Get the tuple (s_t, s_t+1, g_t, g_t-1, c_t, c_t-1, a_t)
        prev_goals = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  # g_t-1
        prev_skills = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  # c_t-1
        states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  # s_t
        env_goal = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  # g_env
        curr_goals = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  # g_t
        curr_skills = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  # c_t
        actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  # a_t
        states_2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  # s_t+1

        successes = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        distances = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        # Initialize the environment
        # if resume_state_dict is not None:
        #     try:
        #         curr_state, _, g_env = self.force_reset_rollout(resume_state_dict)
        #         init_state_dict = resume_state_dict
        #     except Exception as _:
        #         logging.error("Some Error occurred while loading initial state in the environment!")
        #         sys.exit(-1)
        #
        # else:
        if not reset:
            raise ValueError("reset can not be False if resume_state_dict is None!")

        curr_state, _, g_env = self.reset_rollout()  # Get s_0 and g_env
        # init_state_dict = self.env.get_state_dict()

        # For Expert, reset its policy to reset its internal state
        if self.is_expert_worker:
            self.policy.reset(curr_state, g_env)
        else:
            if hasattr(self.policy, 'expert'):
                tf.numpy_function(func=self.policy.expert.reset, inp=[curr_state, g_env], Tout=[])
            else:
                logger.info("Expert guide not found for the model policy!")

        curr_goal = self.policy.get_init_goal(curr_state, g_env)  # g_0 = g_env
        # # InitSkill: Ask the actor to predict the init skill. If expert, must be called after reset
        curr_skill = self.policy.get_init_skill()

        for t in tf.range(self.horizon):
            # # Sleep to slow down the simulation
            # time.sleep(0.1)

            # # Reshape the tensors
            curr_state = tf.reshape(curr_state, shape=(1, -1))
            g_env = tf.reshape(g_env, shape=(1, -1))
            curr_goal = tf.reshape(curr_goal, shape=(1, -1))
            curr_skill = tf.reshape(curr_skill, shape=(1, -1))

            # # [[Save]] g_{t-1}, c_{t-1}, s_t, g_env
            prev_goals = prev_goals.write(t, tf.squeeze(tf.cast(curr_goal, dtype=tf.float32)))
            prev_skills = prev_skills.write(t, tf.squeeze(tf.cast(curr_skill, dtype=tf.float32)))
            states = states.write(t, tf.squeeze(curr_state))
            env_goal = env_goal.write(t, tf.squeeze(g_env))

            # # Act using the policy -> Get g_t, c_t, a_t
            curr_goal, curr_skill, action = self.policy.act(state=curr_state, env_goal=g_env,
                                                            prev_goal=curr_goal, prev_skill=curr_skill,
                                                            epsilon=epsilon, stddev=stddev)

            # # [[Save]] g_t, c_t, a_t
            curr_goals = curr_goals.write(t, tf.squeeze(tf.cast(curr_goal, dtype=tf.float32)))
            curr_skills = curr_skills.write(t, tf.squeeze(tf.cast(curr_skill, dtype=tf.float32)))
            actions = actions.write(t, tf.squeeze(tf.cast(action, dtype=tf.float32)))

            try:
                # # Transition to the next state: s_{t+1}
                curr_state, _, g_env, done, distance = tf.numpy_function(func=self.env.step,
                                                                         inp=[action, self.render],
                                                                         Tout=[tf.float32, tf.float32,
                                                                               tf.float32, tf.int32, tf.float32])

                # # [[Save]] s_{t+1}, done, distance
                states_2 = states_2.write(t, tf.squeeze(curr_state))  # s_t+1
                successes = successes.write(t, float(done))
                distances = distances.write(t, distance)

                if self.rollout_terminate and tf.cast(done, tf.bool):
                    break

            except Exception:
                self.generate_rollout(reset=True)

        # [[Save]] s_T
        states = states.write(t + 1, tf.squeeze(curr_state))
        env_goal = env_goal.write(t + 1, tf.squeeze(g_env))

        # Optional: Save C_T-1, G_T-1 (remember to add 1 to the horizon in buffer_shape)
        # prev_goals = prev_goals.write(self.horizon, tf.squeeze(tf.cast(curr_goal, dtype=tf.float32)))
        # prev_skills = prev_skills.write(self.horizon, tf.squeeze(tf.cast(curr_skill, dtype=tf.float32)))

        # Stack the arrays
        prev_goals = prev_goals.stack()
        prev_skills = prev_skills.stack()
        states = states.stack()
        env_goal = env_goal.stack()
        curr_goals = curr_goals.stack()
        curr_skills = curr_skills.stack()
        actions = actions.stack()
        states_2 = states_2.stack()
        successes = successes.stack()
        distances = distances.stack()

        episode = dict(
            prev_goals=tf.expand_dims(prev_goals, axis=0),  # (T, goal_dim)
            prev_skills=tf.expand_dims(prev_skills, axis=0),  # (T, skill_dim)
            states=tf.expand_dims(states, axis=0),  # (T+1, state_dim)
            env_goals=tf.expand_dims(env_goal, axis=0),  # (T+1, goal_dim)
            curr_goals=tf.expand_dims(curr_goals, axis=0),  # (T, goal_dim)
            curr_skills=tf.expand_dims(curr_skills, axis=0),  # (T, skill_dim)
            actions=tf.expand_dims(actions, axis=0),  # (T, action_dim)
            states_2=tf.expand_dims(states_2, axis=0),  # (T, state_dim)
            successes=tf.expand_dims(successes, axis=0),  # (T, 1)
            distances=tf.expand_dims(distances, axis=0)  # (T, 1)
        )

        # success_rate = tf.reduce_mean(tf.cast(successes, tf.float32)) #
        if tf.reduce_sum(successes) == 0:
            success = 0
        else:
            success = 1
        self.success_history.append(tf.cast(success, tf.float32))

        # Log stats here to make these two functions part of computation graph of generate_rollout since *history vars
        # can't be used when calling the funcs from outside as *history vars are populated in generate_rollout's graph
        stats = {
            # 'init_state_dict': init_state_dict,
            'ep_success': success,
            'ep_length': t + 1,
        }

        self.n_episodes += 1
        return episode, stats

    def clear_history(self):
        """
            Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()
        self.n_episodes = 0

    def current_success_rate(self):
        return tf.add_n(self.success_history) / tf.cast(len(self.success_history), dtype=tf.float32)

    def current_mean_Q(self):
        return tf.add_n(self.Q_history) / tf.cast(len(self.Q_history), dtype=tf.float32)

    def record_rollout(self, save_at, epsilon=0.0, stddev=0.0):

        curr_state, _, g_env = self.reset_rollout()  # Get s_0 and g_env

        # For Expert, reset its policy to reset its internal state
        if self.is_expert_worker:
            self.policy.reset(curr_state, g_env)
        else:
            if hasattr(self.policy, 'expert'):
                tf.numpy_function(func=self.policy.expert.reset, inp=[curr_state, g_env], Tout=[])
            else:
                logger.info("Expert guide not found for the model policy!")

        curr_goal = self.policy.get_init_goal(curr_state, g_env)  # g_0 = g_env
        # # InitSkill: Ask the actor to predict the init skill. If expert, must be called after reset
        curr_skill = self.policy.get_init_skill()

        for t in tf.range(self.horizon):
            # # Sleep to slow down the simulation
            # time.sleep(0.1)

            # Save the environment image
            save_env_img(self.env, path_to_save=os.path.join(save_at, f'{t}.png'))

            # # Reshape the tensors
            curr_state = tf.reshape(curr_state, shape=(1, -1))
            g_env = tf.reshape(g_env, shape=(1, -1))
            curr_goal = tf.reshape(curr_goal, shape=(1, -1))
            curr_skill = tf.reshape(curr_skill, shape=(1, -1))

            # # Act using the policy -> Get g_t, c_t, a_t
            curr_goal, curr_skill, action = self.policy.act(state=curr_state, env_goal=g_env,
                                                            prev_goal=curr_goal, prev_skill=curr_skill,
                                                            epsilon=epsilon, stddev=stddev)

            try:
                # # Transition to the next state: s_{t+1}
                curr_state, _, g_env, done, distance = tf.numpy_function(func=self.env.step,
                                                                         inp=[action, self.render],
                                                                         Tout=[tf.float32, tf.float32,
                                                                               tf.float32, tf.int32, tf.float32])

                if self.rollout_terminate and tf.cast(done, tf.bool):
                    break

            except Exception:
                self.generate_rollout(reset=True)
