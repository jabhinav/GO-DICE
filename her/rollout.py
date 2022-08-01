import logging
import time
import numpy as np
from domains.PnP import MyPnPEnvWrapperForGoalGAIL
from mujoco_py import MujocoException
from collections import deque
import tensorflow as tf

logger = logging.getLogger(__name__)


class RolloutWorker:
    def __init__(self, env: MyPnPEnvWrapperForGoalGAIL, policy, T, rollout_terminate=True,
                 exploit=False, noise_eps=0., random_eps=0., use_target_net=False, render=False,
                 history_len=100):
        """
        Rollout worker generates experience by interacting with policy.
        Args:
            env: List of environments.
            policy: the policy that is used to act
            T: Horizon Length
            rollout_terminate: If true, rollout is terminated when a goal is reached. Otherwise, it continues
            exploit: whether to explore (random action sampling) or exploit (greedy)
            noise_eps: scale of the additive Gaussian noise
            random_eps: probability of selecting a completely random action
            use_target_net: whether to use the target net for action selection
            history_len (int): length of history for statistics smoothing
        """
        self.env = env
        self.policy = policy
        self.horizon = T
        self.rollout_terminate = rollout_terminate
        # self.dims = dims

        self.n_episodes = 0
        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        if not exploit:
            self.noise_eps = noise_eps
            self.random_eps = random_eps
        else:
            self.noise_eps = 0.
            self.random_eps = 0.
        self.use_target_net = use_target_net

        self.render = render
        self.init_state = None

    @tf.function
    def reset_rollout(self, reset=True):
        """
            Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
            and `g` arrays accordingly.
        """
        if reset:
            curr_state, curr_ag, curr_g = tf.numpy_function(func=self.env.reset, inp=[self.render],
                                                            Tout=(tf.float32, tf.float32, tf.float32))
        else:
            if self.init_state is None:
                curr_state, curr_ag, curr_g = tf.numpy_function(func=self.env.reset, inp=[self.render],
                                                                Tout=(tf.float32, tf.float32, tf.float32))
            else:
                # The env won't be reset, the environment would be at its last reached position
                curr_state = self.init_state.copy()
                curr_ag = self.env.transform_to_goal_space(curr_state)
                curr_g = self.env.current_goal

        return curr_state, curr_ag, curr_g

    @tf.function
    def generate_rollout(self, reset=True, slice_goal=None, compute_Q=False):

        # generate episodes
        states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        achieved_goals = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        goals = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        successes = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        quality = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        # Initialize the environment
        curr_state, curr_ag, curr_g = self.reset_rollout(reset=reset)

        # Initialise other variables that will be computed
        action, value = tf.zeros(shape=(self.env.action_space.shape[0],)), tf.zeros(shape=(1,))

        for t in range(self.horizon):

            # Convert state into a batched tensor (batch size = 1)
            curr_state = tf.reshape(curr_state, shape=(1, -1))
            curr_ag = tf.reshape(curr_ag, shape=(1, -1))
            curr_g = tf.reshape(curr_g, shape=(1, -1))

            states = states.write(t, tf.squeeze(curr_state))
            achieved_goals = achieved_goals.write(t, tf.squeeze(curr_ag))
            goals = goals.write(t, tf.squeeze(curr_g))

            # Run the model and to get action probabilities and critic value
            op = self.policy.act(curr_state, curr_ag, curr_g, compute_Q=compute_Q,
                                 noise_eps=self.noise_eps, random_eps=self.random_eps,
                                 use_target_net=self.use_target_net)
            if compute_Q:
                action, value = op
            else:
                action = op
            actions = actions.write(t, tf.squeeze(tf.cast(action, dtype=tf.float32)))
            if compute_Q:
                quality = quality.write(t, tf.squeeze(tf.cast(value, dtype=tf.float32)))

            # Apply action to the environment to get next state and reward
            try:
                curr_state, curr_ag, curr_g, done = tf.numpy_function(func=self.env.step, inp=[action, self.render],
                                                                      Tout=[tf.float32, tf.float32, tf.float32, tf.int32])
                if self.rollout_terminate:
                    done = int(done)
                else:
                    done = 0
                successes = successes.write(t, done)

                # # We will save the done signals even after the episode terminates in order to maintain traj. length
                # if tf.cast(done, tf.bool):
                #     print("Terminate")
                #     break

            except MujocoException:
                self.generate_rollout(reset=True, slice_goal=slice_goal)

        # Save the terminal state and corresponding achieved goal
        states = states.write(self.horizon, tf.squeeze(curr_state))
        achieved_goals = achieved_goals.write(self.horizon, tf.squeeze(curr_ag))

        # Save the last state for the next episode [if reqd.]
        self.init_state = curr_state

        states = states.stack()
        achieved_goals = achieved_goals.stack()
        goals = goals.stack()
        actions = actions.stack()
        quality = quality.stack()
        successes = successes.stack()

        episode = dict(states=tf.expand_dims(states, axis=0),
                       achieved_goals=tf.expand_dims(achieved_goals, axis=0),
                       goals=tf.expand_dims(goals, axis=0),
                       actions=tf.expand_dims(actions, axis=0),
                       successes=tf.expand_dims(successes, axis=0))
        if compute_Q:
            episode['quality'] = tf.expand_dims(quality, axis=0)

        success_rate = tf.reduce_mean(tf.cast(successes, tf.float32))
        self.success_history.append(success_rate)
        if compute_Q:
            self.Q_history.append(tf.reduce_mean(quality))
        self.n_episodes += 1

        return episode

    def clear_history(self):
        """
            Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()
        self.n_episodes = 0

    def current_success_rate(self):
        return tf.reduce_mean(self.success_history)

    def current_mean_Q(self):
        return tf.reduce_mean(self.Q_history)