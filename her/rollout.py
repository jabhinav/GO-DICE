import logging
import sys
import time
import pickle
import numpy as np
from domains.PnP import MyPnPEnvWrapperForGoalGAIL
from mujoco_py import MujocoException
from collections import deque
import tensorflow as tf
from utils.debug import debug

logger = logging.getLogger(__name__)


class RolloutWorker:
    def __init__(self, env: MyPnPEnvWrapperForGoalGAIL, policy, T, rollout_terminate=True, is_expert=False,
                 exploit=False, noise_eps=0., random_eps=0., compute_Q=False, use_target_net=False, render=False,
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
            compute_Q: Whether to compute Q Values
            use_target_net: whether to use the target net for action selection
            history_len (int): length of history for statistics smoothing
        """
        self.env = env
        self.is_expert = is_expert
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

        self.compute_Q = compute_Q
        self.use_target_net = use_target_net

        self.render = render
        self.resume_state = None

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
            if self.resume_state is None:
                curr_state, curr_ag, curr_g = tf.numpy_function(func=self.env.reset, inp=[self.render],
                                                                Tout=(tf.float32, tf.float32, tf.float32))
            else:
                # The env won't be reset, the environment would be at its last reached position
                curr_state = self.resume_state.copy()
                curr_ag = self.env.transform_to_goal_space(curr_state)
                curr_g = self.env.current_goal

        return curr_state, curr_ag, curr_g

    def force_reset_rollout(self, init_state_dict):
        # curr_state, curr_ag, curr_g = tf.numpy_function(func=self.env.forced_reset, inp=[init_state_dict, self.render],
        #                                                 Tout=(tf.float32, tf.float32, tf.float32))
        curr_state, curr_ag, curr_g = self.env.forced_reset(init_state_dict, self.render)
        return curr_state, curr_ag, curr_g

    @tf.function
    def generate_rollout(self, reset=True, slice_goal=None, init_state_dict=None):
        debug("generate_rollout")

        # generate episodes
        states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        achieved_goals = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        states_2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        achieved_goals_2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        goals = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        
        latent_modes = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        
        successes = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        quality = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        distances = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        # Initialize the environment
        if init_state_dict is not None:
            try:
                curr_state, curr_ag, curr_g = self.force_reset_rollout(init_state_dict)
            
            except MujocoException as _:
                logging.error("Some Error occurred while loading initial state in the environment!")
                sys.exit(-1)
            
        else:
            curr_state, curr_ag, curr_g = self.reset_rollout(reset=reset)
            init_state_dict = self.env.get_state_dict()
            
            # print(init_state_dict['goal'])
            # tf.print("Rollout: {}".format(init_state_dict['goal']))
            
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
            op = self.policy.act(curr_state, curr_ag, curr_g, compute_Q=self.compute_Q,
                                 noise_eps=self.noise_eps, random_eps=self.random_eps,
                                 use_target_net=self.use_target_net)
            if self.compute_Q:
                action, value = op
                actions = actions.write(t, tf.squeeze(tf.cast(action, dtype=tf.float32)))
                quality = quality.write(t, tf.squeeze(tf.cast(value, dtype=tf.float32)))
            else:
                if self.is_expert:
                    action, latent_mode = op
                    latent_modes = latent_modes.write(t, tf.squeeze(tf.cast(latent_mode, dtype=tf.float32)))
                else:
                    action = op
                actions = actions.write(t, tf.squeeze(tf.cast(action, dtype=tf.float32)))

            # Apply action to the environment to get next state and reward
            try:
                curr_state, curr_ag, curr_g, done, distance = tf.numpy_function(func=self.env.step,
                                                                                inp=[action, self.render],
                                                                                Tout=[tf.float32, tf.float32,
                                                                                      tf.float32, tf.int32, tf.float32])
                if self.rollout_terminate:
                    done = int(done)
                else:
                    done = 0

                states_2 = states_2.write(t, tf.squeeze(curr_state))
                achieved_goals_2 = achieved_goals_2.write(t, tf.squeeze(curr_ag))
                successes = successes.write(t, float(done))
                distances = distances.write(t, distance)

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
        self.resume_state = curr_state

        states = states.stack()
        achieved_goals = achieved_goals.stack()
        states_2 = states_2.stack()
        achieved_goals_2 = achieved_goals_2.stack()
        goals = goals.stack()
        actions = actions.stack()
        
        latent_modes = latent_modes.stack()
        
        quality = quality.stack()
        successes = successes.stack()
        distances = distances.stack()  # The distance between achieved goal and desired goal

        episode = dict(states=tf.expand_dims(states, axis=0),
                       achieved_goals=tf.expand_dims(achieved_goals, axis=0),
                       states_2=tf.expand_dims(states_2, axis=0),
                       achieved_goals_2=tf.expand_dims(achieved_goals_2, axis=0),
                       goals=tf.expand_dims(goals, axis=0),
                       actions=tf.expand_dims(actions, axis=0),
                       successes=tf.expand_dims(successes, axis=0),
                       distances=tf.expand_dims(distances, axis=0))
        if self.compute_Q:
            episode['quality'] = tf.expand_dims(quality, axis=0)
            
        if self.is_expert:
            episode['latent_modes'] = tf.expand_dims(latent_modes, axis=0)

        # success_rate = tf.reduce_mean(tf.cast(successes, tf.float32)) #
        if tf.math.equal(tf.argmax(successes), 0):  # We want to check if goal is achieved or not i.e. binary
            success = 0
        else:
            success = 1
        self.success_history.append(tf.cast(success, tf.float32))
        
        if self.compute_Q:
            self.Q_history.append(tf.reduce_mean(quality))
        
        # Log stats here to make these two functions part of computation graph of generate_rollout since *history vars
        # can't be used when calling the funcs from outside as *history vars are populated in generate_rollout's graph
        stats = {}
        success_rate = self.current_success_rate()
        stats['success_rate'] = success_rate
        if self.compute_Q:
            mean_Q = self.current_mean_Q()
            stats['mean_Q'] = mean_Q

        self.n_episodes += 1

        stats['init_state_dict'] = init_state_dict
        return episode, stats

    def clear_history(self):
        """
            Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()
        self.n_episodes = 0

    def current_success_rate(self):
        return tf.add_n(self.success_history)/tf.cast(len(self.success_history), dtype=tf.float32)

    def current_mean_Q(self):
        return tf.add_n(self.Q_history)/tf.cast(len(self.Q_history), dtype=tf.float32)
