from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import os.path
import random
import sys
from abc import ABC
from itertools import permutations

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from typing import List
from utils import one_hot_encode
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils as tf_utils
from tf_agents.environments import wrappers
from tf_agents.specs import array_spec
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory, time_step
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.trajectories import time_step as ts
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import network, encoding_network
from tf_agents.utils import common

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_debug = 0


def sample_xy(valid_posns: List):
    picked_pos = random.choice(valid_posns)
    return picked_pos[0], picked_pos[1]


def _find(state, _id):
    x, y = np.where(state[:, :, _id])
    assert len(x) == len(y) == 1
    return [x[0], y[0]]


def debug_data(trajectories: trajectory.Trajectory):
    s = trajectories.observation['grid_world'].numpy()[:, 0, :, :, :]
    s_prime = trajectories.observation['grid_world'].numpy()[:, 1, :, :, :]
    reward = trajectories.reward.numpy()[:, 0]
    action = trajectories.action.numpy()[:, 0]

    pos_rew_id = np.where(reward == 100.)[0]

    for _id in range(128):
        if _find(s[_id], 0) == _find(s[_id], 4):
            print("\n--------- Init State is Terminal")
            print("Agent_t: ", _find(s[_id], 0))
            print("Goal_t: ", _find(s[_id], 4))

            print("Action_t: ", action[_id])
            print("Agent_t+1: ", _find(s_prime[_id], 0))
            print("Goal_t+1: ", _find(s_prime[_id], 4))

    for _id in pos_rew_id:
        print("\n")
        print("Agent_t: ", _find(s[_id], 0))
        print("Goal_t: ", _find(s[_id], 4))

        print("Action_t: ", action[_id])
        print("Agent_t+1: ", _find(s_prime[_id], 0))
        print("Goal_t+1: ", _find(s_prime[_id], 4))


class QNetwork(network.Network):
    """Feed Forward network."""

    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 preprocessing_layers=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=(75, 40),
                 dropout_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 kernel_initializer=None,
                 batch_squash=True,
                 dtype=tf.float32,
                 name='QNetwork'):
        """Creates an instance of `QNetwork`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input observations.
      action_spec: A nest of `tensor_spec.BoundedTensorSpec` representing the
        actions.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations.
        All of these layers must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them. Good options include
        `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
        This layer must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      dropout_layer_params: Optional list of dropout layer parameters, where
        each item is the fraction of input units to drop. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except if the entry in the list is
        None. This list must have the same length of fc_layer_params, or be
        None.
      activation_fn: Activation function, e.g. tf.keras.activations.relu.
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default variance_scaling_initializer
      batch_squash: If True the outer_ranks of the observation are squashed into
        the batch dimension. This allow encoding networks to be used with
        observations with shape [BxTx...].
      dtype: The dtype to use by the convolution and fully connected layers.
      name: A string representing the name of the network.

    Raises:
      ValueError: If `input_tensor_spec` contains more than one observation. Or
        if `action_spec` contains more than one action.
    """
        # validate_specs(action_spec, input_tensor_spec)
        action_spec = tf.nest.flatten(action_spec)[0]
        num_actions = action_spec.maximum - action_spec.minimum + 1
        encoder_input_tensor_spec = input_tensor_spec

        encoder = encoding_network.EncodingNetwork(
            encoder_input_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=batch_squash,
            dtype=dtype)

        q_value_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.constant_initializer(-0.2),
            dtype=dtype)

        super(QNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self._encoder = encoder
        self._q_value_layer = q_value_layer

    def call(self, observation, step_type=None, network_state=(), training=False):
        """Runs the given observation through the network.

        Args:
          observation: The observation to provide to the network.
          step_type: The step type for the given observation. See `StepType` in
            time_step.py.
          network_state: A state tuple to pass to the network, mainly used by RNNs.
          training: Whether the output is being used for training.

        Returns:
          A tuple `(logits, network_state)`.
    """
        state, network_state = self._encoder(
            observation, step_type=step_type, network_state=network_state,
            training=training)
        q_value = self._q_value_layer(state, training=training)
        return q_value, network_state


class pickDropBoxWorld(py_environment.PyEnvironment, ABC):
    """
    Define the Environment
    """

    def __init__(self, grid_x=5, grid_y=5, num_objects=3, discount=1.0, learn_option=None):

        self.grid_length_x = grid_x
        self.grid_length_y = grid_y
        self.num_objects = num_objects
        self.num_actions = 4

        if learn_option is None:
            print("Define The option from pick, drop, pick_drop")
            sys.exit(-1)
        self.learn_option = learn_option  # pick, drop, pick_drop

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.num_actions - 1, name='nav_action')

        # 5x5x6 bit map of the grid-world carrying status of objects, obstacle, agent and goal
        self._observation_spec = {
            'grid_world': array_spec.BoundedArraySpec(
                shape=(grid_x, grid_y, 6), dtype=np.float, minimum=0, maximum=1, name='grid_world'),
            'latent_mode': array_spec.BoundedArraySpec((3,), np.float, minimum=0, maximum=1, name='latent_mode')
        }

        # Attributes of the grid-world
        # self.obstacles = [[2, 0], [0, 2], [1, 2], [3, 2], [4, 2], [2, 4]]

        # Here we remove two obstacles since if a object is spawned at the entrance to a grid region and the object is
        # also spawned in that region, the agent will have to pick the object in order to exit the region which will
        # incur a neg reward if the underlying c does not correspond to that object.
        # NOTE:- If we want to handle such states, we need to introduce a pick action separately
        self.obstacles = [[0, 2], [1, 2], [3, 2], [4, 2]]
        grid_posns = [[x, y] for x in range(grid_x) for y in range(grid_y)]
        self.valid_grid_posns = [x for x in grid_posns if x not in self.obstacles]

        self.grid_regions = {
            0: [[0, 0], [0, 1], [1, 0], [1, 1]],
            1: [[0, 3], [0, 4], [1, 3], [1, 4]],
            2: [[3, 0], [3, 1], [4, 0], [4, 1]],
            3: [[3, 3], [3, 4], [4, 3], [4, 4]]
        }

        self.steps_ref = {
            0: "up",  # Up
            1: "right",  # Right
            2: "left",  # Left
            3: "down",  # Down
        }

        self._state, self.c = None, None
        self.step_count = 0
        self._episode_ended = False
        self.discount = discount

        self.debug = _debug

        super(pickDropBoxWorld, self).__init__()

    def sample_init_state(self):
        if self.debug:
            print("\n-------------------------- New Trajectory --------------------------")
        state = np.zeros((5, 5, 6))

        # Fix the obstacles
        for obstacle in self.obstacles:
            state[obstacle[0], obstacle[1], 5] = 1

        # Randomly select regions
        goal_region, obj1_region, obj2_region, obj3_region = random.choice(
            list(permutations(range(1 + self.num_objects), 1 + self.num_objects)))

        grid_elements = []
        # Spawn the goal
        x, y = sample_xy(self.grid_regions[goal_region])
        if self.debug:
            print("Goal: [{}, {}]".format(x, y))
        state[x, y, 4] = 1
        grid_elements.append([x, y])

        # Spawn the objects
        x, y = sample_xy(self.grid_regions[obj1_region])
        if self.debug:
            print("Object1: [{}, {}]".format(x, y))
        state[x, y, 1] = 1
        grid_elements.append([x, y])

        x, y = sample_xy(self.grid_regions[obj2_region])
        if self.debug:
            print("Object2: [{}, {}]".format(x, y))
        state[x, y, 2] = 1
        grid_elements.append([x, y])

        x, y = sample_xy(self.grid_regions[obj3_region])
        if self.debug:
            print("Object3: [{}, {}]".format(x, y))
        state[x, y, 3] = 1
        grid_elements.append([x, y])

        # Spawn the agent [for initial state, do not spawn agent at any of the object's/goal's position]
        # If pick, spawn anywhere
        if self.learn_option == "pick":
            x, y = sample_xy([pos for pos in self.valid_grid_posns if pos not in grid_elements])
            if self.debug:
                print("Agent: [{}, {}]".format(x, y))
            state[x, y, 0] = 1

            # Set the underlying latent mode
            c = random.randint(0, 2)

        # If drop, spawn agent at object pos determined by c
        else:
            # Set the underlying latent mode
            c = random.randint(0, 2)
            x, y = _find(state, c+1)
            state[x, y, 0] = 1

        self._state, self.c = state, c

    def transition_xssx(self, s, s_prime, force_c=None):
        """
        Args:
            force_c:
            s: s_{t-1}
            s_prime: s_t
        Returns: next_latent_mode: x_{t}
        Note that we will transition to other object's id if -
        (i) it is not already present at the goal
        (ii) the brought object's id matches with current set latent mode (in this cond., we will keep the latent id to
        the object's that needs to be brought -  this will incur huge negative reward and should enforce right behavior)
        """
        # Set the next latent mode as the current
        c = force_c if force_c is not None else self.c
        next_x = c

        # Locate relevant positions
        prev_agent_pos = _find(s, 0)
        curr_agent_pos = _find(s_prime, 0)
        goal_pos = _find(s_prime, 4)

        # The agent should arrive at the goal position for transition to occur
        if prev_agent_pos != curr_agent_pos and curr_agent_pos == goal_pos:
            obj1_pos = _find(s, 1)
            obj2_pos = _find(s, 2)
            obj3_pos = _find(s, 3)

            # If object 1 is brought with x=1: switch to 2 or 3
            if obj1_pos == prev_agent_pos and c == 0:
                if obj2_pos != goal_pos and obj3_pos != goal_pos:
                    next_x = np.random.choice(np.arange(0, 3), p=[0.0, 0.5, 0.5])
                elif obj2_pos != goal_pos and obj3_pos == goal_pos:
                    next_x = np.random.choice(np.arange(0, 3), p=[0.0, 1.0, 0.0])
                elif obj2_pos == goal_pos and obj3_pos != goal_pos:
                    next_x = np.random.choice(np.arange(0, 3), p=[0.0, 0.0, 1.0])

            # If object 2 is brought with x=2: switch to 1 or 3
            elif obj2_pos == prev_agent_pos and c == 1:
                if obj1_pos != goal_pos and obj3_pos != goal_pos:
                    next_x = np.random.choice(np.arange(0, 3), p=[0.5, 0.0, 0.5])
                elif obj1_pos != goal_pos and obj3_pos == goal_pos:
                    next_x = np.random.choice(np.arange(0, 3), p=[1.0, 0.0, 0.0])
                elif obj1_pos == goal_pos and obj3_pos != goal_pos:
                    next_x = np.random.choice(np.arange(0, 3), p=[0.0, 0.0, 1.0])

            # If object 3 is brought with x=3: switch to 1 or 2
            elif obj3_pos == prev_agent_pos and c == 2:
                if obj1_pos != goal_pos and obj2_pos != goal_pos:
                    next_x = np.random.choice(np.arange(0, 3), p=[0.5, 0.5, 0.0])
                elif obj1_pos != goal_pos and obj2_pos == goal_pos:
                    next_x = np.random.choice(np.arange(0, 3), p=[1.0, 0.0, 0.0])
                elif obj1_pos == goal_pos and obj2_pos != goal_pos:
                    next_x = np.random.choice(np.arange(0, 3), p=[0.0, 1.0, 0.0])

        return next_x

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.sample_init_state()
        self.step_count = 0
        self._episode_ended = False
        observation = {
            'grid_world': self._state,
            'latent_mode': one_hot_encode(np.array(self.c, dtype=np.int), dim=self.num_objects)[0]
        }
        return ts.restart(observation)

    def apply_action(self, a, force_state=None):
        """
        Returns: s_{t+1} from applying a_t to s_t
        """
        # neighs = {}
        steps = {
            0: [-1, 0],  # Up
            1: [0, 1],  # Right
            2: [0, -1],  # Left
            3: [1, 0],  # Down
        }

        # Apply the action on current state if provided else get from env: Useful for HER
        s = force_state if force_state is not None else self._state.copy()

        # Locate agent's position
        curr_agent_pos = _find(s, 0)

        new_agent_pos = list(np.array(curr_agent_pos) + np.array(steps[a]))

        # 1] Out-of-space actions - stay
        if new_agent_pos[0] >= self.grid_length_x or new_agent_pos[0] < 0 or new_agent_pos[1] >= self.grid_length_y or \
                new_agent_pos[1] < 0.:
            # neighs[a] = s
            return s
        else:
            # Determine the statuses of agent and objects in next state
            z = s[curr_agent_pos[0], curr_agent_pos[1], :].copy()
            z_prime = s[new_agent_pos[0], new_agent_pos[1], :].copy()

            # 2] Obstacle-hitting actions - stay
            if z_prime[5] == 1:
                # neighs[a] = s
                return s
            else:

                # 3] Object aggregating actions (not at goal) - stay
                if np.count_nonzero(z_prime[1:4] + z[1:4]) > 1 and z_prime[4] != 1 and z[4] != 1:
                    # neighs[a] = s
                    return s

                # 4] Update status of the positions moved from and moved to by the agent
                else:
                    # Move the agent and the objects from z to z_prime
                    if z[4] != 1:
                        z_prime[:4] = z_prime[:4] + z[:4]
                        z[:4] = z[:4] - z[:4]
                    # Move the agent from z to z_prime but leave the objects at z
                    else:
                        z_prime[0] = 1
                        z[0] = 0

                    s[curr_agent_pos[0], curr_agent_pos[1], :] = z
                    s[new_agent_pos[0], new_agent_pos[1], :] = z_prime
                    # neighs[a] = s_prime
                    return s

    def compute_reward(self, s, s_prime, force_c=None):
        """
        Args:
            force_c:
            s: prev_state
            s_prime: curr_state

        Returns: reward

        """

        c = force_c if force_c is not None else self.c
        reward = -1.0

        curr_agent_pos = _find(s, 0)
        next_agent_pos = _find(s_prime, 0)
        z = s[curr_agent_pos[0], curr_agent_pos[1], :]
        z_prime = s[next_agent_pos[0], next_agent_pos[1], :]

        if curr_agent_pos != next_agent_pos:

            # Rewards for reaching object pos (not present at the goal already):
            if not z_prime[4] and not np.count_nonzero(z[1:4]) and np.count_nonzero(z_prime[1:4]):

                collected_obj_id = np.where(z_prime[1:4] == 1.)[0][0]
                if collected_obj_id == c:
                    reward = 100.
                else:
                    reward = -100.

                if self.debug:
                    print("Collected. c_gt: {}, obj_id: {}, reward: {}".format(c, collected_obj_id, reward))
                    # print("-------  {}-> {}".format(curr_agent_pos, next_agent_pos))
                    # print("Object collected with reward: ", reward)

            # Rewards for bring objects to goal
            elif z_prime[4] and np.count_nonzero(z[1:4]):

                dropped_obj_id = np.where(z[1:4] == 1.)[0][0]
                if dropped_obj_id == c:
                    reward = 100.
                else:
                    reward = -100.

                if self.debug:
                    print("Dropped. c_gt: {}, obj_id: {}, reward: {}".format(c, dropped_obj_id, reward))
                    # print("-------  {}-> {}".format(curr_agent_pos, next_agent_pos))
                    # print("Object Dropped with reward: ", reward)

        return reward

    def _step(self, action: np.ndarray):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            print("------ Terminal State Reached -----")
            return self.reset()

        self.step_count += 1
        # [Update the state]
        s = self._state.copy()
        if 0 <= action <= 3:
            self._state = s_prime = self.apply_action(action.item())
            if self.debug:
                print("Action {}: {}->{}".format(self.steps_ref[action.item()], _find(s, 0), _find(s_prime, 0)))

            # if debug:
            #     print("------- {}: {}-> {}".format(action, _find(s, 0), _find(s_prime, 0)))

            # [Check for Termination] Make sure episodes don't go on forever.
            next_agent_pos = _find(s_prime, 0)
            z_prime = s_prime[next_agent_pos[0], next_agent_pos[1], :]

            # Define Terminal State
            # If any object specified by the latent mode is at goal position
            if self.learn_option == 'pick':
                if any(z_prime[1:4] == 1.):
                    self._episode_ended = True
            elif self.learn_option == 'drop':
                if z_prime[1 + self.c] == z_prime[4] == 1.:
                    self._episode_ended = True
            else:
                # All objects are at the goal position
                if all(z_prime[1:5] == 1.):
                    self._episode_ended = True
        else:
            raise ValueError('`action` should be between 0 and 3.')

        # [Compute the reward]
        reward = self.compute_reward(s, s_prime)

        # Stay on the same latent state
        if self.learn_option in ['pick', 'drop']:
            c = self.c
        # Else transit
        else:
            self.c = c = self.transition_xssx(s, s_prime)

        # 1 x obj_dim_1 x ...
        observation = {
            'grid_world': s_prime,
            'latent_mode': one_hot_encode(np.array(c, dtype=np.int), dim=self.num_objects)[0]
        }
        if self._episode_ended:
            return ts.termination(observation, reward)
        else:
            return ts.transition(observation, reward=reward, discount=self.discount)


class myAgentOptions:
    def __init__(self):
        self.num_objects = 3

        # FOR HER
        self.do_her = False

        # Hyper-Parameters
        self.log_interval = 200  # @param {type:"integer"}
        self.eval_interval = 1000  # @param {type:"integer"}

        self.initial_collect_steps = 500  # @param {type:"integer"} Using Random policy
        self.collect_steps_per_iteration = 1  # @param {type:"integer"}
        self.replay_buffer_max_length = 1000000  # @param {type:"integer"}
        self.num_eval_episodes = 10  # @param {type:"integer"}

        self.num_epochs = 3  # Will give num_epochs * num_cycles * num_episodes configurations
        self.num_cycles = 50  # Num cycles per_epoch
        self.num_train_episodes = 20  # Num episodes per cycle
        self.num_opt_steps = 30  # Num optimisation steps per cycle
        self.max_steps_per_episode = 10  # @param {type:"integer"}
        self.max_iterations_per_epoch = self.num_cycles * self.num_opt_steps  # num_cycles * num_optimisation_steps
        self.sample_batch_size = 128  # @param {type:"integer"}
        self.sample_step_size = 2  # @param {type:"integer"} <- DQN agent needs s_t and s_{t+1}

        self.gamma = 0.98
        self.learning_rate = 1e-3  # @param {type:"number"}
        self.epsilon_initial = 0.8
        self.epsilon_min = 0.2
        self.epsilon_prob_decay = 0.05
        self.target_net_update_freq = self.num_train_episodes * self.max_steps_per_episode  # Update after each cycle
        self.target_net_decay_coeff = 0.95

        # Define the Environment
        self.pick_env, self.train_tf_pick_env, self.eval_tf_pick_env = self.define_environment(learn_option='pick')
        self.drop_env, self.train_tf_drop_env, self.eval_tf_drop_env = self.define_environment(learn_option='drop')

        # Define the Agent
        self.pick_agent = self.define_agent()
        self.drop_agent = self.define_agent()

        # Define the random policy
        self.random_pick_policy = random_tf_policy.RandomTFPolicy(self.train_tf_pick_env.time_step_spec(),
                                                                  self.train_tf_pick_env.action_spec())

        self.random_drop_policy = random_tf_policy.RandomTFPolicy(self.train_tf_pick_env.time_step_spec(),
                                                                  self.train_tf_pick_env.action_spec())

        # Policy Saver
        self.tf_pick_policy_saver = policy_saver.PolicySaver(self.pick_agent.policy)
        self.tf_drop_policy_saver = policy_saver.PolicySaver(self.drop_agent.policy)

        # ############################################################################################### #
        # 3] Replay Buffer: To store (s_t, a_t, r_t, s_{t+1})
        # # The TFUniformReplayBuffer stores episodes in `B == batch_size` blocks of size `L == max_length`,
        # # with total frame capacity `C == L * B`. Multiple episodes may be stored within a given block, up to L frames
        self.pick_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.pick_agent.collect_data_spec,
            batch_size=self.train_tf_pick_env.batch_size,  # Note:- This should be environment's batch size
            max_length=self.replay_buffer_max_length, device='cpu:*')

        self.drop_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.pick_agent.collect_data_spec,
            batch_size=self.train_tf_pick_env.batch_size,  # Note:- This should be environment's batch size
            max_length=self.replay_buffer_max_length, device='cpu:*')

        self.initialise_buffers()

        # ############################################################################################### #
        # 4] Dataset: Read the replay buffer as a Dataset,
        # # as_dataset() - returns the replay buffer as a tf.data.Dataset. One can then create a dataset iterator and
        # #                iterate through the samples of the items in the buffer. Returns batch_size * num_steps items
        # # num_steps - specify that sub-episodes (a seq. of consecutive items in the buffer) are desired.
        # #             DQN Agent needs both the current and next observation to compute the loss
        pick_dataset = self.pick_replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.sample_batch_size,
            num_steps=self.sample_step_size).prefetch(self.sample_batch_size)  # prefetches elements from dataset while the current element is being processed
        self.pick_data_iter = iter(pick_dataset)
        drop_dataset = self.drop_replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.sample_batch_size,
            num_steps=self.sample_step_size).prefetch(self.sample_batch_size)
        self.drop_data_iter = iter(drop_dataset)

    def define_environment(self, learn_option):
        # Specify the env
        """
        Important Wrappers:-
            - GoalReplayEnvWrapper: For HER
            - TimeLimit: End episodes after specified number of steps
            - OneHotActionWrapper: Converts discrete action to one_hot format
            - FixedLength: Truncates long episodes and pads short episodes to have a fixed length
        """
        env = pickDropBoxWorld(discount=self.gamma, learn_option=learn_option)

        train_env = pickDropBoxWorld(discount=self.gamma, learn_option=learn_option)
        train_env = wrappers.TimeLimit(train_env, duration=self.max_steps_per_episode)
        # Wrap the py env into Tensorflow to parallelize operations
        train_tf_env = tf_py_environment.TFPyEnvironment(train_env)

        eval_env = pickDropBoxWorld(discount=self.gamma, learn_option=learn_option)
        eval_env = wrappers.TimeLimit(eval_env, duration=self.max_steps_per_episode)
        # Wrap the py env into Tensorflow to parallelize operations
        eval_tf_env = tf_py_environment.TFPyEnvironment(eval_env)
        return env, train_tf_env, eval_tf_env

    def verify_env(self):
        # Verify if the environments works correctly
        env = wrappers.TimeLimit(self.pick_env, duration=self.max_steps_per_episode)
        tf_utils.validate_py_environment(env, episodes=5)
        print("Environment Verified")

    def define_agent(self):
        # ############################################################################################### #
        # 1] Define the Q-Network
        preprocessing_layers = {
            'grid_world': tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(16, 3, activation='relu'), tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu')
            ]),
            'latent_mode': tf.keras.layers.Dense(64, activation='relu')
        }
        preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
        q_net = QNetwork(self.train_tf_pick_env.observation_spec(), self.train_tf_pick_env.action_spec(),
                         preprocessing_layers=preprocessing_layers,
                         preprocessing_combiner=preprocessing_combiner,
                         fc_layer_params=(64, 16))

        # ############################################################################################### #
        # 2] Specify the Agent and Random Policy
        # # Agent contains two policies:
        # - agent.policy — The main policy that is used for evaluation and deployment.
        # - agent.collect_policy — A second policy that is used for data collection.
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        train_step_counter = tf.Variable(0)
        _agent = dqn_agent.DdqnAgent(
            self.train_tf_pick_env.time_step_spec(),
            self.train_tf_pick_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            target_update_period=self.target_net_update_freq,
            target_update_tau=self.target_net_decay_coeff,  # Q_target:= w_t = (1 - tau) * w_t + tau * w_s
            td_errors_loss_fn=common.element_wise_squared_loss,  # takes as input the target and the estimated Q values
            epsilon_greedy=self.epsilon_initial,
            gamma=self.gamma,
            train_step_counter=train_step_counter)  # counter to increment every time the train op is run
        _agent.initialize()
        return _agent

    # Alternate to Driver for recording the trajectories in replay buffer
    @staticmethod
    def collect_step(_policy, _env, curr_time_step=None) -> (ts.TimeStep, trajectory.Transition,
                                                             trajectory.Trajectory):
        """
        Args:
            _policy:
            _env:
            curr_time_step:

        Returns:
            - next_time_step: contains the data (step_type, obs, discount, reward) emitted by an environment at
                                  each step of interaction
            - Transition:
            - Trajectory: represents a 'sequence' of aligned time steps (i.e. seq of <s_t, a_t, r_t>). It captures
                          the observation, step_type from current time step with the computed action and policy_info.
                          Discount, reward and next_step_type come from the next time step. (Here it just provides s_t)
        """
        if not curr_time_step:
            curr_time_step = _env.current_time_step()
        action_step = _policy.action(curr_time_step)
        next_time_step = _env.train(action_step.action)
        _traj = trajectory.from_transition(curr_time_step, action_step, next_time_step)
        transition = trajectory.Transition(curr_time_step, action_step, next_time_step)
        return next_time_step, transition, _traj

    def initialise_buffers(self):
        # execute the random policy in the environment for a few steps, recording the data in the replay buffer.
        for _ in range(self.initial_collect_steps):

            # Populate Pick Policy's Buffer
            next_ts, transition, traj = self.collect_step(_policy=self.random_pick_policy, _env=self.train_tf_pick_env)
            # Add trajectory to the replay buffer
            self.pick_replay_buffer.add_batch(traj)

            # Populate Drop Policy's Buffer
            ext_ts2, transition2, traj2 = self.collect_step(_policy=self.random_drop_policy, _env=self.train_tf_drop_env)
            # Add trajectory to the replay buffer
            self.drop_replay_buffer.add_batch(traj2)

        print("[DEBUG] Running Random Policy to initialise the buffers ({}/{})".format(self.initial_collect_steps,
                                                                                       self.replay_buffer_max_length))
        # # Read all elements in the pick/drop replay buffer:
        # trajectories = self.pick_replay_buffer.gather_all()
        #
        # print("[DEBUG] Initial Buffer:")
        # print(tf.nest.map_structure(lambda t: t.shape, trajectories))

    def hindsight_exp_replay(self, init_state, init_c, transitions: List[trajectory.Transition],
                             learn_option, env, buffer):
        """
        The idea of HER is to achieve the goal with the trajectory that agent has so far followed. In our case, the goal
        is to collect all the objects in the correct order. Thus we sample the possible latent mode transitions first
        and modify the environment so that agent collect objects based on the sampled latent mode transition.
        Args:
            init_state: Initial state of the episode
            init_c: Underlying Latent Mode
            transitions: List of segments where either object is being collected or dropped at the goal
            learn_option: pick/drop
            env: Instance of the pick/drop env
            buffer: Buffer

        Returns: Updated Buffer
        """
        orig_agent_pos = _find(init_state, 0)
        objs_pos = [_find(init_state, 1), _find(init_state, 2), _find(init_state, 3)]
        target_pos = _find(init_state, 4)

        trans_init_state = copy.deepcopy(init_state)
        skipped = 0  # To debug: How many transitions were skipped because agent did not move

        obj_id = init_c
        obj_pos = objs_pos[obj_id]

        if transitions:

            # Event: Collect object specified by the set strategy
            if learn_option == 'pick':
                # [Future] Get the object's position which has to be collected based on the strategy and set
                # it to agent's pos in next_time_step of each transition so that it gets collected

                temp_buffer = []  # To avoid redundant transitions from the same trajectory segment
                for transition_idx, transition in enumerate(transitions):
                    agent_curr_pos = _find(transition.time_step.observation['grid_world'].numpy()[0], 0)
                    agent_next_pos = _find(transition.next_time_step.observation['grid_world'].numpy()[0], 0)

                    # If the entry is already in temp_buffer, skip
                    if (agent_curr_pos, agent_next_pos, obj_id) not in temp_buffer:
                        temp_buffer.append((agent_curr_pos, agent_next_pos, obj_id))
                    else:
                        skipped += 1
                        continue

                    # Define the new modified_state and modified_next_state for each transition
                    modified_state = copy.deepcopy(trans_init_state)
                    modified_state[orig_agent_pos[0], orig_agent_pos[1], 0] = 0.
                    modified_state[agent_curr_pos[0], agent_curr_pos[1], 0] = 1.
                    modified_state[obj_pos[0], obj_pos[1], obj_id + 1] = 0.
                    modified_state[agent_next_pos[0], agent_next_pos[1], obj_id + 1] = 1.

                    modified_next_state = copy.deepcopy(trans_init_state)
                    modified_next_state[orig_agent_pos[0], orig_agent_pos[1], 0] = 0.
                    modified_next_state[agent_next_pos[0], agent_next_pos[1], 0] = 1.
                    modified_next_state[obj_pos[0], obj_pos[1], obj_id+1] = 0.
                    modified_next_state[agent_next_pos[0], agent_next_pos[1], obj_id+1] = 1.

                    reward = env.compute_reward(modified_state, modified_next_state, force_c=init_c)
                    observation = {
                        'grid_world': np.array([modified_state]),
                        'latent_mode': one_hot_encode(np.array(init_c, dtype=np.int), dim=self.num_objects)
                    }
                    # Batched Trajectory
                    curr_time_step = time_step.TimeStep(transition.time_step.step_type, np.array([0.0]),
                                                        np.array([self.gamma]), observation)
                    next_time_step = ts.transition(observation, np.array([reward]), np.array([self.gamma]))
                    _traj = trajectory.from_transition(curr_time_step, transition.action_step, next_time_step)
                    buffer.add_batch(_traj)

            # Event: Drop object (Here the obj_id is fixed to underlying latent mode)
            # We are just dropping objects at target position under different underlying c
            else:
                temp_buffer = []  # To avoid redundant transitions from the same trajectory segment
                for transition_idx, transition in enumerate(transitions):
                    agent_curr_pos = _find(transition.time_step.observation['grid_world'].numpy()[0], 0)
                    agent_next_pos = _find(transition.next_time_step.observation['grid_world'].numpy()[0], 0)

                    # If the entry is already in temp_buffer, skip
                    if (agent_curr_pos, agent_next_pos, obj_id) not in temp_buffer:
                        temp_buffer.append((agent_curr_pos, agent_next_pos, obj_id))
                    else:
                        skipped += 1
                        continue

                    z_goal = trans_init_state[target_pos[0], target_pos[1], :]

                    # Update Current State
                    modified_state = copy.deepcopy(trans_init_state)
                    # Bring the collected object to the agent's current position
                    modified_state[orig_agent_pos[0], orig_agent_pos[1], 0] = 0.
                    modified_state[obj_pos[0], obj_pos[1], obj_id+1] = 0
                    modified_state[agent_curr_pos[0], agent_curr_pos[1], 0] = 1.
                    modified_state[agent_curr_pos[0], agent_curr_pos[1], obj_id+1] = 1.
                    # Bring the goal to the agent's next position
                    modified_state[target_pos[0], target_pos[1], :] = 0.
                    modified_state[agent_next_pos[0], agent_next_pos[1], :] += z_goal

                    # Update Next State
                    modified_next_state = copy.deepcopy(trans_init_state)
                    # Bring the collected object to the agent's next position
                    modified_next_state[orig_agent_pos[0], orig_agent_pos[1], 0] = 0.
                    modified_next_state[obj_pos[0], obj_pos[1], obj_id + 1] = 0
                    modified_next_state[agent_next_pos[0], agent_next_pos[1], 0] = 1.
                    modified_next_state[agent_next_pos[0], agent_next_pos[1], obj_id + 1] = 1.
                    # Bring the goal to the agent's next position
                    modified_next_state[target_pos[0], target_pos[1], :] = 0.
                    modified_next_state[agent_next_pos[0], agent_next_pos[1], :] += z_goal

                    try:
                        reward = env.compute_reward(modified_state, modified_next_state, force_c=init_c)
                    except AssertionError:
                        sys.exit(-1)
                    observation = {
                        'grid_world': np.array([modified_state]),
                        'latent_mode': one_hot_encode(np.array(init_c, dtype=np.int), dim=self.num_objects)
                    }
                    # Batched Trajectory
                    curr_time_step = time_step.TimeStep(transition.time_step.step_type, np.array([0.0]),
                                                        np.array([self.gamma]), observation)  # Contains modified_state with reward=0
                    next_time_step = ts.transition(observation, np.array([reward]), np.array([self.gamma]))  # Contains modified_state with reward=100
                    _traj = trajectory.from_transition(curr_time_step, transition.action_step, next_time_step)  # Contains modified_state with reward=100
                    buffer.add_batch(_traj)

    def learn_option(self, learn_option, _policy_dir):

        if not os.path.exists(_policy_dir):
            os.mkdir(_policy_dir)
        # ############################################################################################### #
        # 6] Training

        if learn_option == "pick":
            _agent = self.pick_agent
            _buffer = self.pick_replay_buffer
            _env = self.pick_env
            train_tf_env = self.train_tf_pick_env
            eval_tf_env = self.eval_tf_pick_env
            _policy_saver = self.tf_pick_policy_saver
            _iterator = self.pick_data_iter
        elif learn_option == 'drop':
            _agent = self.drop_agent
            _buffer = self.drop_replay_buffer
            _env = self.drop_env
            train_tf_env = self.train_tf_drop_env
            eval_tf_env = self.eval_tf_drop_env
            _policy_saver = self.tf_drop_policy_saver
            _iterator = self.drop_data_iter
        else:
            print("Provide a valid learn option")
            sys.exit(-1)

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        _agent.train = common.function(_agent.train)

        # Reset the train step.
        _agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        init_avg_return = self.compute_avg_return(_agent.policy, eval_tf_env, _env)
        max_avg_return = init_avg_return
        returns = [init_avg_return]
        plot_loss = []

        for j in range(self.num_epochs):
            print("\n--------------- Epoch {} ---------------".format(j))
            # self.agent.collect_policy._epsilon = self.epsilon_initial

            with tqdm(total=self.max_iterations_per_epoch, position=0, leave=True) as pbar:
                i = 0
                for _ in range(self.num_cycles):

                    # # # Update Epsilon of Greedy Policy before collecting data
                    _agent.collect_policy._epsilon = max(_agent.collect_policy._epsilon * np.exp(
                            -self.epsilon_prob_decay), self.epsilon_min)

                    # DATA COLLECTION
                    avg_ep_len = 0
                    for _ in range(self.num_train_episodes):

                        curr_time_step = train_tf_env.reset()
                        avg_ep_len += 1  # Average episode length in each cycle
                        transitions = []

                        # Identify the goal and object's initial position in the grid
                        init_state = curr_time_step.observation['grid_world'].numpy()[0]
                        init_c = np.argmax(curr_time_step.observation['latent_mode'].numpy()[0])

                        while not curr_time_step.is_last():
                            # Collect a few steps sing the driver and save to the replay buffer.
                            # time_step, _ = collect_driver.run(time_step)
                            # Collect a few steps using collect_policy and save to the replay buffer.
                            curr_time_step, transition, _traj = self.collect_step(_agent.collect_policy, train_tf_env,
                                                                                  curr_time_step)
                            _buffer.add_batch(_traj)
                            avg_ep_len += 1

                            # NOTE: When using to_transition, the `reward` and `discount` fields of `time_steps` are
                            # filled with zeros because these cannot be deduced [Official Implementation]
                            if transition.next_time_step.reward.numpy()[0] == -1.0:
                                # Add the segment so far collected to overall transitions
                                transitions.append(transition)

                        # Do HER after collecting each episode
                        if self.do_her:
                            self.hindsight_exp_replay(init_state, init_c, transitions, learn_option, _env, _buffer)

                    avg_ep_len /= self.num_train_episodes
                    # POLICY LEARNING
                    for _ in range(self.num_opt_steps):
                        i += 1
                        # Sample a batch of data from the buffer and update the agent's network.
                        # Trajectories: [Batch, Time_steps, dim_1, ...]
                        trajectories, unused_info = next(_iterator)
                        debug_data(trajectories)

                        train_loss = _agent.train(trajectories).loss
                        plot_loss.append(train_loss)

                        _ = _agent.train_step_counter.numpy()

                        pbar.refresh()
                        pbar.set_description("Step {}".format(i + 1))
                        pbar.set_postfix(loss=train_loss.numpy(), epsilon=_agent.collect_policy._epsilon,
                                         avg_ep_len=avg_ep_len)
                        pbar.update(1)

            avg_return = self.compute_avg_return(_agent.policy, eval_tf_env, _env)
            if max_avg_return < avg_return:
                print("Saving Model")
                _policy_saver.save(os.path.join(_policy_dir, 'best_policy'))

            _policy_saver.save(os.path.join(_policy_dir, 'Epoch{}_policy'.format(j)))
            print('epoch = {0}: Average Return = {1}'.format(j, avg_return))
            returns.append(avg_return)

        self.plot(returns, plot_loss, _policy_dir, learn_option)

    def learn(self):
        # Learn the policy to pick an object
        # self.learn_option(learn_option='pick', _policy_dir='./saved_pick_policy')

        # Learn the policy to drop an object
        self.learn_option(learn_option='drop', _policy_dir='./saved_drop_policy')

    @staticmethod
    def test_policy(eval_tf_env, env, _policy_dir):

        saved_policy = tf.saved_model.load(os.path.join(_policy_dir, 'Epoch1_policy'))

        for _ in range(5):
            _time_step = eval_tf_env.reset()

            episode_return = 0.0

            # Print the Initial Configuration
            curr_agent_pos = _find(_time_step.observation['grid_world'].numpy()[0], 0)
            obj1_pos = _find(_time_step.observation['grid_world'].numpy()[0], 1)
            obj2_pos = _find(_time_step.observation['grid_world'].numpy()[0], 2)
            obj3_pos = _find(_time_step.observation['grid_world'].numpy()[0], 3)
            goal_pos = _find(_time_step.observation['grid_world'].numpy()[0], 4)
            print("Agent: ", curr_agent_pos, " Obj1: ", obj1_pos, " Obj2: ", obj2_pos, " Obj3: ", obj3_pos, " Goal: ",
                  goal_pos)

            while not _time_step.is_last():
                action_step = saved_policy.action(_time_step)
                _time_step = eval_tf_env.train(action_step.action)
                episode_return += _time_step.reward
                print("Action {}: ".format(env.step_to_id[action_step.action.numpy()[0]]),
                      _find(_time_step.observation['grid_world'].numpy()[0], 0))

            print("Episode Overall Return: ", episode_return.numpy()[0])

    def compute_avg_return(self, eval_policy, eval_tf_env, env):
        # See also the metrics module for standard implementations of different metrics.
        # https://github.com/tensorflow/agents/tree/master/tf_agents/metrics

        total_return = 0.0
        for _ in range(self.num_eval_episodes):
            print("--------------------------- Eval Trajectory ---------------------------")
            _time_step = eval_tf_env.reset()
            curr_agent_pos = _find(_time_step.observation['grid_world'].numpy()[0], 0)
            obj1_pos = _find(_time_step.observation['grid_world'].numpy()[0], 1)
            obj2_pos = _find(_time_step.observation['grid_world'].numpy()[0], 2)
            obj3_pos = _find(_time_step.observation['grid_world'].numpy()[0], 3)
            goal_pos = _find(_time_step.observation['grid_world'].numpy()[0], 4)
            print("Agent: ", curr_agent_pos, " Obj1: ", obj1_pos, " Obj2: ", obj2_pos, " Obj3: ", obj3_pos, " Goal: ",
                  goal_pos)

            episode_return = 0.0

            while not _time_step.is_last():
                action_step = eval_policy.action(_time_step)
                _time_step = eval_tf_env.train(action_step.action)
                print("Action {}: ".format(env.step_to_id[action_step.action.numpy()[0]]),
                      _find(_time_step.observation['grid_world'].numpy()[0], 0))
                episode_return += _time_step.reward
            total_return += episode_return

        avg_return = total_return / self.num_eval_episodes
        # print("Average Return = ", avg_return.numpy()[0])
        return avg_return.numpy()[0]

    def plot(self, returns, plot_loss, _policy_dir, learn_option):
        # ############################################################################################### #
        # 6] Plot Visualisation
        iterations = range(self.num_epochs + 1)
        plt.plot(iterations, returns)
        plt.ylabel('Average Return ({} episodes)'.format(self.num_eval_episodes))
        plt.xlabel('Epochs')
        plt.savefig(os.path.join(_policy_dir, "{}BoxWorld_avgReturn.png".format(learn_option)))
        plt.clf()

        iterations = range(len(plot_loss))  #
        plt.plot(iterations, plot_loss)
        plt.ylabel('Loss')
        plt.xlabel('Train Steps')
        plt.savefig(os.path.join(_policy_dir, "{}BoxWorld_Loss.png".format(learn_option)))


if __name__ == "__main__":
    agent = myAgentOptions()
    agent.learn()
    # agent.test_policy(policy_dir)
