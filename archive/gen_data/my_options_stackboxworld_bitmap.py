import copy
import json
import os
import random
from itertools import permutations
from typing import List
from utils import one_hot_encode, yield_batched_indexes
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, Input, Add, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_squared_error
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tqdm import tqdm


def sample_xy(valid_posns: List):
    picked_pos = random.choice(valid_posns)
    return picked_pos[0], picked_pos[1]


def _find(state, _id):
    x, y = np.where(state[:, :, _id])
    assert len(x) == len(y) == 1
    return [x[0], y[0]]


def plot(returns, plot_loss, _policy_dir, _learn_option):
    # ############################################################################################### #
    # 6] Plot Visualisation
    iterations = range(len(returns))
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Epochs')
    plt.savefig(os.path.join(_policy_dir, "{}BoxWorld_avgReturn.png".format(learn_option)))
    plt.clf()

    iterations = range(len(plot_loss))  #
    plt.plot(iterations, plot_loss)
    plt.ylabel('Loss')
    plt.xlabel('Train Steps')
    plt.savefig(os.path.join(_policy_dir, "{}BoxWorld_Loss.png".format(learn_option)))


class stackBoxWorld:
    """
    Define the Environment
    """

    def __init__(self, grid_x=5, grid_y=5, num_objects=3, max_step_count=10, _learn_option=None):

        self.learn_option = _learn_option

        self.grid_length_x = grid_x
        self.grid_length_y = grid_y
        self.num_objects = num_objects
        self.num_actions = 4

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

        self._state, self.c, self.observation = None, None, None

        self.unique_state_gen = self.unique_world_sampler()
        self.max_step_count = max_step_count
        self.step_count = 0
        self.episode_ended = False
        self.debug = 0

    @property
    def latent_size(self):
        return self.num_objects

    @property
    def action_size(self):
        return self.num_actions

    @property
    def state_size(self):
        return [self.grid_length_x, self.grid_length_y, 1+self.num_objects+1+1]

    def unique_world_sampler(self):
        while True:
            for perm in permutations(range(4), 4):
                goal_reg, ob1_reg, ob2_reg, ob3_reg = perm
                for goal in self.grid_regions[goal_reg]:
                    for ob1 in self.grid_regions[ob1_reg]:
                        for ob2 in self.grid_regions[ob2_reg]:
                            for ob3 in self.grid_regions[ob3_reg]:
                                if self.learn_option == 'drop':

                                    for c, agent_pos in enumerate([ob1, ob2, ob3]):
                                        state = np.zeros((5, 5, 6))
                                        # Fix the obstacles
                                        for obstacle in self.obstacles:
                                            state[obstacle[0], obstacle[1], 5] = 1
                                        # Spawn the goal
                                        state[goal[0], goal[1], 4] = 1
                                        # Spawn the objects
                                        state[ob1[0], ob1[1], 1] = 1
                                        state[ob2[0], ob2[1], 2] = 1
                                        state[ob3[0], ob3[1], 3] = 1
                                        # Spawn the agent at object's position
                                        state[agent_pos[0], agent_pos[1], 0] = 1

                                        yield state, c

                                else:
                                    raise NotImplementedError

    def sample_init_state(self):
        if self.debug:
            print("\n-------------------------- New Trajectory --------------------------")
        state = np.zeros((5, 5, 6))

        # Fix the obstacles
        for obstacle in self.obstacles:
            state[obstacle[0], obstacle[1], 5] = 1

        # Randomly select regions
        goal_region, obj1_region, obj2_region, obj3_region = random.choice(list(permutations(range(1+self.num_objects), 1+self.num_objects)))

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

        return state, c

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

    def reset(self):

        self.step_count = 0
        self.episode_ended = False

        # state, c = self.sample_init_state()
        state, c = self.unique_state_gen.__next__()
        self._state, self.c = state, c
        self.observation = {
            'grid_world': state,
            'latent_mode': one_hot_encode(np.array(c, dtype=np.int), dim=self.num_objects)[0]
        }
        return self.observation, 0, self.episode_ended

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
                    reward = 0.
                else:
                    reward = -1.

                if self.debug:
                    print("Collected. c_gt: {}, obj_id: {}, reward: {}".format(c, collected_obj_id, reward))
                    # print("-------  {}-> {}".format(curr_agent_pos, next_agent_pos))
                    # print("Object collected with reward: ", reward)

            # Rewards for bring objects to goal
            elif z_prime[4] and np.count_nonzero(z[1:4]):

                dropped_obj_id = np.where(z[1:4] == 1.)[0][0]
                if dropped_obj_id == c:
                    reward = 0.
                else:
                    reward = -1.

                if self.debug:
                    print("Dropped. c_gt: {}, obj_id: {}, reward: {}".format(c, dropped_obj_id, reward))
                    # print("-------  {}-> {}".format(curr_agent_pos, next_agent_pos))
                    # print("Object Dropped with reward: ", reward)

        return reward

    def step(self, action: np.ndarray):

        if self.episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            print("------ Terminal State Reached -----")
            return self.reset()
        self.step_count += 1
        reached_terminal = False
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
                    reached_terminal = True
            elif self.learn_option == 'drop':
                if z_prime[1 + self.c] == z_prime[4] == 1.:
                    reached_terminal = True
            else:
                # All objects are at the goal position
                if all(z_prime[1:5] == 1.):
                    reached_terminal = True
        else:
            raise ValueError('`action` should be between 0 and 3.')

        # [Compute the reward]
        reward = self.compute_reward(s, s_prime)

        # Stay on the same latent state
        if self.learn_option in ['pick', 'drop']:
            c = self.c
        # Else transit
        else:
            raise NotImplementedError

        # 1 x obj_dim_1 x ...
        self.observation = {
            'grid_world': s_prime,
            'latent_mode': one_hot_encode(np.array(c, dtype=np.int), dim=self.num_objects)[0]
        }

        # Check for Terminal State:
        if reached_terminal or self.step_count == self.max_step_count:
            self.episode_ended = True

        return self.observation, reward, self.episode_ended


class Buffer:
    def __init__(self):
        self.curr_state = None
        self.curr_c = None
        self.next_state = None
        self.next_c = None

        self.done = None
        self.action = None
        self.reward = None
        self.size = 0

    def add_transition(self, current_obs, action, reward, next_obs, done):
        if self.size == 0:
            self.curr_state = np.array([current_obs['grid_world']])
            self.curr_c = np.array([current_obs['latent_mode']])
            self.next_state = np.array([next_obs['grid_world']])
            self.next_c = np.array([next_obs['latent_mode']])

            self.action = np.array([action])
            self.reward = np.array([reward])
            self.done = np.array([done])
        else:
            self.curr_state = np.append(self.curr_state, np.array([current_obs['grid_world']]), axis=0)
            self.curr_c = np.append(self.curr_c, np.array([current_obs['latent_mode']]), axis=0)
            self.next_state = np.append(self.next_state, np.array([next_obs['grid_world']]), axis=0)
            self.next_c = np.append(self.next_c, np.array([next_obs['latent_mode']]), axis=0)

            self.action = np.append(self.action, np.array([action]), axis=0)
            self.reward = np.append(self.reward, np.array([reward]), axis=0)
            self.done = np.append(self.done, np.array([done]), axis=0)

        self.size += 1

    def remove_transition(self):
        np.delete(self.curr_state, 0, axis=0)
        np.delete(self.curr_c, 0, axis=0)
        np.delete(self.next_state, 0, axis=0)
        np.delete(self.next_c, 0, axis=0)

        np.delete(self.done, 0, axis=0)
        np.delete(self.action, 0, axis=0)
        np.delete(self.reward, 0, axis=0)

        self.size -= 1

    def shuffle(self):
        idx_d = np.arange(self.size)
        np.random.shuffle(idx_d)

        self.curr_state = self.curr_state[idx_d]
        self.curr_c = self.curr_c[idx_d]
        self.next_state = self.next_state[idx_d]
        self.next_c = self.next_c[idx_d]

        self.done = self.done[idx_d]
        self.action = self.action[idx_d]
        self.reward = self.reward[idx_d]


class DDQNAgent:
    # constructor: we need both the state size and the number of actions that the agent can take
    #              to initialize the DNN
    # https://rubikscode.net/2021/07/20/introduction-to-double-q-learning/
    def __init__(self, state_size, action_size, latent_size):
        # we define some hyper_parameters
        self.n_actions = action_size

        self.lr = 0.001
        self.gamma = 0.99
        self.max_exploration_proba = self.exploration_proba = 1.0
        self.min_exploration_proba = 0.2
        self.exploration_proba_decay = 0.00001

        self.memory_buffer = Buffer()
        self.max_memory_buffer = 1000000  # Should me more than enough to accommodate all possible world config.

        self.q_model = self.build_model(state_size, action_size, latent_size)
        self.q_model.compile(loss="mse", optimizer=Adam(lr=self.lr))

        self.q_target_decay_coeff = 0.08  # A low value updates target network parameters slowly from primary network

        self.q_target_model = self.build_model(state_size, action_size, latent_size)

    # building a model of 2 hidden later of 24 units each
    def build_model(self, state_size, action_size, latent_size):
        # 1] Define the Q-Network
        state = Input(shape=state_size)
        x = Flatten()(state)
        x = Dense(128)(x)
        x = LeakyReLU()(x)
        x = Dense(128)(x)
        x = LeakyReLU()(x)

        # x = Conv2D(16, 3, activation='relu')(state)
        # x = Flatten()(x)
        # x = Dense(units=64, activation='relu')(x)

        encodes = Input(shape=[latent_size])
        c = Dense(128)(encodes)
        c = LeakyReLU()(c)
        # c = Dense(64)(encodes)

        h = Add()([x, c])
        h = LeakyReLU()(h)
        # h = Dense(64)(h)
        # h = LeakyReLU()(h)
        # h = Dense(16)(h)
        # h = LeakyReLU()(h)
        # h = Dense(units=64, activation='relu')(h)
        # h = Dense(units=16, activation='relu')(h)
        actions = Dense(units=action_size, activation='linear')(h)

        q_net = Model(inputs=[state, encodes], outputs=actions)

        return q_net

    # the agent computes the action to take given a state
    def compute_action(self, curr_obs, _test=False):
        if not _test:
            if np.random.uniform(0, 1) < self.exploration_proba:
                return np.random.choice(range(self.n_actions))
        q_values = self.q_model.predict([np.array([curr_obs['grid_world']]), np.array([curr_obs['latent_mode']])])[0]
        return np.argmax(q_values)

    # we store all experiences
    def store_transition(self, current_obs, action, reward, next_obs, done):
        self.memory_buffer.add_transition(current_obs, action, reward, next_obs, done)

        if self.memory_buffer.size > self.max_memory_buffer:
            self.memory_buffer.remove_transition()

    # when an episode is finished, we update the exploration probability
    def align_epsilon(self, step):
        self.exploration_proba = self.min_exploration_proba + (self.max_exploration_proba - self.min_exploration_proba) * np.exp(-self.exploration_proba_decay * step)

    # train the model using the replayed memory
    def train(self, batched_idxs):
        batch_size = len(batched_idxs)

        if self.memory_buffer.size < batch_size*3:
            return 0

        # Predict Q(s,a) and Q(s',a') given the batch of states
        q_current_state = self.q_model.predict([self.memory_buffer.curr_state[batched_idxs],
                                                self.memory_buffer.curr_c[batched_idxs]])
        q_next_state = self.q_model.predict([self.memory_buffer.next_state[batched_idxs],
                                                self.memory_buffer.next_c[batched_idxs]])

        reward = self.memory_buffer.reward[batched_idxs]

        # Copy the q_values_state into the target: Target <- Q(s,a)
        target = q_current_state
        # Obtain the best action from current estimate of Q Net at next step, a_max <- argmax Q(s', a')
        next_actions = np.argmax(q_next_state, axis=1)
        q_next_state_target = self.q_target_model.predict([self.memory_buffer.next_state[batched_idxs], self.memory_buffer.next_c[batched_idxs]])
        # Update the Q(s, a) for taken action a with R + Gamma*Q_{target}(s', a_max)
        updates = reward + self.gamma*q_next_state_target[np.arange(batch_size), next_actions]
        target[np.arange(batch_size), self.memory_buffer.action[batched_idxs]] = updates

        # q_pred = np.choose(list(next_actions), q_next_state_target.T)
        loss = self.q_model.train_on_batch([self.memory_buffer.curr_state[batched_idxs],
                                            self.memory_buffer.curr_c[batched_idxs]], target)
        self.align_q_target_network()
        return loss

    # we update the weights of the Q-target DNN
    def align_q_target_network(self):
        # Soft-Update: w_t = (1 - tau) * w_t + tau * w_s
        TAU = self.q_target_decay_coeff
        for t, e in zip(self.q_target_model.trainable_variables,
                        self.q_model.trainable_variables): t.assign(t * (1 - TAU) + e * TAU)

    def hindsight_exp_replay(self, init_state, init_c, transitions, _learn_option, env):
        """
        The idea of HER is to achieve the goal with the trajectory that agent has so far followed. In our case, the goal
        is to collect all the objects in the correct order. Thus we sample the possible latent mode transitions first
        and modify the environment so that agent collect objects based on the sampled latent mode transition.
        Args:
            init_state: Initial state of the episode
            init_c: Underlying Latent Mode
            transitions: List of segments where either object is being collected or dropped at the goal
            _learn_option: pick/drop
            env: Instance of the pick/drop env

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
            if _learn_option == 'pick':
                # [Future] Get the object's position which has to be collected based on the strategy and set
                # it to agent's pos in next_time_step of each transition so that it gets collected

                temp_buffer = []  # To avoid redundant transitions from the same trajectory segment
                for transition_idx, transition in enumerate(transitions):
                    curr_obs, action, next_obs = transition
                    agent_curr_pos = _find(curr_obs['grid_world'], 0)
                    agent_next_pos = _find(next_obs['grid_world'], 0)

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
                    new_curr_obs = {
                        'grid_world': modified_state,
                        'latent_mode': one_hot_encode(np.array(init_c, dtype=np.int), dim=env.num_objects)[0]
                    }
                    new_next_obs = {
                        'grid_world': modified_next_state,
                        'latent_mode': one_hot_encode(np.array(init_c, dtype=np.int), dim=env.num_objects)[0]
                    }
                    # Batched Trajectory
                    self.store_transition(new_curr_obs, action, reward, new_next_obs, True)

            # Event: Drop object (Here the obj_id is fixed to underlying latent mode)
            # We are just dropping objects at target position under different underlying c
            else:
                temp_buffer = []  # To avoid redundant transitions from the same trajectory segment
                for transition_idx, transition in enumerate(transitions):
                    curr_obs, action, next_obs = transition
                    agent_curr_pos = _find(curr_obs['grid_world'], 0)
                    agent_next_pos = _find(next_obs['grid_world'], 0)

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

                    reward = env.compute_reward(modified_state, modified_next_state, force_c=init_c)

                    new_curr_obs = {
                        'grid_world': modified_state,
                        'latent_mode': one_hot_encode(np.array(init_c, dtype=np.int), dim=env.num_objects)[0]
                    }
                    new_next_obs = {
                        'grid_world': modified_next_state,
                        'latent_mode': one_hot_encode(np.array(init_c, dtype=np.int), dim=env.num_objects)[0]
                    }
                    # Batched Trajectory
                    self.store_transition(new_curr_obs, action, reward, new_next_obs, True)


def learn_option(option, _policy_dir):
    if not os.path.exists(_policy_dir):
        os.mkdir(_policy_dir)

    # Hyper-parameters
    num_eval_episodes = 10

    num_epochs = 10
    num_cycles = 2000
    num_train_episodes = 10
    num_opt_steps = 2
    max_iterations_per_epoch = num_cycles * num_opt_steps
    max_steps_per_episode = 10
    batch_size = 128

    # Declare Env
    train_env = stackBoxWorld(max_step_count=max_steps_per_episode, _learn_option=option)
    eval_env = stackBoxWorld(max_step_count=max_steps_per_episode, _learn_option=option)
    # Declare Agent
    agent = DDQNAgent(train_env.state_size, train_env.action_size, train_env.latent_size)

    # HER
    do_her = True

    incurred_loss = []
    epoch_avg_return = []
    total_time_steps = 0

    for j in range(num_epochs):
        print("\n--------------- Epoch {} ---------------".format(j))
        with tqdm(total=max_iterations_per_epoch, position=0, leave=True) as pbar:
            opt_step = 0
            for _ in range(num_cycles):

                # Data Collection
                avg_ep_len = 0
                for _ in range(num_train_episodes):
                    curr_obs, _, _ = train_env.reset()
                    # Identify the goal and object's initial position in the grid
                    init_state = curr_obs['grid_world']
                    init_c = np.argmax(curr_obs['latent_mode'])

                    transitions = []
                    while not train_env.episode_ended:
                        avg_ep_len += 1  # Average episode length in each cycle

                        # Update exploration probability before each step
                        agent.align_epsilon(total_time_steps)
                        total_time_steps += 1

                        # Take action and observe the next step
                        action = agent.compute_action(curr_obs)
                        next_obs, reward, done = train_env.step(action)
                        agent.store_transition(curr_obs, action, reward, next_obs, done)

                        if reward == -1.0:
                            transitions.append((curr_obs, action, next_obs))
                        curr_obs = next_obs

                    # Do HER after collecting each episode
                    if do_her:
                        agent.hindsight_exp_replay(init_state, init_c, transitions, learn_option, train_env)

                avg_ep_len /= num_train_episodes
                # Model Training
                agent.memory_buffer.shuffle()
                data_iterator = yield_batched_indexes(start=0, b_size=batch_size, n_samples=agent.memory_buffer.size)
                for step in range(num_opt_steps):
                    batched_idxs = data_iterator.__next__()
                    loss = agent.train(batched_idxs)
                    incurred_loss.append(loss)

                    pbar.refresh()
                    pbar.set_description("Step {}".format(opt_step + 1))
                    pbar.set_postfix(loss=loss, epsilon=agent.exploration_proba,
                                     avg_ep_len=avg_ep_len)

                    opt_step += 1
                    pbar.update(1)

        # Evaluate after each epoch ends
        total_return = 0
        for _ in range(num_eval_episodes):
            print("--------------------------- Eval Trajectory ---------------------------")
            curr_obs, _, _ = eval_env.reset()
            curr_agent_pos = _find(curr_obs['grid_world'], 0)
            obj1_pos = _find(curr_obs['grid_world'], 1)
            obj2_pos = _find(curr_obs['grid_world'], 2)
            obj3_pos = _find(curr_obs['grid_world'], 3)
            goal_pos = _find(curr_obs['grid_world'], 4)
            print("Agent: ", curr_agent_pos, " Obj1: ", obj1_pos, " Obj2: ", obj2_pos, " Obj3: ", obj3_pos, " Goal: ",
                  goal_pos)

            episode_return = 0
            while not eval_env.episode_ended:
                action = agent.compute_action(curr_obs)
                next_obs, reward, done = eval_env.step(action)
                episode_return += reward
                curr_obs = next_obs

            total_return += episode_return
        avg_return = total_return/num_eval_episodes
        epoch_avg_return.append(avg_return)
        print("Average Return: ", avg_return)

        # Save the model
        with open(os.path.join(_policy_dir, "Q_model_%d.json" % j), "w") as outfile:
            json.dump(agent.q_model.to_json(), outfile)
        agent.q_model.save_weights(os.path.join(_policy_dir, "Q_model_%d.h5" % j), overwrite=True)

        with open(os.path.join(_policy_dir, "Qtarget_model_%d.json" % j), "w") as outfile:
            json.dump(agent.q_target_model.to_json(), outfile)
        agent.q_target_model.save_weights(os.path.join(_policy_dir, "Qtarget_model_%d.h5" % j), overwrite=True)

    # Plot the loss and average return per epoch
    plot(epoch_avg_return, incurred_loss, _policy_dir, learn_option)


def eval_policy(option, _policy_dir):

    epoch_num = 6
    eval_env = stackBoxWorld(max_step_count=10, _learn_option=option)

    # Load Model
    agent = DDQNAgent(eval_env.state_size, eval_env.action_size, eval_env.latent_size)
    agent.q_model.load_weights(os.path.join(_policy_dir, "Q_model_{}.h5".format(epoch_num)))
    agent.q_target_model.load_weights(os.path.join(_policy_dir, "Qtarget_model_{}.h5".format(epoch_num)))

    for _ in range(5):
        eval_env.reset()

        episode_return = 0.0

        # Print the Initial Configuration
        eval_env.reset()
        curr_obs = eval_env.observation
        curr_agent_pos = _find(curr_obs['grid_world'], 0)
        obj1_pos = _find(curr_obs['grid_world'], 1)
        obj2_pos = _find(curr_obs['grid_world'], 2)
        obj3_pos = _find(curr_obs['grid_world'], 3)
        goal_pos = _find(curr_obs['grid_world'], 4)
        print("Agent: ", curr_agent_pos, " Obj1: ", obj1_pos, " Obj2: ", obj2_pos, " Obj3: ", obj3_pos, " Goal: ",
              goal_pos)

        while not eval_env.episode_ended:
            action = agent.compute_action(curr_obs, _test=True)
            next_obs, reward, done = eval_env.step(action)
            episode_return += reward
            curr_obs = next_obs
            print("Action {}: ".format(eval_env.steps_ref[action]), _find(next_obs['grid_world'], 0))

        print("Episode Overall Return: ", episode_return)

def main():
    learn_option(option='drop', _policy_dir='./saved_drop_policy')
    # eval_policy(option='drop', _policy_dir='./saved_drop_policy')


if __name__ == "__main__":
    main()
