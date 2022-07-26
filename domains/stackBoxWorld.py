import copy
import json
import os
import numpy as np
import random
from itertools import permutations
from typing import List, Dict, Tuple
from utils.misc import one_hot_encode


def sample_xy(valid_posns: List):
    picked_pos = random.choice(valid_posns)
    return picked_pos[0], picked_pos[1]


def _find(state, _id):
    x, y = np.where(state[:, :, _id])
    assert len(x) == len(y) == 1
    return [x[0], y[0]]


class stackBoxWorld:
    """
    Define the Environment
    """

    def __init__(self, random_transition=True):

        self.grid_length_x = 5
        self.grid_length_y = 5
        self.num_objects = 3
        self.num_actions = 4

        self._max_step_count = 50
        self._max_stuck_count = 3

        # Here we remove two obstacles since if a object is spawned at the entrance to a grid region and the object is
        # also spawned in that region, the agent will have to pick the object in order to exit the region which will
        # incur a neg reward if the underlying c does not correspond to that object.
        # NOTE:- If we want to handle such states, we need to introduce a pick action separately
        self.obstacles = [[0, 2], [1, 2], [3, 2], [4, 2]]
        grid_posns = [[x, y] for x in range(5) for y in range(5)]
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
        self.pick_latent_modes = [0, 2, 4]
        self.drop_latent_modes = [1, 3, 5]

        # FIXED PICK-DROP: This will test if agent correctly transits to object's drop after picking it up
        self.random_transition = random_transition
        if not random_transition:
            # FIXED PICK-DROP-PICK: This will test if agent collects objects in specific order
            self.follow_pick_transition = [2, 0, 4]  # Obj2 -> Obj1 -> Obj3

        # Observations
        self._state, self.c, self.observation = None, None, None

        # Initial positions of the spawned elements
        self.obj1, self.obj2, self.obj3, self.goal = None, None, None, None

        self.info: Dict = {}
        self.step_count = 0
        self.stuck_count = 0
        self.episode_ended = False
        self.debug = 0

    @property
    def latent_size(self):
        # For pick-drop of each object
        return 2*self.num_objects

    @property
    def action_size(self):
        return self.num_actions

    @property
    def state_size(self):
        return [self.grid_length_x, self.grid_length_y, 1+self.num_objects+1+1]

    def sample_init_state(self):
        if self.debug:
            print("\n-------------------------- New Trajectory --------------------------")
        state = np.zeros((5, 5, 6))

        # Fix the obstacles
        for obstacle in self.obstacles:
            state[obstacle[0], obstacle[1], 5] = 1

        # Randomly select regions
        goal_region, obj1_region, obj2_region, obj3_region = random.choice(list(permutations(range(1+self.num_objects),
                                                                                             1+self.num_objects)))

        grid_elements = []
        # Spawn the goal
        x, y = sample_xy(self.grid_regions[goal_region])
        self.goal = [x, y]
        if self.debug:
            print("Goal: [{}, {}]".format(x, y))
        state[x, y, 4] = 1
        grid_elements.append([x, y])

        # Spawn the objects
        x, y = sample_xy(self.grid_regions[obj1_region])
        self.obj1 = [x, y]
        if self.debug:
            print("Object1: [{}, {}]".format(x, y))
        state[x, y, 1] = 1
        grid_elements.append([x, y])

        x, y = sample_xy(self.grid_regions[obj2_region])
        self.obj2 = [x, y]
        if self.debug:
            print("Object2: [{}, {}]".format(x, y))
        state[x, y, 2] = 1
        grid_elements.append([x, y])

        x, y = sample_xy(self.grid_regions[obj3_region])
        self.obj3 = [x, y]
        if self.debug:
            print("Object3: [{}, {}]".format(x, y))
        state[x, y, 3] = 1
        grid_elements.append([x, y])

        # Spawn the agent [for initial state, do not spawn agent at any of the object's/goal's position]
        # If pick, spawn anywhere
        x, y = sample_xy([pos for pos in self.valid_grid_posns if pos not in grid_elements])
        if self.debug:
            print("Agent: [{}, {}]".format(x, y))
        state[x, y, 0] = 1

        # Set the underlying latent mode
        if self.random_transition:
            c = random.choice(self.pick_latent_modes)
        else:
            c = self.follow_pick_transition[0]

        return state, c

    def random_transition_xssx(self, s, s_prime):
        """
        Args:
            s: s_{t-1}
            s_prime: s_t
        Returns: next_latent_mode: x_{t}
        Pick_Latent_modes = [0, 2, 4]
        Drop_Latent_modes = [1, 3, 5]
        (i) If object collected, transition to collected object's drop latent mode
        (i) If object dropped, transition to any remaining object's pick latent mode
        """
        # Set the next latent mode as the current
        c = self.c
        next_x = c

        # Locate relevant positions
        curr_agent_pos = _find(s, 0)
        next_agent_pos = _find(s_prime, 0)
        goal_pos = self.goal

        # object's positions in the last state
        obj1_pos = _find(s, 1)
        obj2_pos = _find(s, 2)
        obj3_pos = _find(s, 3)
        init_obj_pos = [self.obj1, self.obj2, self.obj3]  # SHOULD BE ORDERED

        if curr_agent_pos != next_agent_pos:

            # Object Picked: Transition to its drop
            if curr_agent_pos not in init_obj_pos and next_agent_pos in init_obj_pos and next_agent_pos != goal_pos:
                obj_id = init_obj_pos.index(next_agent_pos)
                next_x = self.drop_latent_modes[obj_id]

            # Object Dropped (if agent is carrying the object): Transition to pick of remaining objects
            elif next_agent_pos == goal_pos:
                # If object 0 is brought with x=1: switch to 2 or 4
                if obj1_pos == curr_agent_pos and c == 1:
                    if obj2_pos != goal_pos and obj3_pos != goal_pos:
                        next_x = np.random.choice(self.pick_latent_modes, p=[0.0, 0.5, 0.5])
                    elif obj2_pos != goal_pos and obj3_pos == goal_pos:
                        next_x = np.random.choice(self.pick_latent_modes, p=[0.0, 1.0, 0.0])
                    elif obj2_pos == goal_pos and obj3_pos != goal_pos:
                        next_x = np.random.choice(self.pick_latent_modes, p=[0.0, 0.0, 1.0])

                # If object 1 is brought with x=3: switch to 0 or 4
                elif obj2_pos == curr_agent_pos and c == 3:
                    if obj1_pos != goal_pos and obj3_pos != goal_pos:
                        next_x = np.random.choice(self.pick_latent_modes, p=[0.5, 0.0, 0.5])
                    elif obj1_pos != goal_pos and obj3_pos == goal_pos:
                        next_x = np.random.choice(self.pick_latent_modes, p=[1.0, 0.0, 0.0])
                    elif obj1_pos == goal_pos and obj3_pos != goal_pos:
                        next_x = np.random.choice(self.pick_latent_modes, p=[0.0, 0.0, 1.0])

                # If object 2 is brought with x=5: switch to 0 or 2
                elif obj3_pos == curr_agent_pos and c == 5:
                    if obj1_pos != goal_pos and obj2_pos != goal_pos:
                        next_x = np.random.choice(self.pick_latent_modes, p=[0.5, 0.5, 0.0])
                    elif obj1_pos != goal_pos and obj2_pos == goal_pos:
                        next_x = np.random.choice(self.pick_latent_modes, p=[1.0, 0.0, 0.0])
                    elif obj1_pos == goal_pos and obj2_pos != goal_pos:
                        next_x = np.random.choice(self.pick_latent_modes, p=[0.0, 1.0, 0.0])

        return next_x

    def fixed_transition_xssx(self, s, s_prime):
        """
        FOR object_id picking order [1, 0, 2]
        Args:
            s: s_{t-1}
            s_prime: s_t
        Returns: next_latent_mode: x_{t}
        Pick_Latent_modes = [0, 2, 4]
        Drop_Latent_modes = [1, 3, 5]
        (i) If object collected, transition to collected object's drop latent mode
        (i) If object dropped, transition to specific object's pick latent mode
        """
        # Set the next latent mode as the current
        c = self.c
        next_x = c

        # Locate relevant positions
        curr_agent_pos = _find(s, 0)
        next_agent_pos = _find(s_prime, 0)
        goal_pos = self.goal

        # object's positions in the last state
        obj1_pos = _find(s, 1)
        obj2_pos = _find(s, 2)
        obj3_pos = _find(s, 3)
        init_obj_pos = [self.obj1, self.obj2, self.obj3]  # SHOULD BE ORDERED

        if curr_agent_pos != next_agent_pos:

            # Object Picked: Transition to its drop
            if curr_agent_pos not in init_obj_pos and next_agent_pos in init_obj_pos and next_agent_pos != goal_pos:
                obj_id = init_obj_pos.index(next_agent_pos)
                next_x = self.drop_latent_modes[obj_id]

            # Object Dropped (if agent is carrying the object): Transition to pick of remaining objects
            elif next_agent_pos == goal_pos:
                # If object 0 is brought with x=1: switch to object 2 with pick_id=4
                if obj1_pos == curr_agent_pos and c == 1:
                    next_x = 4

                # If object 1 is brought with x=3: switch to object 0 with pick_id=0
                elif obj2_pos == curr_agent_pos and c == 3:
                    next_x = 0

                # If object 2 is brought with x=5: do nothing
                elif obj3_pos == curr_agent_pos and c == 5:
                    next_x = c

        return next_x

    def reset(self) -> Tuple[Dict, float, bool, Dict]:

        # Reset the env variables
        self.step_count = 0
        self.stuck_count = 0
        self.episode_ended = False
        self.info = {
            'is_success': False
        }

        # Sample Initial state
        state, c = self.sample_init_state()

        # Update Env's state
        self._state, self.c = state, c
        self.observation = {
            'grid_world': state,
            'latent_mode': one_hot_encode(np.array(c, dtype=np.int), dim=self.latent_size)[0]
        }
        return self.observation, 0, self.episode_ended, self.info

    def force_set(self, state: np.ndarray, c: np.ndarray) -> Tuple[Dict, float, bool, Dict]:

        # Reset the env variables
        self.step_count = 0
        self.stuck_count = 0
        self.episode_ended = False
        self.info = {
            'is_success': False
        }

        # Update Env's state
        self._state, self.c = state, c
        self.observation = {
            'grid_world': state,
            'latent_mode': c
        }
        return self.observation, 0, self.episode_ended, self.info

    def render_init_config(self):
        if self._state is None:
            print("Environment not initiated")
            raise ValueError
        else:
            s = self._state

            # Get all the element positions
            curr_agent_pos = _find(s, 0)
            obj1_pos = _find(s, 1)
            obj2_pos = _find(s, 2)
            obj3_pos = _find(s, 3)
            goal_pos = _find(s, 4)
            obstacles = self.obstacles

            if curr_agent_pos in [obj1_pos, obj2_pos, obj3_pos, goal_pos]:
                print("Environment cannot be displayed with overlapping elements")
                raise NotImplementedError

            legend = {
                'agent': 'o',
                'obj1': 'P',
                'obj2': 'Q',
                'obj3': 'R',
                'goal': 'G',
                'obs': 'x'
            }

            positions = {
                str(curr_agent_pos): 'agent',
                str(obj1_pos): 'obj1',
                str(obj2_pos): 'obj2',
                str(obj3_pos): 'obj3',
                str(goal_pos): 'goal',
            }
            for obs_pos in obstacles:
                positions[str(obs_pos)] = 'obs'

            grid_str = ''
            for i in range(5):
                row_str = '|'
                for j in range(5):
                    query = str([i, j])
                    if query in positions.keys():
                        row_str += legend[positions[query]]
                    else:
                        row_str += ' '
                    row_str += '|'
                grid_str += row_str + "\n"

            return grid_str

    def apply_action(self, a):
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
        s = self._state.copy()

        # Locate agent's position
        curr_agent_pos = _find(s, 0)

        # Compute agent's next position
        new_agent_pos = list(np.array(curr_agent_pos) + np.array(steps[a]))

        # 1] Out-of-space actions - stay
        if new_agent_pos[0] >= self.grid_length_x or new_agent_pos[0] < 0 or new_agent_pos[1] >= self.grid_length_y or \
                new_agent_pos[1] < 0.:
            return s
        else:
            # Determine the statuses of agent and objects in next state
            z = s[curr_agent_pos[0], curr_agent_pos[1], :].copy()
            z_prime = s[new_agent_pos[0], new_agent_pos[1], :].copy()

            # 2] Obstacle-hitting actions - stay
            if z_prime[5] == 1:
                return s
            else:

                # 3] Object aggregating actions (not at goal) - stay
                if np.count_nonzero(z_prime[1:4] + z[1:4]) > 1 and z_prime[4] != 1 and z[4] != 1:
                    return s

                # 4] Update status of the positions moved from and moved to by the agent
                else:
                    # Move the agent
                    # Move the object's along with the agent if not at goal
                    if z[4] != 1:
                        z_prime[:4] = z_prime[:4] + z[:4]
                        z[:4] = z[:4] - z[:4]

                    # Move the agent from z to z_prime but leave the objects at z
                    else:
                        z_prime[0] = 1
                        z[0] = 0

                    s[curr_agent_pos[0], curr_agent_pos[1], :] = z
                    s[new_agent_pos[0], new_agent_pos[1], :] = z_prime
                    return s

    def compute_reward(self, s, s_prime):
        """
        Args:
            s: prev_state
            s_prime: curr_state

        Returns: reward

        """

        c = self.c
        reward = -1.0

        curr_agent_pos = _find(s, 0)
        next_agent_pos = _find(s_prime, 0)
        z = s[curr_agent_pos[0], curr_agent_pos[1], :]
        z_prime = s[next_agent_pos[0], next_agent_pos[1], :]

        if curr_agent_pos != next_agent_pos:

            # Rewards for reaching object pos (not present at the goal already):
            if not z_prime[4] and not np.count_nonzero(z[1:4]) and np.count_nonzero(z_prime[1:4]):

                collected_obj_id = np.where(z_prime[1:4] == 1.)[0][0]
                if self.pick_latent_modes[collected_obj_id] == c:
                    reward = 0.
                else:
                    reward = -1.

                if self.debug:
                    print("Collected. c_gt: {}, obj_id: {}, reward: {}".format(c, collected_obj_id, reward))
                    # print("-------  {}-> {}".format(curr_agent_pos, next_agent_pos))
                    # print("Object collected with reward: ", reward)

            # Rewards for bringing objects to goal
            elif z_prime[4] and np.count_nonzero(z[1:4]):

                dropped_obj_id = np.where(z[1:4] == 1.)[0][0]
                if self.drop_latent_modes[dropped_obj_id] == c:
                    reward = 0.
                else:
                    reward = -1.

                if self.debug:
                    print("Dropped. c_gt: {}, obj_id: {}, reward: {}".format(c, dropped_obj_id, reward))
                    # print("-------  {}-> {}".format(curr_agent_pos, next_agent_pos))
                    # print("Object Dropped with reward: ", reward)

        return reward

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:

        if self.episode_ended:
            # The last action ended the episode. Ignore the current action and start a new episode.
            if self.debug:
                print("------ Terminal State Reached -----")
            return self.reset()

        self.step_count += 1
        objective_achieved = False

        # [Update the state]
        s = self._state.copy()
        if 0 <= action <= 3:
            self._state = s_prime = self.apply_action(action.item())
            if np.all(s_prime == s):
                self.stuck_count += 1
            if self.debug:
                print("Action {}: {}->{}".format(self.steps_ref[action.item()], _find(s, 0), _find(s_prime, 0)))

            # [Check for Termination] Make sure episodes don't go on forever.
            next_agent_pos = _find(s_prime, 0)
            z_prime = s_prime[next_agent_pos[0], next_agent_pos[1], :]

            # Define Terminal State
            # All objects are at the goal position
            if all(z_prime[1:5] == 1.):
                objective_achieved = True
        else:
            raise ValueError('`action` should be between 0 and 3.')

        # [Compute the reward]
        reward = self.compute_reward(s, s_prime)

        # Update the latent mode
        if self.random_transition:
            self.c = c = self.random_transition_xssx(s, s_prime)
        else:
            self.c = c = self.fixed_transition_xssx(s, s_prime)

        # 1 x obj_dim_1 x ...
        self.observation = {
            'grid_world': s_prime,
            'latent_mode': one_hot_encode(np.array(c, dtype=np.int), dim=self.latent_size)[0]
        }

        # Check for Terminal State:
        if objective_achieved:
            self.episode_ended = True
            self.info = {'is_success': True}
        elif self.step_count == self._max_step_count:
            self.episode_ended = True
            self.info = {'is_success': False, 'reached_max': True}
        elif self.stuck_count > self._max_stuck_count:
            self.episode_ended = True
            reward = -100.
            self.info = {'is_success': False, 'stuck': True}

        return self.observation, reward, self.episode_ended, self.info
