import os
import random

import numpy as np
from tqdm import tqdm

from np_mdp.algorithms import planning as mdp_plan
from np_mdp.models import mdp as mdp_lib
from py_amm.amm import Amm
from utils.misc import check_and_load, save


class PickBoxWorld(Amm):
    """
    Objective:
    There are 3 objects spawned randomly in the grid-world. The agent needs to collect them one-by-one
    and stack them up at the spawned goal position.
    Rules:
    - The agent can not carry two or more objects at the same time.
    - The agent collects and drops the object if it arrives at the object location and then reaches the
      goal with the object.
    - [FUTURE] The agent can carry the object to another object's position and swap them but this will need
       adding object status (i.e. whether it is in possession or not) to the state and add/drop actions
    - A restricted grid-world divides the grid into regions separated by obstacles and spawns objects in each region
      for the agent to collect
    """
    def __init__(self, grid_length_x=5, grid_length_y=5, num_objects=3, root_dir=None):
        """
        Args:
            grid_length_x: grid dim
            grid_length_y: grid dim
            num_objects: number of objects that will be spawned randomly on the grid to be collected by an agent
        """
        self.grid_length_x = grid_length_x
        self.grid_length_y = grid_length_y
        self.num_objects = num_objects
        self.num_actions = 4

        self.nx1 = self.latent1 = num_objects
        self.latent_dict = self.inv_latent_dict = {0: 0, 1: 1, 2: 2}

        # Define the layout of the 5x5 grid-world
        self.obstacles = [[2, 0], [0, 2], [1, 2], [3, 2], [4, 2], [2, 4]]  # For 5x5 Grid
        grid_posns = [[x, y] for x in range(self.grid_length_x) for y in range(self.grid_length_y)]
        self.valid_grid_posns = [pos for pos in grid_posns if pos not in self.obstacles]

        self.grid_regions = {
            0: [[0, 0], [0, 1], [1, 0], [1, 1]],
            1: [[0, 3], [0, 4], [1, 3], [1, 4]],
            2: [[3, 0], [3, 1], [4, 0], [4, 1]],
            3: [[3, 3], [3, 4], [4, 3], [4, 4]]
        }

        # Define the layout of the 4x4 grid-world
        # self.obstacles = [[0, 1], [2, 1], [2, 3]]  # For 4x4 Grid
        # self.grid_regions = {
        #     0: [[0, 2], [0, 3], [1, 2], [1, 3]],
        # }

        self.temperatures = [0.01 for _ in range(num_objects)]
        self.states = self.enumerate_states()

        # Compute state transition matrix
        obstate_transition_matrix_path = os.path.join(root_dir, 'obstate_transition_matrix.pkl')
        obstate_transition_matrix = check_and_load(obstate_transition_matrix_path)
        if obstate_transition_matrix:
            self.obstate_transition_matrix = obstate_transition_matrix
        else:
            self.obstate_transition_matrix = self.compute_transition_sas()
            save(self.obstate_transition_matrix, obstate_transition_matrix_path)

        # bx is the temperature that sets the prob. of sampling a latent mode in a given demo
        self.bx = np.zeros(self.nx1)
        self.bx[0] = 0.33
        self.bx[1] = 0.33
        self.bx[2] = 0.34

        # Compute latent mode transition matrix
        trans_xsax_path = os.path.join(root_dir, 'trans_xsax.pkl')
        trans_xsax = check_and_load(trans_xsax_path)
        if trans_xsax:
            self.trans_xsax = trans_xsax
        else:
            self.trans_xsax = self.compute_transition_xsax()
            save(self.trans_xsax, trans_xsax_path)

        # Compute Policy
        policy_matrix_path = os.path.join(root_dir, 'policy_matrix.pkl')
        policy_matrix = check_and_load(policy_matrix_path)
        if policy_matrix:
            self.policy_matrix = policy_matrix
        else:
            self.policy_matrix = self.compute_policy()
            save(self.policy_matrix, policy_matrix_path)

        super(PickBoxWorld, self).__init__(
            obstate_trans_matrix=self.obstate_transition_matrix,
            num_obstates=len(self.states),
            num_actions=self.num_actions,
            num_states=[self.nx1],
            init_state_weights=self.bx,
            transition_matrix=self.trans_xsax,
            policy_matrix=self.policy_matrix
        )

    def get_initial_state(self):
        """
        Returns:
        State: [agent_x, agent_y, box1_x, box1_y, ....., goal_x, goal_y]
        """

        i = 0
        to_sample = self.num_objects + 2
        # # Sample initial state
        sampled_elements = []
        # Sample Goal
        goal_region = np.random.randint(4)
        # Sample Object Regions
        obj1_region = (goal_region + 1) % 4
        obj2_region = (goal_region + 2) % 4
        obj3_region = (goal_region + 3) % 4

        agent_posn = random.choice(self.valid_grid_posns)
        # Add state positions
        sampled_elements.append(agent_posn)
        sampled_elements.append(random.choice(self.grid_regions[obj1_region]))
        sampled_elements.append(random.choice(self.grid_regions[obj2_region]))
        sampled_elements.append(random.choice(self.grid_regions[obj3_region]))
        sampled_elements.append(random.choice(self.grid_regions[goal_region]))

        goal = sampled_elements[-1]

        # Terminal state corresponds to when all object positions and agent position coincide with goal
        terminal = [goal for _ in range(to_sample)]

        init_state = [v for grid_pos in sampled_elements for v in grid_pos]
        terminal_state = [v for grid_pos in terminal for v in grid_pos]
        return init_state, terminal_state

    def enumerate_states(self):
        """
        Returns: The valid state-space
        Note:- Allowing objects to span anywhere in a 5x5 grid-world with obstacles gives OOM error. Approx states = (~16x19x19x19x19)
        In 4x4 world with obstacles, we get ~ 4x13x13x13
        """
        states = []

        # # Logic 2: The following version results in 1882672 states (~16x19x19x19x19)
        for region in self.grid_regions.keys():
            for goal_pos in self.grid_regions[region]:
                for obj1_pos in self.valid_grid_posns:
                    for obj2_pos in self.valid_grid_posns:
                        for obj3_pos in self.valid_grid_posns:

                            # We only allow objects to be stacked up at terminal state
                            if obj1_pos == obj2_pos == obj3_pos == goal_pos:
                                for agent_pos in self.valid_grid_posns:
                                    state = agent_pos + obj1_pos + obj2_pos + obj3_pos + goal_pos
                                    states.append(tuple(state))

                            elif obj1_pos == obj2_pos == goal_pos != obj3_pos or obj2_pos == obj3_pos == goal_pos != obj1_pos or obj1_pos == obj3_pos == goal_pos != obj2_pos:
                                for agent_pos in self.valid_grid_posns:
                                    state = agent_pos + obj1_pos + obj2_pos + obj3_pos + goal_pos
                                    states.append(tuple(state))

                            elif obj1_pos != obj2_pos and obj2_pos != obj3_pos and obj1_pos != obj3_pos:
                                for agent_pos in self.valid_grid_posns:
                                    state = agent_pos + obj1_pos + obj2_pos + obj3_pos + goal_pos
                                    states.append(tuple(state))

        return states

    @property
    def states_dict(self):
        states_dict = {self.states[j]: j for j in range(len(self.states))}
        return states_dict

    @property
    def inv_states_dict(self):
        inv_states_dict = {j: self.states[j] for j in range(len(self.states))}
        return inv_states_dict

    def set_latent1(self, latent1):
        self.latent1 = latent1

    @property
    def latent_state_dict(self):
        return self.latent_dict

    @property
    def action_dict(self):
        act_dict = {
            'up': 0,
            'down': 1,
            'right': 2,
            'left': 3,
            # 'pick': 4,
            # 'drop': 5,
        }
        return act_dict

    @property
    def inv_action_dict(self):
        inv_act_dict = {
            0: 'up',
            1: 'down',
            2: 'right',
            3: 'left',
            # 4: 'pick',
            # 5: 'drop'
        }
        return inv_act_dict

    def opposite_action(self, a):
        opposite = {
            'up': 'down',
            'down': 'up',
            'right': 'left',
            'left': 'right',
            # 'pick': 'drop',
            # 'drop': 'pick'
        }
        return opposite[a]

    def neighbors(self, s):
        # if not isinstance(s, np.ndarray):
        #     s = np.array(s)
        neighs = {}
        out_of_space = {}
        steps = {
            'up': [-1, 0],
            'right': [0, 1],
            'left': [0, -1],
            'down': [1, 0],
            # 'pick': [0, 0],
            # 'drop': [0, 0]
        }

        curr_agent_pos = s[:2]
        obj1_pos, obj2_pos, obj3_pos = s[2:4], s[4:6], s[6:8]
        goal_pos = s[8:]
        for a in steps.keys():
            new_agent_pos = list(np.array(curr_agent_pos) + np.array(steps[a]))
            # Note this logic works only if the agent does not have two objects in possession
            # else with current logic it will carry both
            # Move Object 1 (If not at goal)
            if curr_agent_pos == obj1_pos and obj1_pos != goal_pos:
                obj1_pos = list(np.array(obj1_pos) + np.array(steps[a]))
            # Move Object 2 (If not at goal)
            elif curr_agent_pos == obj2_pos and obj2_pos != goal_pos:
                obj2_pos = list(np.array(obj2_pos) + np.array(steps[a]))
            # Move Object 3 (If not at goal)
            elif curr_agent_pos == obj3_pos and obj3_pos != goal_pos:
                obj3_pos = list(np.array(obj3_pos) + np.array(steps[a]))

            new_s = new_agent_pos + obj1_pos + obj2_pos + obj3_pos + goal_pos

            if tuple(new_s) in self.states:
                neighs[a] = tuple(new_s)
            else:
                # This means out of space
                out_of_space[a] = tuple(new_s)

        return neighs, out_of_space

    def transition(self, s, a, s_prime):

        # If all objects are stacked up at the goal position, do not transition
        if s[2:4] == s[4:6] == s[6:8] == s[8:]:
            return 1 if s_prime == s else 0

        prob = 0
        neighs, out_of_space = self.neighbors(s)

        # ### Non diagonal Movements ####
        if a in ['left', 'right', 'up', 'down']:
            if a in out_of_space.keys():
                if s_prime == s:
                    prob = 1
                else:
                    prob = 0
            else:
                if tuple(s_prime) == neighs[a]:
                    prob = 1
                else:
                    prob = 0
        return prob

    def compute_transition_sas(self):
        """Creates transition matrix T(s,a,s') = T(s'|s, a)"""
        trans_matrix = np.zeros([
            len(self.states_dict), self.num_actions, len(self.states_dict)]
        )

        # Computing Transition Matrix
        with tqdm(total=self.num_actions*(len(self.states_dict)**2)) as pbar:
            for s in self.states_dict:
                for sprime in self.states_dict:
                    for a in self.action_dict:
                        sidx = self.states_dict[s]
                        spidx = self.states_dict[sprime]
                        aidx = self.action_dict[a]
                        trans_matrix[sidx, aidx, spidx] = self.transition(list(s), a, list(sprime))
                        pbar.update(1)

        return trans_matrix

    def compute_transition_xsax(self):
        trans_matrix = np.zeros([self.nx1, len(self.states_dict), self.num_actions, self.nx1])

        for s in self.states:
            sidx = self.states_dict[s]
            agent_pos, obj1_pos, obj2_pos, obj3_pos, goal_pos = s[:2], s[2:4], s[4:6], s[6:8], s[8:]

            if agent_pos == goal_pos:
                # TODO: How to assign unique identity to objects being collected? Below logic assumes if x=0 agent has brought obj1 <- Motivation for using Semi-Supervision?

                # If only one object has been brought to the goal, switch the latent mode to one of the remaining two
                if goal_pos == obj1_pos != obj2_pos != obj3_pos:
                    trans_matrix[0, sidx, :] = [0, 0.5, 0.5]  # Not assigning prob. for trans_matrix[1/2, sidx, :]
                elif goal_pos == obj2_pos != obj1_pos != obj3_pos:
                    trans_matrix[1, sidx, :] = [0.5, 0, 0.5]
                elif goal_pos == obj3_pos != obj1_pos != obj2_pos:
                    trans_matrix[2, sidx, :] = [0.5, 0.5, 0.]
                # If two objects have been brought to the goal, switch the latent mode to remaining one
                elif goal_pos == obj1_pos == obj2_pos != obj3_pos:
                    trans_matrix[0, sidx, :] = [0, 0, 1.]
                    trans_matrix[1, sidx, :] = [0, 0, 1.]
                elif goal_pos == obj1_pos == obj3_pos != obj2_pos:
                    trans_matrix[0, sidx, :] = [0, 1., 0]
                    trans_matrix[2, sidx, :] = [0, 1., 0]
                elif goal_pos == obj2_pos == obj3_pos != obj1_pos:
                    trans_matrix[1, sidx, :] = [1., 0, 0]
                    trans_matrix[2, sidx, :] = [1., 0, 0]

                # If agent does not bring any object to goal
                else:
                    trans_matrix[0, sidx, :, 0] = 1.
                    trans_matrix[1, sidx, :, 1] = 1.
                    trans_matrix[2, sidx, :, 2] = 1.
            else:
                trans_matrix[0, sidx, :, 0] = 1.
                trans_matrix[1, sidx, :, 1] = 1.
                trans_matrix[2, sidx, :, 2] = 1.

        return trans_matrix

    def reward(self, obj_id):
        """
        Args:
            obj_id: 0/1/2
        """

        reward = -np.ones([len(self.states), self.num_actions])
        # Positive reward for picking up the object indicated by obj_id

        rel_obj_x = 2 + obj_id*2
        irrel_obj1_x = (rel_obj_x+2) if rel_obj_x+2 <= 6 else (rel_obj_x+2) % 6
        irrel_obj2_x = (rel_obj_x + 4) if rel_obj_x+4 <= 6 else (rel_obj_x+4) % 6

        for s in self.states:
            sidx = self.states_dict[s]
            agent_pos, rel_obj, irr_obj1, irr_obj2, goal_pos = s[:2], s[rel_obj_x:rel_obj_x+2], \
                                                               s[irrel_obj1_x:irrel_obj1_x+2], \
                                                               s[irrel_obj2_x:irrel_obj2_x+2], \
                                                               s[8:]

            # Positive reward for agents collecting the reward
            if agent_pos != rel_obj and agent_pos != goal_pos:
                nr_s, _ = self.neighbors(s)
                for action, s_prime in nr_s.items():
                    new_agent_pos = list(s_prime)[:2]
                    if new_agent_pos == rel_obj:
                        aidx = self.action_dict[action]
                        reward[sidx, aidx] = 100.

            # positive reward for bringing the object to goal
            elif agent_pos == rel_obj and agent_pos != goal_pos:
                nr_s, _ = self.neighbors(s)
                for action, s_prime in nr_s.items():
                    new_agent_pos = list(s_prime)[:2]
                    aidx = self.action_dict[action]
                    if new_agent_pos == goal_pos:
                        reward[sidx, aidx] = 100.
                    # else:
                    #     reward[sidx, aidx] = 0.

            # Negative reward for collecting and depositing the object not matching the object ID
            for irr_obj in [irr_obj1, irr_obj2]:
                if agent_pos != irr_obj and agent_pos != goal_pos:
                    nr_s, _ = self.neighbors(s)
                    for action, s_prime in nr_s.items():
                        new_agent_pos = list(s_prime)[:2]
                        if new_agent_pos == irr_obj:
                            aidx = self.action_dict[action]
                            reward[sidx, aidx] = -100.

        return reward

    def compute_policy(self):
        """pi(x1, s, a) = pi(a|s, x1)"""
        policy = []
        for x in range(self.nx1):
            reward = self.reward(x)
            _, _, q_value = mdp_plan.value_iteration(
                self.obstate_transition_matrix,
                reward,
                max_iteration=1000
            )
            x_pol = mdp_lib.softmax_policy_from_q_value(q_value, self.temperatures[x])
            policy.append(x_pol)
        return np.array(policy)

    def get_demonstrations(self, max_len=100, quantity=1, start=None):
        demos = []
        latent = []
        terminal = [(*obj, 0, 0) for obj in self.obj_interest]
        terminal = [self.states_dict[j] for j in terminal]
        for _ in range(quantity):
            start_s, terminal_s = self.get_initial_state()
            sidx = self.states_dict[start_s]
            demo, lat = self.generate_normal(max_len, sidx)

            for j in range(len(demo)):
                if demo[j, 0] in terminal:
                    break
            demo = demo[:j + 1]
            lat = lat[:j + 1]
            demos.append(demo)
            latent.append(lat)

        return demos, latent