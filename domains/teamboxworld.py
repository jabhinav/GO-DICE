import itertools
from copy import deepcopy as copy

import numpy as np
from tqdm import tqdm

from np_mdp.algorithms import planning as mdp_plan
from np_mdp.models import mdp as mdp_lib
from py_amm.amm import Amm


class TeamBoxWorld(Amm):
    def __init__(
            self,
            grid_length_x=5, grid_length_y=5,
            obj_interest=None,
            danger=None,
            obstacles=None, static=False,
            transition_x=None,
            transition_s=None,
            policy=None,
            temperatures=None, root_dir=None):
        """Create simple domain class where agent and a robot achieves the desired objective together
        Params:
            length (int): length of grid
            latent1 (int): Fatigue.
                           0 -- low
                           1 -- medium
                           2 -- high
            latent2 (int): Trust
                           0 -- low
                           1 -- high

        """
        if temperatures is None:
            temperatures = [0.01, 7, 12]
        if obstacles is None:
            obstacles = [[2, 0], [1, 2], [2, 2], [4, 2]]
        if danger is None:
            danger = [[2, 1], [3, 4]]
        if obj_interest is None:
            obj_interest = [[0, 0], [1, 1]]
        self.nx1 = 3
        self.nx2 = 2
        self.temperatures = temperatures
        self.grid_length_x = grid_length_x
        self.grid_length_y = grid_length_y

        # Probability that the robot picks up the object: Larger the grid, lower the prob
        self.robot_pickup_param = 1. / float(grid_length_x)

        self.obj_interest = obj_interest
        self.danger = danger
        self.obstacles = obstacles
        perm = (range(j) for j in [2] * len(obj_interest))
        obj_statuses = list(itertools.product(*perm))
        self.states = [(i, j, status[0], status[1], g)
                       for i in range(grid_length_x)
                       for j in range(grid_length_y)
                       for status in obj_statuses  # presence of object of interest at location 2
                       for g in range(-1, len(obj_interest))
                       if [i, j] not in obstacles]

        if transition_s is not None:
            self.obstate_transition_matrix = transition_s
        else:
            self.obstate_transition_matrix = self.compute_transition_sas()

        self.bx = np.zeros(6)
        tmp = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3, (2, 0): 4, (2, 1): 5}
        # self.bx[tmp[(0, 0)]] = 0.25
        # self.bx[tmp[(0, 1)]] = 0.25
        # self.bx[tmp[(1, 0)]] = 0.25
        # self.bx[tmp[(1, 1)]] = 0.25

        self.bx[tmp[(0, 0)]] = 0.17
        self.bx[tmp[(0, 1)]] = 0.17
        self.bx[tmp[(1, 0)]] = 0.17
        self.bx[tmp[(1, 1)]] = 0.17
        self.bx[tmp[(2, 0)]] = 0.16
        self.bx[tmp[(2, 1)]] = 0.16

        self.latent_dict = tmp
        self.inv_latent_dict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1), 4: (2, 0), 5: (2, 1)}

        if transition_x is not None:
            self.trans_xsax = transition_x
        else:
            self.trans_xsax = self.compute_transition_xsax(static)
        if policy is not None:
            self.policy_matrix = policy
        else:
            self.policy_matrix = self.compute_policy()

        super(TeamBoxWorld, self).__init__(
            obstate_trans_matrix=self.obstate_transition_matrix,
            num_obstates=len(self.states),
            num_actions=len(self.action_dict),
            num_states=[self.nx1, self.nx2],
            init_state_weights=self.bx,
            transition_matrix=self.trans_xsax,
            policy_matrix=self.policy_matrix
        )

    def get_initial_state(self, start=None):
        sidx = None
        terminal_status_objs = [0 for _ in range(len(self.obj_interest))]
        terminal = []
        for obj in self.obj_interest:
            for r_goal in range(-1, len(self.obj_interest)):
                terminal.append((obj[0], obj[1], terminal_status_objs[0], terminal_status_objs[1], r_goal))
        terminal = [self.states_dict[j] for j in terminal]

        start_obstate = start if start is not None else self.obstacles[0]
        while start_obstate in self.obstacles or start_obstate in self.danger or start_obstate in self.obj_interest:
            x_start = np.random.randint(np.ceil(self.grid_length_x / 2).astype('int8'), self.grid_length_x)
            y_start = np.random.randint(self.grid_length_y)
            start_obstate = [x_start, y_start]

        starting_status_objs = [1 for _ in range(len(self.obj_interest))]
        sidx = self.states_dict[(start_obstate[0], start_obstate[1], starting_status_objs[0], starting_status_objs[1], -1)]

        return sidx, terminal

    def get_demonstrations(self, max_len=500, quantity=1, start=None):
        demos = []
        latent = []
        status_objs = [0 for j in range(len(self.obj_interest))]
        terminal = [
            (*obj, *status_objs, r_goal)
            for obj in self.obj_interest
            for r_goal in range(-1, len(self.obj_interest))
        ]
        terminal = [self.states_dict[j] for j in terminal]
        for _ in range(quantity):
            start_obstate = start if start is not None else self.obstacles[0]
            while start_obstate in self.obstacles or \
                    start_obstate in self.danger or \
                    start_obstate in self.obj_interest:
                x_start = np.random.randint(np.ceil(self.grid_length_x / 2).astype('int8'), self.grid_length_x)
                y_start = np.random.randint(self.grid_length_y)
                start_obstate = [x_start, y_start]

            all_objs_on = [1 for _ in range(len(self.obj_interest))]
            sidx = self.states_dict[(*start_obstate, *all_objs_on, -1)]

            demo, lat = self.generate_normal(max_len, sidx)

            for j in range(len(demo)):
                if demo[j, 0] in terminal:
                    break
            demo = demo[:j + 1]
            lat = lat[:j + 1]
            demos.append(demo)
            latent.append(lat)

        return demos, latent

    def set_latent1(self, latent1):
        self.latent1 = latent1

    def set_latent2(self, latent2):
        self.latent2 = latent2

    @property
    def action_dict(self):
        act_dict = {
            'up': 0,
            'down': 1,
            'right': 2,
            'left': 3
        }
        for obj in range(len(self.obj_interest)):
            act_dict['toObj' + str(obj)] = 4 + obj

        return act_dict

    @property
    def inv_action_dict(self):
        inv_act_dict = {
            0: 'up',
            1: 'down',
            2: 'right',
            3: 'left',
        }
        for obj in range(len(self.obj_interest)):
            inv_act_dict[4 + obj] = 'toObj' + str(obj)

        return inv_act_dict

    @property
    def states_dict(self):
        states_dict = {self.states[j]: j for j in range(len(self.states))}
        return states_dict

    @property
    def inv_states_dict(self):
        inv_states_dict = {j: self.states[j] for j in range(len(self.states))}
        return inv_states_dict

    @property
    def latent_state_dict(self):
        return self.latent_dict

    @property
    def sendto(self):
        sendto_dict = {}
        for obj in range(len(self.obj_interest)):
            sendto_dict['toObj' + str(obj)] = obj
        return sendto_dict

    def neighbors(self, s):
        """Finds the neighbors on the four directions of the agent.
        If an object of interest is in the neighborhood, it is automatically
        picked up when moving to that grid."""
        if not isinstance(s, np.ndarray):
            s = np.array(s)

        neighs = {}
        out_of_space = {}
        steps = {
            'up': [-1, 0],
            'right': [0, 1],
            'left': [0, -1],
            'down': [1, 0]
        }

        pos = s[:2]
        objs = s[2:-1]
        goal_robot = [s[-1]]
        for a in steps.keys():
            new_s = pos + steps[a]
            # Check if by moving we enter a grid with an object of interest
            # This is to automatically pick it up.
            if list(new_s) in self.obj_interest:
                idx = self.obj_interest.index(list(new_s))
                tmp = copy(objs)
                tmp[idx] = 0
                new_s = np.concatenate([new_s, tmp, goal_robot])
            # If not a grid with object of interest, just move to the grid
            else:
                new_s = np.concatenate([new_s, objs, goal_robot])

            # Check if the new state is inside the grid
            if tuple(new_s) in self.states:
                neighs[a] = tuple(new_s)
            else:
                # This means out of space
                out_of_space[a] = tuple(new_s)
        return neighs, out_of_space

    def transition(self, s, a, s_prime):
        if s[2:-1] == np.zeros(len(self.obj_interest)).tolist():
            return 1. if s_prime == s else 0.

        prob = 0
        neighs, out_of_space = self.neighbors(s)

        pos_c = s[:2]
        objs_c = s[2:-1]
        goal_c = s[-1]
        pos_p = s_prime[:2]
        objs_p = s_prime[2:-1]
        goal_p = s_prime[-1]

        # ## Move robot actions ## #
        # P(s_prime = (x,y,o1,o2,i) | s = (x,y,o1,o2,_), a = toObj_i)
        if a in self.sendto:
            if s[:-1] == s_prime[:-1]:
                if goal_p == self.sendto[a]:
                    prob = 1.

        # ### Non diagonal Movements ### #

        elif a in ['left', 'right', 'up', 'down']:
            if a in out_of_space.keys():
                if goal_c == -1:
                    if s == s_prime:
                        prob = 1.
                # If the object is already picked up
                elif objs_c[goal_c] == 0:
                    if s[:-1] == s_prime[:-1] and goal_p == -1:
                        prob = 1.
                else:
                    pickedup = copy(objs_c)
                    pickedup[goal_c] = 0
                    # P((x, y, o1, o2, 0/1) | (x, y, o1, o2, 0/1), a)
                    if s == s_prime:
                        prob = 1. - self.robot_pickup_param
                    # P((x, y, 0, _, -1) | (x, y, 1, _, 0), a) = P((x, y, _, 0, -1) | (x, y, _, 1, 1), a) = some_prob
                    elif objs_p == pickedup and pos_c == pos_p and goal_p == -1:
                        prob = self.robot_pickup_param
            else:
                # If nav actions take to plausible state then transit with prob = 1 (when robot is not instructed)
                if goal_c == -1:
                    if tuple(s_prime) == neighs[a]:
                        prob = 1.
                # If the object specified to robot is already picked up
                elif objs_c[goal_c] == 0:
                    if tuple(s_prime)[:-1] == neighs[a][:-1] and goal_p == -1:
                        prob = 1.
                # If obj is not picked up and robot is instructed
                else:
                    pickedup = copy(objs_c)
                    pickedup[goal_c] = 0
                    # P((x, y, o1, o2, 0/1) | (x, y, o1, o2, 0/1), a)
                    # i.e. when instructed, robot does not pickup the obj with some prob (state stays the same)
                    if tuple(s_prime) == neighs[a]:
                        prob = 1. - self.robot_pickup_param
                    # P((x, y, 0, _, -1) | (x, y, 1, _, 0), a) = P((x, y, _, 0, -1) | (x, y, _, 1, 1), a) = some_prob
                    # i.e. robot picks up the obj with some prob
                    elif objs_p == pickedup and neighs[a][:2] == tuple(pos_p) and goal_p == -1:
                        prob = self.robot_pickup_param

        return prob

    def compute_transition_sas(self):
        """Creates transition matrix T(s,a,s') = T(s'|s, a)"""
        trans_matrix = np.zeros([
            len(self.states_dict), len(self.action_dict), len(self.states_dict)]
        )
        count = 0
        with tqdm(total=len(self.states_dict) ** 2 * len(self.action_dict)) as pbar:
            for s in self.states_dict:
                for sprime in self.states_dict:
                    for a in self.action_dict:
                        # print('Computed' + str(count))
                        count += 1
                        sidx = self.states_dict[s]
                        spidx = self.states_dict[sprime]
                        aidx = self.action_dict[a]
                        trans_matrix[sidx, aidx, spidx] = self.transition(list(s), a, list(sprime))
                        pbar.update(1)

        return trans_matrix

    def compute_transition_xsax(self, static=False):
        """Creates transition matrix T(x1, x2,s,a,x1',x2') = T(x1',x2'|x1,x2, s, a)"""

        trans_matrix = np.zeros([self.nx1, self.nx2, len(self.states_dict),
                                 len(self.action_dict), self.nx1, self.nx2])

        if static:
            trans_matrix[0, 0, :, :, 0, 0] = 1
            trans_matrix[0, 1, :, :, 0, 1] = 1
            trans_matrix[1, 0, :, :, 1, 0] = 1
            trans_matrix[1, 1, :, :, 1, 1] = 1
            trans_matrix[2, 0, :, :, 2, 0] = 1
            trans_matrix[2, 1, :, :, 2, 1] = 1

        else:
            for s in self.states:
                sidx = self.states_dict[s]
                if list(s[:2]) in self.danger:
                    trans_matrix[0, 0, sidx, :, 1, 0] = 1
                    trans_matrix[0, 1, sidx, :, 1, 1] = 1
                    trans_matrix[1, 0, sidx, :, 2, 0] = 1
                    trans_matrix[1, 1, sidx, :, 2, 1] = 1
                    trans_matrix[2, 0, sidx, :, 2, 0] = 1
                    trans_matrix[2, 1, sidx, :, 2, 1] = 1
                else:
                    trans_matrix[0, 0, sidx, :] = [[0.7, 0.1], [0.1, 0.1], [0, 0]]
                    trans_matrix[0, 1, sidx, :] = [[0.1, 0.8], [0.05, 0.05], [0, 0]]
                    trans_matrix[1, 0, sidx, :] = [[0, 0], [0.8, 0.1], [0.1, 0]]
                    trans_matrix[1, 1, sidx, :] = [[0.1, 0.1], [0, 0.8], [0, 0]]
                    trans_matrix[2, 0, sidx, :] = [[0, 0], [0, 0], [0.8, 0.2]]
                    trans_matrix[2, 1, sidx, :] = [[0, 0], [0.1, 0.1], [0, 0.8]]

        return trans_matrix

    def reward(self, trust):
        """
        Args:
        avoid_danger:

        Returns: Reward for each (state, action) i.e.
        (a) +100 for (s, a) that collects obj of interests.
        (b) 0 for (s, a) where s[2:] = [0, 0] i.e. objects have been collected already (no reward)
        (c) -100 for (s, a) for states where action taken beings the agent to dangerous cell
        (d) -100 for (s, a) for taking 'sendTo' action that directs robot towards an obj when no trust on robot
        (e) +100 for (s, a) for taking 'sendTo' action that directs robot towards an uncollected obj when trust on robot (robot should not have been tasked to do something i.e. s[-1] == -1)
        (f) -100 for (s, a) for taking 'sendTo' action that directs robot towards an already collected obj when trust on robot (robot should not have been tasked to do something i.e. s[-1] == -1)
        (g) -100 for (s, a) for taking 'sendTo' action when robot has already been tasked with something i.e. when s[-1] != -1 and trust on robot [No overrides]
        (h) -1 otherwise
        """
        reward = -np.ones([len(self.states), len(self.action_dict)])

        # Positive reward for picking up the object
        n_objs = len(self.obj_interest)
        for j in range(n_objs):
            obj = self.obj_interest[j]
            status = [0] * j + [1] + [0] * (n_objs - j - 1)
            for r_goal in range(-1, len(self.obj_interest)):
                state_interest = []
                state_interest.extend(obj)
                state_interest.extend(status)
                state_interest.append(r_goal)

                # state_interest = [*obj, *status, r_goal]
                neighs, _ = self.neighbors(state_interest)
                for a in neighs:
                    a_op = self.opposite_action(a)
                    aidx = self.action_dict[a_op]
                    sidxp = self.states_dict[tuple(neighs[a])]
                    reward[sidxp, aidx] = 100

        # Reward for trust and for danger grids
        for s in self.states:
            pos = s[:2]
            status = s[2:-1]
            sidx = self.states_dict[s]
            if list(status) == list(np.zeros(len(self.obj_interest))):
                reward[sidx] = 0
            elif list(pos) in self.danger:
                reward[sidx] = -100
            elif not trust:
                for a in list(self.sendto.keys()):
                    aidx = self.action_dict[a]
                    reward[sidx, aidx] = -100
            elif trust:
                for a in list(self.sendto.keys()):
                    aidx = self.action_dict[a]
                    if s[-1] == -1:
                        if status[self.sendto[a]] == 1:
                            reward[sidx, aidx] = 100
                        else:
                            reward[sidx, aidx] = -100
                    else:
                        reward[sidx, aidx] = -100

        return reward

    def compute_policy(self):
        """pi(x1,x2, s, a) = pi(a|s, x1,x2)"""
        policy = []
        for temperature in self.temperatures:
            temp_pol = []
            for b in [False, True]:
                reward = self.reward(b)
                _, _, q_value = mdp_plan.value_iteration(
                    self.obstate_transition_matrix,
                    reward,
                    max_iteration=1000
                )
                pol = mdp_lib.softmax_policy_from_q_value(q_value, temperature)
                temp_pol.append(pol)
            policy.append(temp_pol)
        return np.array(policy)

    def opposite_action(self, a):
        opposite = {
            'up': 'down',
            'down': 'up',
            'right': 'left',
            'left': 'right'
        }
        return opposite[a]