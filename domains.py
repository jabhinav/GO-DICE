import itertools
import numpy as np
from tqdm import tqdm
import np_mdp.models.mdp as mdp_lib
import np_mdp.algorithms.planning as mdp_plan
from copy import deepcopy as copy


class RoadWorld(object):

    def __init__(self, grid_length_x=2, grid_length_y=5):
        """Create simple domain class
        Params:
            length (int): length of grid
            latent1 (int): which goal to collect; the one on the
                           right or the one on the left.
                           0 -- left
                           1 -- right
            latent2 (int): which route to take; upper or lower route
                           0 -- upper
                           1 -- lower
            [[0,0], [0,1], [1,0], [1,1]] -> [0, 1, 2, 3]


        """
        self.grid_length_x = grid_length_x
        self.grid_length_y = grid_length_y
        self.goals = [[1, 0], [1, grid_length_y - 1]]

        self.pos = [0, int(grid_length_y / 2)]
        # self.latent1 = latent1
        # self.latent2 = latent2

        # self.bx = np.zeros(4)
        tmp = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
        self.latent_dict = tmp
        self.inv_latent_dict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
        # self.bx[tmp[(latent1, latent2)]] = 1
        self.bx = [0.25, 0.25, 0.25, 0.25]

        self.obstate_transition_matrix = self.compute_transition_sas()
        self.trans_xsax = self.compute_transition_xsax()
        self.policy_matrix = self.compute_policy()
        self.latent_modes_x1 = 2
        self.latent_modes_x2 = 2
        # super(Simple, self).__init__(
        #     obstate_trans_matrix=self.obstate_transition_matrix,
        #     num_obstates=len(self.states_dict),
        #     num_actions=5,
        #     num_states=2,
        #     num_features=2,
        #     init_state_weights=self.bx,
        #     transition_matrix=self.trans_xsax,
        #     policy_matrix=self.policy_matrix)

    def get_initial_state(self, start=None):
        x_start = np.random.randint(np.ceil(self.grid_length_x / 2).astype('int8'), self.grid_length_x)
        y_start = np.random.randint(self.grid_length_y)
        sidx = self.states_dict[(x_start, y_start)]

        # Here the terminal state gets fixed once the latent mode is known <- will not work for InfoGAIL
        # goal = self.goals[0] if self.latent1 == 0 else self.goals[1]
        # goal = self.states_dict[tuple(goal)]
        return sidx, [self.states_dict[tuple(goal)] for goal in self.goals]

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
            'left': 3,
            'stay': 4
        }
        return act_dict

    @property
    def inv_action_dict(self):
        inv_act_dict = {
            0: 'up',
            1: 'down',
            2: 'right',
            3: 'left',
            4: 'stay'
        }
        return inv_act_dict

    @property
    def states_dict(self):
        st_list = [[i, j] for i in range(self.grid_length_x) for j in range(self.grid_length_y)]
        st_dict = {}
        count = 0
        for st in st_list:
            st_dict[tuple(st)] = count
            count += 1
        return st_dict

    @property
    def inv_states_dict(self):
        st_list = [[i, j] for i in range(self.grid_length_x) for j in range(self.grid_length_y)]
        inv_st_dict = {}
        count = 0
        for st in st_list:
            inv_st_dict[count] = st
            count += 1
        return inv_st_dict

    @property
    def latent_state_dict(self):
        # latent_states = [[i, j] for i in range(self.latent_modes_x1) for j in range(self.latent_modes_x2)]
        # latent_dict = {}
        # count = 0
        # for st in latent_states:
        #     latent_dict[tuple(st)] = count
        #     count += 1
        return self.latent_dict

    def neighbors(self, s):
        neighs = {}
        out_of_space = {}
        steps = {
            'up': [-1, 0],
            'right': [0, 1],
            'left': [0, -1],
            'down': [1, 0]
        }
        for a in steps.keys():
            new_s = s + steps[a]
            if tuple(new_s) in self.states_dict:
                neighs[a] = tuple(new_s)
            else:
                # This means out of space
                out_of_space[a] = tuple(new_s)

        return neighs, out_of_space

    # def learner_transition(self, s, a):
    #     """
    #     :param s: Tuple
    #     :param a: action_id
    #     :return: next state if available based on env. dynamics
    #     """
    #     a = self.inv_action_dict[a]
    #     if a == 'stay':
    #         return s
    #     neighs, out_of_space = self.neighbors(np.array(s))
    #     if a in ['left', 'right', 'up', 'down']:
    #         if a in out_of_space.keys():
    #             return None
    #         else:
    #             return neighs[a]

    def transition(self, s, a, s_prime):
        # Although the transition function defined in the book
        # depends also on r, it is not necessary to put r for
        # this MDP because it is deterministic
        if a == 'stay':
            # return 0
            return 1 if s_prime == s else 0

        prob = 0
        neighs, out_of_space = self.neighbors(np.array(s))

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
        trans_matrix = np.zeros([len(self.states_dict), 5, len(self.states_dict)])
        for s in self.states_dict:
            for sprime in self.states_dict:
                for a in self.action_dict:
                    sidx = self.states_dict[s]
                    spidx = self.states_dict[sprime]
                    aidx = self.action_dict[a]
                    trans_matrix[sidx, aidx, spidx] = self.transition(
                        list(s),
                        a,
                        list(sprime)
                    )

        return trans_matrix

    def compute_transition_xsax(self):
        """Creates transition matrix T(x1, x2,s,a,x1',x2') = T(x1',x2'|x1,x2, s, a)"""

        trans_matrix = np.zeros([2, 2, len(self.states_dict), 5, 2, 2])
        trans_matrix[0, 0, :, :, 0, 0] = 1
        trans_matrix[0, 1, :, :, 0, 1] = 1
        trans_matrix[1, 0, :, :, 1, 0] = 1
        trans_matrix[1, 1, :, :, 1, 1] = 1
        return trans_matrix

    def compute_policy(self):
        """pi(s,x1,x2,a) = pi(a|s, x1,x2)"""
        pol = np.zeros(shape=[2, 2, len(self.states_dict), 5])  # [latent1, latent2, states, actions]

        # Pick left goal taking upper route
        for s in self.states_dict:
            sidx = self.states_dict[s]
            if list(s) == self.goals[0]:
                pol[0, 0, sidx, self.action_dict['stay']] = 1
            elif list(s) == [0, 0]:
                pol[0, 0, sidx, self.action_dict['down']] = 1
            elif s[0] == 0:
                pol[0, 0, sidx, self.action_dict['left']] = 1
            else:
                pol[0, 0, sidx, self.action_dict['up']] = 1

        # Pick left goal taking lower route
        for s in self.states_dict:
            sidx = self.states_dict[s]
            if list(s) == self.goals[0]:
                pol[0, 1, sidx, self.action_dict['stay']] = 1
            elif list(s) == [self.grid_length_x - 1, 0]:
                pol[0, 1, sidx, self.action_dict['up']] = 1
            elif s[0] == self.grid_length_x - 1:
                pol[0, 1, sidx, self.action_dict['left']] = 1
            else:
                pol[0, 1, sidx, self.action_dict['down']] = 1

        # Pick right goal taking upper route
        for s in self.states_dict:
            sidx = self.states_dict[s]
            if list(s) == self.goals[1]:
                pol[1, 0, sidx, self.action_dict['stay']] = 1
            elif list(s) == [0, self.grid_length_y - 1]:
                pol[1, 0, sidx, self.action_dict['down']] = 1
            elif s[0] == 0:
                pol[1, 0, sidx, self.action_dict['right']] = 1
            else:
                pol[1, 0, sidx, self.action_dict['up']] = 1

        # Pick right goal taking lower route
        for s in self.states_dict:
            sidx = self.states_dict[s]
            if list(s) == self.goals[1]:
                pol[1, 1, sidx, self.action_dict['stay']] = 1
            elif list(s) == [self.grid_length_x - 1, self.grid_length_y - 1]:
                pol[1, 1, sidx, self.action_dict['up']] = 1
            elif s[0] == self.grid_length_x - 1:
                pol[1, 1, sidx, self.action_dict['right']] = 1
            else:
                pol[1, 1, sidx, self.action_dict['down']] = 1

        return pol


class BoxWorld(object):
    def __init__(
            self,
            grid_length_x=2, grid_length_y=5,
            obj_interest=None,
            danger=None,
            obstacles=None):
        """Create simple domain class
        Params:
            length (int): length of grid
            latent1 (int): tired or not. This affects sensors
                           0 -- not tired
                           1 -- tired
                           2 -- very tired
            latent2 (int): protocol to follow
                           0 -- don't avoid danger
                           1 -- avoid danger

        """
        # Pick objects of interest and avoid obstacles and danger [DEFAULT SETTING]
        if obj_interest is None:
            obj_interest = [[0, 0], [1, 1]]
        if danger is None:
            danger = [[2, 1], [3, 4]]
        if obstacles is None:
            obstacles = [[2, 0], [1, 2], [2, 2], [4, 2]]

        self.grid_length_x = grid_length_x
        self.grid_length_y = grid_length_y
        self.nx1 = 3
        self.nx2 = 2
        self.temperatures = [0.01, 1.5, 2]

        self.obj_interest = obj_interest
        self.danger = danger
        self.obstacles = obstacles

        self.states = [(i, j, k, l) for i in range(grid_length_x)
                       for j in range(grid_length_y)
                       for k in [0, 1]  # presence of object of interest at location 1
                       for l in [0, 1]  # presence of object of interest at location 2
                       if [i, j] not in obstacles]  # Obstacles do not constitute a valid state

        # self.states_dict = {self.states[j]: j for j in range(len(self.states))}

        self.obstate_transition_matrix = self.compute_transition_sas()

        self.bx = np.zeros(6)
        tmp = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3, (2, 0): 4, (2, 1): 5}
        self.bx[tmp[(0, 0)]] = 0.25
        self.bx[tmp[(0, 1)]] = 0.25
        self.bx[tmp[(1, 0)]] = 0.25
        self.bx[tmp[(1, 1)]] = 0.25

        self.latent_dict = tmp
        self.inv_latent_dict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1), 4: (2, 0), 5: (2, 1)}
        self.latent_modes_x1 = 3
        self.latent_modes_x2 = 2

        self.trans_xsax = self.compute_transition_xsax()
        self.policy_matrix = self.compute_policy()

        # super(BoxWorld, self).__init__(
        #     obstate_trans_matrix=self.obstate_transition_matrix,
        #     num_obstates=len(self.states),
        #     num_actions=4,
        #     num_states=[self.nx1, self.nx2],
        #     init_state_weights=self.bx,
        #     transition_matrix=self.trans_xsax,
        #     policy_matrix=self.policy_matrix
        # )

    def get_initial_state(self, start=None):
        sidx = None
        status_objs = [0 for _ in range(len(self.obj_interest))]
        terminal = []

        # NOTE: Make sure that obj_interest is rep. in 2d since *obj is not available in python2 to unwrap its values
        for obj in self.obj_interest:
            terminal.append((obj[0], obj[1], 0, 0))
        terminal = [self.states_dict[j] for j in terminal]

        start_obstate = start if start is not None else self.obstacles[0]
        # The following loop makes sure that we don't start from positions of obj of interest, obstacles and dangers
        while start_obstate in self.obstacles or start_obstate in self.danger or start_obstate in self.obj_interest:
            x_start = np.random.randint(np.ceil(self.grid_length_x / 2).astype('int8'), self.grid_length_x)
            y_start = np.random.randint(self.grid_length_y)
            start_obstate = [x_start, y_start]

        sidx = self.states_dict[(x_start, y_start, 1, 1)]

        return sidx, terminal

    # def get_demonstrations(self, max_len=100, quantity=1, start=None):
    #     demos = []
    #     latent = []
    #     terminal = [(*obj, 0, 0) for obj in self.obj_interest]
    #     terminal = [self.states_dict[j] for j in terminal]
    #     for _ in range(quantity):
    #         start_obstate = start if start is not None else self.obstacles[0]
    #         while start_obstate in self.obstacles or \
    #                 start_obstate in self.danger or \
    #                 start_obstate in self.obj_interest:
    #             x_start = np.random.randint(
    #                 np.ceil(self.side_len / 2).astype('int8'),
    #                 self.side_len)
    #             y_start = np.random.randint(self.side_len)
    #             start_obstate = [x_start, y_start]
    #         sidx = self.states_dict[(x_start, y_start, 1, 1)]
    #
    #         demo, lat = self.generate_normal(max_len, sidx)
    #
    #         for j in range(len(demo)):
    #             if demo[j, 0] in terminal:
    #                 break
    #         demo = demo[:j + 1]
    #         lat = lat[:j + 1]
    #         demos.append(demo)
    #         latent.append(lat)
    #
    #     return demos, latent

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
        return act_dict

    @property
    def inv_action_dict(self):
        inv_act_dict = {
            0: 'up',
            1: 'down',
            2: 'right',
            3: 'left',
        }
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
        # latent_states = [[i, j] for i in range(self.latent_modes_x1) for j in range(self.latent_modes_x2)]
        # latent_dict = {}
        # count = 0
        # for st in latent_states:
        #     latent_dict[tuple(st)] = count
        #     count += 1
        return self.latent_dict

    def neighbors(self, s):
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
        objs = s[2:]
        for a in steps.keys():
            new_s = pos + steps[a]
            if list(new_s) == self.obj_interest[0]:
                new_s = np.concatenate([new_s, [0, objs[1]]])
            elif list(new_s) == self.obj_interest[1]:
                new_s = np.concatenate([new_s, [objs[0], 0]])
            else:
                new_s = np.concatenate([new_s, objs])

            if tuple(new_s) in self.states:
                neighs[a] = tuple(new_s)
            else:
                # This means out of space
                out_of_space[a] = tuple(new_s)

        return neighs, out_of_space

    def transition(self, s, a, s_prime):
        if s[2:] == [0, 0]:
            return 1 if s_prime == s else 0

        prob = 0
        neighs, out_of_space = self.neighbors(np.array(s))

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
            len(self.states_dict), 4, len(self.states_dict)]
        )

        for s in self.states_dict:
            for sprime in self.states_dict:
                for a in self.action_dict:
                    sidx = self.states_dict[s]
                    spidx = self.states_dict[sprime]
                    aidx = self.action_dict[a]
                    trans_matrix[sidx, aidx, spidx] = self.transition(list(s), a, list(sprime))

        return trans_matrix

    def compute_transition_xsax(self):
        """Creates transition matrix T(x1, x2,s,a,x1',x2') = T(x1',x2'|x1,x2, s, a)

        When x2 = 0, which is avoid the danger squares, the latent states remain constant
        When x2 = 1, then x1 increases when in a danger square"""

        trans_matrix = np.zeros([self.nx1, self.nx2, len(self.states_dict), 4, self.nx1, self.nx2])
        # trans_matrix[2, 0, :, :, 2, 0] = 1
        # trans_matrix[2, 1, :, :, 2, 1] = 1
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

    def reward(self, avoid_danger):
        reward = -np.ones([len(self.states), 4])
        # Positive reward for picking up the object
        n_objs = len(self.obj_interest)
        for j in range(n_objs):

            obj = self.obj_interest[j]
            status = [0] * j + [1] + [0] * (n_objs - j - 1)
            state_interest = []
            state_interest.extend(obj)
            state_interest.extend(status)

            neighs, _ = self.neighbors(state_interest)
            for a in neighs:
                a_op = self.opposite_action(a)
                aidx = self.action_dict[a_op]
                sidxp = self.states_dict[tuple(neighs[a])]
                reward[sidxp, aidx] = 100

        for s in self.states:
            pos = s[:2]
            status = s[2:]
            sidx = self.states_dict[s]
            if list(status) == [0, 0]:
                reward[sidx] = 0
            elif list(pos) in self.danger:
                if avoid_danger:
                    reward[sidx] = -100

        return reward

    def compute_policy(self):
        """pi(x1,x2, s, a) = pi(a|s, x1,x2)"""
        policy = []
        for temperature in self.temperatures:
            temp_pol = []
            for b in [True, False]:
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


class TeamBoxWorld(object):
    def __init__(
            self,
            grid_length_x=5, grid_length_y=5,
            obj_interest=None,
            danger=None,
            obstacles=None, static=False,
            transition_x=None,
            transition_s=None,
            policy=None):
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
        if obstacles is None:
            obstacles = [[2, 0], [1, 2], [2, 2], [4, 2]]
        if danger is None:
            danger = [[2, 1], [3, 4]]
        if obj_interest is None:
            obj_interest = [[0, 0], [1, 1]]
        self.nx1 = 3
        self.nx2 = 2
        self.temperatures = [0.01, 7, 12]
        self.grid_length_x = grid_length_x
        self.grid_length_y = grid_length_y

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
        self.bx[tmp[(0, 0)]] = 0.25
        self.bx[tmp[(0, 1)]] = 0.25
        self.bx[tmp[(1, 0)]] = 0.25
        self.bx[tmp[(1, 1)]] = 0.25

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

        # super(TeamBoxWorld, self).__init__(
        #     obstate_trans_matrix=self.obstate_transition_matrix,
        #     num_obstates=len(self.states),
        #     num_actions=len(self.action_dict),
        #     num_states=[self.nx1, self.nx2],
        #     init_state_weights=self.bx,
        #     transition_matrix=self.trans_xsax,
        #     policy_matrix=self.policy_matrix
        # )

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

    # def get_demonstrations(self, max_len=500, quantity=1, start=None):
    #     demos = []
    #     latent = []
    #     status_objs = [0 for j in range(len(self.obj_interest))]
    #     terminal = [
    #         (*obj, *status_objs, r_goal)
    #         for obj in self.obj_interest
    #         for r_goal in range(-1, len(self.obj_interest))
    #     ]
    #     terminal = [self.states_dict[j] for j in terminal]
    #     for _ in range(quantity):
    #         start_obstate = start if start is not None else self.obstacles[0]
    #         while start_obstate in self.obstacles or \
    #                 start_obstate in self.danger or \
    #                 start_obstate in self.obj_interest:
    #             x_start = np.random.randint(
    #                 np.ceil(self.side_len / 2).astype('int8'),
    #                 self.side_len)
    #             y_start = np.random.randint(self.side_len)
    #             start_obstate = [x_start, y_start]
    #
    #         all_objs_on = [1 for _ in range(len(self.obj_interest))]
    #         sidx = self.states_dict[(*start_obstate, *all_objs_on, -1)]
    #
    #         demo, lat = self.generate_normal(max_len, sidx)
    #
    #         for j in range(len(demo)):
    #             if demo[j, 0] in terminal:
    #                 break
    #         demo = demo[:j + 1]
    #         lat = lat[:j + 1]
    #         demos.append(demo)
    #         latent.append(lat)
    #
    #     return demos, latent

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
                    if s == s_prime:
                        prob = 1. - self.robot_pickup_param
                    elif objs_p == pickedup and pos_c == pos_p and goal_p == -1:
                        prob = self.robot_pickup_param
            else:
                if goal_c == -1:
                    if tuple(s_prime) == neighs[a]:
                        prob = 1.
                # If the object is already picked up
                elif objs_c[goal_c] == 0:
                    if tuple(s_prime)[:-1] == neighs[a][:-1] and goal_p == -1:
                        prob = 1.
                else:
                    pickedup = copy(objs_c)
                    pickedup[goal_c] = 0
                    if tuple(s_prime) == neighs[a]:
                        prob = 1. - self.robot_pickup_param
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
        """Creates transition matrix T(x1, x2,s,a,x1',x2') = T(x1',x2'|x1,x2, s, a)

        When x2 = 0, which is avoid the danger squares, the latent states remain constant
        When x2 = 1, then x1 increases when in a danger square"""

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
