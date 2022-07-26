import numpy as np

from py_amm.amm import Amm


class RoadWorld(Amm):

    def __init__(self, grid_length_x=2, grid_length_y=5, root_dir=None):
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
        super(RoadWorld, self).__init__(
            obstate_trans_matrix=self.obstate_transition_matrix,
            num_obstates=len(self.states_dict),
            num_actions=5,
            num_states=[2, 2],
            init_state_weights=self.bx,
            transition_matrix=self.trans_xsax,
            policy_matrix=self.policy_matrix)

    def get_demonstrations(self, max_len=100, quantity=1, start=None):
        demos = []
        latent = []

        for _ in range(quantity):
            start_obstate = np.random.randint([self.grid_length_x, self.grid_length_y])
            sidx = self.states_dict[tuple(start_obstate)]
            demo, lat = self.generate_normal(max_len, sidx)
            goal = self.goals[0] if lat[0, 0] == 0 else self.goals[1]
            goal = self.states_dict[tuple(goal)]

            idx = list(demo[:, 0]).index(goal)
            demo = demo[:idx + 1]
            lat = lat[:idx + 1]
            demos.append(demo)
            latent.append(lat)

        return demos, latent

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
        # trans_matrix = np.zeros([len(self.states_dict), 4, len(self.states_dict)])
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
        # trans_matrix = np.zeros([2, 2, len(self.states_dict), 4, 2, 2])
        trans_matrix[0, 0, :, :, 0, 0] = 1
        trans_matrix[0, 1, :, :, 0, 1] = 1
        trans_matrix[1, 0, :, :, 1, 0] = 1
        trans_matrix[1, 1, :, :, 1, 1] = 1
        return trans_matrix

    def compute_policy(self):
        """pi(s,x1,x2,a) = pi(a|s, x1,x2)"""
        pol = np.zeros(shape=[2, 2, len(self.states_dict), 5])  # [latent1, latent2, states, actions]
        # pol = np.zeros(shape=[2, 2, len(self.states_dict), 4])  # [latent1, latent2, states, actions]

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