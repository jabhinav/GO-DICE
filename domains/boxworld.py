import numpy as np

from np_mdp.algorithms import planning as mdp_plan
from np_mdp.models import mdp as mdp_lib
from py_amm.amm import Amm


class BoxWorld(Amm):
    def __init__(
            self,
            grid_length_x=2, grid_length_y=5,
            obj_interest=None,
            danger=None,
            obstacles=None,
            static=True,
            temperatures=None, root_dir=None):
        """Create simple domain class
        Params:
            length (int): length of grid
            latent1 (int): tired or not. This affects sensors
                           0 -- not tired
                           1 -- tired
                           2 -- very tired [avoid for static]
            latent2 (int): protocol to follow
                           0 -- avoid danger
                           1 -- don't avoid danger

        """
        # [DEFAULT SETTING] temperature for stochastic policies, objects of interest and avoid obstacles and danger
        if temperatures is None:
            if static:
                temperatures = [0.01, 1.5]
            else:
                temperatures = [0.01, 1.5, 2]
        if obj_interest is None:
            obj_interest = [[0, 0], [1, 1]]
        if danger is None:
            danger = [[2, 1], [3, 4]]
        if obstacles is None:
            obstacles = [[2, 0], [1, 2], [2, 2], [4, 2]]

        self.grid_length_x = grid_length_x
        self.grid_length_y = grid_length_y

        if static:
            self.nx1 = 2
            self.latent_modes_x1 = 2
        else:
            self.nx1 = 3
            self.latent_modes_x1 = 3
        self.nx2 = 2
        self.latent_modes_x2 = 2

        self.temperatures = temperatures
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

        # bx is the temperature that sets the prob. of sampling a latent mode in a given demo
        self.bx = np.zeros(self.nx1*self.nx2)

        if static:
            tmp = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
            self.bx[tmp[(0, 0)]] = 0.25
            self.bx[tmp[(0, 1)]] = 0.25
            self.bx[tmp[(1, 0)]] = 0.25
            self.bx[tmp[(1, 1)]] = 0.25
            self.inv_latent_dict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
        else:
            tmp = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3, (2, 0): 4, (2, 1): 5}
            # In Dynamic Case, we will have dim of fatigue,
            # nx1=3 but we will sample only nx1=2 initially allow it change during agent movement
            self.bx[tmp[(0, 0)]] = 0.25
            self.bx[tmp[(0, 1)]] = 0.25
            self.bx[tmp[(1, 0)]] = 0.25
            self.bx[tmp[(1, 1)]] = 0.25
            self.inv_latent_dict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1), 4: (2, 0), 5: (2, 1)}

        self.latent_dict = tmp
        self.trans_xsax = self.compute_transition_xsax(static=static)
        self.policy_matrix = self.compute_policy()

        super(BoxWorld, self).__init__(
            obstate_trans_matrix=self.obstate_transition_matrix,
            num_obstates=len(self.states),
            num_actions=4,
            num_states=[self.nx1, self.nx2],
            init_state_weights=self.bx,
            transition_matrix=self.trans_xsax,
            policy_matrix=self.policy_matrix
        )

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

    def get_demonstrations(self, max_len=100, quantity=1, start=None):
        demos = []
        latent = []
        terminal = [(*obj, 0, 0) for obj in self.obj_interest]
        terminal = [self.states_dict[j] for j in terminal]
        for _ in range(quantity):
            start_obstate = start if start is not None else self.obstacles[0]
            while start_obstate in self.obstacles or \
                    start_obstate in self.danger or \
                    start_obstate in self.obj_interest:
                x_start = np.random.randint(np.ceil(self.grid_length_x / 2).astype('int8'), self.grid_length_x)
                y_start = np.random.randint(self.grid_length_y)
                start_obstate = [x_start, y_start]
            sidx = self.states_dict[(x_start, y_start, 1, 1)]

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

    def compute_transition_xsax(self, static=True):
        """Creates transition matrix T(x1, x2,s,a,x1',x2') = T(x1',x2'|x1,x2, s, a)

        When x2 = 0, which is avoid the danger squares, the latent states remain constant
        When x2 = 1, then x1 increases when in a danger square"""

        trans_matrix = np.zeros([self.nx1, self.nx2, len(self.states_dict), 4, self.nx1, self.nx2])
        # trans_matrix[2, 0, :, :, 2, 0] = 1
        # trans_matrix[2, 1, :, :, 2, 1] = 1

        if static:
            trans_matrix[0, 0, :, :, 0, 0] = 1.
            trans_matrix[0, 1, :, :, 0, 1] = 1.
            trans_matrix[1, 0, :, :, 1, 0] = 1.
            trans_matrix[1, 1, :, :, 1, 1] = 1.
            # trans_matrix[2, 0, :, :, 2, 0] = 1
            # trans_matrix[2, 1, :, :, 2, 1] = 1

        else:
            for s in self.states:
                sidx = self.states_dict[s]
                if list(s[:2]) in self.danger:
                    # Increase the tired level 'x1' if you are in a dangerous state
                    trans_matrix[0, 0, sidx, :, 1, 0] = 1.
                    trans_matrix[0, 1, sidx, :, 1, 1] = 1.
                    trans_matrix[1, 0, sidx, :, 2, 0] = 1.
                    trans_matrix[1, 1, sidx, :, 2, 1] = 1.
                    trans_matrix[2, 0, sidx, :, 2, 0] = 1.
                    trans_matrix[2, 1, sidx, :, 2, 1] = 1.
                else:
                    # Declare 3x2 matrix specifying x1', x2' probs for given x1, x2
                    trans_matrix[0, 0, sidx, :] = [[0.7, 0.1], [0.1, 0.1], [0, 0]]  # [0, 0] highest prob: can change to [0, 1] [1, 0] and [1, 1]
                    trans_matrix[0, 1, sidx, :] = [[0.1, 0.8], [0.05, 0.05], [0, 0]]  # [0, 1] highest prob: can change to [0, 0] [1, 0] and [1, 1]
                    trans_matrix[1, 0, sidx, :] = [[0, 0], [0.8, 0.1], [0.1, 0]]  # [1, 0] highest prob: can change to [1, 1] and [2, 0]
                    trans_matrix[1, 1, sidx, :] = [[0.1, 0.1], [0, 0.8], [0, 0]]  # [1, 1] highest prob: can change to [0, 0] and [0, 1]
                    trans_matrix[2, 0, sidx, :] = [[0, 0], [0, 0], [0.8, 0.2]]  # [2, 0] highest prob: can change to [2, 1]
                    trans_matrix[2, 1, sidx, :] = [[0, 0], [0.1, 0.1], [0, 0.8]]  # [2, 1] highest prob: can change to [1, 1] and [1, 0]

        return trans_matrix

    def reward(self, avoid_danger):
        """
        Args:
            avoid_danger:

        Returns: Reward for each (state, action) i.e.
        (a) +100 for (s, a) that collects obj of interests.
        (b) 0 for (s, a) where s[2:] = [0, 0] i.e. objects have been collected already (no reward)
        (c) -100 for (s, a) for states where action taken beings the agent to dangerous cell
        (d) -1 otherwise
        """
        reward = -np.ones([len(self.states), 4])
        # Positive reward for picking up the object
        n_objs = len(self.obj_interest)
        for j in range(n_objs):

            obj = self.obj_interest[j]  # Obj of Interest at [0, 0] and [1, 1]
            status = [0] * j + [1] + [0] * (n_objs - j - 1)  # Gives states of interest for reaching the obj: [1, 0] or [0, 1]
            state_interest = []
            state_interest.extend(obj)
            state_interest.extend(status)

            neighs, _ = self.neighbors(state_interest)
            for a in neighs:
                a_op = self.opposite_action(a)  # For all the states in the Nr of state of interest, identify the action to reach the state of interest
                aidx = self.action_dict[a_op]
                sidxp = self.states_dict[tuple(neighs[a])]
                reward[sidxp, aidx] = 100

        # Todo: Here the neg reward should be for states where action taken beings the agent to dangerous cell but below the neg reward is for (s, a) where s is already dangerous
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
        for temperature in self.temperatures:  # Iterate over X1 i.e. fatigue level corresponding to temperature
            temp_pol = []
            for b in [True, False]:  # Iterate over X2 i.e. Protocol (b is avoid_danger)
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