"""
Create a SimpleDomain class where the agent has to collect
an object of interest.
The grid is 2 X lenght, and contains two objects of interest,
one located at [0,0] and the second at [0,length-1].
The goal object to be collected is encoded as a latent state. 
Also the route to be taken is a latent state as well.
We have then:
S = 
A = 
X1 = {0,1} (0 - left, 1 - right)
X2 = {0,1} (0 - up route, 1 - down route)
Tx = T(x1', x2' | s, a, x1, x2) = 1 if x1' = x1, and x2' = x2, else 0
Ts = T(s'| s, a)
bx = (predefined)
pi = pi(a| s, x1, x2) = 
"""
import numpy as np
import pickle
import json
from copy import deepcopy as copy

# from py_amm.amm import Amm
# from py_amm.internals.amm_transitions import AmmTransition
# from py_amm.internals.amm_policy import AmmPolicy
# from py_amm.internals.amm_init_state import AmmInitialState
# from py_amm.utils.stats import AmmMetrics


# class Simple(Amm):
class Simple(object):
    def __init__(self, length=5, latent1=0, latent2=0):
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

        self.length = length
        self.pos = [0, int(length / 2)]
        self.goal = [0, 0] if latent1 == 0 else [0, length - 1]
        self.latent1 = latent1
        self.latent2 = latent2

        self.obstate_transition_matrix = self.transition_sas()
        self.bx = np.zeros(4)
        tmp = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
        self.bx[tmp[(latent1, latent2)]] = 1
        self.trans_xsax = self.transition_xsax()
        self.policy_matrix = self.policy()
        # super(Simple, self).__init__(
        #     obstate_trans_matrix=self.obstate_transition_matrix,
        #     num_obstates=len(self.states_dict),
        #     num_actions=5,
        #     num_states=2,
        #     num_features=2,
        #     init_state_weights=self.bx,
        #     transition_matrix=self.trans_xsax,
        #     policy_matrix=self.policy_matrix)

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
    def states_dict(self):
        st_list = [[i, j] for i in range(2) for j in range(self.length)]
        st_dict = {}
        count = 0
        for st in st_list:
            st_dict[tuple(st)] = count
            count += 1
        return st_dict

    @property
    def inv_states_dict(self):
        st_list = [[i, j] for i in range(2) for j in range(self.length)]
        inv_st_dict = {}
        count = 0
        for st in st_list:
            inv_st_dict[count] = st
            count += 1
        return inv_st_dict

    @property
    def latent_state_dict(self):
        latent_states = [[i, j] for i in range(2) for j in range(2)]
        latent_dict = {}
        count = 0
        for st in latent_states:
            latent_dict[tuple(st)] = count
            count += 1
        return latent_dict

    def sample_demonstrations(self):
        if self.latent1 == 0 and self.latent2 == 0:
            # Pick left goal taking upper route
            y_coords = list(reversed(range(self.pos[1] + 1)))
            x_coords = np.zeros(len(y_coords))
            actions = np.ones(len(x_coords)) * self.action_dict['left']

        elif self.latent1 == 0 and self.latent2 == 1:
            # Pick left goal taking lower route
            y_coords = list(reversed(range(self.pos[1] + 1)))
            y_coords = [self.pos[1]] + y_coords + [0]
            x_coords = np.ones(len(y_coords))
            x_coords[0] = x_coords[-1] = y_coords[-1] = 0
            actions = np.ones(len(x_coords)) * self.action_dict['left']
            actions[0] = self.action_dict['down']
            actions[-1] = self.action_dict['up']

        elif self.latent1 == 1 and self.latent2 == 0:
            # Pick right goal taking upper route
            y_coords = list(range(self.pos[1], self.length))
            x_coords = np.zeros(len(y_coords))
            actions = np.ones(len(x_coords)) * self.action_dict['right']

        elif self.latent1 == 1 and self.latent2 == 1:
            # Pick right goal taking lower route
            y_coords = list(range(self.pos[1], self.length))
            y_coords = [self.pos[1]] + y_coords + [self.length - 1]
            x_coords = np.ones(len(y_coords))
            x_coords[0] = x_coords[-1] = y_coords[-1] = 0
            actions = np.ones(len(x_coords)) * self.action_dict['right']
            actions[0] = self.action_dict['down']
            actions[-1] = self.action_dict['up']

        position_list = np.stack((x_coords, y_coords), axis=1)

        return position_list, actions

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

    def transition(self, s, a, s_prime):
        # Although the transition function defined in the book
        # depends also on r, it is not necessary to put r for 
        # this MDP because it is deterministic
        if a == 'stay':
            # return 0
            return 1 if s_prime == s else 0

        prob = 0
        neighs, out_of_space = self.neighbors(np.array(s))

        #### Non diagonal Movements ####

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

    def transition_sas(self):
        """Creates transition matrix T(s,a,s') = T(s'|s, a)"""
        trans_matrix = np.zeros([len(self.states_dict), 5, len(self.states_dict)])
        for s in self.states_dict:
            for sprime in self.states_dict:
                for a in self.action_dict:
                    sidx = self.states_dict[s]
                    spidx = self.states_dict[sprime]
                    aidx = self.action_dict[a]
                    trans_matrix[sidx, aidx, spidx] = self.transition(list(s), a, list(sprime))

        return trans_matrix

    def transition_xsax(self):
        """Creates transition matrix T(x1, x2,s,a,x1',x2') = T(x1',x2'|x1,x2, s, a)"""

        trans_matrix = np.zeros([2, 2, len(self.states_dict), 5, 2, 2])
        trans_matrix[0, 0, :, :, 0, 0] = 1
        trans_matrix[0, 1, :, :, 0, 1] = 1
        trans_matrix[1, 0, :, :, 1, 0] = 1
        trans_matrix[1, 1, :, :, 1, 1] = 1
        return trans_matrix

    def policy(self):
        """pi(s,x1,x2,a) = pi(a|s, x1,x2)"""
        pol = np.zeros(shape=[2, 2, len(self.states_dict), 5])  # [states, latent1, latent2, actions]

        # Pick left goal taking upper route
        for s in self.states_dict:
            sidx = self.states_dict[s]
            if list(s) == self.goal:
                pol[0, 0, sidx, self.action_dict['stay']] = 1
            elif s[0] == 1:
                pol[0, 0, sidx, self.action_dict['up']] = 1
            else:
                pol[0, 0, sidx, self.action_dict['left']] = 1

            # if list(s) == self.goal:
            #     var = pol[0, 0, sidx, self.action_dict['stay']]

        # Pick left goal taking lower route
        for s in self.states_dict:
            sidx = self.states_dict[s]

            if list(s) == self.goal:
                pol[0, 1, sidx, self.action_dict['stay']] = 1
            elif s[0] == 0:
                pol[0, 1, sidx, self.action_dict['down']] = 1
            elif s[1] == 0:
                pol[0, 1, sidx, self.action_dict['up']] = 1
            else:
                pol[0, 1, sidx, self.action_dict['left']] = 1

        # Pick right goal taking upper route
        for s in self.states_dict:
            sidx = self.states_dict[s]
            if list(s) == self.goal:
                pol[1, 0, sidx, self.action_dict['stay']] = 1
            elif s[0] == 1:

                pol[1, 0, sidx, self.action_dict['up']] = 1
            else:
                pol[1, 0, sidx, self.action_dict['right']] = 1

        # Pick right goal taking lower route
        for s in self.states_dict:
            sidx = self.states_dict[s]
            if list(s) == self.goal:
                pol[1, 1, sidx, self.action_dict['stay']] = 1
            elif s[0] == 0:
                pol[1, 1, sidx, self.action_dict['down']] = 1
            elif s[1] == self.length - 1:
                pol[1, 1, sidx, self.action_dict['up']] = 1
            else:
                pol[1, 1, sidx, self.action_dict['right']] = 1

        return pol


def convertST2ST2D(states, state_dict):
    state2D = []
    state_dict = list(state_dict.keys())
    for st in states:
        state2D.append(list(state_dict[st]))
    return state2D


def get_demonstrations(simple_env, initial_pos):
    simple_env.pos = initial_pos
    goal = simple_env.goal
    goal = simple_env.states_dict[tuple(goal)]
    initial_state = simple_env.states_dict[tuple(simple_env.pos)]
    stateaction, latentstates = simple_env.generate_normal(20, start_obstate=initial_state)
    idx = list(stateaction[:, 0]).index(goal)
    return stateaction[:idx + 1], latentstates[:idx + 1]


def get_multiple_demonstrations(grid_length, initial_positions, latent1, latent2):
    m = Simple(length=grid_length,
               latent1=latent1,
               latent2=latent2)
    stateaction = []
    latentstates = []
    for j in initial_positions:
        st, ls = get_demonstrations(m, j)
        stateaction.append(st)
        latentstates.append(ls)
    return m, stateaction, latentstates

#
# def apply_munkrees(posterior_amm, true_amm, dataobs, stateobs):
#     nx = posterior_amm.n_states
#     nf = posterior_amm.n_features
#     ns = posterior_amm.n_obstates
#     na = posterior_amm.n_actions
#     updated_posterior = copy(posterior_amm)
#     metrics = AmmMetrics(posterior_amm, true_amm, dataobs, stateobs)
#     new_transition = metrics.order_matrix('transition', metrics.order_trans)
#     new_init = metrics.order_matrix('init', metrics.order_init)
#     new_policy = metrics.order_matrix('policy', metrics.order_policy)
#     updated_posterior.init_state_distn = AmmInitialState(
#         nx,
#         nf,
#         weights=new_init)
#     updated_posterior.transitions = AmmTransition(
#         nx, ns, na, nf, trans_matrix=new_transition
#     )
#     updated_posterior.policy = AmmPolicy(
#         ns, na, nx, nf, policy_matrix=new_policy
#     )
#     return updated_posterior

def create_data():
    # Specify the length of the grid, the initial position, and the latent state values.
    grid_length = 10
    # single = False
    # if single:
    #     initial_positions = [[0, 5]]
    # else:
    #     initial_positions = [[0, 0], [0, 3], [0, 5], [0, 7], [0, 9]]

    np.random.seed(2)

    num_latent_modes = 2
    num_trajectories = 80

    # state_action = [] # List of trajectories
    states = []
    actions = []
    latent_states = []
    traj_length = []
    envs = []
    for i in range(num_trajectories):

        initial_position = list(np.random.randint(0, [2, 10]))
        latent_mode = list(np.random.randint(num_latent_modes, size=2))

        print("\n-----------------xxxx-----------------")
        print("-- Initial Pos: ", initial_position)
        print("-- Latent Mode: {}".format(latent_mode))
        m = Simple(length=grid_length,
                   latent1=latent_mode[0],
                   latent2=latent_mode[1])
        st, ls = get_demonstrations(m, initial_position)

        # Book-keeping purpose
        print("-- Trajectory {}: {}".format(i, st))
        envs.append(m)
        traj_length.append(len(st))
        for (s, a) in st:
            states.append(m.inv_states_dict[s])
            actions.append(a)
            latent_states.append(m.latent_state_dict[tuple(latent_mode)])

    ls = np.array(latent_states)
    one_hot_latent_states = np.zeros((ls.size, ls.max()+1))
    one_hot_latent_states[np.arange(ls.size), ls] = 1

    demo ={
        # 'envs': envs,
        'states': np.array(states),
        'actions':  np.array(actions),
        'traj_length':  np.array(traj_length),
        'latent_states': one_hot_latent_states
    }
    with open('demo.pkl', 'wb') as f:
        pickle.dump(demo, f, protocol=2)  # To maintain compatibility of pickled files with both Python 2 and 3, pickle files should be generated with protocol level 2
        # state_action.append(st)

if __name__ == "__main__":
    create_data()