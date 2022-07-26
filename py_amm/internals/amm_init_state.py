"""Create initial state distribution class

Classes:
    AmmInitialState
"""
import numpy as np
import itertools

from pybasicbayes.distributions import Categorical

from py_amm.utils.stats import sample_discrete


class AmmInitialState(Categorical):
    """
    Functions:
        exp_expected_log
        meanfieldupdate
        meanfield_sgdstep
        max_likelihood
    """
    def __init__(self, num_states, init_state_concentration=None,
                 alphav_0=None, weights=None):
        """Create initial distributions for each feature.

        Arguments:
        num_states (int): number of latent states for each feature
        init_state_concetration (int):
        weights (list): 
        """
        if isinstance(num_states, int):
            self.n_states = [num_states]
        else:
            self.n_states = num_states

        self.nrows = np.prod(self.n_states)
            
        if weights is not None:
            if isinstance(weights, list):
                weights = np.array(weights)
            if weights.ndim > 1:
                weights = weights.flatten()
            self.mf = False
            super(AmmInitialState, self).__init__(
                    alpha_0=init_state_concentration,
                    K=self.nrows,
                    alphav_0=alphav_0,
                    weights=weights)
        elif init_state_concentration is not None or alphav_0 is not None:
            self.mf = True
            super(AmmInitialState, self).__init__(
                    alpha_0=init_state_concentration,
                    K=self.nrows,
                    alphav_0=alphav_0,
                    weights=weights)
        else:
            raise TypeError("Specify `init_state_concentration` "
                            "`alphav_0` or `weights`")
        self.compute_row_idx()

    @property
    def initial_distn(self):
        """Return values of transition matrix in list form.

        The shape of the list is `np.prod(self.n_states)`
        """
        return np.array(self.weights)

    @property
    def initial_distn_matrix(self):
        """Return initial distribution in matrix form.
        
        Matrix shape is specified by `self.n_states`
        """
        shape = self.n_states
        matrix = self.initial_distn
        matrix = matrix.reshape(shape)
        return matrix

    @property
    def mf_initial_distn(self):
        """Return values of transition matrix in list form.
        
        The shape of the list is `np.prod(self.n_states)`
        """
        if self.mf:
            return self.exp_expected_log
        else:
            return self.initial_distn

    @property
    def mf_initial_distn_matrix(self):
        """Return initial distribution in matrix form.
        
        Matrix shape is `self.n_features x self.n_states`
        """
        shape = self.n_states
        matrix = self.mf_initial_distn
        matrix = matrix.reshape(shape)
        return matrix

    def generate(self):
        """Sample latent state."""
        distn = self.initial_distn
        sample = sample_discrete(distn)
        x = self.get_x(sample)
        return x
        
    # Methods to map row idx to x state and vice-versa

    def get_x(self, row_idx):
        """Get latent state given row index."""
        return self.row_idx_to_xstate[row_idx]

    def get_row_idx(self, x):
        """Get row index of x."""
        nx = self.n_states
        factors = [np.prod(nx[j+1:], dtype=np.int8) for j in range(len(nx))]
        return np.dot(factors, x)

    def compute_row_idx(self):
        """Generate easy access from idx to x = x1...xn and vice-versa.
        
        Generates:
        row_idx_to_xstate -- access the index corresponding to xstate
        row_xstate_to_idx -- dictionary with xstate keys to find index
        """
        nx = self.n_states
        self.row_idx_to_xstate = np.zeros((self.nrows, len(nx)), dtype=np.int32)
        self.row_xstate_to_idx = {}
        perm = (range(j) for j in nx)
        xset = itertools.product(*perm)
        for x in xset:
            row_idx = self.get_row_idx(x)
            self.row_idx_to_xstate[row_idx] = np.asarray(x)
            self.row_xstate_to_idx[x] = row_idx

    @property
    def exp_expected_log(self):
        #TODO: how to handle underflow in this case?
        return np.exp(self.expected_log_likelihood())

    #TODO: Check why following three functions have None as argument
    # def meanfieldupdate(self, expected_initial_states_list):
    #     # super(AmmInitialState, self).meanfieldupdate(
    #     #     None, expected_initial_states_list.flatten())
    #     counts = expected_initial_states_list.flatten()
    #     alphav = self.base_distn + counts
    #     super(AmmInitialState, self).__init__(
    #                 alphav_0=alphav)

    def meanfieldupdate(self, alphal_list, betal_list, normalizer_list):
        counts = 0
        # if isinstance(alphal_list, list):
        #     alphal_list = np.array(alphal_list)
        #     betal_list = np.array(betal_list)
        #     normalizer_list = np.array(normalizer_list)
        # if alphal_list.ndim == 2:
        #     alphal_list = np.expand_dims(alphal_list, axis=0)
        #     betal_list = np.expand_dims(betall_list, axis=0)
        #     normalizer_list = np.expand_dims(normalizer_list, axis=0)
        for j in range(len(alphal_list)):
            counts += np.exp(
                alphal_list[j][0]
                + betal_list[j][0]
                - normalizer_list[j])
        
        alphav = self.base_distn + counts.flatten()
        super(AmmInitialState, self).__init__(
                    alphav_0=alphav)

    def meanfield_sgdstep(self, expected_initial_states_list, prob, stepsize):
        super(AmmInitialState, self).meanfield_sgdstep(
                None, expected_initial_states_list, prob, stepsize)

    def max_likelihood(self, samples=None, expected_states_list=None):
        super(AmmInitialState, self).max_likelihood(
                data=samples, weights=expected_states_list)
