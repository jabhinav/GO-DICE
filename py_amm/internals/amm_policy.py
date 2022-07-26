"""Create AmmPolicy class

Classes:
AmmPolicy
"""
import numpy as np
import itertools

from pybasicbayes.distributions import Categorical

from py_amm.utils.stats import sample_discrete


class AmmPolicy():
    """
    """
    def __init__(self, num_obstates, num_actions, num_states,
                 rho=None, rho_v=None,
                 policy_matrix=None):
        """ """
        if isinstance(num_states, int):
            self.n_states = [num_states]
        else:
            self.n_states = num_states
        self.n_obstates = num_obstates
        self.n_actions = num_actions
        self.nrows = num_obstates * np.prod(self.n_states)

        if policy_matrix is not None:
            self.mf = False  # The distribution is known with certainty
            if policy_matrix.ndim > 2:
                policy_matrix = policy_matrix.reshape(
                    [self.nrows, self.n_actions],
                    # order='F'
                    )
            self.obs_distns = [Categorical(alpha_0=rho,
                                           K=self.n_actions,
                                           alphav_0=rho_v,
                                           weights=row)
                               for row in policy_matrix]
        elif rho is not None or rho_v is not None:
            self.mf = True
            self.obs_distns = [Categorical(alpha_0=rho,
                                           K=self.n_actions,
                                           alphav_0=rho_v)
                               for r in range(self.nrows)]
        else:
            raise TypeError("Specify `rho`, `rho_v` or `policy_matrix`")

        self.compute_row_idx()
    
    @property
    def rho(self):
        return self.obs_distns[0].alpha_0
    
    @property
    def rho_v(self):
        return self.obs_distns[0].alphav_0

    @property
    def policy_matrix(self):
        """Return values of policy for all state - latent state pairs.

        Returns a matrix of shape `self.nrows x self.n_actions`.
        """
        return np.array([d.weights for d in self.obs_distns])

    @property
    def policy_matrix_xs(self):
        """Return policy matrix in tensor form.
        
        Returns a tensor of shape:
        `[self.n_states]*self.n_features x self.n_obstates`
        """
        policy_matrix = self.policy_matrix
        nx, ns = self.n_states, self.n_obstates
        na = self.n_actions
        policy_matrix = policy_matrix.reshape((*nx, ns, na))  #, order='A')
        return policy_matrix

    def get_vlb(self):
        return sum(o.get_vlb() for o in self.obs_distns)
        
    def conditionaldistn_a(self, x, s):
        return self.policy_matrix_xs[tuple(x)][s]

    def generate(self, x, s):
        distn = self.conditionaldistn_a(x, s)
        sample = sample_discrete(distn)
        return sample

    def log_likelihood(self, data, mf = False):
        """Compute log likelihood of action sequence observed in data.

        Parameters:
        mf (bool): True if policy known, False if it is to be learned
        data (list): (s, a) sequence

        Returns:
        numpy.array
        """
        if not self.mf:
            # Avoid any meanfield estimation since matrix is fully known
            mf = False
        shape = self.n_states
        aBl = np.zeros((data.shape[0], *shape))
        s_seq, a_seq = data[:,0], data[:,1]
        for obsrow_idx, obs_distn in enumerate(self.obs_distns):
            #note: idx = s*num_states + x
            x = self.get_x(obsrow_idx)
            s = self.get_s(obsrow_idx)
            if mf:
                temp_col = obs_distn.expected_log_likelihood(a_seq).ravel()
            else:
                temp_col = obs_distn.log_likelihood(a_seq).ravel()
            s_indices = (s_seq == s)
            try:
                s = (s_indices, *x)
                aBl[s] += temp_col[s_indices]
            except:
                raise ValueError

        aBl[np.isnan(aBl)] = 0.

        return aBl

    # Methods to map row idx to xs and vice-versa

    def get_x(self, row_idx):
        """Get latent state given row index."""
        return self.row_idx_to_xs[row_idx, :-1]

    def get_s(self, row_idx):
        """Get latent and observed state corresponding to row index."""
        return self.row_idx_to_xs[row_idx, -1]

    def get_xs(self, row_idx):
        return self.row_idx_to_xs[row_idx]

    def get_row_idx(self, x, s):
        """Get index of action distribution given x and s.
        
        Parameters:
        x (int, list or np.array): latent state
        s (int): observable state
        
        Returns:
        int
        """
        nx = self.n_states
        ns = self.n_obstates
        factors = [ns * np.prod(nx[j+1:], dtype=np.int8) for j in range(len(nx))]
        
        return np.dot(factors, x) + s

    def compute_row_idx(self):
        """Generate easy access from idx to xs and vice-versa.
        
        Generates:
        row_idx_to_xs -- access the index corresponding to xs
        row_xs_to_idx -- dictionary with xs keys to find index
        """
        nx = self.n_states
        conditional_vars = len(nx) + 1  # account for observable state
        self.row_idx_to_xs = np.zeros((self.nrows, conditional_vars))
        self.row_idx_to_xs = self.row_idx_to_xs.astype(np.int32)
        self.row_xs_to_idx = {}
        perm = (range(j) for j in nx)
        xset = itertools.product(*perm)
        for x in xset:
            for s in range(self.n_obstates):
                row_idx = self.get_row_idx(x, s)
                self.row_idx_to_xs[row_idx] = np.asarray([*x, s])
                self.row_xs_to_idx[(*x, s)] = row_idx

    @property
    def exp_expected_log(self):
        exps = [np.exp(obs.expected_log_likelihood())
                for obs in self.obs_distns]
        return exps

    # Variational inference methods
    def meanfieldupdate(self, dataobs, expected_states):
        """ """
        if not isinstance(dataobs, list):
            dataobs = [dataobs]

        for obsrow_idx, distn in enumerate(self.obs_distns):
            state = self.get_xs(obsrow_idx)[:-1]
            obstate = self.get_xs(obsrow_idx)[-1]
            # counts = []
            # for action in range(self.n_actions):
            #     counts_peridx = 0
            #     for j in range(len(dataobs)):
            #         idxs = np.all(dataobs[j] == [obstate, action], axis=1)
            #         idx = (idxs, *(state))
            #         action_weight = sum(expected_states[j][idx])
            #         counts_peridx += action_weight
            #     counts.append(counts_peridx)

            # Faster alternative. Is it?? Feels quite slow 
            dataobs2 = np.concatenate(dataobs, axis=0)
            expected_states2 = np.concatenate(expected_states, axis=0)            
            idxs = dataobs2[:, 0] == obstate
            actionseq = dataobs2[idxs][:, 1]
            stateidxs = expected_states2[idxs]
            state = tuple(state)
            tmp = (slice(None), *state)
            stateidxs = stateidxs[tmp]
            counts = np.bincount(actionseq, stateidxs, minlength=self.n_actions)

            # assert((np.array(counts2) == np.array(counts)).all()), 'not equal'
            newlambda = self.base_distn + np.array(counts)
            self.obs_distns[obsrow_idx] = Categorical(alphav_0=newlambda)
        
        # obstateseq = data[:,0]
        # actionseq = data[:,1]
        # for obsrow_idx, distn in enumerate(self.obs_distns):
        #     state = self.get_xs(obsrow_idx)[:-1]
        #     obstate = self.get_xs(obsrow_idx)[-1]
        #     idx = ((obstateseq == obstate), *(state))
        #     distn.meanfieldupdate(
        #         [actionseq[(obstateseq == obstate)]],
        #         [expected_states[idx]]
        #         )

    def meanfield_sgdstep(self, mb_data, expected_states, prob, stepsize):
        obstateseq = mb_data[:,0]
        actionseq = mb_data[:,1]

        for obsrow_idx, distn in enumerate(self.obs_distns):
            state, obstate = self.get_xs(obsrow_idx)
            distn.meanfield_sgdstep(
                [actionseq[(obstateseq == obstate)]],
                [expected_states[(obstateseq == obstate), state]],
                prob,
                stepsize)
