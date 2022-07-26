"""
"""
import numpy as np
import itertools

from pybasicbayes.distributions import Multinomial

from py_amm.utils.stats import sample_discrete


class AmmTransition():
    """AMM Transition class."""
    def __init__(self, num_states, num_obstates, num_actions,
                 alpha=None, alphav=None,
                 trans_matrix=None):
        """Create AmmTransition instance."""
        #TODO: do variable number of states for each latent feature
        if isinstance(num_states, int):
            self.n_states = [num_states]
        else:
            self.n_states = num_states
        self.n_obstates = num_obstates
        self.n_actions = num_actions

        self.nrows = np.prod(self.n_states) * num_obstates * num_actions
        self.totalfeat = np.prod(self.n_states)  # total latent feats
        
        # Create rows of transition matrix.
        if trans_matrix is not None:
            self.mf = False
            if trans_matrix.ndim > 2:
                trans_matrix = trans_matrix.reshape(
                    [self.nrows, self.totalfeat],
                    #order='F'
                    )
            self._row_distns = [Multinomial(
                alpha_0=alpha, K=self.totalfeat, alphav_0=alphav, weights=row)
                for row in trans_matrix] # set based on trans_matrix
        elif alpha is not None or alphav is not None:
            self.mf = True
            self._row_distns = [Multinomial(
                alpha_0=alpha, K=self.totalfeat, alphav_0=alphav)
                for n in range(self.nrows)]
        else:
            raise TypeError("Specify `alpha` or `alphav`")

        self.compute_row_idx() # build vars to interchange xsa with idx
        self.compute_col_idx() # build vars to interchange x with idx

    @property
    def trans_matrix(self):
        """Return values of transition matrix in list form.
        
        The shape of the list is `self.Nrows x self.totalfeat`
        """
        return np.array([d.weights for d in self._row_distns])

    @property
    def trans_matrix_xsax(self):
        """Return transition matrix in matrix form.
        #TODO: verify shape of matrix
        Matrix shape is `self.Nx x self.Na x self.Ns x self.Nx`
        """
        trans_matrix = self.trans_matrix
        nx, ns = self.n_states, self.n_obstates
        na = self.n_actions
        trans_matrix_xsax = trans_matrix.reshape((*nx, ns, na, *nx))  # , order='F')
        return trans_matrix_xsax
    
    @property
    def mf_trans_matrix(self):
        """Return values of transition matrix in list form.
        
        The shape of the list is `self.Nrows x self.totalfeat`
        """
        if self.mf:
            return np.exp(np.array([d.expected_log_likelihood()
                                    for d in self._row_distns]))
        else:
            return self.trans_matrix
    @property
    def mf_trans_matrix_xsax(self):
        """Return transition matrix in matrix form.
        #TODO: verify shape of matrix
        Matrix shape is `self.Nx x self.Na x self.Ns x self.Nx`
        """
        trans_matrix = self.mf_trans_matrix
        nx, ns = self.n_states, self.n_obstates
        na = self.n_actions
        trans_matrix_xsax = trans_matrix.reshape((*nx, ns, na, *nx))#, order='F')
        return trans_matrix_xsax
    
    def get_vlb(self):
        return sum(distn.get_vlb() for distn in self._row_distns)
    # Sampling functions

    def conditionaldistn_x(self, x, s, a):
        """Return the distribution T_x(.|s, a, x).
        
        Returns the conditional distribution of `x` at next time step
        given the state, action and latent state at current time step.

        Parameters:
        s (int): state
        a (int): action
        x (int, list, or numpy.array): latent state

        Returns:
        list of lists
        """
        row_idx = self.get_row_idx(x, s, a)
        distn = self.trans_matrix[row_idx]
        return distn

    def generate_x(self, x, s, a):
        """Generate next latent state `x_next` given `x`, `s`, `a`."""
        distn = self.conditionaldistn_x(x, s, a)
        sample = sample_discrete(distn)
        xnext = self.get_x_column(sample)
        return xnext

    # Methods to map latent variables to column idx and vice-versa

    def get_x_column(self, idx: int):
        """Get the feature values corresponding to column index.

        Parameters:
        idx (int): column index

        Returns:
        numpy.array
        """
        return self.col_idx_to_x[idx]
    
    def get_col_idx(self, x):
        """Get column index of feature values.

        Parameters:
        features (int, list or np.array): feature values

        Returns:
        int
        """
        #TODO: return Error when x values are higher than self.n_states
        nx = self.n_states
        factors = [np.prod(nx[j+1:], dtype=np.int8) for j in range(len(nx))]

        return np.dot(factors, x)

    def compute_col_idx(self):
        """Generate easy access from column idx to x and vice-versa.
        
        Generates:
        col_idx_to_x -- to access the index corresponding to x
        col_x_to_idx -- dictionary with x keys to find index
        """
        nx = self.n_states
        nrows = self.totalfeat
        self.col_idx_to_x = np.zeros((nrows, len(nx)), dtype = np.int32)
        self.col_x_to_idx = {}
        perm = (range(j) for j in nx)
        xset = itertools.product(*perm)
        for x in xset:
            row_idx = self.get_col_idx(x)
            self.col_idx_to_x[row_idx] = np.asarray(x)
            self.col_x_to_idx[x] = row_idx

    # Methods to map row idx to xsa and vice-versa

    def get_x(self, row_idx):
        """Get latent state given row index."""
        return self.row_idx_to_xsa[row_idx, :-2]

    def get_s(self, row_idx):
        """Get observed state given row index."""
        return self.row_idx_to_xsa[row_idx, -2]

    def get_a(self, row_idx):
        """Get action given row index."""
        return self.row_idx_to_xsa[row_idx, -1]

    def get_row_idx(self, x, s, a):
        """Get index of distribution given x, s and a.
        
        Parameters:
        x (int, list or np.array): latent state
        s (int): observable state
        a (int): action
        
        Returns:
        int
        """
        if hasattr(x, 'astype'):
            x = x.astype(np.int32) # make sure it is integer valued
        nx = self.n_states
        ns = self.n_obstates
        na = self.n_actions
        factors = [ns * na * np.prod(nx[j+1:], dtype=np.int8) for j in range(len(nx))]
                
        return np.dot(factors, x) + (s*na) + a

    def compute_row_idx(self):
        """Generate easy access from idx to xsa and vice-versa.
        
        Generates:
        row_idx_to_xsa -- to access the index corresponding to xsa
        row_xsa_to_idx -- dictionary with xsa keys to find index
        """
        nx = self.n_states
        condition_vars = len(nx) + 2 # add action and state
        self.row_idx_to_xsa = np.zeros((self.nrows, condition_vars))
        self.row_idx_to_xsa = self.row_idx_to_xsa.astype(np.int32)
        self.row_xsa_to_idx = {}
        perm = (range(j) for j in nx)
        xset = itertools.product(*perm)
        for x in xset:
            for s in range(self.n_obstates):
                for a in range(self.n_actions):
                    row_idx = self.get_row_idx(x, s, a)
                    self.row_idx_to_xsa[row_idx] = np.asarray([*x, s, a])
                    self.row_xsa_to_idx[(*x, s, a)] = row_idx

    # Variational inference methods
    
    def meanfieldupdate(self, expected_transcounts_list):
        """ """
        #assert isinstance(expected_transcounts,list) and len(expected_transcounts) > 0
        #trans_softcounts = sum(expected_transcounts)
        if expected_transcounts_list.ndim == 2:
            expected_transcounts_list = np.expand_dims(
                expected_transcounts_list, axis=0
                )
                
        expected_transcounts = np.array(expected_transcounts_list).sum(0)
        for idx in range(len(expected_transcounts)):
            newlambda = self.base_distn + expected_transcounts[idx].flatten()
            self._row_distns[idx] = Multinomial(alphav_0=newlambda)
        #for distn, counts in zip(self._row_distns, expected_transcounts):
        #    distn.meanfieldupdate(None,counts.flatten())
        return self

    def meanfield_sgdstep(self,expected_transcounts,prob,stepsize):
        """ """
        assert isinstance(expected_transcounts,list)
        if len(expected_transcounts) > 0:
            trans_softcounts = sum(expected_transcounts)
            # NOTE: see note in meanfieldupdate()
        else:
            return self

        side_info = self.side_info
        row_idx = 0
        for distn, counts in zip(self._row_distns,trans_softcounts):
            row_key = tuple(self.row_idx_to_xsa[row_idx])
            if (side_info is None) or (side_info.get(row_key) is None):
                distn.meanfield_sgdstep(None,counts,prob,stepsize)
            else:
                constraints = side_info[row_key].get('constraints',[])
                bounds = side_info[row_key].get('bounds',None)
                init_guess = side_info[row_key].get('init_guess',None)
                distn.meanfield_sgdstep_constrained(None,counts,
                    prob,stepsize,
                    constraints,bounds,init_guess)
            row_idx += 1
        return self

    def _count_transitions(self, stateseqs, obstateseqs, actionseqs):
        assert isinstance(stateseqs,list) and \
               all(isinstance(s,np.ndarray) for s in stateseqs)
        assert isinstance(obstateseqs,list) and \
               all(isinstance(s,np.ndarray) for s in obstateseqs)
        assert isinstance(stateseqs,list) and \
               all(isinstance(s,np.ndarray) for s in actionseqs)

        Nrows, Nx, Ns, Na = self.Nrows, self.Nx, self.Ns, self.Na
        xsax_counts = np.zeros((Nrows, Nx), dtype= np.int32)
        for seq_idx in range(len(stateseqs)):
            xsax_counts += count_transitions_xsax(stateseq= stateseqs[seq_idx],
            obstateseq= obstateseqs[seq_idx],actionseq= actionseqs[seq_idx],
            num_states= Nx,num_obstates= Ns,num_actions= Na)
        # note: xsax_counts havee shape of the form (Nrows,Nx)
        return xsax_counts
