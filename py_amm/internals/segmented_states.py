"""First tested on scalar latent state"""
from numpy.random import beta
from numpy.random import binomial
import numpy as np


class SegmentedStates():
    def __init__(self, n_states, a_0, b_0):
        self.n_states = n_states
        self.a_0, self.b_0 = a_0, b_0
        self.w = self.w_vector()

    def w_vector(self):
        w = []
        for _ in range(self.n_states):
            w.append(beta(self.a_0, self.b_0))
        return w

    def sample(self, x):
        """Return the distribution T_x(.|s, a, x).

        Returns the conditional distribution of `r` given latent state.

        Parameters:
        x (int, list, or numpy.array): latent state

        Returns:
        list of lists
        """
        #NOTE: only receives scalar latent states for now
        if isinstance(x, list) or isinstance(x, np.ndarray):
            x = x[0]
        w_x = self.w[x]
        r = binomial(1, w_x)
        return r

class GEM():
    def __init__(self, alpha):
        self.alpha = alpha
        self.sticks = [ 0 ]
        self._remaining = 1

    @property
    def params(self):
        return self.alpha

    @params.setter
    def params(self, value):
        self.alpha = value

    def permute(self, perm):
        self.alpha = self.alpha[perm]

    def next_sample(self):
        V = np.random.beta(1, self.alpha)
        new = self._remaining*V                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        self.sticks.append(new)
        self._remaining -= new
    
    def clear(self):
        self.sticks = [0]
        self._remaining = 1
    
    def generate_samples(self, n, keep = True):
        # n - number of samples to generate
        if not keep:
            print('Not implemented yet')
            return None

        for j in range(n):
            self.next_sample()
        return self.sticks
    
    def rvs(self, size = 1):
        sample = self.generate_samples(n = size)
        return sample

    def get_sticks(self):
        return np.array(self.sticks[1:])
