""" """
import numpy as np
import numpy.random as npr
import itertools
from numpy import newaxis as na
from scipy.optimize import linear_sum_assignment
from copy import deepcopy as copy


def count_initial(stateseqs, num_states):
    if not isinstance(stateseqs, list):
        stateseqs = [stateseqs]
    total = len(stateseqs)
    out = np.zeros((num_states))
    for stateseq in stateseqs:
        s = tuple(stateseq[0])
        out[s] += 1/total
    return out


def count_xsa(stateseqs, dataobsseqs, num_states, num_obstates, num_actions):
    if not isinstance(stateseqs, list):
        stateseqs = [stateseqs]
    if not isinstance(dataobsseqs, list):
        dataobsseqs = [dataobsseqs]

    dataobsseqs = np.concatenate(dataobsseqs)
    stateseqs = np.concatenate(stateseqs)
    data = np.concatenate([stateseqs, dataobsseqs], axis=1)
    total = len(data)
    out = np.zeros((*num_states, num_obstates, num_actions, *num_states))
    perm = (range(j) for j in num_states)
    xset = itertools.product(*perm)
    for x in xset:
        for s in range(num_obstates):
            for a in range(num_actions):
                out[(*x, s, a)] = np.all(data == (*x, s, a), axis=1).sum()/total
    return out


def count_xs(stateseqs, dataobsseqs, num_states, num_obstates, num_actions):
    if not isinstance(stateseqs, list):
        stateseqs = [stateseqs]
    if not isinstance(dataobsseqs, list):
        dataobsseqs = [dataobsseqs]

    dataobsseqs = np.concatenate(dataobsseqs)
    stateseqs = np.concatenate(stateseqs)
    data = np.concatenate([stateseqs, dataobsseqs[:, 0].reshape([-1, 1])], axis=1)
    total = len(data)

    out = np.zeros((*num_states, num_obstates, num_actions))

    perm = (range(j) for j in num_states)
    xset = itertools.product(*perm)
    for x in xset:
        for s in range(num_obstates):
            out[(*x, s)] = np.all(data == (*x, s), axis=1).sum()/total
    return out


def sample_discrete(distn, size=[], dtype=np.int32):
    'Samples from a one-dimensional finite pmf'
    distn = np.atleast_1d(distn)
    assert (distn >=0).all() and distn.ndim == 1
    if (0 == distn).all():
        return npr.randint(distn.shape[0],size=size)
    cumvals = np.cumsum(distn)
    return np.sum(np.array(npr.random(size))[...,na] * cumvals[-1] > cumvals, axis=-1,dtype=dtype)


def maximum_prob_metric(distn1, distn2, weights=None):
    # index1 = distn1.index(np.max(distn1, axis=-1))
    # index2 = distn2.index(np.max(distn2, axis=-1))
    indexmax1 = [np.where(dist == np.max(dist)) for dist in distn1]
    indexmax2 = [np.where(dist == np.max(dist)) for dist in distn2]
    intersections = [np.intersect1d(
        indexmax1[row], indexmax2[row]) for row in range(len(distn1))]
    len_intersections = np.array([len(inter) for inter in intersections])

    counts = np.zeros(len(distn1))
    success_idxs = np.where(len_intersections > 0)[0]
    counts[success_idxs] = 1
    if weights is not None:
        weights = weights.flatten()
        weights = weights * len(distn1)
        counts = counts * weights

    return counts.sum()


def kl_divergence(distn1, distn2, weights=None):
    """Returns D_{KL}( distn1 || distn2 ).
    
    Parameters:
        distn1: numpy.ndarray
        distn2: numpy.ndarray of same dimension as distn1
    
    Returns:
        int
    """
    errs = np.seterr('ignore') # ignore np.log(0) error
    # Ensure working with numpy.arrays
    if not isinstance(distn1, np.ndarray):
        distn1 = np.array(distn1)
    if not isinstance(distn2, np.ndarray):
        distn2 = np.array(distn2)
    # KL distance
    m = distn1 * np.log(distn1/distn2)

    if weights is not None:
        m = m * weights

    np.seterr(**errs)  # Return to previous warnings

    return np.nansum(m)


def count_transitions(stateseq, num_states):
    out = np.zeros((num_states,num_states),dtype=np.int32)
    for i,j in zip(stateseq[:-1],stateseq[1:]):
        out[i,j] += 1
    return out


def count_transitions_xsax(stateseq, obstateseq, actionseq,
    num_states, num_obstates, num_actions):
    out = np.zeros((num_states * num_obstates * num_actions, num_states),
        dtype= np.int32)
    for x,s,a,xnext in zip(
        stateseq[:-1], obstateseq[:-1], actionseq[:-1], stateseq[1:]):
        row_idx = a * (num_obstates*num_states) + s * (num_states) + x
        col_idx = xnext
        out[row_idx,col_idx] += 1
    return out


# class AmmMetrics():
#     """Class that finds metrics of the estimated AMM (v1 and v2) model.
    
#     Metrics that are calculated:
#         - KL(T_x, \hat{T}_x)
#         - KL(b_x, \hat{b}_x)
#         - KL(pi, \hat{pi})
#         - Hamming distance between true and expected latent states
#         - Hamming distance between true and max viterbi latent states
#     """
#     def __init__(self, posterior_amm, true_amm=None, dataobs=None, stateobs=None):
#         """Creates AmmMetrics class.

#         Initializes true and estimated amm. Calculates all metrics.

#         Params:
#             - true_amm (AMM)
#             - posterior_amm
#             - dataobs
#             - stateobs

#         Generates the KL(b_x,\hat{b}_x) and KL(\pi, \hat{\pi}).
#         If dataobs and stateobs is given, then the Hamming distances
#         and KL(T_x, \hat{T}_x) are also calculated.
#         """
#         self.true_amm = true_amm
#         self.posterior_amm = posterior_amm
#         self.n_feat = self.posterior_amm.n_features
#         self.n_latent = self.posterior_amm.n_states
        
#         if true_amm is not None:
#             self.kl_init = self.kl_divergence('init')
#             self.kl_policy = self.kl_divergence('policy')
        
#             self.order_init = self._order_latent('init')[1]
#             self.order_policy = self._order_latent('policy')[1]

#         # Calculate Hamming distances and KL(T_x, \hat{T}_x) 
#         if dataobs is not None and stateobs is not None:
#             self.dataobs = dataobs
#             self.stateobs = stateobs
#             self.hamming_expected, _ = self.hamming_distance_latentstates(
#                 dataobs,
#                 stateobs,
#                 'expectation'
#                 )
#             if self.n_feat == 1: # Max viterbi does not work in vector latent states yet.
#                 self.hamming_viterbi, _ = self.hamming_distance_latentstates(
#                     dataobs,
#                     stateobs,
#                     'maxviterbi'
#                     )

#             if true_amm is not None:
#                 self.kl_trans = self.kl_divergence('transition')
#                 self.order_trans = self._order_latent('transition')[1]
#     def update_rest_metrics(self, dataobs, stateobs):
#         """Recalculate Hamming distances and KL(T_x, \hat{T}_x)."""
#         self.dataobs = dataobs
#         self.stateobs = stateobs
#         self.kl_trans = self.kl_divergence('transition')
#         self.hamming_viterbi, _ = self.hamming_distance_latentstates(
#             dataobs,               
#             stateobs,              
#             'maxviterbi'           
#             )
#         self.hamming_expected, _ = self.hamming_distance_latentstates(
#             dataobs,
#             stateobs,
#             'expectation'
#             )

#     def kl_divergence(self, which):
#         """Get KL divergence.

#         Params:
#             which (str): either 'transition', 'policy' or 'init'
#         Output:
#             int
#         """
#         row_ind, col_ind, cost = self._order_latent(which)
#         new_matrix = self.order_matrix(which, col_ind)

#         if which == 'transition':
#             m = self.true_amm.transitions.trans_matrix_xsax
#         elif which == 'policy':
#             m = self.true_amm.policy.policy_matrix_xs
#         elif which == 'init':
#             m = self.true_amm.init_state_distn.initial_distn_matrix
#         return kl_divergence(m, new_matrix)

#     def order_matrix(self, which, order):
#         """Reaccomodate distribution matrix according to order.
        
#         Params:
#             which (str): either 'transition', 'policy' or 'init'
#             order (list): order that the latent states need to follow
#         Output:
#             new_m (numpy.ndarray): reordered distribution matrix.
        
#         """
#         if which == 'transition':
#             new_m = self.posterior_amm.transitions.trans_matrix_xsax
#             for j in range(len(order)):
#                 s = (*([slice(None)]*j), tuple(order[j]))
#                 new_m = new_m[s]
#                 s2 = (*([slice(None)]*(self.n_feat + 2)), *s)
#                 new_m = new_m[s2]
#         elif which == 'policy':
#             new_m = self.posterior_amm.policy.policy_matrix_xs
#             for j in range(len(order)):
#                 s = (*([slice(None)]*j), tuple(order[j]))
#                 new_m = new_m[s]
#         elif which == 'init':
#             new_m = self.posterior_amm.init_state_distn.initial_distn_matrix
#             for j in range(len(order)):
#                 s = (*([slice(None)]*j), tuple(order[j]))
#                 new_m = new_m[s]
#         return new_m

#     def _construct_cost(self, which):
#         """Constructs the cost matrix for the assignment problem.
        
#         Params:
#             which (str): either 'transition', 'policy' or 'init'
#         Output:
#             numpy.ndarray of size (`self.n_latent` X `self.n_latent`)
#         """
#         rows = self.n_latent

#         if which == 'transition':
#             # For the transition, the order of the expected/maxviterbi
#             # latent states in sequence is used
#             pass
#         elif which == 'init':
#             matrix_est = self.posterior_amm.init_state_distn.initial_distn_matrix
#             matrix_true = self.true_amm.init_state_distn.initial_distn_matrix
#         elif which == 'policy':
#             matrix_est = self.posterior_amm.policy.policy_matrix_xs
#             matrix_true = self.true_amm.policy.policy_matrix_xs

#         costs = []

#         for j in range(self.n_feat):
#             cost = np.zeros([rows, rows])
#             for row in range(rows):
#                 for col in range(rows):
#                     r = (*([slice(None)]*j), row)
#                     c = (*([slice(None)]*j), col)
#                     tmp = copy(matrix_est)
#                     tmp[r] = matrix_est[c]
#                     tmp[c] = matrix_est[r]
#                     cost[row, col] = kl_divergence(matrix_true,tmp)
#                     #cost[row, col] = norm(matrix_true[r] - matrix_est[c])
#             costs.append(cost)
#         return costs

#     def _order_latent(self, which):
#         """Find the order of latent states for distribution matrices.

#         Uses the Jonker-Volgenant algorithm.
#         Params:
#             which (str): one of `transition`, `init`, `policy`
#         """
#         if which == 'transition':
#             expected = self.posterior_amm.expected_stateseq(self.dataobs)
#             expected = np.concatenate(expected)
#             stateobs = np.concatenate(self.stateobs)
#             costs = self._costlist_vectors(stateobs, expected)
#             _, col_ind = self.order_latent(costs, expected)
#             return None, col_ind, None
#         costs = self._construct_cost(which)
#         row_inds = []
#         col_inds = []
#         optimal_c = []
#         for cost in costs:
#             row_ind, col_ind = linear_sum_assignment(cost)
#             row_inds.append(row_ind)
#             col_inds.append(col_ind)
#             optimal_c.append(cost[row_ind, col_ind].sum())
#         return row_inds, col_inds, costs

#     def _cost_vectors(self, true_states, est_states):
#         """Constructs the cost matrix for latent state estimation of stateseq.
        
#         This cost matrix is used to find the optimal order
#         of the latent states for the estimated sequence

#         Params:
#             true_states: true latent state sequence
#             est_states: estimated latent state sequence
#         Output:
#             numpy.ndarray of size (`self.n_latent` X `self.n_latent`)
#         """
#         rows = self.n_latent
#         cost = np.zeros([rows, rows])
        
#         for row in range(rows):
#             for col in range(rows):
#                 idx1 = est_states == row
#                 idx1 = [i for i, x in enumerate(idx1) if x]
#                 idx2 = true_states == col
#                 idx2 = [i for i, x in enumerate(idx2) if x]

#                 m = len(np.intersect1d(idx1, idx2))
#                 false_negatives = len(idx1) - m
#                 false_positives = len(idx2) - m
#                 cost[row, col] = false_positives + false_negatives
#         return cost

#     def _costlist_vectors(self, true_states, est_states):
#         costs = []
#         for j in range(self.n_feat):
#             scalar_latent_true = true_states[:, j]
#             scalar_latent_est = est_states[:, j]
#             cost = self._cost_vectors(scalar_latent_true, scalar_latent_est)
#             costs.append(cost)
#         return costs

#     def order_latent(self, costs, expected_states):
#         N = len(costs)
#         new_expected = np.zeros_like(expected_states)
#         orders = []
#         for j in range(N):
#             _, order = linear_sum_assignment(costs[j])
#             new_expected[:, j] = [order[i] for i in expected_states[:, j]]
#             orders.append(order)
#         return new_expected, orders
            
#     def hamming_distance_latentstates(self, dataobs, stateobs, type='expectation'):
#         """Find Hamming distance between estimated and true latent states.
        
#         The Hamming distance is calculated after applying
#         the Jonker-Volgenant algorithm on the estimated state sequence

#         Params:
#             dataobs 
#             stateobs
#             type (str): either `expectation` or `maxviterbi`
#         Output:
#             int: Hamming distance
#             list: order obtained from the Jonker-Volgenant algorithm
#         """
        
#         if type == 'expectation':
#             expected = self.posterior_amm.expected_stateseq(dataobs)

#         else: # TODO: generalize to vector latent states.
#             expected = self.posterior_amm.max_viterbi(dataobs).reshape(-1,1)
        
#         if isinstance(dataobs, list):
#             stateobs = np.concatenate(stateobs)
#             expected = np.concatenate(expected)
        
#         data_len = len(stateobs)
#         costs = self._costlist_vectors(stateobs, expected)
#         new_expected, order = self.order_latent(costs, expected)
#         acc = (new_expected == stateobs).sum() / (data_len * self.n_feat)
#         #acc = (new_expected == stateobs.flatten()).sum() / (data_len * self.n_feat)

#         return acc, order

#     def hamming_distance_predictedstates(self):
#         return NotImplementedError


