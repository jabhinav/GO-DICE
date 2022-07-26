import numpy as np
from scipy.optimize import linear_sum_assignment
from copy import deepcopy as copy

from py_amm.amm import Amm
from py_amm.utils.stats import kl_divergence
from py_amm.utils.stats import maximum_prob_metric
from py_amm.utils.stats import count_xs, count_xsa, count_initial


def find_best_order(true_latent: list, est_latent: list, num_states):
    """Find the order of est_latent that best approximates true_latent.

    Uses the Jonker-Volgenant algorithm.
    Params:
        - true_latent (list of numpy.ndarray):
            list of true latent states during demonstrations

        - est_latent (list of numpy.ndarray):
            list of estimated latent states for observed demonstrations

        - num_states (list or np.ndarray):
            number of states accross each feature.
            One dimensional vector of shape D
    
    Output:
        - new_est_latent (list of numpy.ndarray):
            a list of the same dimensions as est_latent, with reordered
            elements to better approximate true_latent
        - order (list of numpy.ndarray):
            the order used to reaccomodate elements in new_est_latent
            for each feature d
    """
    if not isinstance(est_latent, list):
        est_latent = [est_latent]
    if not isinstance(true_latent, list):
        true_latent = [true_latent]
    # Concatenate all demonstrations into one
    est_latent_concat = np.concatenate(est_latent)
    true_latent_concat = np.concatenate(true_latent)
    n_feats = true_latent_concat.shape[1]
    costs = get_cost(true_latent_concat, est_latent_concat, num_states)

    orders = []
    for feat in range(n_feats):
        _, order = linear_sum_assignment(costs[feat])
        orders.append(order)

    return orders


def get_cost(true_latent, est_latent, num_states):
    """Gets the cost for each possible reordering of latent states.

    Params:
        - true_latent (np.ndarray):
            true latent state sequence of shape N X D
        - est_latent (np.ndarray):
            estimated latent state seq. of shape N X D
        - num_states (np.ndarray):
            number of states accross each feature.
            One dimensional vector of shape D
    
    Output:
        - Matrices of costs (list of np.ndarray):
            One matrix for each feature d.
            Its shape is num_states[d] X num_states[d]
    """
    costs = []
    for feat in range(len(num_states)):
        scalar_true = true_latent[:, feat]
        scalar_est = est_latent[:, feat]
        rows = num_states[feat]
        cost = np.zeros([rows, rows])

        for row in range(rows):
            for col in range(rows):
                reordered_est_states = [
                    col if scalar_est[j] == row
                    else row if scalar_est[j] == col
                    else scalar_est[j]
                    for j in range(len(scalar_est))
                    ]

                cost[row, col] = np.sum(reordered_est_states != scalar_true)
        costs.append(cost)
    return costs


def order_latent_seq(est_latent, orders):
    est_latent_concat = np.concatenate(est_latent)
    new_est_concat = np.zeros_like(est_latent_concat)
    for feat in range(len(orders)):
        order = orders[feat]
        new_est_concat[:, feat] = [order[i] for i in est_latent_concat[:, feat]]

    # Get a list of same shape and dimensions as est_latent
    new_est = []
    init = 0
    for lat_seq in est_latent:
        end = len(lat_seq) + init
        new_est.append(new_est_concat[init:end])
        init = end
    return new_est


def order_init_distn(init_distn, order):
    new_matrix = copy(init_distn)
    for j in range(len(order)):
        s = (*([slice(None)]*j), tuple(order[j]))
        new_matrix = new_matrix[s]
    return new_matrix


def order_transition(transition_xsax, order):
    new_m = copy(transition_xsax)
    n_feat = len(order)
    for j in range(len(order)):
        s = (*([slice(None)]*j), tuple(order[j]))
        new_m = new_m[s]
        s2 = (*([slice(None)]*(n_feat + 2)), *s)
        new_m = new_m[s2]
    return new_m


def order_policy(policy, order):
    new_matrix = copy(policy)
    for j in range(len(order)):
        s = (*([slice(None)]*j), tuple(order[j]))
        new_matrix = new_matrix[s]
    return new_matrix


class MunkreesMetrics():
    def __init__(self, true_amm, posterior_amm,
                 dataobs_train, stateobs_train,
                 dataobs_test, stateobs_test):
        self.true_amm = true_amm
        self.posterior_amm = posterior_amm
        self.dataobs_train = dataobs_train
        self.stateobs_train = stateobs_train
        self.dataobs_test = dataobs_test
        self.stateobs_test = stateobs_test

        self.order = self.get_order()

        self.improved_amm = self.improve_posterior(self.order)

        self.metrics_raw = self.calculate_metrics(posterior_amm)
        self.metrics_improved = self.calculate_metrics(self.improved_amm)

    def get_order(self):
        estimated = self.posterior_amm.expected_stateseq(self.dataobs_test)
        order = find_best_order(self.stateobs_test,
                                estimated,
                                self.true_amm.n_states)
        return order

    def improve_posterior(self, order, only_policy=True):
        if not isinstance(order, list):
            order = [order]
        ts = self.posterior_amm.obstate_trans_matrix
        ns = self.posterior_amm.n_obstates
        nx = self.posterior_amm.n_states
        na = self.posterior_amm.n_actions
        
        if only_policy:
            tx = self.posterior_amm.transitions.trans_matrix_xsax
            bx = self.posterior_amm.init_state_distn.initial_distn_matrix
        else:
            tx = order_transition(
                self.posterior_amm.transitions.trans_matrix_xsax, order
                )
            bx = order_init_distn(
                self.posterior_amm.init_state_distn.initial_distn_matrix, order
                )
        policy = order_policy(
            self.posterior_amm.policy.policy_matrix_xs, order)

        improved_amm = Amm(
            obstate_trans_matrix=ts,
            num_obstates=ns,
            num_actions=na,
            num_states=nx,
            init_state_weights=bx.flatten(),
            transition_matrix=tx,
            policy_matrix=policy
            )

        return improved_amm

    def calculate_metrics(self, amm):
        weights_init = count_initial(self.stateobs_train, amm.n_states)
        weights_tx = count_xsa(self.stateobs_train, self.dataobs_train,
                               amm.n_states, amm.n_obstates, amm.n_actions)
        weights_policy = count_xs(self.stateobs_train, self.dataobs_train,
                                  amm.n_states, amm.n_obstates, amm.n_actions)

        # Weighted metrics
        weighted_kl_init = kl_divergence(
            self.true_amm.init_state_distn.initial_distn_matrix,
            amm.init_state_distn.initial_distn_matrix,
            weights_init)
        weighted_kl_policy = kl_divergence(
            self.true_amm.policy.policy_matrix_xs,
            amm.policy.policy_matrix_xs,
            weights_policy)
        weighted_kl_tx = kl_divergence(
            self.true_amm.transitions.trans_matrix_xsax,
            amm.transitions.trans_matrix_xsax,
            weights_tx)
        weighted_maxprob = maximum_prob_metric(
            self.true_amm.policy.policy_matrix,
            amm.policy.policy_matrix,
            weights_policy[..., 0])

        # Not weighted metrics
        kl_init = kl_divergence(
            self.true_amm.init_state_distn.initial_distn_matrix,
            amm.init_state_distn.initial_distn_matrix)
        kl_policy = kl_divergence(
            self.true_amm.policy.policy_matrix_xs,
            amm.policy.policy_matrix_xs)
        kl_tx = kl_divergence(
            self.true_amm.transitions.trans_matrix_xsax,
            amm.transitions.trans_matrix_xsax)
        maxprob = maximum_prob_metric(
            self.true_amm.policy.policy_matrix,
            amm.policy.policy_matrix)

        # Hamming distance
        self.expected = amm.expected_stateseq(self.dataobs_test)
        hamming = self.hamming_distance_latentstates(self.stateobs_test,
                                                     self.expected)

        metrics_dic = {
            'weighted_kl_init': weighted_kl_init,
            'weighted_kl_policy': weighted_kl_policy,
            'weighted_kl_transition': weighted_kl_tx,
            'weighted_maximum_prob_metric': weighted_maxprob,
            'kl_init': kl_init,
            'kl_policy': kl_policy,
            'kl_transition': kl_tx,
            'maximum_prob_metric': maxprob,
            'hamming': hamming}
        return metrics_dic

    def hamming_distance_latentstates(self, true_latent, estimated):
        """Find Hamming distance between estimated and true latent states.

        Params:
            true_latent
        Output:
            int: Hamming distance
        """
        if isinstance(true_latent, list):
            true_latent = np.concatenate(true_latent)
        if isinstance(estimated, list):
            estimated = np.concatenate(estimated)

        data_len, n_feat = estimated.shape
        acc = (estimated == true_latent).sum() / (data_len * n_feat)

        return 1 - acc
