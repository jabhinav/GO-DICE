import unittest
import numpy as np

from pybasicbayes.distributions import Categorical
from py_amm.amm import Amm
import py_amm.utils.munkrees as munkrees


class TestAmmInitial(unittest.TestCase):

    def all_equal(self, list1, list2):
        """Compare list of numpy arrays."""
        for j in range(len(list1)):
            if (list1[j] != list2[j]).any():
                return False
        return True

    def test_find_best_order(self):
        """Test `find_best_order()`.
        
        Perform various tests for relabeling latent states.
        The tests performed are on:
        1- 1D array with two values
        2- 1D array with three values and several latent seqs
        3- 1D array with three values where mapping is not exact
        4- 2D array with three values and several latent seqs
        5- 2D array with different quantity of values across dimensions
        6- 2D array with distinct number of values across dims and
           inexact estimation
        This test automatically tests `get_cost()` and
        `order_latent_seq` functions.
        """
        # First: 1D array with two values
        true_1d = [np.array([0, 1, 0, 1, 1, 0]).reshape(-1, 1)]
        est_1d = [np.array([1, 0, 1, 0, 0, 1]).reshape(-1, 1)]
        real_order = [[1, 0]]
        order1d = munkrees.find_best_order(true_1d, est_1d, num_states=[2])
        res1d = munkrees.order_latent_seq(est_1d, order1d)
        self.assertTrue(self.all_equal(res1d, true_1d))
        self.assertTrue(self.all_equal(real_order, order1d))
        
        # Second: 1D array with three values and several latent seqs
        true_1d = [
            np.array([0, 1, 2, 0, 1, 1, 0, 2]).reshape(-1, 1),
            np.array([0, 1, 2, 0, 1, 1, 0, 2]).reshape(-1, 1)]
        est_1d = [
            np.array([2, 0, 1, 2, 0, 0, 2, 1]).reshape(-1, 1),
            np.array([2, 0, 1, 2, 0, 0, 2, 1]).reshape(-1, 1)]

        real_order = [[1, 2, 0]]

        order1d = munkrees.find_best_order(
            true_1d, est_1d, [3])
        res1d = munkrees.order_latent_seq(est_1d, order1d)
        self.assertTrue(self.all_equal(real_order, order1d))
        self.assertTrue(self.all_equal(res1d, true_1d))

        # Third: 1D array with three values where mapping is not exact
        true_1d = [
            np.array([0, 1, 2, 0, 1, 1, 0, 2]).reshape(-1, 1),
            np.array([0, 1, 2, 0, 1, 1, 0, 2]).reshape(-1, 1)]
        est_1d = [
            np.array([2, 0, 2, 2, 0, 0, 2, 1]).reshape(-1, 1),
            np.array([2, 0, 1, 2, 0, 0, 2, 1]).reshape(-1, 1)]

        real_order = [[1, 2, 0]]

        order1d = munkrees.find_best_order(
            true_1d, est_1d, [3])
        true_1d[0][2] = 0
        res1d = munkrees.order_latent_seq(est_1d, order1d)

        self.assertTrue(self.all_equal(real_order, order1d))
        self.assertTrue(self.all_equal(res1d, true_1d))

        # Fourth: 2D array with three values and several latent seqs
        true_2d = [np.array([0, 2, 1, 2, 0, 1, 0, 2, 1, 1, 0, 2]).reshape(-1, 2)]
        est_2d = [np.array([1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0]).reshape(-1, 2)]

        real_order = [[1, 0, 2], [2, 1, 0]]
        order2d = munkrees.find_best_order(
            true_2d, est_2d, [3, 3])
        
        res2d = munkrees.order_latent_seq(est_2d, order2d)
        self.assertTrue(self.all_equal(real_order, order2d))
        self.assertTrue(self.all_equal(res2d, true_2d))

        # Fifth: 2D array with distinct number of values across dims
        true_2d = [
            np.array([0, 2, 1, 2, 0, 1, 0, 2, 1, 1, 0, 2]).reshape(-1, 2),
            np.array([0, 2, 1, 2, 0, 1, 0, 2, 1, 1, 0, 2]).reshape(-1, 2)]
        est_2d = [
            np.array([1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0]).reshape(-1, 2),
            np.array([1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0]).reshape(-1, 2)]

        real_order = [[1, 0, 2], [2, 1, 0]]
        order2d = munkrees.find_best_order(
            true_2d, est_2d, [2, 3])
        
        res2d = munkrees.order_latent_seq(est_2d, order2d)
        self.assertTrue(self.all_equal(real_order, order2d))
        self.assertTrue(self.all_equal(res2d, true_2d))

        # Sixth: 2D array with distinct number of values across dims and
        # inexact estimation
        true_2d = [
            np.array([0, 2, 1, 2, 0, 1, 0, 2, 1, 1, 0, 2]).reshape(-1, 2),
            np.array([0, 2, 1, 2, 0, 1, 0, 2, 1, 1, 0, 2]).reshape(-1, 2)]
        est_2d = [
            np.array([1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0]).reshape(-1, 2),
            np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0]).reshape(-1, 2)]

        real_order = [[1, 0, 2], [2, 1, 0]]
        order2d = munkrees.find_best_order(
            true_2d, est_2d, [2, 3])
        true_2d = [
            np.array([[0, 2],
                      [0, 2],
                      [0, 1],
                      [0, 2],
                      [0, 1],
                      [0, 2]]),
            np.array([[0, 1],
                      [1, 2],
                      [0, 1],
                      [0, 1],
                      [1, 1],
                      [0, 2]])]

        res2d = munkrees.order_latent_seq(est_2d, order2d)
        self.assertTrue(self.all_equal(real_order, order2d))
        self.assertTrue(self.all_equal(res2d, true_2d))

    def test_order_initial_distn(self):
        bx = np.array([0.1, 0.2, 0.3, 0.4, 0, 0]).reshape([3, 2])
        bx_new = np.array([0.4, 0.3, 0, 0, 0.2, 0.1]).reshape([3, 2])
        order = [[1, 2, 0], [1, 0]]
        reordered = munkrees.order_init_distn(bx, order)
        self.assertTrue(self.all_equal([reordered], [bx_new]))

    def test_order_transition(self):
        """Test reordering of transition function.
        Tests:
        1. 1 latent state
        2. 2 latent states"""

        tx = np.zeros([2, 4, 2, 2])
        tx[0, :, :, 0] = 0.1
        tx[0, :, :, 1] = 0.9
        tx[1, :, :, 0] = 0.8
        tx[1, :, :, 1] = 0.2
        tx_new = np.zeros([2, 4, 2, 2])
        tx_new[1, :, :, 1] = 0.1
        tx_new[1, :, :, 0] = 0.9
        tx_new[0, :, :, 1] = 0.8
        tx_new[0, :, :, 0] = 0.2
        
        order = [[1, 0]]

        reordered = munkrees.order_transition(tx, order)
        self.assertTrue(self.all_equal(reordered, tx_new))

        # Second: 2 latent states
        tx = np.zeros([3, 2, 4, 2, 3, 2])
        tx[0, 0, :, :, 0, 0] = 1
        tx[0, 1, :, :, 0, 1] = 1
        tx[1, 0, :, :, 1] = 0.5
        tx[1, 1, :, :, 0] = 0.5
        tx[2, 0, :, :, :, 1] = 0.3
        tx[2, 1, :, :] = 1/6
        tx_new = np.zeros([3, 2, 4, 2, 3, 2])
        tx_new[2, 1, :, :, 2, 1] = 1
        tx_new[2, 0, :, :, 2, 0] = 1
        tx_new[0, 1, :, :, 0] = 0.5
        tx_new[0, 0, :, :, 2] = 0.5
        tx_new[1, 1, :, :, :, 0] = 0.3
        tx_new[1, 0, :, :] = 1/6

        order = [[1, 2, 0], [1, 0]]
        reordered = munkrees.order_transition(tx, order)
        self.assertTrue(self.all_equal(reordered, tx_new))

    def test_order_policy(self):
        """Test reordering of policy matrix.
        Tests:
        1. 1 latent state
        2. 2 latent states
        """
        pi = np.zeros([2, 4, 2])
        pi[0, :, 0] = 0.1
        pi[0, :, 1] = 0.9
        pi[1, :, 0] = 0.8
        pi[1, :, 1] = 0.2
        pi_new = np.zeros([2, 4, 2, 2])
        pi_new[1, :, 0] = 0.1
        pi_new[1, :, 1] = 0.9
        pi_new[0, :, 0] = 0.8
        pi_new[0, :, 1] = 0.2
        
        order = [[1, 0]]

        reordered = munkrees.order_transition(pi, order)
        self.assertTrue(self.all_equal(reordered, pi_new))

        # Second: 2 latent states
        pi = np.zeros([3, 2, 4, 2])
        pi[0, 0, :, 0] = 1
        pi[0, 1, :, 1] = 1
        pi[1, 0, :, :] = 0.5
        pi[2, 0, :, :] = 0.5
        pi[2, 1, :, 1] = 1
        pi_new = np.zeros([3, 2, 4, 2])
        pi_new[2, 1, :, 0] = 1
        pi_new[2, 0, :, 1] = 1
        pi_new[0, 1, :, :] = 0.5
        pi_new[1, 1, :, :] = 0.5
        pi_new[1, 0, :, 1] = 1

        order = [[1, 2, 0], [1, 0]]
        reordered = munkrees.order_policy(pi, order)
        self.assertTrue(self.all_equal(reordered, pi_new))

    def test_class(self):
        # check if the new estimated transition is the same as the reordered
        
        # Hand craft initial distribution
        bx = np.array([0.1, 0.2, 0.3, 0.4, 0, 0]).reshape([3, 2])
        bx_new = np.array([0.4, 0.3, 0, 0, 0.2, 0.1]).reshape([3, 2])

        # Hand craft transition matrix
        tx = np.zeros([3, 2, 4, 2, 3, 2])
        tx[0, 0, :, :, 0, 0] = 1
        tx[0, 1, :, :, 0, 1] = 1
        tx[1, 0, :, :, 1] = 0.5
        tx[1, 1, :, :, 0] = 0.5
        tx[2, 0, :, :, :, 1] = 0.3
        tx[2, 1, :, :] = 1/6
        tx_new = np.zeros([3, 2, 4, 2, 3, 2])
        tx_new[2, 1, :, :, 2, 1] = 1
        tx_new[2, 0, :, :, 2, 0] = 1
        tx_new[0, 1, :, :, 0] = 0.5
        tx_new[0, 0, :, :, 2] = 0.5
        tx_new[1, 1, :, :, :, 0] = 0.3
        tx_new[1, 0, :, :] = 1/6

        # Hand craft policy
        pi = np.zeros([3, 2, 4, 2])
        pi[0, 0, :, 0] = 1
        pi[0, 1, :, 1] = 1
        pi[1, 0, :, :] = 0.5
        pi[2, 0, :, :] = 0.5
        pi[2, 1, :, 1] = 1
        pi_new = np.zeros([3, 2, 4, 2])
        pi_new[2, 1, :, 0] = 1
        pi_new[2, 0, :, 1] = 1
        pi_new[0, 1, :, :] = 0.5
        pi_new[1, 1, :, :] = 0.5
        pi_new[1, 0, :, 1] = 1
        # Build state transition matrix
        obstate_trans_matrix = np.zeros((4, 2, 4))
        # obstate_trans_matrix[:,:,1] = 1
        for obstate in range(4):
            for action in range(2):
                obstate_trans_matrix[obstate, action] = Categorical(
                    alpha_0=10, K=4
                ).weights

        order = [[1, 2, 0], [1, 0]]
        amm_true = Amm(
            obstate_trans_matrix=obstate_trans_matrix,
            num_obstates=4,
            num_actions=2,
            num_states=[3, 2],
            init_state_weights=bx_new,
            transition_matrix=tx_new,
            policy_matrix=pi_new)
        posterior = Amm(
            obstate_trans_matrix=obstate_trans_matrix,
            num_obstates=4,
            num_actions=2,
            num_states=[3, 2],
            init_state_weights=bx,
            transition_matrix=tx,
            policy_matrix=pi
        )

        np.random.seed(200)
        dataobs, stateobs = amm_true.generate_normal(50)
        metrics = munkrees.MunkreesMetrics(amm_true, posterior,
                                           dataobs, stateobs)
        stateobs2 = metrics.improved_amm.expected_stateseq(dataobs)
        self.assertTrue(self.all_equal(stateobs, stateobs2))
