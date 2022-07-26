import unittest
from py_amm.amm.internals.amm_init_state import AmmInitialState


class TestAmmInitial(unittest.TestCase):

    def test__init__(self):
        raise NotImplementedError
    
    def test_initial_distn(self):
        raise NotImplementedError

    def test_initial_distn_matrix(self):
        raise NotImplementedError
    
    def test_mf_initial_distn(self):
        raise NotImplementedError

    def test_mf_initial_distn_matrix(self):
        raise NotImplementedError

    def test_generate(self):
        raise NotImplementedError

    def test_get_x(self):
        raise NotImplementedError
    
    def test_get_row_idx(self):
        raise NotImplementedError
    
    def compute_row_idx(self):
        raise NotImplementedError

    def test_exp_expected_log(self):
        raise NotImplementedError

    def test_meanfieldupdate(self):
        raise NotImplementedError
    
    def test_meanfield_sgdstep(self):
        raise NotImplementedError
    
    def test_max_likelihood(self):
        raise NotImplementedError