  
"""
Module description
"""
import itertools

import numpy as np
from numpy import newaxis as na
# from pybasicbayes.distributions import Categorical  # Third-party imports
from scipy.special import logsumexp

from py_amm.internals.amm_init_state import AmmInitialState
from py_amm.internals.amm_policy import AmmPolicy
# from py_amm.internals.segmented_states import SegmentedStates
from py_amm.internals.amm_transitions import AmmTransition
from py_amm.utils.stats import sample_discrete


# TODO: two blank lines or one?
class Amm():
    """
    """
    def __init__(self, obstate_trans_matrix, num_obstates,
                 num_actions, num_states,
                 init_state_concentration=1, init_state_weights=None,
                 transition_alpha=1, transition_alphav=None,
                 transition_matrix=None,
                 rho=1, rho_v=None,
                 policy_matrix=None,
                 base_distn=None
                 ):
        """
        num_states (int or list): each entry corresponds to the number of
        latent values for each latent state
        """
        if isinstance(num_states, int):
            self.n_states = [num_states]
        else:
            self.n_states = num_states
        self.n_actions = num_actions
        self.n_obstates = num_obstates
        self.obstate_trans_matrix = obstate_trans_matrix
        self.totalfeats = np.prod(self.n_states)

        self.init_state_distn = AmmInitialState(num_states,
                                                init_state_concentration,
                                                weights=init_state_weights)

        self.transitions = AmmTransition(num_states, num_obstates,
                                         num_actions,
                                         transition_alpha, transition_alphav,
                                         transition_matrix)

        self.policy = AmmPolicy(num_obstates, num_actions, num_states,
                                rho, rho_v, policy_matrix)

        #self.segmented_states = SegmentedStates(2, a_0=1, b_0=1)  # IN PROGRESS

        if base_distn is not None:
            self.base_distn = base_distn
        else:
            self.base_distn = self.transitions._row_distns[0].alphav_0
        self._instantiate_base_distn(self.base_distn)

    def generate(self, T, start_obstate=0):
        """Sample demonstrations of length T."""
        As = self.obstate_trans_matrix

        x = self.init_state_distn.generate()
        s = start_obstate
        r = self.segmented_states.sample(x)

        stateseq = np.zeros((T, len(self.n_states)), dtype=np.int32)
        obs_dataseq = np.zeros((T, 2), dtype=np.int32)
        segmentedseq = np.zeros((T, len(self.n_states)), dtype=np.int32)

        for t in range(T):
            a = self.policy.generate(x, s)
            
            obs_dataseq[t] = [s, a]
            stateseq[t] = x
            segmentedseq[t] = r
            if r == 0:
                x = self.transitions.generate_x(x, s, a)
            else:
                x = self.init_state_distn.generate()

            r = self.segmented_states.sample(x)
            next_obstate_distn = As[s, a]
            s = sample_discrete(next_obstate_distn)
        
        return obs_dataseq, stateseq, segmentedseq

    def generate_normal(self, T, start_obstate=0):
        """Sample demonstrations of length T."""
        As = self.obstate_trans_matrix
        # Sample latent state
        x = self.init_state_distn.generate()
        s = start_obstate

        stateseq = np.zeros((T, len(self.n_states)), dtype=np.int32)
        obs_dataseq = np.zeros((T, 2), dtype=np.int32)

        for t in range(T):
            a = self.policy.generate(x, s)
            
            obs_dataseq[t] = [s, a]
            stateseq[t] = x
            
            x = self.transitions.generate_x(x, s, a)
            next_obstate_distn = As[s, a]
            s = sample_discrete(next_obstate_distn)
        
        return obs_dataseq, stateseq
    # Forward and backward log messages
    
    def messages_forwards_log(self, obs_dataseq, mf=True):
        """Compute log forward messages F(t, j) for all t, j.
        
        Forward messages are used to calculate the expected state
        sequence. It is also useful to compute the updates for
        global factors and the partition function.
        Parameters:
        obs_dataseq (list): sequence of state action pairs of length T
        Returns:
        numpy.array of shape `T x self.totalfeats`
        """
        if mf:
            transitions_matrix = self.transitions.mf_trans_matrix_xsax
            init_matrix = self.init_state_distn.mf_initial_distn_matrix
        else:
            transitions_matrix = self.transitions.trans_matrix_xsax
            init_matrix = self.init_state_distn.initial_distn_matrix

        # Apply log function to AMM parameters

        bxl = np.log(init_matrix)
        Axl = np.log(transitions_matrix)
        Asl = np.log(self.obstate_trans_matrix)
        
        aBl = self.policy.log_likelihood(obs_dataseq, mf=mf)

        alphal = np.zeros_like(aBl) # placeholder

        alphal[0] = bxl + aBl[0] # compute first forward message
        for t in range(1, len(obs_dataseq)): # for each time step
            st, at = obs_dataseq[t] # current state-action pair
            sp, ap = obs_dataseq[t-1] # previous state-action pair
            xsa_slice_p = (*([slice(None)]*len(self.n_states)), sp, ap)
            Axl_term = Axl[xsa_slice_p]
            Asl_term = Asl[sp, ap, st]
            alphal_term = np.repeat(alphal[t-1], np.prod(self.n_states)).reshape(Axl_term.shape)
            # Sum over all latent states (appearing as conditional)
            ax = (0, 1)
            alphal[t] = logsumexp(
                alphal_term + Axl_term + Asl_term, axis=ax) + aBl[t]
        return alphal

    def messages_backwards_log(self, obs_dataseq, mf=True):
        """Compute the log backward messages B(t, j) for all t, j.
        
        Similar to forward messages, backward messages are used to
        calculate the expected state sequence, updates for global
        factors, and the partition function.
        Parameters:
        obs_dataseq (list): sequence of state action pairs of length T
        Returns:
        numpy.array of shape `T x self.totalfeats`
        """
        # Apply log function to AMM parameters
        if mf:
            transitions_matrix = self.transitions.mf_trans_matrix_xsax
        else:
            transitions_matrix = self.transitions.trans_matrix_xsax

        Axl = np.log(transitions_matrix)
        Asl = np.log(self.obstate_trans_matrix)
        aBl = self.policy.log_likelihood(obs_dataseq, mf)

        betal = np.zeros_like(aBl) # placeholder

        for t in reversed(range(len(obs_dataseq)-1)):
            st, at = obs_dataseq[t]
            sn, an = obs_dataseq[t+1]
            xsa_slice = (*([slice(None)]*len(self.n_states)), st, at)
            Axl_term = Axl[xsa_slice]
            Asl_term = Asl[st, at, sn]
            # Sum over all latent states (appearing as conditional)
            ax = tuple(range(-1, -(len(self.n_states)+1), -1))
            betal[t] = logsumexp(
                Axl_term + Asl_term + betal[t+1] + aBl[t+1],
                axis=ax)
        return betal

    # Expected state values

    def expected_stateprobs(self, alphal, betal):
        """Get the expected probability for each latent state sequence.
        
        Parameters:
        alphal (numpy.array): log forward messages
        betal (numpy.array): log backward messages
        Returns #TODO: check shape
        numpy.array of shape `T x [self.n_states] ** self.n_features`
        """
        nf = len(self.n_states)
        na_slice = (slice(None), *[na]*nf)
        ax = tuple(range(1, nf + 1))

        expected_stateprobs = alphal + betal
        expected_stateprobs -= expected_stateprobs.max(ax)[na_slice]
        np.exp(expected_stateprobs, out=expected_stateprobs)
        expected_stateprobs /= expected_stateprobs.sum(ax)[na_slice]

        assert not np.any(np.isnan(expected_stateprobs)), 'nan appears in state probs'
        
        return expected_stateprobs
    
    def _instantiate_base_distn(self, base_distn):
        """Instantiantes base distribution for all Dirichlet distns."""
        self.transitions.base_distn = base_distn
        self.init_state_distn.base_distn = base_distn
        self.policy.base_distn = self.policy.obs_distns[0].alphav_0

    def _expected_stateseq(self, alphal, betal):
        """Get the expected latent state sequence.
        
        Parameters:
        alphal (numpy.array): log forward messages
        betal (numpy.array): log backward messages
        Returns
        numpy.array of shape `T x self.n_features`
        """
        stateprobs = self.expected_stateprobs(alphal, betal)
        out = [np.unravel_index(np.argmax(block, axis=None),
                                block.shape)
                                for block in stateprobs]

        expected_xseq = np.array(out).astype('int32')
        return expected_xseq

    def expected_stateseq(self, dataobs, mf=False):
        """Get the expected latent state sequence.
        
        Parameters:
        dataobs (numpy.array): state-action sequences
        Returns
        numpy.array of shape `T x self.n_features`
        """
        if not isinstance(dataobs, list):
            dataobs = [dataobs]
        expected_xseq_list = []

        for data in dataobs:
            alphal = self.messages_forwards_log(data, mf)
            betal = self.messages_backwards_log(data, mf)
            stateprobs = self.expected_stateprobs(alphal, betal)
            out = [np.unravel_index(np.argmax(block, axis=None),
                                    block.shape)
                   for block in stateprobs]

            expected_xseq = np.array(out).astype('int32')
            expected_xseq_list.append(expected_xseq)
        return expected_xseq_list

    def _normalizer(self, alphal, betal):
        self.normalizer = logsumexp(alphal[0] + betal[0])
        return self.normalizer

    def vlb(self, alphal, betal):
        # Calculate normalizer
        '''nf = self.n_features
        na_slice = (slice(None), *[na]*nf)
        ax = tuple(range(1, nf + 1))
        expected_stateprobs = alphal + betal
        expected_stateprobs -= expected_stateprobs.max(ax)[na_slice]
        np.exp(expected_stateprobs, out=expected_stateprobs)
        # log normalizer
        normalizer = np.log(expected_stateprobs.sum(ax)[na_slice]).sum()'''
        vlb = 0.
        #vlb += normalizer # vlb = normalizer: log or not log?
        vlb += self.transitions.get_vlb() # sum of vlb of each row transition
        vlb += self.init_state_distn.get_vlb() # vlb calculated by Categorical
        vlb += self.policy.get_vlb() # sum of vlb of each row policy
        return vlb

    def max_viterbi_msg(self, obs_dataseq, mf=True):
        errs = np.seterr(divide='ignore')
        N = len(obs_dataseq)
        if mf:
            transitions_matrix = self.transitions.mf_trans_matrix_xsax
            init_matrix = self.init_state_distn.mf_initial_distn_matrix
        else:
            transitions_matrix = self.transitions.trans_matrix_xsax
            init_matrix = self.init_state_distn.initial_distn_matrix

        # Apply log function to AMM parameters

        bxl = np.log(init_matrix)
        Axl = np.log(transitions_matrix)
        Asl = np.log(self.obstate_trans_matrix)
        
        aBl = self.policy.log_likelihood(obs_dataseq, mf=mf)

        np.seterr(**errs)

        msgsl = np.zeros_like(aBl) # placeholder
        maxseq = np.zeros(aBl.shape, dtype=np.int32)

        msgsl[0] = bxl + aBl[0]

        for t in range(1, N): # for each time step
            st, _ = obs_dataseq[t] # current state-action pair
            sp, ap = obs_dataseq[t-1] # previous state-action pair
            Asl_term = Asl[sp, ap, st]
            aBl_term = aBl[t]
            xsa_slice_p = (*([slice(None)]*len(self.n_states)), sp, ap)
            Axl_term = Axl[xsa_slice_p] 
            #Axl_term = Axl[:, sp, ap]
            recursion = msgsl[t-1].reshape([-1, *([1]*len(self.n_states))])
            vals = aBl_term + recursion + Axl_term + Asl_term
            msgsl[t] = vals.max(axis=tuple(range(len(self.n_states))))
            maxseq[t] = vals.argmax(axis=tuple(range(len(self.n_states)))[0]) #TODO: correct this to have it generalized to several latent statess
        return msgsl, maxseq

        
        # # initialization
        # t = 0
        # vals = aBl[t] + bxl
        # scores[t] = copy.deepcopy(vals)
        # args[t] = np.arange(num_states)
        # # recursion
        # for t in range(1,scores.shape[0]):
        #     # extract relevant data
        #     sp,ap = data[t-1,0],data[t-1,1]
        #     st = data[t,0]
        #     # compute relevant terms
        #     # note xp is axis=0, xt is axis=1
        #     aBl_term = aBl[t]
        #     scores_term = scores[t-1]
        #     scores_term = scores_term[:,na]
        #     Axl_term = Axl[:,sp,ap]
        #     Asl_term = Asl[sp,ap,st]
        #     # maxum message computation
        #     vals = aBl_term + scores_term + Axl_term + Asl_term
        #     vals.argmax(axis=0,out=args[t])
        #     vals.max(axis=0, out=scores[t])

        # return scores, args
    
    def _maximize_backwards(self, scores, args):
        T = scores.shape[0]

        stateseq = np.empty(T,dtype=np.int32)

        stateseq[T-1] = (scores[T-1]).argmax()
        for idx in range(T-2,-1,-1):
            stateseq[idx] = args[idx, stateseq[idx+1]]

        return stateseq
    
    def max_viterbi(self, dataobs, mf=True):
        msgsl, maxseq = self.max_viterbi_msg(dataobs, mf=mf)
        return self._maximize_backwards(msgsl, maxseq)

    def meanfieldupdate(self, dataobslist, only_policy=True, mf=True):
        """Do meanfield update.
        Params:
            dataobs - can be either 2D list or list of 2D lists
        """
        if not isinstance(dataobslist, list):
            dataobslist = [dataobslist]
        
        alphal_list = []
        betal_list = []
        stateprobs_list = []
        expected_transcounts_list = []
        normalizer_list = []
        for dataobs in dataobslist:
            alphal = self.messages_forwards_log(dataobs, mf)
            betal = self.messages_backwards_log(dataobs, mf)
            stateprobs = self.expected_stateprobs(alphal, betal)
            assert not np.any(np.isnan(stateprobs)), 'nan appears in state probs'

            expected_transcounts = self._expected_transcounts(dataobs,
                                                              alphal,
                                                              betal)
            normalizer = self._normalizer(alphal, betal)

            alphal_list.append(alphal)
            betal_list.append(betal)
            stateprobs_list.append(stateprobs)
            expected_transcounts_list.append(expected_transcounts)
            normalizer_list.append(normalizer)

        self.policy.meanfieldupdate(dataobslist, stateprobs_list)
        if not only_policy:
            self.init_state_distn.meanfieldupdate(
                alphal_list,
                betal_list,
                normalizer_list
                )
            self.transitions.meanfieldupdate(
                np.array(expected_transcounts_list)
                )
        return self.vlb(alphal, betal)
    
    def supervised_meanfieldupdate(self, dataobslist, statelist,
                                   only_policy=True):
        """Do meanfield update.
        Params:
            dataobs - can be either 2D list or list of 2D lists
            statelist - 
        """
        if not isinstance(dataobslist, list):
            dataobslist = [dataobslist]
            #statelist = [statelist]

        stateprobs_list = []
        for stateobs in statelist:
            stateprobs = np.zeros([len(stateobs), *self.n_states])
            for j in range(len(stateobs)):
                stateprobs[j][tuple(stateobs[j])] = 1

            stateprobs_list.append(stateprobs)

        self.policy.meanfieldupdate(dataobslist, stateprobs_list)
        if not only_policy:
            raise NotImplementedError

    def semisupervised_meanfieldupdate(self,
                                       dataobslist_labeled,
                                       statelist,
                                       dataobslist_unlabeled,
                                       only_policy=True,
                                       mf=True):
        # Calculate state probabilities for the known latent states
        if not isinstance(dataobslist_labeled, list):
            dataobslist_labeled = [dataobslist_labeled]
        if not isinstance(dataobslist_unlabeled, list):
            dataobslist_unlabeled = [dataobslist_unlabeled]
        if not isinstance(statelist, list):
            statelist = [statelist]

        # Statelist to probability representation
        stateprobs_list1 = []
        for stateobs in statelist:
            stateprobs = np.zeros([len(stateobs), *self.n_states])
            for j in range(len(stateobs)):
                stateprobs[j][tuple(stateobs[j])] = 1

            stateprobs_list1.append(stateprobs)

        # Estimate state probabilitied for unknown latent states
        alphal_list = []
        betal_list = []
        stateprobs_list2 = []
        # expected_transcounts_list = []
        # normalizer_list = []
        for dataobs in dataobslist_unlabeled:
            alphal = self.messages_forwards_log(dataobs, mf)
            betal = self.messages_backwards_log(dataobs, mf)
            stateprobs = self.expected_stateprobs(alphal, betal)
            assert not np.any(np.isnan(stateprobs)), 'nan appears in state probs'
            alphal_list.append(alphal)
            betal_list.append(betal)
            stateprobs_list2.append(stateprobs)
            # expected_transcounts = self._expected_transcounts(dataobs,
            #                                                   alphal,
            #                                                   betal)
            # normalizer = self._normalizer(alphal, betal)


            # expected_transcounts_list.append(expected_transcounts)
            # normalizer_list.append(normalizer)
        
        # Concatenate results
        dataobslist = dataobslist_labeled + dataobslist_unlabeled
        stateprobs_list = stateprobs_list1 + stateprobs_list2
        self.policy.meanfieldupdate(dataobslist, stateprobs_list)
        if not only_policy:
            raise NotImplementedError
        # TODO: update initial_state_distn and transitionmatrix
        # if not only_policy:
        #     self.init_state_distn.meanfieldupdate(
        #         alphal_list,
        #         betal_list,
        #         normalizer_list
        #         )
        #     self.transitions.meanfieldupdate(expected_transcounts_list)
        # return self.vlb(alphal, betal)

    def _expected_transcounts(self, dataobs, alphal, betal, mf=False):
        aBl = self.policy.log_likelihood(dataobs, mf)
        if mf:
            transitions_matrix = self.transitions.mf_trans_matrix_xsax
        else:
            transitions_matrix = self.transitions.trans_matrix_xsax

        Axl = np.log(transitions_matrix)
        Asl = np.log(self.obstate_trans_matrix)
        ax1 = [-i for i in range(1, len(self.n_states) + 1)]
        ax2 = [i for i in range(1, len(self.n_states) + 1)]
        alphaexpanded = np.expand_dims(alphal[:-1], axis = ax1)
        betaexpanded = np.expand_dims(betal[1:], axis = ax2)
        aBlexpanded = np.expand_dims(aBl[1:], axis = ax2)
        log_joints = alphaexpanded + (betaexpanded + aBlexpanded)
        for idx in range(alphal.shape[0]-1):
            st, at = dataobs[idx]
            sn, an = dataobs[idx+1]
            xsa_slice = (*([slice(None)]*len(self.n_states)), st, at)
            log_joints[idx] += Axl[xsa_slice]
            log_joints[idx] += Asl[st, at, sn]
        log_joints -= log_joints.max((1,2))[:, na, na]
        # NaN values are originated when substracting np.inf - np.inf
        # Convert those values to 0 (although this is not true)
        # TODO: check how to avoid NaN in the first place
        log_joints = np.nan_to_num(log_joints)
        joints = np.exp(log_joints)
        ax = [i for i in range(1, 2*len(self.n_states)+1)]
        joints /= np.expand_dims(joints.sum(tuple(ax)), axis=ax)

        # converting transcounts to the same dimension as trans_matrix
        rows = self.transitions.trans_matrix.shape[0]
        num_steps = joints.shape[0]
        joints_row = np.zeros(
            (num_steps, rows, *self.n_states))
        for idx in range(num_steps):
            st, at = dataobs[idx]
            nx = self.n_states
            perm = (range(j) for j in nx)
            xset = itertools.product(*perm)
            for xt in xset:
                row = self.transitions.get_row_idx(xt, st, at)
                m = (idx, row)
                m2 = (idx, *(xt))
                joints_row[m] = joints[m2]

        self.expected_transcounts = joints_row.sum(0)

        return self.expected_transcounts

    '''
    def svi_meanfieldupdate(self):
        """Do stochastic variational inference update."""
        '''