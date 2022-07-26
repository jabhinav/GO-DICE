import numpy as np
from pybasicbayes.distributions import Categorical

from py_amm.utils.stats import kl_divergence
from py_amm.amm import Amm
from py_amm.utils.stats import AmmMetrics
from py_amm.internals.segmented_states import GEM


num_obstates = 4
num_actions = 4
num_states = 4
num_features = 2
data_len = 2000
alpha0_policy = 1
alpha0_transitions = 1
#Generate base distribution
alpha_GEM = 10
gem = GEM(alpha_GEM)
alphav_0 = []
for _ in range(num_states**num_features-1):
    gem.rvs()
alphav_0 = gem.sticks[1:]
alphav_0.append(1 - sum(alphav_0))
alphav_0 = np.array(alphav_0)

# Build state transition matrix
obstate_trans_matrix = np.zeros((num_obstates, num_actions, num_obstates))
# obstate_trans_matrix[:,:,1] = 1
for obstate in range(num_obstates):
    for action in range(num_actions):
        obstate_trans_matrix[obstate, action] = Categorical(
            alpha_0=alpha0_transitions, K=num_obstates
        ).weights

amm = Amm(
    obstate_trans_matrix,
    num_obstates,
    num_actions,
    num_states,
    num_features,
    init_state_weights=alphav_0,
    transition_alphav=alphav_0,
    rho=alpha0_policy,
)

# trans = np.zeros_like(amm.transitions.trans_matrix)
# trans[:, 1] = 1
# policy = np.zeros_like(amm.policy.policy_matrix)
# policy[:, 1] = 1
# amm = Amm(
#     obstate_trans_matrix,
#     num_obstates,
#     num_actions,
#     num_states,
#     num_features,
#     init_state_concentration=initial_concentration,
#     transition_matrix=trans,
#     policy_matrix = policy,
#     rho=alpha0_policy,
#     transition_alpha=alpha_model,
# )

#Generate base distribution again
#alpha_GEM = 10
#gem = GEM(alpha_GEM)
#alphav_0 = []
#for _ in range(num_states**num_features-1):
#    gem.rvs()
#alphav_0 = gem.sticks[1:]
#alphav_0.append(1 - sum(alphav_0))
#alphav_0 = np.array(alphav_0)
post_amm = Amm(
    obstate_trans_matrix,
    num_obstates,
    num_actions,
    num_states,
    num_features,
    init_state_weights=alphav_0,
    transition_alphav=alphav_0,
    rho=alpha0_policy,
)
from copy import deepcopy as copy
post_amm.policy = copy(amm.policy)


data = amm.generate_normal(data_len)
dataobs = data[0]
stateobs = data[1]

metrics = AmmMetrics(amm, post_amm)
kl = metrics.kl_divergence('transition')
kl_divergence(amm.transitions.trans_matrix_xsax, post_amm.transitions.trans_matrix_xsax)


kl = metrics.kl_divergence('policy')
kl_divergence(amm.policy.policy_matrix, post_amm.policy.policy_matrix)

kl = metrics.kl_divergence('init')
kl_divergence(amm.init_state_distn.initial_distn, post_amm.init_state_distn.initial_distn)
pol_divergence = metrics.kl_policy
init_divergence = metrics.kl_init
tx_divergence = metrics.kl_trans


# Train
counter = 0
while True:
    print(counter)
    counter += 1
    post_amm.meanfieldupdate(dataobs, mf=True)
 
    metrics = AmmMetrics(amm, post_amm, preserve_order=False)
    print(metrics.kl_trans)
    #init_divergence = kl_divergence(post_amm.init_state_distn.initial_distn, amm.init_state_distn.initial_distn)
    #tx_divergence = kl_divergence(post_amm.transitions.trans_matrix, amm.transitions.trans_matrix)
    #print('tx_divergence: ', tx_divergence, '\n init_divergence: ', init_divergence)
    alphal = post_amm.messages_forwards_log(dataobs)
    betal = post_amm.messages_backwards_log(dataobs)
    expected_transcounts2 = post_amm._expected_transcounts(dataobs, alphal, betal)

metrics5 = AmmMetrics(amm, post_amm, preserve_order=False)
# Error of expected state estimation of amm
expected = amm.expected_stateseq(dataobs)
acc = (expected == stateobs).sum() / (data_len * num_features)
error = 1 - acc

# Error of expected state estimation of post_amm
expected2 = post_amm.expected_stateseq(dataobs)
acc = (expected2 == stateobs).sum() / (data_len * num_features)

# Error of Viterbi state estimation of post_amm
expected2 = post_amm.max_viterbi(dataobs)
expected = amm.max_viterbi(dataobs)
acc = (expected2 == expected).sum() / (data_len * num_features)

acc2 = (expected2 == stateobs.flatten()).sum() / (data_len * num_features)
error = 1 - acc


# Trying out new metric
res = 0
alphal = post_amm.messages_forwards_log(dataobs)
betal = post_amm.messages_forwards_log(dataobs)
exp = post_amm.expected_stateprobs(alphal, betal)

for j in range(len(stateobs)):
    arr = stateobs[j]
    res += exp[j,arr[0]]



# Error of expected state estimation of post_amm
expected3 = post_amm.expected_stateseq(dataobs)
acc = (expected3 == stateobs).sum() / (data_len * num_features)
