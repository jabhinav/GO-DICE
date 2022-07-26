""" """
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from pybasicbayes.distributions import Categorical

from py_amm.amm import Amm


def fwdbwd_test(num_features, num_states, num_obstates, num_actions, data_len):
    """Test with initialization params same as true_model."""
    # Build state transition matrix
    obstate_trans_matrix = np.zeros((num_obstates, num_actions, num_obstates))
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
        init_state_concentration=initial_concentration,
        transition_alpha=alpha_model,
        rho=alpha0_policy,
    )
    data = amm.generate(data_len)
    dataobs = data[0]
    stateobs = data[1]
    expected = amm.expected_stateseq(dataobs)
    acc = (expected == stateobs).sum() / (data_len * num_features)
    error = 1 - acc
    return error


def unittest(
    num_features, num_states, num_obstates, num_actions, data_len, test_type="A"
):
    """Test with initialization params same as true_model."""
    # Build state transition matrix
    obstate_trans_matrix = np.zeros((num_obstates, num_actions, num_obstates))
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
        init_state_concentration=initial_concentration,
        transition_alpha=alpha_model,
        rho=alpha0_policy,
    )

    if test_type == "A":
        post_amm = deepcopy(amm)
    elif test_type == "B":
        post_amm = Amm(
            obstate_trans_matrix,
            num_obstates,
            num_actions,
            num_states,
            num_features,
            init_state_concentration=initial_concentration,
            transition_alpha=alpha_model,
            rho=alpha0_policy,
        )
        post_amm.policy = deepcopy(amm.policy)
    elif test_type == "C":
        post_amm = Amm(
            obstate_trans_matrix,
            num_obstates,
            num_actions,
            num_states,
            num_features,
            init_state_concentration=initial_concentration,
            transition_alpha=alpha_model,
            rho=alpha0_policy,
        )

    data = amm.generate(data_len)
    dataobs = data[0]
    stateobs = data[1]
    for j in range(100):  # training iterations.
        post_amm.meanfieldupdate(dataobs, mf=False)

    expected = post_amm.expected_stateseq(dataobs)
    acc = (expected == stateobs).sum() / (data_len * num_features)
    error = 1 - acc
    return error


def plot_results(average_errors, min_errors, max_errors, title, xlabel, ylabel):
    N = len(average_errors)
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.plot(average_errors, "--", color="red", alpha=0.5, label="Average error")
    ax.plot(max_errors, color="royalblue", label="Max error")
    ax.plot(min_errors, "-o", color="royalblue", label="Min error")

    ax.set_ylim([0, 1])
    ax.set_xlim([1, N - 1])

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.annotate(
        "Expert demonstration length: " + str(data_len),
        xy=(2.8, 0.7),
        xycoords="data",
        xytext=(-100, 60),
        textcoords="offset points",
    )
    ax.fill_between(x=range(N), y1=max_errors, y2=min_errors, alpha=0.2)
    fig.suptitle(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.show()


def runtests_on_features(max_features=7, test_type="A"):
    for num_features in range(1, max_features + 1):
        print("Testing on ", num_features, " features")
        errors_tmp = []
        for j in range(10):  # repeat 100 times for stats
            print("Training No", j)
            err = unittest(
                num_features, num_states, num_obstates, num_actions, data_len, test_type
            )
            errors_tmp.append(err)
        av_err.append(np.average(errors_tmp))
        mi_err.append(np.min(errors_tmp))
        ma_err.append(np.max(errors_tmp))
    return av_err, mi_err, ma_err


def runtests_on_states(max_states=8, test_type="A"):
    for s in range(1, max_states + 1):
        num_obstates = s
        num_actions = s
        num_states = s
        print("Testing on ", num_states, " states")
        errors_tmp = []
        for j in range(10):  # repeat test 100 times
            print("Training No", j)
            err = unittest(
                num_features, num_states, num_obstates, num_actions, data_len, test_type
            )
            errors_tmp.append(err)
        av_err.append(np.average(errors_tmp))
        mi_err.append(np.min(errors_tmp))
        ma_err.append(np.max(errors_tmp))
    return av_err, mi_err, ma_err


num_obstates = 2
num_actions = 2
num_states = 2
num_features = 1
data_len = 2000
av_err = [0]
mi_err = [0]
ma_err = [0]
alpha0_policy = 1
alpha0_transitions = 1
alpha_model = 1
initial_concentration = 1
av_err, mi_err, ma_err = runtests_on_features(max_features=1)
av_err, mi_err, ma_err = runtests_on_states(max_states=2)

# NOTE: next tests are for varying size of S, X and A set.
av_err, mi_err, ma_err = runtests_on_states(test_type="C", max_states=2)
plot_results(
    av_err,
    mi_err,
    ma_err,
    title="AMM Decoding",
    xlabel="Problem Size",
    ylabel="Normalized Hamming Distance",
)

# NOTE: next tests are with varying X dimensionality,
#       with |S| = |A| = 2, and two latent states
#       across each dimension
av_err, mi_err, ma_err = runtests_on_features()
plot_results(
    av_err,
    mi_err,
    ma_err,
    title="AMM Decoding",
    xlabel="Latent State Dimension",
    ylabel="Normalized Hamming Distance",
)
