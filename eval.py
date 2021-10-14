import os
import sys
import numpy as np
import tensorflow as tf
import pickle as pkl
import itertools
from models import Agent
from utils import parse_data, one_hot_encode, plot_infogail_results
from domains import BoxWorld, RoadWorld, TeamBoxWorld
from config import training_data_config, data_dir, param_dir, env_name, seed_values, model_name, state_action_path, latent_path, kl_figpath
from config import InfoGAIL_config as model_config
from config import env_roadWorld_config, env_boxWorld_config, env_teamBoxWorld_config
from BC import Generator as BC_Generator
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

selected_model_id = 499  # For InfoGAIL, we pick the last model

if env_name == "BoxWorld":
    env_config = env_boxWorld_config
elif env_name == "RoadWorld":
    env_config = env_roadWorld_config
elif env_name == "TeamBoxWorld":
    env_config = env_teamBoxWorld_config
else:
    print "Specify a valid environment name in the config file"
    sys.exit(-1)

# initialize the env
if env_name == "BoxWorld":
    env = BoxWorld(grid_length_x=env_config.grid_length_x, grid_length_y=env_config.grid_length_y)
elif env_name == "RoadWorld":
    env = RoadWorld(grid_length_x=env_config.grid_length_x, grid_length_y=env_config.grid_length_y)
elif env_name == "TeamBoxWorld":
    env = TeamBoxWorld(grid_length_x=env_config.grid_length_x, grid_length_y=env_config.grid_length_y)
else:
    print "Specify a valid environment name in the config file"
    sys.exit(-1)


def pred_kl_div(act_gt, act_pred, weights=None, epsilon=10e-9):  # KL(p,q) = plog(p) - plog(q)
    if weights is not None:
        kl_div = np.sum(np.multiply(act_gt, np.log(act_gt+epsilon)) - np.multiply(act_gt, np.log(act_pred+epsilon)),
                        axis=1)
        avg_div = np.sum(np.multiply(kl_div, weights))  # Since, its weighted just sum it. No averaging
    else:
        kl_div = np.sum(np.multiply(act_gt, np.log(act_gt+epsilon)) - np.multiply(act_gt, np.log(act_pred+epsilon)),
                        axis=1)
        avg_div = np.average(kl_div)  # Avg. across samples
    # steer_loss = np.average(np.abs(act_gt[:, 0] - act_pred[:, 0]))
    # accel_loss = np.average(np.abs(act_gt[:, 1] - act_pred[:, 1]))
    # brake_loss = np.average(np.abs(act_gt[:, 2] - act_pred[:, 2]))
    return np.array([avg_div])


def get_index_of_all_max(preds):
    sample_max_indexes = []
    for _i in range(preds.shape[0]):
        sample_max_indexes.append(set(np.where(preds[_i] == np.max(preds[_i]))[0]))
    return sample_max_indexes


def pred_0_1_loss(_gt, _pred, weights=None):
    #  Choose maximum prob action which matches GT if multiple actions are predicted with high and equal prob
    _gt_max_indexes = get_index_of_all_max(_gt)
    _pred_max_indexes = get_index_of_all_max(_pred)
    # act_gt = np.argmax(act_gt, axis=1)
    # act_pred = np.argmax(act_pred, axis=1)
    num_t = len(_gt)
    count = 0
    if weights is not None:
        for i in xrange(num_t):
            if _gt_max_indexes[i] & _pred_max_indexes[i]:
                count += 1.*weights[i]
        return 1. - float(count)
    else:
        for i in xrange(num_t):
            if _gt_max_indexes[i] & _pred_max_indexes[i]:
                count += 1.
        return 1. - (float(count)/num_t)


def compute_weights(env, demo):
    weights = {(s, c): 0. for s in xrange(len(env.states_dict.keys())) for c in xrange(len(env.latent_state_dict.keys()))}

    # From training data
    states_d, actions_d, encodes_d = demo["states"], demo["actions"], demo['latent_states']

    num_pairs = states_d.shape[0]
    for i in xrange(num_pairs):
        state = env.states_dict[tuple(states_d[i])]
        encode = np.argmax(encodes_d[i])
        weights[(state, encode)] += 1.

    for state_encode in weights.keys():
        weights[state_encode] /= float(num_pairs)

    return weights


def get_data():
    state_dim = env_config.state_dim
    encode_dim = env_config.encode_dim
    action_dim = env_config.action_dim
    # actions_one_hot_encoded = False

    # define the model
    agent = Agent(env, sess, state_dim, action_dim, encode_dim, model_config)

    # Load expert (state, action) pairs and G.T latent modes
    print "Loading data ..."
    with open(state_action_path, 'rb') as f:
        trajectories = pkl.load(f)
    with open(latent_path, 'rb') as f:
        latent_modes_traj = pkl.load(f)
    demo_train = parse_data(env, trajectories, latent_modes_traj, lower_bound=0, upper_bound=training_data_config.num_traj)

    # Enumerate all possible states-latent-action tuples for a given env.
    states_d, actions_d, encodes_d = [], [], []
    expert_policy_matrix = env.compute_policy()
    for state in env.states_dict.keys():
        for encode in env.latent_state_dict.keys():
            states_d.append(state)
            encodes_d.append(env.latent_state_dict[encode])
            # The expert policy is non-deterministic, so we have the expert's prob distribution over actions
            actions_d.append(expert_policy_matrix[encode[0], encode[1], env.states_dict[state], :])
    states_d = np.array(states_d)
    encodes_d = one_hot_encode(np.array(encodes_d), dim=encode_dim)
    actions_d = np.array(actions_d)

    # For evaluating on test data
    # states_d, actions_d, encodes_d = demo["states"], demo["actions"], demo['latent_states']

    # Compute weights for state-encode pairs based on their freq. of occurrence in training data w[(s,c)]=weight
    weights = compute_weights(env, demo_train)
    sc_weights = np.array([weights[(env.states_dict[tuple(state)], np.argmax(encode))]
                           for (state, encode) in zip(states_d, encodes_d)])  # Note order of w(s,c) is maintained
            # [weights[(env.states_dict[tuple(states_d[nd])], np.argmax(encodes_d[nd]))] for nd in xrange(states_d.shape[0])])

    data = {
        'agent': agent,
        'states_d': states_d,
        'encodes_d': encodes_d,
        'actions_d': actions_d,
        'weights': sc_weights
    }
    return data


def get_test_data():
    state_dim = env_config.state_dim
    encode_dim = env_config.encode_dim
    action_dim = env_config.action_dim
    # actions_one_hot_encoded = False

    # define the model
    agent = Agent(env, sess, state_dim, action_dim, encode_dim, model_config)

    # Load expert (state, action) pairs and G.T latent modes
    print "Loading data ..."
    with open(state_action_path, 'rb') as f:
        trajectories = pkl.load(f)
    with open(latent_path, 'rb') as f:
        latent_modes_traj = pkl.load(f)

    # Get last 100 traj for the test data
    demo_test = parse_data(env, trajectories, latent_modes_traj, lower_bound=900, upper_bound=1000)

    # For evaluating on test data
    states_d, actions_d, encodes_d = demo_test["states"], demo_test["actions"], demo_test['latent_states']

    data = {
        'agent': agent,
        'states_d': states_d,
        'encodes_d': encodes_d,
        'actions_d': actions_d,
    }
    return data


def eval(exp_num, data, permute=None):
    # print "\n------- Exp No. {} -------".format(exp_num)

    # Load Generator weights
    # model_path = param_dir + "params0_v1/generator_model_{}.h5".format(exp_num)
    # model_path = param_dir + "generator_bc_model_{}.h5".format(exp_num)

    encodes_d = data['encodes_d']
    if permute:
        encodes_d = encodes_d[:, permute]

    def load_predict_and_evaluate(path, generator_model):
        generator_model.load_weights(path)
        if model_name == "BC":
            act_pred = generator_model.predict([data['states_d']])
        else:
            act_pred = generator_model.predict([data['states_d'], encodes_d])
        kl = pred_kl_div(data['actions_d'], act_pred)
        acc = pred_0_1_loss(data['actions_d'], act_pred)
        w_kl = pred_kl_div(data['actions_d'], act_pred, weights=data['weights'])
        w_acc = pred_0_1_loss(data['actions_d'], act_pred, weights=data['weights'])
        return kl, acc, w_kl, w_acc

    if model_name == "BC":
        model_path = param_dir + "generator_bc_model_{}.h5".format(exp_num)
        generator = BC_Generator(sess, env_config.state_dim, env_config.action_dim)
        kl, acc, w_kl, w_acc = load_predict_and_evaluate(model_path, generator.model)

        # generator.model.load_weights(model_path)
        # act_pred = generator.model.predict([data['states_d']])
        # kl = kl_div(data['actions_d'], act_pred)
        # acc = pred_acc(data['actions_d'], act_pred)
        # w_kl = kl_div(data['actions_d'], act_pred, weights=data['weights'])
        # w_acc = pred_acc(data['actions_d'], act_pred, weights=data['weights'])
    elif model_name == 'BC_InfoGAIL':
        model_path = param_dir + "generator_bc_model_{}.h5".format(exp_num)
        agent = data['agent']
        kl, acc, w_kl, w_acc = load_predict_and_evaluate(model_path, agent.generator)

        # agent.generator.load_weights(model_path)
        # act_pred = agent.generator.predict([data['states_d'], encodes_d])
        # kl = kl_div(data['actions_d'], act_pred)
        # acc = pred_acc(data['actions_d'], act_pred)
        # w_kl = kl_div(data['actions_d'], act_pred, weights=data['weights'])
        # w_acc = pred_acc(data['actions_d'], act_pred, weights=data['weights'])
    else:
        model_dir = param_dir + "params{}/".format(exp_num)
        model_path = os.path.join(model_dir, 'generator_model_{}.h5'.format(selected_model_id))
        agent = data['agent']
        kl, acc, w_kl, w_acc = load_predict_and_evaluate(model_path, agent.generator)
        # kl_models, acc_models, w_kl_models, w_acc_models = [], [], [], []
        # import os
        # import re
        # models = os.listdir(model_dir)
        # gen_models = [m for m in models if re.match(re.compile('generator_model_' + "[0-9]*" + ".h5"), m)]
        # for model in gen_models:
        #     model_path = os.path.join(model_dir, model)
        #     agent = data['agent']
        #
        #     kl, acc, w_kl, w_acc = load_predict_and_evaluate(model_path, agent.generator)
        #     # agent.generator.load_weights(model_path)
        #     # act_pred = agent.generator.predict([data['states_d'], encodes_d])
        #     #
        #     # kl = kl_div(data['actions_d'], act_pred)
        #     # acc = pred_acc(data['actions_d'], act_pred)
        #     # w_kl = kl_div(data['actions_d'], act_pred, weights=data['weights'])
        #     # w_acc = pred_acc(data['actions_d'], act_pred, weights=data['weights'])
        #     kl_models.append(kl)
        #     acc_models.append(acc)
        #     w_kl_models.append(w_kl)
        #     w_acc_models.append(w_acc)
        # selected_model_id =
        # print "Found min weighted KL Div at", selected_model_id
        # print "Found max weighted Acc at", np.argmax(w_acc)
        # kl = kl_models[selected_model_id]
        # w_kl = w_kl_models[selected_model_id]
        # acc = acc_models[selected_model_id]
        # w_acc = w_acc_models[selected_model_id]
        # plot_infogail_results(kl_models, acc_models, w_kl_models, w_acc_models,
        #                       InfoGAIL_figpath + "InfoGAIL_perf{}.png".format(exp_num))

    # kl = kl_div(data['actions_d'], act_pred)
    # acc = pred_acc(data['actions_d'], act_pred)
    # print "KL Divergence is:", kl
    # print "Acc:", acc, '%'
    # #
    # # w_kl = kl_div(data['actions_d'], act_pred, weights=data['weights'])
    # # w_acc = pred_acc(data['actions_d'], act_pred, weights=data['weights'])
    # print "Weighted KL Divergence is:", w_kl
    # print "Weighted Acc:", w_acc, '%'
    return kl, acc, w_kl, w_acc


def eval_across_runs(range_, data, plot=False, fig_path=None, permute=None, metrics_permute=[]):

    kl_avg = []
    acc_avg = []
    wkl_avg = []
    wacc_avg = []
    for exp_num, seed_value in enumerate(xrange(range_)):
        # print "\n ------- Exp No. {} -------".format(exp_num)
        # Set a seed value
        np.random.seed(seed_value)
        tf.set_random_seed(seed_value)

        kl, acc, w_kl, w_acc = eval(exp_num, data, permute)
        kl_avg.append(kl)
        acc_avg.append(acc)
        wkl_avg.append(w_kl)
        wacc_avg.append(w_acc)

    kl_mean, kl_std = np.mean(np.array(kl_avg)), np.std(np.array(kl_avg))
    acc_mean, acc_std = np.mean(np.array(acc_avg)), np.std(np.array(acc_avg))
    wkl_mean, wkl_std = np.mean(np.array(wkl_avg)), np.std(np.array(wkl_avg))
    wacc_mean, wacc_std = np.mean(np.array(wacc_avg)), np.std(np.array(wacc_avg))
    print "\nKL: {} +- {}".format(kl_mean, kl_std)
    print "0-1 Loss: {} +- {}".format(acc_mean, acc_std)
    print "Weighted-KL: {} +- {}".format(wkl_mean, wkl_std)
    print "Weighted 0-1 Loss: {} +- {}".format(wacc_mean, wacc_std)

    if plot and fig_path:
        plot_infogail_results(wkl_avg, fig_path)

    if permute:
        metrics_permute.append((permute,
                                [(kl_mean, kl_std), (acc_mean, acc_std), (wkl_mean, wkl_std), (wacc_mean, wacc_std)]))

    return metrics_permute


def get_decoding_error(exp_num, data, permute):
    encodes_d = data['encodes_d']
    encodes_d = encodes_d[:, permute]
    if model_name != "InfoGAIL":
        print "Decoding error not supported on {}".format(model_name)
        return None, None
    else:
        model_dir = param_dir + "params{}/".format(exp_num)
        model_path = os.path.join(model_dir, 'posterior_model_{}.h5'.format(selected_model_id))
        agent = data['agent']
        agent.posterior_target.load_weights(model_path)
        encodes_pred = agent.posterior_target.predict([data['states_d'], data['actions_d']])
        decoding_error = pred_0_1_loss(encodes_d, encodes_pred)
        return decoding_error


def compute_avg_decoding_error(range_):
    data = get_test_data()
    # permutations = itertools.permutations(list(range(len(env.latent_state_dict))))
    permute = [1, 3, 2, 0]
    avg_w_decode, avg_decode = [], []
    for exp_num, seed_value in enumerate(xrange(range_)):
        # print "\n ------- Exp No. {} -------".format(exp_num)
        # Set a seed value
        np.random.seed(seed_value)
        tf.set_random_seed(seed_value)
        decoding_error = get_decoding_error(exp_num, data, permute)
        avg_decode.append(decoding_error)

    dec_mean, dec_std = np.mean(np.array(avg_decode)), np.std(np.array(avg_decode))
    print "Decoding Error: {} +- {}".format(dec_mean, dec_std)


if __name__ == "__main__":

    use_perm = True
    num_exps = 10
    compute_avg_decoding_error(range_=num_exps)

    # data = get_data()
    # if use_perm:
    #     metrics_permute = []
    #     permutations = itertools.permutations(list(range(len(env.latent_state_dict))))
    #     for i, perm in enumerate(permutations):
    #         print "\n{}. perm".format(i), list(perm)
    #         metrics_permute = eval_across_runs(range_=num_exps, data=data, permute=list(perm), metrics_permute=metrics_permute)
    #     # Choose the perm with minimum weighted KL divergence
    #     best_metric = min(metrics_permute, key=lambda(metric): metric[1][2][0])
    #     print "\nBest Permutation {}'s Results:-".format(best_metric[0])
    #     print "KL: {} +- {}".format(best_metric[1][0][0], best_metric[1][0][1])
    #     print "0-1 Loss: {} +- {}".format(best_metric[1][1][0], best_metric[1][1][1])
    #     print "Weighted-KL: {} +- {}".format(best_metric[1][2][0], best_metric[1][2][1])
    #     print "Weighted 0-1 Loss: {} +- {}".format(best_metric[1][3][0], best_metric[1][3][1])
    # else:
    #     _ = eval_across_runs(range_=num_exps, data=data)
