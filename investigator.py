import numpy as np
import sys
import os
import re
import pickle as pkl
import json
import tensorflow as tf
from BC import Generator
from config import env_name, env_boxWorld_config, env_teamBoxWorld_config, env_roadWorld_config, InfoGAIL_config, latent_setting, supervision_setting
from config import param_dir, kl_figpath
from domains import BoxWorld, RoadWorld, TeamBoxWorld
from eval import pred_kl_div, get_data
from utils import plot_infogail_results, one_hot_encode
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

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


def measure_KL_with_expert(data, exp_num=0):
    kl_models, w_kl_models = [], []
    model_dir = param_dir + "params{}/".format(exp_num)
    # models = os.listdir(model_dir)
    # gen_models = [m for m in models if re.match(re.compile('generator_model_' + "[0-9]*" + ".h5"), m)]
    for model_id in range(InfoGAIL_config.n_epochs):
        model_path = os.path.join(model_dir, 'generator_model_' + str(model_id) + ".h5")
        agent = data['agent']
        agent.generator.load_weights(model_path)
        act_pred = agent.generator.predict([data['states_d'], data['encodes_d']])
        # kl = pred_kl_div(data['actions_d'], act_pred)
        w_kl = pred_kl_div(data['actions_d'], act_pred, weights=data['weights'])
        # kl_models.append(kl[0])
        w_kl_models.append(w_kl[0])

    plot_infogail_results(w_kl_models, kl_figpath + "perf{}.png".format(exp_num))


def gen_policy_path(data, exp_num=0):
    start_state = np.array([[0, 3, 1, 1]])
    latent_code = np.array([[1, 0, 0, 0, 0, 0]])
    model_id = 1

    model_dir = param_dir + "params{}/".format(exp_num)
    model_path = os.path.join(model_dir, 'generator_model_' + str(model_id) + ".h5")

    state = start_state
    agent = data['agent']
    env = agent.env
    agent.generator.load_weights(model_path)

    (latent1, latent2) = env.inv_latent_dict[np.argmax(latent_code)]
    print start_state
    print "X1: ", latent1, " X2:", latent2
    env.set_latent1(latent1)
    env.set_latent1(latent2)
    _, terminal_states = env.get_initial_state()

    for i in range(InfoGAIL_config.max_step_limit):
        action = agent.act(state, latent_code)
        # Get action id
        action_id = np.where(action[0] == 1.0)[0][0]

        # Compute the next state using environment's transition dynamics
        next_state_prob_dist = env.obstate_transition_matrix[env.states_dict[tuple(state[0])], action_id, :]
        next_state_idx = np.random.choice(np.array(range(len(env.states_dict))), p=next_state_prob_dist)
        next_state = env.inv_states_dict[next_state_idx]
        next_state = np.expand_dims(np.array(next_state), axis=0)

        print "{}: {} - {} - {}".format(i, state, env.inv_action_dict[action_id], next_state)

        if next_state_idx in terminal_states:
            print "Terminal state reached"
            break
        state = next_state


def save_policy():
    actions_pred_exp = []
    for exp_num in range(10):
        model_path = param_dir + "generator_bc_model_{}.h5".format(exp_num)
        generator = Generator(sess, env_config.state_dim, env_config.action_dim)
        generator.model.load_weights(model_path)

        encode_dim = env_config.encode_dim
        states, actions_d, encodes, actions_pred = [], [], [], []
        expert_policy_matrix = env.compute_policy()
        for state in env.states_dict.keys():
            for encode in env.latent_state_dict.keys():
                states.append(state)
                encodes.append(env.latent_state_dict[encode])
                # The expert policy is non-deterministic, so we have the expert's prob distribution over actions
                actions_d.append(expert_policy_matrix[encode[0], encode[1], env.states_dict[state], :])
                act_pred = generator.model.predict(np.expand_dims(np.array(state), axis=0))
                actions_pred.append(act_pred[0])
        states = np.array(states)
        encodes = one_hot_encode(np.array(encodes), dim=encode_dim)
        actions_d = np.array(actions_d)

        actions_pred_exp.append(np.array(actions_pred))

    policy = {
        'states': states,
        'encodes': encodes,
        'actions_e': actions_d,
        'actions_p': np.array(actions_pred_exp)
    }
    with open("{}_{}_{}_policy.pkl".format(env_name, supervision_setting, latent_setting), 'wb') as f:
        pkl.dump(policy, f, protocol=2)


def main():
    data = get_data()
    measure_KL_with_expert(data, exp_num=2)
    # gen_policy_path(data)
    # save_policy()


if __name__ == "__main__":
    main()
