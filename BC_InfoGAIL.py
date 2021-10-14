import sys
import numpy as np
import tensorflow as tf
import json
import pickle as pkl
from tqdm import tqdm
from models import Generator
from domains import RoadWorld, BoxWorld, TeamBoxWorld
from config import env_roadWorld_config, env_boxWorld_config, env_teamBoxWorld_config, training_data_config
from config import state_action_path, latent_path, param_dir, fig_path
from config import BC_InfoGAIL_config, env_name, seed_values, supervision_setting
from utils import parse_data, plot_BC_loss

print "Model running in Env:", env_name


def calc_loss(act_gt, act_pred):
    xe_loss = np.average(-np.sum(np.multiply(act_gt, np.log(act_pred+10e-9)), axis=1))
    # steer_loss = np.average(np.abs(act_gt[:, 0] - act_pred[:, 0]))
    # accel_loss = np.average(np.abs(act_gt[:, 1] - act_pred[:, 1]))
    # brake_loss = np.average(np.abs(act_gt[:, 2] - act_pred[:, 2]))
    return np.array([xe_loss])


def run(exp_num, env, env_config):

    print "------- Exp No. {} -------".format(exp_num)

    state_dim = env_config.state_dim
    encode_dim = env_config.encode_dim
    action_dim = env_config.action_dim

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    # print "Now we build generator"
    generator = Generator(sess, state_dim, encode_dim, action_dim)
    # print generator.model.summary()
    # print "Now we build feature extractor"
    # base_model = ResNet50(weights='imagenet', include_top=False)
    # feat_extractor = Model(
    #     input=base_model.input,
    #     output=base_model.get_layer('activation_40').output
    # )

    print "Loading data ..."
    with open(state_action_path, 'rb') as f:
        trajectories = pkl.load(f)
    with open(latent_path, 'rb') as f:
        latent_modes_traj = pkl.load(f)

    # Create demo for InfoGAIL
    demo = parse_data(env, trajectories, latent_modes_traj, lower_bound=0, upper_bound=training_data_config.num_traj)

    states_d, actions_d, encodes_d = demo["states"], demo["actions"], demo['latent_states']
    num_data = states_d.shape[0]
    idx = np.arange(num_data)
    np.random.shuffle(idx)
    print "Number of samples: ", num_data

    states_d_train = states_d[idx][:int(num_data * BC_InfoGAIL_config.train_val_ratio)]
    actions_train = actions_d[idx][:int(num_data * BC_InfoGAIL_config.train_val_ratio)]
    encodes_d_train = encodes_d[idx][:int(num_data * BC_InfoGAIL_config.train_val_ratio)]
    states_d_val = states_d[idx][int(num_data * BC_InfoGAIL_config.train_val_ratio):]
    actions_val = actions_d[idx][int(num_data * BC_InfoGAIL_config.train_val_ratio):]

    # print "Getting feature for training set ..."
    # feats_train = get_feat(imgs_train, feat_extractor)
    # print "Getting feature for validation set ..."
    # feats_val = get_feat(imgs_val, feat_extractor)

    cur_min_loss_val = np.inf

    lr = BC_InfoGAIL_config.initial_lr
    plot_train_loss = []
    plot_val_loss = []
    with tqdm(total=BC_InfoGAIL_config.n_epochs, position=0, leave=True) as pbar:
        for i in xrange(BC_InfoGAIL_config.n_epochs):
            total_step = states_d_train.shape[0] // BC_InfoGAIL_config.batch_size

            train_loss = np.array([0.])

            # Model Training
            for j in xrange(total_step):
                states = states_d_train[j * BC_InfoGAIL_config.batch_size: (j + 1) * BC_InfoGAIL_config.batch_size]

                if supervision_setting == "unsupervised":
                    encodes_cur = np.zeros([BC_InfoGAIL_config.batch_size, encode_dim], dtype=np.float32)
                    idx = np.random.randint(0, encode_dim, BC_InfoGAIL_config.batch_size)
                    encodes_cur[np.arange(BC_InfoGAIL_config.batch_size), idx] = 1  # Prior encoded latent vector (using uniform dis.)
                elif supervision_setting == "supervised":
                    encodes_cur = encodes_d_train[j * BC_InfoGAIL_config.batch_size: (j + 1) * BC_InfoGAIL_config.batch_size]

                act_pred = generator.model.predict([states, encodes_cur])
                act_gt = actions_train[j * BC_InfoGAIL_config.batch_size: (j + 1) * BC_InfoGAIL_config.batch_size]

                generator.train(states, encodes_cur, act_pred - act_gt, lr)
                batch_loss = calc_loss(act_gt, act_pred)
                # print "Episode:", i, "Batch:", j, "/", total_step, \
                #         np.round(batch_loss, 6), np.sum(batch_loss)

                train_loss += batch_loss / total_step

            if i % 20 == 0 and i > 0:
                lr *= BC_InfoGAIL_config.lr_decay_factor

            # Model Validation
            if supervision_setting == "unsupervised":
                encodes_val = np.zeros([states_d_val.shape[0], encode_dim], dtype=np.float32)
                idx = np.random.randint(0, encode_dim, states_d_val.shape[0])
                encodes_val[idx] = 1
            else:
                encodes_val = encodes_d[idx][int(num_data * BC_InfoGAIL_config.train_val_ratio):]

            act_pred = generator.model.predict([states_d_val, encodes_val])
            val_loss = calc_loss(actions_val, act_pred)
            plot_train_loss.append(np.sum(train_loss))
            plot_val_loss.append(np.sum(val_loss))

            # print "Episode:", i, \
            #     "Train Loss: ", np.round(train_loss, 6), np.sum(train_loss), \
            #     "Val Loss:", np.round(val_loss, 6), np.sum(val_loss), cur_min_loss_val, \
            #     "LR:", lr

            if cur_min_loss_val > np.sum(val_loss):
                cur_min_loss_val = np.sum(val_loss)
                # print("Now we save the model")
                generator.model.save_weights(param_dir + "generator_bc_model_{}.h5".format(exp_num), overwrite=True)
                with open(param_dir + "generator_bc_model_{}.json".format(exp_num), "w") as outfile:
                    json.dump(generator.model.to_json(), outfile)

            pbar.refresh()
            pbar.set_description("Epoch {}".format(i + 1))
            pbar.set_postfix(train_loss=np.sum(train_loss), val_loss=np.sum(val_loss), lr=lr)
            pbar.update(1)

    plot_BC_loss(BC_InfoGAIL_config.n_epochs, plot_train_loss, plot_val_loss, fig_path=fig_path + "_{}.png".format(exp_num))


def run_multiple_runs():
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
    grid_length_y = env_config.grid_length_y
    grid_length_x = env_config.grid_length_x
    if env_name == "BoxWorld":
        env = BoxWorld(grid_length_x=grid_length_x, grid_length_y=grid_length_y)
    elif env_name == "RoadWorld":
        env = RoadWorld(grid_length_x=grid_length_x, grid_length_y=grid_length_y)
    elif env_name == "TeamBoxWorld":
        env = TeamBoxWorld(grid_length_x=grid_length_x, grid_length_y=grid_length_y)
    else:
        print "Specify a valid environment name in the config file"
        sys.exit(-1)

    # For averaging across different runs
    for exp_num, seed_value in enumerate(seed_values):
        # Set a seed value
        np.random.seed(seed_value)
        tf.set_random_seed(seed_value)

        # Run multiple experiments
        run(exp_num, env, env_config)


if __name__ == "__main__":
    run_multiple_runs()
