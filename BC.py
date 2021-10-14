import sys
import numpy as np
import tensorflow as tf
import json
import pickle as pkl
import keras.backend as K
from tqdm import tqdm
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.advanced_activations import LeakyReLU
from domains import RoadWorld, BoxWorld, TeamBoxWorld
from config import state_action_path, latent_path, param_dir, fig_path
from config import env_roadWorld_config, env_boxWorld_config, env_teamBoxWorld_config
from config import BC_config, env_name, seed_values, training_data_config
from utils import parse_data, plot_BC_loss


def calc_loss(act_gt, act_pred):
    xe_loss = np.average(-np.sum(np.multiply(act_gt, np.log(act_pred+10e-9)), axis=1))
    return np.array([xe_loss])


class Generator(object):
    def __init__(self, sess, state_dim, action_dim):
        self.sess = sess
        self.lr = tf.placeholder(tf.float32, shape=[])

        K.set_session(sess)

        self.model, self.weights, self.state = self.create_generator(state_dim, action_dim)

        self.action_gradient = tf.placeholder(tf.float32, [None, action_dim])
        self.params_grad = tf.gradients(self.model.output, self.weights, self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)

        self.sess.run(tf.global_variables_initializer())

    def train(self, state, action_grads, lr):
        self.sess.run(self.optimize, feed_dict={
            self.state: state,
            self.lr: lr,
            self.action_gradient: action_grads,
            K.learning_phase(): 1
        })

    @staticmethod
    def create_generator(state_dim, action_dim):
        state = Input(shape=[state_dim])
        x = Dense(128)(state)
        x = LeakyReLU()(x)
        x = Dense(128)(x)
        x = LeakyReLU()(x)

        # encodes = Input(shape=[encode_dim])
        # c = Dense(128)(encodes)
        # h = merge([x, c], mode='sum')
        # h = LeakyReLU()(h)

        actions = Dense(action_dim, activation='softmax')(x)
        model = Model(input=[state], output=actions)
        return model, model.trainable_weights, state


def run(exp_num, env, env_config):

    print "------- Exp No. {} -------".format(exp_num)

    state_dim = env_config.state_dim
    action_dim = env_config.action_dim

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    # print "Now we build generator"
    generator = Generator(sess, state_dim, action_dim)
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

    states_d, actions_d = demo["states"], demo["actions"]
    num_data = states_d.shape[0]
    idx = np.arange(num_data)
    np.random.shuffle(idx)
    print "Number of samples: ", num_data

    states_d_train = states_d[idx][:int(num_data * BC_config.train_val_ratio)]
    actions_train = actions_d[idx][:int(num_data * BC_config.train_val_ratio)]
    states_d_val = states_d[idx][int(num_data * BC_config.train_val_ratio):]
    actions_val = actions_d[idx][int(num_data * BC_config.train_val_ratio):]

    # print "Getting feature for training set ..."
    # feats_train = get_feat(imgs_train, feat_extractor)
    # print "Getting feature for validation set ..."
    # feats_val = get_feat(imgs_val, feat_extractor)

    cur_min_loss_val = np.inf

    lr = BC_config.initial_lr
    plot_train_loss = []
    plot_val_loss = []
    with tqdm(total=BC_config.n_epochs, position=0, leave=True) as pbar:
        for i in range(BC_config.n_epochs):

            total_step = states_d_train.shape[0] // BC_config.batch_size

            train_loss = np.array([0.])

            # Model Training
            for j in xrange(total_step):
                states = states_d_train[j * BC_config.batch_size: (j + 1) * BC_config.batch_size]

                act_pred = generator.model.predict([states])
                act_gt = actions_train[j * BC_config.batch_size: (j + 1) * BC_config.batch_size]

                generator.train(states, act_pred - act_gt, lr)
                batch_loss = calc_loss(act_gt, act_pred)

                train_loss += batch_loss / total_step

            if i % 20 == 0 and i > 0:
                lr *= BC_config.lr_decay_factor

            act_pred = generator.model.predict([states_d_val])
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
            pbar.set_description("Epoch {}".format(i+1))
            pbar.set_postfix(train_loss=np.round(train_loss[0], 6), val_loss=np.round(val_loss[0], 6), lr=lr)
            pbar.update(1)

    plot_BC_loss(BC_config.n_epochs, plot_train_loss, plot_val_loss, fig_path=fig_path + "_{}.png".format(exp_num))


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
