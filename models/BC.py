import os
import random
import time
import numpy as np
import pickle as pkl
import tensorflow as tf
from tqdm import tqdm
from typing import Dict
from domains.stackBoxWorld import stackBoxWorld
from evaluation.eval import evaluate_sup_model_discrete
from utils.plot import plot_metric
from utils.misc import causally_parse_dynamic_data_v2, yield_batched_indexes
from keras.layers import Dense, Flatten, Add
from tensorflow_probability.python.distributions import Categorical

global file_txt_results_path
file_txt_results_path = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/BC_sup.txt'


class Actor(tf.keras.Model):
    def __init__(self, a_dim):
        super(Actor, self).__init__()
        # Adding recommended Orthogonal initialization with scaling that varies from layer to layer
        relu_gain = tf.math.sqrt(2.0)
        relu_init = tf.initializers.Orthogonal(gain=relu_gain)

        self.flatten = Flatten()
        self.fc1 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=relu_init)
        self.fc2 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=relu_init)
        self.fc3 = Dense(units=128, activation=tf.nn.relu, kernel_initializer=relu_init)
        self.fc4 = Dense(units=128, activation=tf.nn.relu, kernel_initializer=relu_init)
        self.add = Add()
        # self.leaky_relu = LeakyReLU()
        self.a_out = Dense(units=a_dim, activation=tf.nn.softmax,
                           kernel_initializer=tf.keras.initializers.Orthogonal(gain=0.01))

    # @tf.function
    def call(self, curr_state, curr_encode_z):
        s = self.flatten(curr_state)
        s = self.fc1(s)
        s = self.fc2(s)
        s = self.fc3(s)

        c = self.fc4(curr_encode_z)
        h = self.add([s, c])
        # h = self.leaky_relu(h)
        actions_prob = self.a_out(h)
        return actions_prob


class Agent(object):
    def __init__(self, a_dim: int, y_dim: int, z_dim: int, env: stackBoxWorld, train_config: Dict):
        self.a_dim: int = a_dim
        self.y_dim: int = y_dim
        self.z_dim: int = z_dim
        self.config: Dict = train_config

        # Declare Environment
        self.env: stackBoxWorld = env

        # Declare Networks
        self.actor = Actor(a_dim)

        # Define Optimisers
        self.a_lr = tf.Variable(train_config['a_lr'])
        self.a_opt = tf.keras.optimizers.Adam(self.a_lr)

        # Define Losses
        self.cat_cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        self.debug = 0

    def set_learning_rate(self, actor_learning_rate=None, critic_learning_rate=None):
        """Update learning rate."""
        if actor_learning_rate:
            self.a_lr.assign(actor_learning_rate)

    def load_model(self, param_dir):
        # BUILD First
        _ = self.actor(np.ones([1, 5, 5, 6]), np.ones([1, self.z_dim]))

        # Load Models
        self.actor.load_weights(os.path.join(param_dir, "actor.h5"))

    def save_model(self, param_dir):
        # Save weights
        self.actor.save_weights(os.path.join(param_dir, "actor.h5"), overwrite=True)

    def act(self, state, encode_z, action=None):
        action_prob = self.actor(state, encode_z)
        dist = Categorical(probs=action_prob)
        if action is None:
            action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action, action_log_prob, dist.entropy()

    def env_reset(self):
        obs, _, _, _ = self.env.reset()
        return obs['grid_world'].astype(np.float32), obs['latent_mode'].astype(np.float32)

    def env_step(self, action: np.ndarray):
        next_obs, reward, done, info = self.env.step(action)
        # To only send specific signals: surrogate state-based reward
        raw = np.array(-100.) if 'stuck' in info.keys() else np.array(0.)
        obj_achieved = np.array(1.0) if info['is_success'] else np.array(0.)
        done = np.array(1.0) if done else np.array(0.)
        return next_obs['grid_world'].astype(np.float32), next_obs['latent_mode'].astype(np.float32), \
               raw.astype(np.float32), done.astype(np.float32), \
               obj_achieved.astype(np.float32)

    @tf.function
    def actor_step(self, expert_data, expert_actions):
        with tf.GradientTape() as disc_tape:

            pred_actions = tf.clip_by_value(self.actor(*expert_data), 0.01, 1.0)
            # Compute Loss
            pred_loss = self.cat_cross_entropy(expert_actions, pred_actions)  # Prob=1: expert data from expert

        gradients_of_discriminator = disc_tape.gradient(pred_loss, self.actor.trainable_variables)
        self.a_opt.apply_gradients(zip(gradients_of_discriminator, self.actor.trainable_variables))
        return pred_loss

    def train(self, expert_data: Dict, param_dir: str, fig_path: str, log_dir: str,
              exp_num: int = 0):

        num_expert_trans = expert_data['states'].shape[0]
        expert_gen = yield_batched_indexes(start=0, b_size=self.config['batch_size'], n_samples=num_expert_trans)

        total_actor_loss = []
        with tqdm(total=self.config['num_epochs'] * self.config['num_cycles'], position=0, leave=True) as pbar:
            for epoch in range(self.config['num_epochs']):

                # # Shuffle Expert Data
                # idx_d = np.arange(num_expert_trans)
                # np.random.shuffle(idx_d)
                for cycle in range(self.config['num_cycles']):

                    iter_num = epoch * self.config['num_cycles'] + cycle

                    #  ############################################################################################### #
                    #  ###################################### Perform Optimisation ################################### #
                    #  ############################################################################################### #
                    start = time.time()
                    it_ALoss = []

                    #  ####################################### Train Actor ################################### #
                    num_iter = num_expert_trans // self.config['batch_size']  # ~ 500/50 = 10 iter
                    for i in range(num_iter):

                        e_idxs = expert_gen.__next__()
                        random.shuffle(e_idxs)

                        a_loss = self.actor_step(expert_data=[tf.gather(expert_data[key], tf.constant(e_idxs))
                                                              for key in ['states', 'encodes']],
                                                 expert_actions=tf.gather(expert_data['one_hot_actions'],
                                                                          tf.constant(e_idxs)))
                        it_ALoss.append(a_loss)

                    opt_time = round(time.time() - start, 3)
                    total_actor_loss.extend(it_ALoss)

                    pbar.refresh()
                    pbar.set_description("Cycle {}".format(iter_num))
                    pbar.set_postfix(LossA=np.average(it_ALoss), TimeOpt='{}s'.format(opt_time))
                    pbar.update(1)

        plot_metric(total_actor_loss, fig_path, exp_num, name='ActorLoss')

        self.save_model(param_dir)


def run(env_name, exp_num=0, sup=0.1, random_transition=True):

    train_config = {'num_epochs': 10, 'num_cycles': 20, 'batch_size': 50, 'w_size': 1, 'num_traj': 100,
                    'a_lr': 4e-4, 'v_clip': 0.2, 'grad_clip_norm': 0.5}

    print("\n\n---------------------------------------------------------------------------------------------",
          file=open(file_txt_results_path, 'a'))
    print("---------------------------------- Supervision {} : Exp {} ----------------------------------".format(
        sup, exp_num),
          file=open(file_txt_results_path, 'a'))
    print("---------------------------------------------------------------------------------------------",
          file=open(file_txt_results_path, 'a'))
    print("Train Config: ", train_config, file=open(file_txt_results_path, 'a'))

    env_config = {
        'a_dim':  4,
        'z_dim': 6,
        'y_dim': 6,
        'env': stackBoxWorld(random_transition)
    }

    # ################################################ Specify Paths ################################################ #
    # Root Directory
    root_dir = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/'

    # Data Directory
    data_dir = os.path.join(root_dir, 'training_data/{}'.format(env_name))
    if random_transition:
        train_data_path = os.path.join(data_dir, 'stateaction_latent_dynamic.pkl')
        test_data_path = os.path.join(data_dir, 'stateaction_latent_dynamic_test.pkl')
        model_dir = os.path.join(root_dir, 'pretrained_params/Project', env_name, 'BC_{}'.format(sup))
    else:
        train_data_path = os.path.join(data_dir, 'stateaction_fixed_latent_dynamic.pkl')
        test_data_path = os.path.join(data_dir, 'stateaction_fixed_latent_dynamic_test.pkl')
        model_dir = os.path.join(root_dir, 'pretrained_params/Project', env_name, 'fixedC_BC_{}'.format(sup))

    # Model Directory
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    param_dir = os.path.join(model_dir, "params{}".format(exp_num))
    fig_path = os.path.join(model_dir, '{}_loss'.format('BC'))

    # Log Directory
    log_dir = os.path.join(model_dir, "logs{}".format(exp_num))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # ################################################ Load Data ################################################ #
    print("Loading data", file=open(file_txt_results_path, 'a'))
    with open(train_data_path, 'rb') as f:
        traj_sac = pkl.load(f)

    # ################################################ Parse Data ################################################ #
    demos_un = causally_parse_dynamic_data_v2(traj_sac, lower_bound=0,
                                              upper_bound=train_config["num_traj"],
                                              window_size=train_config["w_size"])

    expert_data = {
        'states': tf.cast(demos_un['curr_states'], dtype=tf.float32),
        'encodes': tf.cast(demos_un['curr_latent_states'], dtype=tf.float32),
        'one_hot_actions': tf.cast(demos_un['next_actions'], dtype=tf.float32),
    }

    # ################################################ Train Model ###########################################
    agent = Agent(env_config['a_dim'], env_config['y_dim'], env_config['z_dim'], env_config['env'],
                  train_config)

    if not os.path.exists(param_dir):
        os.makedirs(param_dir)
        try:
            agent.train(expert_data, param_dir, fig_path, log_dir)
        except KeyboardInterrupt:
            agent.save_model(param_dir)
        print("Finish.", file=open(file_txt_results_path, 'a'))
    else:
        print("Skipping Training", file=open(file_txt_results_path, 'a'))

    # ################################################ Test Model ################################################ #
    with open(test_data_path, 'rb') as f:
        test_traj_sac = pkl.load(f)

    agent = Agent(env_config['a_dim'], env_config['y_dim'], env_config['z_dim'], env_config['env'], train_config)
    agent.load_model(param_dir)
    print("\nTraining Data Results", file=open(file_txt_results_path, 'a'))
    evaluate_sup_model_discrete(agent, "GAIL", traj_sac, train_config, file_txt_results_path)
    print("\nTesting Data Results", file=open(file_txt_results_path, 'a'))
    evaluate_sup_model_discrete(agent, "GAIL", test_traj_sac, train_config, file_txt_results_path)


if __name__ == "__main__":
    supervision_settings = [1.0]
    for perc_supervision in supervision_settings:
        run(env_name='StackBoxWorld', sup=perc_supervision, random_transition=False)






