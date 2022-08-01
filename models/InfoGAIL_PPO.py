import os
import random
import time
import numpy as np
import pickle as pkl
import tensorflow as tf
from tqdm import tqdm
from typing import Dict, List, Tuple
from domains.stackBoxWorld import stackBoxWorld
from utils.plot import plot_metric
from utils.ppo import discount
from utils.misc import causally_parse_dynamic_data_v2, yield_batched_indexes
from keras.layers import Dense, Flatten, Add, Concatenate, LeakyReLU
from tensorflow_probability.python.distributions import Categorical

global file_txt_results_path
file_txt_results_path = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/InfoGAIL_results.txt'


class Conditional_Prior(tf.keras.Model):
    def __init__(self, z_dim):
        super(Conditional_Prior, self).__init__()
        self.locs_out = Dense(units=z_dim, activation=tf.nn.relu)
        self.std_out = Dense(units=z_dim, activation=tf.nn.softplus)

    def call(self, curr_encode_y):
        locs = self.locs_out(curr_encode_y)
        scale = self.std_out(curr_encode_y)
        scale = tf.clip_by_value(scale, clip_value_min=1e-3, clip_value_max=1e3)
        return locs, scale


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
        self.a_out = Dense(units=a_dim, activation=None, kernel_initializer=tf.keras.initializers.Orthogonal(gain=0.01))

    # @tf.function
    def call(self, curr_state, curr_encode_z, action=None):
        s = self.flatten(curr_state)
        s = self.fc1(s)
        s = self.fc2(s)
        s = self.fc3(s)

        c = self.fc4(curr_encode_z)
        h = self.add([s, c])
        # h = self.leaky_relu(h)
        action_logits = self.a_out(h)
        dist = Categorical(logits=action_logits)
        if action is None:
            action = dist.sample()
        actions_log_probs = dist.log_prob(action)
        return action, actions_log_probs, dist.entropy()


class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        relu_gain = tf.math.sqrt(2.0)
        relu_init = tf.initializers.Orthogonal(gain=relu_gain)

        self.flatten = Flatten()
        self.fc1 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=relu_init)
        self.fc2 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=relu_init)
        self.fc3 = Dense(units=128, activation=tf.nn.relu, kernel_initializer=relu_init)
        self.fc4 = Dense(units=128, activation=None, kernel_initializer=relu_init)
        self.add = Add()
        self.leaky_relu = LeakyReLU()
        self.v_out = Dense(units=1, activation=None, kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0))

    # @tf.function
    def call(self, curr_state, curr_encode_y):
        s = self.flatten(curr_state)
        s = self.fc1(s)
        s = self.fc2(s)
        s = self.fc3(s)
        c = self.fc4(curr_encode_y)
        h = self.add([s, c])
        h = self.leaky_relu(h)
        v_s = tf.squeeze(self.v_out(h), axis=1)
        return v_s


class ssDiscriminator(tf.keras.Model):
    def __init__(self):
        super(ssDiscriminator, self).__init__()
        self.flatten = Flatten()
        self.concat = Concatenate()
        self.fc1 = Dense(units=256, activation=tf.nn.relu)
        self.fc2 = Dense(units=256, activation=tf.nn.relu)
        self.fc3 = Dense(units=128, activation=tf.nn.relu)
        self.d_out = Dense(units=1, activation=tf.nn.sigmoid)

    def call(self, curr_state, curr_action, curr_encode_y):
        curr_state = self.flatten(curr_state)
        h = self.concat([curr_state, curr_action, curr_encode_y])
        h = self.fc1(h)
        h = self.fc2(h)
        h = self.fc3(h)
        d_prob = self.d_out(h)
        return d_prob


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.flatten = Flatten()
        self.concat = Concatenate()
        self.fc1 = Dense(units=256, activation=tf.nn.relu)
        self.fc2 = Dense(units=256, activation=tf.nn.relu)
        self.fc3 = Dense(units=128, activation=tf.nn.relu)
        self.d_out = Dense(units=1, activation=tf.nn.sigmoid)

    def call(self, curr_state, curr_action):
        curr_state = self.flatten(curr_state)
        h = self.concat([curr_state, curr_action])
        h = self.fc1(h)
        h = self.fc2(h)
        h = self.fc3(h)
        d_prob = self.d_out(h)
        return d_prob


class PostEncoder(tf.keras.Model):
    def __init__(self, z_dim):
        super(PostEncoder, self).__init__()
        self.flatten = Flatten()
        self.fc1 = Dense(units=256, activation=tf.nn.relu)
        self.fc2 = Dense(units=256, activation=tf.nn.relu)
        self.fc3 = Dense(units=128, activation=tf.nn.relu)

        self.fc4 = Dense(units=128, activation=tf.nn.relu)

        self.add = Add()

        self.locs_out = Dense(units=z_dim, activation=tf.nn.relu)
        self.std_out = Dense(units=z_dim, activation=tf.nn.softplus)

    def call(self, stack_states, prev_encode_y):
        s = self.flatten(stack_states)
        s = self.fc1(s)
        s = self.fc2(s)
        s = self.fc3(s)
        c = self.fc4(prev_encode_y)
        h = self.add([s, c])
        locs = self.locs_out(h)
        scale = self.std_out(h)
        scale = tf.clip_by_value(scale, clip_value_min=1e-3, clip_value_max=1e3)
        return locs, scale


class PostClassifier(tf.keras.Model):
    def __init__(self, y_dim):
        super(PostClassifier, self).__init__()
        self.out_prob_y = Dense(units=y_dim, activation=tf.nn.softmax)

    def call(self, encodes_z):
        prob_y = self.out_prob_y(encodes_z)
        return prob_y


class Posterior(tf.keras.Model):
    def __init__(self, z_dim, y_dim, k_samples):
        super(Posterior, self).__init__()
        self.encoder = PostEncoder(z_dim)
        self.classifier = PostClassifier(y_dim)

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.k = k_samples
        self.one_k = tf.constant(1/k_samples, dtype=tf.float32)

    # @tf.function
    def multi_sample_normal(self, locs, scales):
        samples = []
        for _ in range(self.k):
            epsilon = tf.random.normal(tf.shape(scales), mean=0.0, stddev=1.0, )
            z = locs + tf.math.multiply(scales, epsilon)
            samples.append(z)
        return samples

    def call(self, stack_states, prev_encode_y):
        [post_locs, post_scales] = self.encoder(stack_states, prev_encode_y)
        qy_z_k = [(lambda x: self.classifier(x))(_z) for _z in self.multi_sample_normal(post_locs, post_scales)]
        qy_z_k = tf.concat(values=[tf.expand_dims(qy_z, axis=0) for qy_z in qy_z_k], axis=0)
        qy_x = self.one_k*tf.reduce_sum(qy_z_k, axis=0)
        # qy_x = tf.reshape(qy_x, shape=[tf.shape(prev_encode_y)[0], ])
        dist = Categorical(probs=qy_x, dtype=tf.float32)
        next_encode_y = dist.sample().numpy()
        return tf.one_hot(next_encode_y, self.y_dim), qy_x


class Agent(object):
    def __init__(self, a_dim: int, y_dim: int, z_dim: int, env: stackBoxWorld, train_config: Dict):
        self.a_dim: int = a_dim
        self.y_dim: int = y_dim
        self.z_dim: int = z_dim
        self.self_supervision = train_config['self_supervision']
        self.config: Dict = train_config

        self.disc_coeff = train_config['d_coeff']
        self.post_coeff = train_config['p_coeff']
        self.raw_reward_coeff = train_config['raw_coeff']

        self.ent_coeff = train_config['ent_coeff']
        self.vf_coeff = train_config['vf_coeff']

        # Declare Environment
        self.env: stackBoxWorld = env

        # Declare Networks
        self.actor = Actor(a_dim)
        self.critic = Critic()
        if self.self_supervision:
            self.discriminator = ssDiscriminator()
            raise NotImplementedError
        else:
            self.discriminator = Discriminator()
        self.cond_prior = Conditional_Prior(z_dim)
        self.posterior = Posterior(z_dim, y_dim, k_samples=train_config['post_samples'])  # k determined empirically

        # Define Optimisers
        self.d_opt = tf.keras.optimizers.Adam(train_config['d_lr'])

        self.a_lr = tf.Variable(train_config['a_lr'])
        self.a_opt = tf.keras.optimizers.Adam(self.a_lr)

        self.c_lr = tf.Variable(train_config['c_lr'])
        self.c_opt = tf.keras.optimizers.Adam(self.c_lr)

        # Define Losses
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.mse = tf.keras.losses.MeanSquaredError()

        self.debug = 0

    def set_learning_rate(self, actor_learning_rate=None, critic_learning_rate=None):
        """Update learning rate."""
        if actor_learning_rate:
            self.a_lr.assign(actor_learning_rate)
        if critic_learning_rate:
            self.c_lr.assign(critic_learning_rate)

    def load_preTrainedModels(self, use_pretrained_actor, param_dir, model_id):
        # BUILD First
        _ = self.posterior.encoder(np.ones([1, self.config['w_size'] * 5, 5, 6]), np.ones([1, self.y_dim]))
        _ = self.posterior.classifier(np.ones([1, self.z_dim]))
        if use_pretrained_actor:
            _ = self.cond_prior(np.ones([1, self.y_dim]))
            _ = self.actor(np.ones([1, 5, 5, 6]), np.ones([1, self.z_dim]))

        # Load Models
        self.posterior.encoder.load_weights(os.path.join(param_dir, "encoder_model_{}.h5".format(model_id)))
        self.posterior.classifier.load_weights(os.path.join(param_dir, "classifier_{}.h5".format(model_id)))
        if use_pretrained_actor:
            self.cond_prior.load_weights(os.path.join(param_dir, "cond_prior_{}.h5".format(model_id)))
            self.actor.load_weights(os.path.join(param_dir, "decoder_model_{}.h5".format(model_id)))

        # # [DEBUG] Test pre-trained actor [Assume z is y].
        # # Careful the decoder predicts prob. of actions while actor predicts logits [Should not be an issue]
        # curr_state, curr_encode_y = tf.numpy_function(func=self.env_reset, inp=[], Tout=(tf.float32, tf.float32))
        # curr_state = tf.expand_dims(curr_state, axis=0)
        # curr_encode_y = tf.expand_dims(curr_encode_y, axis=0)
        # op = self.actor(curr_state, curr_encode_y)
        # print("Working Actor")

    def act(self, state, encode):
        action, action_log_prob, entropy = self.actor(state, encode)
        value = self.critic(state, encode)
        return action, action_log_prob, entropy, value

    def env_reset(self):
        obs, _, _, _ = self.env.reset()
        return obs['grid_world'].astype(np.float32), obs['latent_mode'].astype(np.float32)

    def env_step(self, action: np.ndarray):
        next_obs, raw, done, info = self.env.step(action)
        obj_achieved = np.array(1.0) if info['is_success'] else np.array(0.)
        done = np.array(1.0) if done else np.array(0.)
        return next_obs['grid_world'].astype(np.float32), done.astype(np.float32), obj_achieved.astype(np.float32)

    @staticmethod
    def wrap_data(trajectories: List[Dict]) -> Tuple[Dict, float]:
        avg_traj_len = 0
        for traj in trajectories:
            avg_traj_len += traj['states'].shape[0]
        keys = trajectories[0].keys()
        gen_data = {}
        for key in keys:
            gen_data[key] = tf.concat([traj[key] for traj in trajectories], axis=0)

        return gen_data, avg_traj_len/len(trajectories)

    # # Discriminator Loss
    # def discriminator_loss(self, real_output, fake_output):
    #     real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)  # Prob=1: expert data is from expert
    #     fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)  # Prob=0: sampled data is from expert
    #     total_loss = real_loss + fake_loss
    #     return total_loss

    # @tf.function
    def disc_train_step(self, sampled_data, expert_data):
        with tf.GradientTape() as disc_tape:
            # Prob that expert data is from expert
            real_output = tf.clip_by_value(self.discriminator(*expert_data, training=True), 0.01, 1.0)
            # Probability that sampled data is from expert
            fake_output = tf.clip_by_value(self.discriminator(*sampled_data, training=True), 0.01, 1.0)
            # Compute Loss
            real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)  # Prob=1: expert data from expert
            fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)  # Prob=0: sampled data from expert
            d_loss = real_loss + fake_loss
            # d_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return d_loss

    # @tf.function
    def actor_train_step(self, data: Dict):
        with tf.GradientTape() as actor_tape:
            # #################################### Actor Loss ####################################
            actions, actions_log_prob, dist_entropy = self.actor(data['states'], data['encodes'], data['actions'])
            ratio = tf.exp(actions_log_prob - data['old_action_logprob'])
            ppo_term1 = tf.math.multiply(ratio, data['advants'])
            ppo_term2 = tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.config['clip_param'],
                                                          1.0 + self.config['clip_param']), data['advants'])
            a_loss = tf.reduce_mean(tf.math.minimum(ppo_term1, ppo_term2))

            # ################## Entropy Loss (being maximised will encourage exploration) ##################
            entropy_loss = tf.reduce_mean(dist_entropy)

            total_aloss = tf.math.negative(a_loss + self.ent_coeff * entropy_loss)

        actor_gradients = actor_tape.gradient(total_aloss, self.actor.trainable_variables)
        # To perform global norm based gradient clipping
        # gradients, _ = tf.clip_by_global_norm(actor_gradients, self.train_config['grad_clip_norm'])
        self.a_opt.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        return a_loss, entropy_loss

    # @tf.function
    def critic_train_step(self, data: Dict, clip_v=False):
        with tf.GradientTape() as critic_tape:

            # #################################### Critic Loss ####################################
            values = self.critic(data['states'], data['encodes'])
            # Need to multiply the mse with vf_coeff if actor and critic share param.
            # for balance else no need since critic loss will be minimised separately
            # c_loss = self.vf_coeff*self.mse(data['returns'], values)

            c_loss = self.mse(data['returns'], values)
            # # For clipped Critic loss
            if clip_v:
                value_pred_clipped = data['values'] + tf.clip_by_value(values - data['values'],
                                                                       - self.config['clip_param'],
                                                                       self.config['clip_param'])
                c_loss_clipped = self.mse(data['returns'], value_pred_clipped)
                c_loss = tf.maximum(c_loss, c_loss_clipped)

        critic_gradients = critic_tape.gradient(c_loss, self.critic.trainable_variables)
        # To perform global norm based gradient clipping
        gradients, _ = tf.clip_by_global_norm(critic_gradients, self.config['grad_clip_norm'])
        self.c_opt.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        return c_loss

    def process_episodes(self, trajectories: List[Dict], _iter, log_dir) -> List[Dict]:

        for path_idx, path in enumerate(trajectories):
            file_path = os.path.join(log_dir, "iter_%d_path_%d.txt" % (_iter, path_idx))
            f = open(file_path, "a")

            #  #################################################################################################### #
            #  ######################################## Compute Reward ############################################ #
            #  #################################################################################################### #

            # Disc should predict whether the sampled data belongs to expert, which if high will correspond to high rew
            if self.self_supervision:  # D(s,a,c)
                d_pred = self.discriminator(path['states'], path['one_hot_actions'], path['encodes'])
            else:  # D(s, a)
                d_pred = self.discriminator(path['states'], path['one_hot_actions'])
            r_d = self.disc_coeff * tf.math.log(tf.clip_by_value(d_pred, 1e-10, 1))
            # Post should predict how probable the underlying latent mode is for given state [IMPLICATION???]
            p_pred = tf.reduce_sum(tf.math.multiply(path['encodes'], path['encodes_prob']), axis=-1)
            r_p = self.post_coeff * tf.math.log(tf.clip_by_value(p_pred, 1e-10, 1))
            rewards = tf.reshape(r_d, shape=[-1, ]) + r_p
            rewards = tf.cast(rewards, dtype=tf.float32)
            path['rewards'] = rewards

            #  #################################################################################################### #
            #  ####################################### Compute Advantage ########################################## #
            #  #################################################################################################### #
            returns = discount(rewards, self.config['gamma'])
            path['returns'] = tf.cast(returns, dtype=tf.float32)

            deltas = rewards + self.config['gamma'] * path["values"][1:] - path["values"][:-1]
            advants = discount(deltas, self.config['gamma'] * self.config['lambda'])
            path['advants'] = tf.cast(advants, dtype=tf.float32)
            # Normalise advantage
            path['advants'] = (path['advants'] - tf.math.reduce_mean(path['advants'], axis=-1)) / (
                    tf.math.reduce_std(path['advants']) + 1e-8)

            # Remove the terminal state value added to the list
            path['values'] = path['values'][:-1]

            # Book-keeping
            f.write(
                "\n#  ######################################################################################## #\n")
            f.write("Actions:\n" + str([self.env.steps_ref[a] for a in list(path['actions'].numpy())]) + "\n")
            f.write(
                "\n#  ######################################################################################## #\n")
            f.write("Latent Modes:\n" + np.array_str(path['encodes'].numpy()) + "\n")
            f.write(
                "\n#  ######################################################################################## #\n")
            f.write("Rewards ({}*log(D) + {}*logP):\n".format(self.config['d_coeff'], self.config['p_coeff']) + np.array_str(path['rewards'].numpy()) + "\n")
            f.write("Returns:\n" + np.array_str(path['returns'].numpy()) + "\n")
            f.write("Values:\n" + np.array_str(path['values'].numpy()) + "\n")
            f.write("Advants:\n" + np.array_str(path['advants'].numpy()) + "\n")
            f.close()

        return trajectories

    # @tf.function
    def collect_episodesv1(self, _iter, num_episodes, log_dir) -> List[Dict]:
        trajectories: List[Dict] = []
        t1, t2 = 0.0, 0.0
        for path_idx in range(num_episodes):
            file_path = os.path.join(log_dir, "iter_%d_path_%d.txt" % (_iter, path_idx))
            f = open(file_path, "w")

            states, encodes, actions, old_actions_logprob, one_hot_actions, encodes_prob, values = [], [], [], [], [], \
                                                                                                   [], []

            # Observe the initial states
            curr_state, curr_encode_y = tf.numpy_function(func=self.env_reset, inp=[],
                                                          Tout=(tf.float32, tf.float32))

            f.write("Init_State:\n" + np.array_str(curr_state.numpy()) + "\n")
            f.write("Init_Encode:\n" + np.array_str(curr_encode_y.numpy()) + "\n")

            encode_prob = tf.one_hot(tf.argmax(curr_encode_y), self.y_dim)
            encode_prob = tf.expand_dims(encode_prob, axis=0)
            curr_state = tf.expand_dims(curr_state, axis=0)
            curr_encode_y = tf.expand_dims(curr_encode_y, axis=0)

            #  #################################################################################################### #
            #  ###################################### Unroll Trajectories ######################################### #
            #  #################################################################################################### #

            obj_achieved = tf.cast([0.], dtype=tf.float32)
            while not self.env.episode_ended:
                # Computer action and value of the state
                start = time.time()
                action, action_log_prob, entropy, value = self.act(curr_state, curr_encode_y)
                t1 += round(time.time()-start, 3)

                # Add to the stack
                one_hot_actions.append(tf.cast(tf.one_hot(action, self.a_dim), dtype=tf.float32))
                states.append(tf.cast(curr_state, dtype=tf.float32))
                encodes.append(tf.cast(curr_encode_y, dtype=tf.float32))
                actions.append(tf.cast(action, dtype=tf.float32))
                old_actions_logprob.append(tf.cast(action_log_prob, dtype=tf.float32))
                encodes_prob.append(tf.cast(encode_prob, dtype=tf.float32))
                values.append(tf.cast(value, dtype=tf.float32))

                # Take action
                start = time.time()
                next_state, done, obj_achieved = tf.numpy_function(func=self.env_step, inp=[action],
                                                                   Tout=(tf.float32,
                                                                         tf.float32,
                                                                         tf.float32))
                t2 += round(time.time() - start, 3)
                next_state = tf.expand_dims(next_state, axis=0)

                # Predict next encode
                next_encode, encode_prob = self.posterior(next_state, curr_encode_y)

                # Update the curr_state and curr_encode
                curr_state = next_state
                curr_encode_y = next_encode

            # Add the value of terminal state
            if obj_achieved:
                values.append(tf.cast([0.], dtype=tf.float32))
            else:
                values.append(values[-1])

            path = {
                'states': tf.concat(states, axis=0),
                'encodes': tf.concat(encodes, axis=0),
                'actions': tf.concat(actions, axis=0),
                'old_action_logprob': tf.concat(old_actions_logprob, axis=0),
                'encodes_prob': tf.concat(encodes_prob, axis=0),
                'values': tf.concat(values, axis=0),  # Leave the value of terminal state when storing
                'one_hot_actions': tf.concat(one_hot_actions, axis=0)
            }
            trajectories.append(path)
            f.close()

        print("[DEBUG TIME] Take action: ", t1)
        print("[DEBUG TIME] Take step: ", t2)
        return trajectories

    def train(self, expert_data: Dict, use_pretrained_actor: bool,
              param_dir: str, fig_path: str, log_dir: str, exp_num: int = 0):

        num_expert_trans = expert_data['states'].shape[0]
        expert_gen = yield_batched_indexes(start=0, b_size=self.config['batch_size'], n_samples=num_expert_trans)

        total_disc_loss, total_actor_loss, total_critic_loss, total_ent_loss = [], [], [], []
        with tqdm(total=self.config['num_epochs'] * self.config['num_cycles'], position=0, leave=True) as pbar:
            for epoch in range(self.config['num_epochs']):

                # # Shuffle Expert Data
                # idx_d = np.arange(num_expert_trans)
                # np.random.shuffle(idx_d)
                for cycle in range(self.config['num_cycles']):

                    iter_num = epoch * self.config['num_cycles'] + cycle

                    #  ############################################################################################### #
                    #  ########################################## Collect Data ####################################### #
                    #  ############################################################################################### #
                    start = time.time()
                    if iter_num <= 5:
                        num_episodes = 2 * self.config['num_episodes']  # 20ep x 50s = 1000 (20 iter w batch size of 50)
                    else:
                        num_episodes = self.config['num_episodes']  # 10ep x 50s = 500 (~10 iter w batch size of 50)

                    if use_pretrained_actor:
                        trajectories = self.collect_episodesv2(iter_num, num_episodes, log_dir)
                    else:
                        trajectories = self.collect_episodesv1(iter_num, num_episodes, log_dir)
                    datac_time = round(time.time()-start, 3)

                    #  ############################################################################################### #
                    #  ########################################## Process Data ####################################### #
                    #  ############################################################################################### #
                    start = time.time()
                    trajectories = self.process_episodes(trajectories, iter_num, log_dir)
                    sampled_data, avg_traj_len = self.wrap_data(trajectories)
                    datap_time = round(time.time() - start, 3)

                    num_gen_trans = sampled_data['states'].shape[0]

                    #  ############################################################################################### #
                    #  ###################################### Perform Optimisation ################################### #
                    #  ############################################################################################### #
                    start = time.time()
                    it_DLoss, it_ALoss, it_CLoss, it_ELoss = [], [], [], []

                    #  ####################################### Train Discriminator ################################### #
                    sampled_gen_D = yield_batched_indexes(start=0, b_size=self.config['batch_size'],
                                                          n_samples=num_gen_trans)
                    if iter_num <= 5:
                        num_diter = 120 - iter_num*20  # 120, 100, 80, 60, 40, 20
                    else:
                        num_diter = num_gen_trans // self.config['batch_size']  # ~ 500/50 = 10 iter
                    for i in range(num_diter):
                        e_idxs = expert_gen.__next__()
                        random.shuffle(e_idxs)
                        s_idxs = sampled_gen_D.__next__()
                        random.shuffle(s_idxs)
                        d_loss = self.disc_train_step(sampled_data=[tf.gather(sampled_data[key],
                                                                              tf.constant(s_idxs))
                                                                    for key in ['states', 'one_hot_actions']],
                                                      expert_data=[tf.gather(expert_data[key],
                                                                             tf.constant(e_idxs))
                                                                   for key in ['states', 'one_hot_actions']])

                        it_DLoss.append(d_loss)

                    #  ########################################### Train Critic ###################################### #
                    sampled_gen_C = yield_batched_indexes(start=0, b_size=self.config['batch_size'],
                                                          n_samples=num_gen_trans)
                    num_citer = num_gen_trans // self.config['batch_size']
                    for j in range(num_citer):
                        s_idxs = sampled_gen_C.__next__()
                        random.shuffle(s_idxs)
                        c_loss = self.critic_train_step({key: tf.gather(sampled_data[key], tf.constant(s_idxs))
                                                         for key in sampled_data.keys()}, clip_v=True)
                        it_CLoss.append(c_loss)

                    #  ########################################### Train Actor ####################################### #

                    a_loss, ent_loss = self.actor_train_step(sampled_data)
                    it_ALoss.append(a_loss)
                    it_ELoss.append(ent_loss)

                    opt_time = round(time.time() - start, 3)

                    total_disc_loss.extend(it_DLoss)
                    total_actor_loss.extend(it_ALoss)
                    total_critic_loss.extend(it_CLoss)
                    total_ent_loss.extend(it_ELoss)

                    pbar.refresh()
                    pbar.set_description("Cycle {}".format(iter_num))
                    pbar.set_postfix(LossD=np.average(it_DLoss), LossA=np.average(it_ALoss), LossC=np.average(it_CLoss),
                                     LossE=np.average(it_ELoss), TimeDataC='{}s'.format(datac_time),
                                     TimeDataP='{}s'.format(datap_time), TimeOpt='{}s'.format(opt_time),
                                     AvgEpLen=avg_traj_len,)
                    pbar.update(1)

        plot_metric(total_disc_loss, fig_path, exp_num, name='Disc')
        plot_metric(total_actor_loss, fig_path, exp_num, name='Actor')
        plot_metric(total_critic_loss, fig_path, exp_num, name='Critic')
        plot_metric(total_ent_loss, fig_path, exp_num, name='Entropy')

        # Save weights
        self.discriminator.save_weights(os.path.join(param_dir, "discriminator.h5"), overwrite=True)
        self.actor.save_weights(os.path.join(param_dir, "actor.h5"), overwrite=True)
        self.critic.save_weights(os.path.join(param_dir, "critic.h5"), overwrite=True)


def run(sup, exp_num):

    env_name = 'StackBoxWorld'
    use_pretrained_actor = True

    train_config = {
        'num_epochs': 25,
        'num_cycles': 50,
        'num_episodes': 10,
        'batch_size': 50,
        'w_size': 1,
        'num_traj': 100,
        'post_samples': 5,
        'perc_supervision': sup,
        'gamma': 0.99,
        'lambda': 0.95,
        'd_iter': 10,
        'ac_iter': 5,
        'd_lr': 5e-3,
        'a_lr': 4e-4,
        'c_lr': 3e-4,
        'd_coeff': 1.0,
        'p_coeff': 0.0,
        'raw_coeff': 0.01,
        'ent_coeff': 0.0,
        'vf_coeff': 0.5,
        'pretrained_post': 'CCVAE',
        'self_supervision': False,
        'clip_param': 0.2,  # 0.1 or 0.2
        'grad_clip_norm': 0.5
    }

    print("\n\n---------------------------------- Supervision {} : Exp {} ----------------------------------".format(
        sup, exp_num),
          file=open(file_txt_results_path, 'a'))

    env_config = {
        'a_dim':  4,
        'z_dim': 6,
        'y_dim': 6,
        'env': stackBoxWorld()
    }

    # # Specify Paths
    # Root Directory
    root_dir = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/'

    # Data Directory
    data_dir = os.path.join(root_dir, 'training_data/{}'.format(env_name))
    train_data_path = os.path.join(data_dir, 'stateaction_fixed_latent_dynamic.pkl')
    print("Loading data from - ",  train_data_path, file=open(file_txt_results_path, 'a'))
    with open(train_data_path, 'rb') as f:
        traj_sac = pkl.load(f)

    # Parse Data
    demos_expert = causally_parse_dynamic_data_v2(traj_sac, lower_bound=0, upper_bound=train_config['num_traj'],
                                                  window_size=train_config["w_size"])
    expert_data = {
        'states': tf.cast(demos_expert['curr_states'], dtype=tf.float32),
        'one_hot_actions': tf.cast(demos_expert['next_actions'], dtype=tf.float32),
    }

    # Model Directory
    model_dir = os.path.join(root_dir, 'pretrained_params/Project', env_name, 'DirectedInfoGAIL_{}'.format(sup))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    param_dir = os.path.join(model_dir, "params{}".format(exp_num))
    log_dir = os.path.join(model_dir, "logs{}".format(exp_num))
    fig_path = os.path.join(model_dir, '{}_loss'.format('DirectedInfoGAIL'))

    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Initiate the agent
    agent = Agent(env_config['a_dim'], env_config['y_dim'], env_config['z_dim'], env_config['env'],
                  train_config)

    # CCVAE-Load Models
    pretrained_model_dir = os.path.join(root_dir, 'pretrained_params/Project', env_name,
                                        '{}_{}'.format(train_config['pretrained_model'], sup))
    pretrained_param_dir = os.path.join(pretrained_model_dir, "params{}".format(0))
    agent.load_preTrainedModels(use_pretrained_actor, pretrained_param_dir, model_id='best')

    agent.train(expert_data, use_pretrained_actor, param_dir, fig_path, log_dir)
    print("Finish.", file=open(file_txt_results_path, 'a'))
    # else:
    #     print("Skipping Training", file=open(file_txt_results_path, 'a'))


if __name__ == "__main__":
    run(sup=1., exp_num=0)






