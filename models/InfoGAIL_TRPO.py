import os
import random
import sys
import time
import numpy as np
import pickle as pkl
import tensorflow as tf
from tqdm import tqdm
from typing import Dict, List, Tuple
from domains.stackBoxWorld import stackBoxWorld
from evaluation.eval import evaluate_model_discrete
from utils.plot import plot_metric
from utils.ppo import discount
from utils.trpo import linesearch, get_flat, conjugate_gradient, set_from_flat
from utils.misc import causally_parse_dynamic_data_v2, yield_batched_indexes
from utils.gail import ReplayBuffer
from keras.layers import Dense, Flatten, Add, Concatenate, LeakyReLU, Softmax, Lambda
from tensorflow_probability.python.distributions import Categorical, Normal

global file_txt_results_path
file_txt_results_path = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/temp_results.txt'


class Conditional_Prior(tf.keras.Model):
    def __init__(self, z_dim):
        super(Conditional_Prior, self).__init__()
        self.locs_out = Dense(units=z_dim, activation=tf.nn.relu)
        self.std_out = Dense(units=z_dim, activation=tf.nn.softplus)

    def call(self, curr_encode_y, k=None):
        if not k:
            k = 1
        locs = self.locs_out(curr_encode_y)
        scale = self.std_out(curr_encode_y)
        scale = tf.clip_by_value(scale, clip_value_min=1e-3, clip_value_max=1e3)
        prior_z_y = Normal(loc=locs, scale=scale)
        return locs, scale, prior_z_y.sample(sample_shape=k)


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
        self.a_out = Dense(units=a_dim, activation=tf.nn.softmax, kernel_initializer=tf.keras.initializers.Orthogonal(gain=0.01))

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
    def __init__(self, y_dim):
        super(ssDiscriminator, self).__init__()
        self.flatten = Flatten()
        self.concat = Concatenate()
        self.fc1 = Dense(units=256, activation=tf.nn.relu)
        self.fc2 = Dense(units=256, activation=tf.nn.relu)
        self.fc3 = Dense(units=128, activation=tf.nn.relu)
        self.d_out = Dense(units=y_dim, activation=None)

        self.sd_out = Softmax()
        self.usd_out = Lambda(lambda x: 1.0 - (1.0/(1.0 + tf.reduce_sum(tf.math.exp(x), axis=-1, keepdims=True))))

    def call(self, curr_state, curr_action, sup):
        curr_state = self.flatten(curr_state)
        h = self.concat([curr_state, curr_action])
        h = self.fc1(h)
        h = self.fc2(h)
        h = self.fc3(h)
        h = self.d_out(h)
        if sup:
            return self.sd_out(h)
        else:
            return self.usd_out(h)


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.flatten = Flatten()
        self.concat = Concatenate()
        self.fc1 = Dense(units=256, activation=tf.nn.relu)
        self.fc2 = Dense(units=256, activation=tf.nn.relu)
        self.fc3 = Dense(units=128, activation=tf.nn.relu)
        self.d_out = Dense(units=1, activation=tf.nn.sigmoid)

    def call(self, curr_state, curr_action, sup=False):
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
    def __init__(self, z_dim, y_dim):
        super(Posterior, self).__init__()
        self.encoder = PostEncoder(z_dim)
        self.classifier = PostClassifier(y_dim)

        self.y_dim = y_dim
        self.z_dim = z_dim

    # @tf.function
    def multi_sample_normal(self, locs, scales, k):
        samples = []
        for _ in tf.range(k):
            epsilon = tf.random.normal(tf.shape(scales), mean=0.0, stddev=1.0, )
            z = locs + tf.math.multiply(scales, epsilon)
            samples.append(z)
        return samples

    def call(self, stack_states, prev_encode_y, k_samples=1):

        [post_locs, post_scales] = self.encoder(stack_states, prev_encode_y)
        sampled_zs = [_z for _z in self.multi_sample_normal(post_locs, post_scales, k_samples)]
        qy_z_k = [(lambda x: self.classifier(x))(_z) for _z in sampled_zs]
        qy_z_k = tf.concat(values=[tf.expand_dims(qy_z, axis=0) for qy_z in qy_z_k], axis=0)
        qy_x = tf.reduce_mean(qy_z_k, axis=0)
        # qy_x = tf.reshape(qy_x, shape=[tf.shape(prev_encode_y)[0], ])
        dist = Categorical(probs=qy_x, dtype=tf.float32)
        next_encode_y = dist.sample().numpy()
        return tf.one_hot(next_encode_y, self.y_dim), qy_x, post_locs


class Agent(object):
    def __init__(self, a_dim: int, y_dim: int, z_dim: int, env: stackBoxWorld, train_config: Dict):
        self.a_dim: int = a_dim
        self.y_dim: int = y_dim
        self.z_dim: int = z_dim
        self.use_SSD = train_config['use_SSD']
        self.config: Dict = train_config

        self.disc_coeff = train_config['d_coeff']
        self.post_coeff = train_config['p_coeff']
        self.raw_reward_coeff = train_config['raw_coeff']

        self.ent_coeff = train_config['ent_coeff']
        self.vf_coeff = train_config['vf_coeff']

        # Declare Environment
        self.env: stackBoxWorld = env

        # Define the Buffer
        self.buffer = ReplayBuffer(buffer_size=train_config['buffer_size'])

        # Declare Networks
        self.actor = Actor(a_dim)
        self.critic = Critic()
        if self.use_SSD:
            self.discriminator = ssDiscriminator(self.y_dim)
        else:
            self.discriminator = Discriminator()
        self.cond_prior = Conditional_Prior(z_dim)
        self.posterior = Posterior(z_dim, y_dim)

        # Define Optimisers
        self.d_opt = tf.keras.optimizers.Adam(train_config['d_lr'])

        self.a_lr = tf.Variable(train_config['a_lr'])
        self.a_opt = tf.keras.optimizers.Adam(self.a_lr)

        self.c_lr = tf.Variable(train_config['c_lr'])
        self.c_opt = tf.keras.optimizers.Adam(self.c_lr)

        # Define Losses
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.cat_cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.mse = tf.keras.losses.MeanSquaredError()

        self.debug = 0

    def set_learning_rate(self, actor_learning_rate=None, critic_learning_rate=None):
        """Update learning rate."""
        if actor_learning_rate:
            self.a_lr.assign(actor_learning_rate)
        if critic_learning_rate:
            self.c_lr.assign(critic_learning_rate)

    def load_pretrained_model(self, param_dir, model_id):
        # BUILD First
        _ = self.posterior.encoder(np.ones([1, self.config['w_size'] * 5, 5, 6]), np.ones([1, self.y_dim]))
        _ = self.posterior.classifier(np.ones([1, self.z_dim]))
        _ = self.cond_prior(np.ones([1, self.y_dim]))
        _ = self.actor(np.ones([1, 5, 5, 6]), np.ones([1, self.z_dim]))

        # Load Models
        self.posterior.encoder.load_weights(os.path.join(param_dir, "encoder_model_{}.h5".format(model_id)))
        self.posterior.classifier.load_weights(os.path.join(param_dir, "classifier_{}.h5".format(model_id)))
        self.cond_prior.load_weights(os.path.join(param_dir, "cond_prior_{}.h5".format(model_id)))
        self.actor.load_weights(os.path.join(param_dir, "decoder_model_{}.h5".format(model_id)))

        # # [DEBUG] Test pre-trained actor [Assume z is y].
        # # Careful the decoder predicts prob. of actions while actor predicts logits [Should not be an issue]
        # curr_state, curr_encode_y = tf.numpy_function(func=self.env_reset, inp=[], Tout=(tf.float32, tf.float32))
        # curr_state = tf.expand_dims(curr_state, axis=0)
        # curr_encode_y = tf.expand_dims(curr_encode_y, axis=0)
        # op = self.actor(curr_state, curr_encode_y)
        # print("Working Actor")

    def load_model(self, param_dir):
        # BUILD First
        _ = self.posterior.encoder(np.ones([1, self.config['w_size'] * 5, 5, 6]), np.ones([1, self.y_dim]))
        _ = self.posterior.classifier(np.ones([1, self.z_dim]))
        _ = self.cond_prior(np.ones([1, self.y_dim]))
        _ = self.actor(np.ones([1, 5, 5, 6]), np.ones([1, self.z_dim]))

        # Load Models
        self.posterior.encoder.load_weights(os.path.join(param_dir, "encoder_model.h5"))
        self.posterior.classifier.load_weights(os.path.join(param_dir, "classifier.h5"))
        self.cond_prior.load_weights(os.path.join(param_dir, "cond_prior.h5"))
        self.actor.load_weights(os.path.join(param_dir, "actor.h5"))

    def save_model(self, param_dir):
        # Save weights
        self.discriminator.save_weights(os.path.join(param_dir, "discriminator.h5"), overwrite=True)
        self.actor.save_weights(os.path.join(param_dir, "actor.h5"), overwrite=True)
        self.critic.save_weights(os.path.join(param_dir, "critic.h5"), overwrite=True)

        self.posterior.encoder.save_weights(os.path.join(param_dir, "encoder_model.h5"))
        self.posterior.classifier.save_weights(os.path.join(param_dir, "classifier.h5"))
        self.cond_prior.save_weights(os.path.join(param_dir, "cond_prior.h5"))

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
        return next_obs['grid_world'].astype(np.float32), raw.astype(np.float32), done.astype(np.float32), \
               obj_achieved.astype(np.float32)

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

    # @tf.function
    def un_disc_step(self, sampled_data, expert_data):
        with tf.GradientTape() as disc_tape:

            # Probability that sampled data is from expert
            fake_output = tf.clip_by_value(self.discriminator(*sampled_data, sup=False), 0.01, 1.0)
            d_loss = self.cross_entropy(tf.zeros_like(fake_output),
                                        fake_output)  # Prob=0: sampled data from expert

            if expert_data:
                # Prob that expert data is from expert
                real_output = tf.clip_by_value(self.discriminator(*expert_data, sup=False), 0.01, 1.0)
                # Compute Loss
                real_loss = self.cross_entropy(tf.ones_like(real_output),
                                               real_output)  # Prob=1: expert data from expert
                d_loss += real_loss

        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return d_loss

    def ln_disc_step(self, expert_data, expert_labels):
        with tf.GradientTape() as disc_tape:
            # Prob that expert data is from expert
            pred_labels = tf.clip_by_value(self.discriminator(*expert_data, sup=True), 0.01, 1.0)
            # Compute Loss
            d_loss = self.cat_cross_entropy(expert_labels, pred_labels)
            # d_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return d_loss

    # @tf.function
    def actor_trpo_step(self, data: Dict):

        # TRPO Step: Find estimate of policy gradients i.e. g by differentiating surrogate advantage func.
        with tf.GradientTape() as tape_surr:
            action, action_logprob, dist_entropy = self.act(data['states'], data['encodes_z'], data['actions'])
            ratio = tf.exp(action_logprob - data['old_action_logprob'])
            surr_loss = tf.reduce_mean(tf.math.multiply(ratio, data['advants']))
            mean_ent = tf.reduce_mean(dist_entropy)
            optim_gain = surr_loss + self.ent_coeff*mean_ent

        grads = tape_surr.gradient(optim_gain, self.actor.trainable_variables)
        policy_gradient = get_flat(grads)  # Flatten

        # TRPO Step 4: Find the step direction i.e. X = inv(H) dot g using conjugate gradients that solves Hx=g
        def fisher_vector_product(p):
            """
            :param p: conjugate direction p_k to be used for computing A*p_k (flattened)
            :return:
            """
            # Compute the Second Gradient (of the product with p constant gives H*p)
            with tf.GradientTape() as tapekl_2:
                # Compute the H, Hessian which is the double derivative of KL_div(current policy || prev fixed policy)
                with tf.GradientTape() as tapekl_1:
                    _, _action_logprob, _ = self.act(data['states'], data['encodes_z'], data['actions'])
                    mean_kl = tf.reduce_mean(
                        tf.math.multiply(tf.exp(_action_logprob), _action_logprob - data['old_action_logprob']))

                grad_kl = tapekl_1.gradient(mean_kl, self.actor.trainable_variables)
                grad_kl = tf.concat([tf.reshape(t, [-1]) for t in grad_kl], axis=0)

                # Computing the product between grad and tangents i.e first derivative of KL and p resp.
                gvp = tf.reduce_sum(grad_kl*p)

            gvp_grads = tapekl_2.gradient(gvp, self.actor.trainable_variables)
            fvp = get_flat(gvp_grads)
            return fvp + p * self.config['cg_damping']  # Damping used for stability

        step_dir = conjugate_gradient(fisher_vector_product, policy_gradient, self.config['cg_iters'])

        # TRPO Step 5: Find the Approximate step-size (Delta_k)
        shs = 0.5 * tf.reduce_sum(step_dir*fisher_vector_product(step_dir))  # (0.5*X^T*Hessian*X) where X is step_dir
        lagrange_multiplier = np.sqrt(shs / self.config['max_kl'])  # 1 / sqrt( (2*delta) / (X^T*Hessian*X) )
        full_step = step_dir / lagrange_multiplier  # Delta

        # TRPO Step 6[POLICY UPDATE]: Perform back-tracking line search with expo. decay to update param.
        theta_prev = get_flat(self.actor.trainable_variables)  # Flatten the vector
        expected_improve_rate = tf.reduce_sum(policy_gradient*step_dir)/lagrange_multiplier
        success, theta = linesearch(self.actor, self.act, theta_prev, full_step, expected_improve_rate, data)

        if np.isnan(theta).any():
            print("NaN detected. Skipping update...")
        else:
            set_from_flat(self.actor, theta)

        # Compute the Generator losses based on current estimate of policy
        action, action_logprob, dist_entropy = self.act(data['states'], data['encodes_z'], data['actions'])
        ratio = tf.exp(action_logprob - data['old_action_logprob'])
        surr_loss = tf.reduce_mean(tf.math.multiply(ratio, data['advants']))
        mean_kl = tf.reduce_mean(
            tf.math.multiply(tf.exp(action_logprob), action_logprob - data['old_action_logprob']))
        mean_ent = tf.reduce_mean(dist_entropy)
        return surr_loss, mean_ent, mean_kl

    # @tf.function
    def critic_step(self, data: Dict, clip_v=False):
        with tf.GradientTape() as critic_tape:

            # #################################### Critic Loss ####################################
            values = self.critic(data['states'], data['encodes'])
            # Need to multiply the mse with vf_coeff if actor and critic share param.
            # for balance else no need since critic loss will be minimised separately
            # c_loss = self.vf_coeff*self.mse(data['returns'], values)

            c_loss = self.vf_coeff*self.mse(data['returns'], values)
            # # For clipped Critic loss
            if clip_v:
                value_pred_clipped = data['values'] + tf.clip_by_value(values - data['values'],
                                                                       - self.config['v_clip'],
                                                                       self.config['v_clip'])
                c_loss_clipped = self.mse(data['returns'], value_pred_clipped)
                c_loss = self.vf_coeff*tf.maximum(c_loss, c_loss_clipped)

        critic_gradients = critic_tape.gradient(c_loss, self.critic.trainable_variables)
        # To perform global norm based gradient clipping
        # gradients, _ = tf.clip_by_global_norm(critic_gradients, self.config['grad_clip_norm'])
        self.c_opt.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        return c_loss

    # @tf.function
    def process_episodes(self, trajectories: List[Dict], _iter, log_dir) -> List[Dict]:

        for path_idx, path in enumerate(trajectories):
            file_path = os.path.join(log_dir, "iter_%d_path_%d.txt" % (_iter, path_idx))
            f = open(file_path, "a")

            #  #################################################################################################### #
            #  ######################################## Compute Reward ############################################ #
            #  #################################################################################################### #

            # Disc should predict whether the sampled data belongs to expert, which if high will correspond to high rew
            if self.use_SSD:  # (s, a) -> c -> Prob(data is from expert)
                d_pred = self.discriminator(path['states'], path['one_hot_actions'], sup=False)
            else:  # (s, a) -> Prob(data is from expert)
                d_pred = self.discriminator(path['states'], path['one_hot_actions'])
            r_d = self.disc_coeff * tf.math.log(tf.clip_by_value(d_pred, 1e-10, 1))
            # Post should predict how probable the underlying latent mode is for given state [IMPLICATION???]
            p_pred = tf.reduce_sum(tf.math.multiply(path['encodes'], path['encodes_prob']), axis=-1)
            r_p = self.post_coeff * tf.math.log(tf.clip_by_value(p_pred, 1e-10, 1))
            # Reward Augmentation
            r_raw = self.raw_reward_coeff*tf.reshape(path['raw_rewards'], shape=[-1, ])
            rewards = tf.reshape(r_d, shape=[-1, ]) + r_p + r_raw
            rewards = tf.cast(rewards, dtype=tf.float32)

            # With reward scaling
            # rewards = tf.clip_by_value(rewards, -5, 5)
            # rewards = (rewards - tf.math.reduce_mean(rewards)) / (tf.math.reduce_std(rewards) + 1e-8)

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
            print_acts = [self.env.steps_ref[a] for a in list(path['actions'].numpy())]
            f.write("Actions [{}]:\n".format(len(print_acts)) + str(print_acts) + "\n")
            f.write(
                "\n#  ######################################################################################## #\n")
            f.write("Latent Modes:\n" + np.array_str(path['encodes'].numpy()) + "\n")
            f.write(
                "\n#  ######################################################################################## #\n")
            f.write("Rewards ({}*logD(s,a) + {}*logP(s,c) + {}*R(s):\n".format(self.config['d_coeff'], self.config['p_coeff'],
                                                                               self.config['raw_coeff']) +
                    np.array_str(path['rewards'].numpy()) + "\n")
            f.write("Returns:\n" + np.array_str(path['returns'].numpy()) + "\n")
            f.write("Values:\n" + np.array_str(path['values'].numpy()) + "\n")
            f.write("Advants:\n" + np.array_str(path['advants'].numpy()) + "\n")
            f.close()

        return trajectories

    # @tf.function
    def collect_episodesv2(self, _iter, num_episodes, log_dir) -> Tuple[List[Dict], float]:
        trajectories: List[Dict] = []
        success_rate = 0
        for path_idx in range(num_episodes):
            file_path = os.path.join(log_dir, "iter_%d_path_%d.txt" % (_iter, path_idx))
            f = open(file_path, "w")

            states, encodes, encodes_z, actions, old_actions_logprob, one_hot_actions, encodes_prob, values = [], [],\
                                                                                                              [], [], \
                                                                                                              [], [], \
                                                                                                              [], []
            # For reward augmentation
            raw_rewards = []

            # Observe the initial state and take the first action using condition prior on the observed latent mode
            curr_state, curr_encode_y = tf.numpy_function(func=self.env_reset, inp=[], Tout=(tf.float32, tf.float32))

            grid_world = self.env.render_init_config()
            # f.write("Init_State:\n" + np.array_str(curr_state.numpy()) + "\n")
            f.write("Init_State:\n" + grid_world + "\n")
            f.write("Init_Encode:\n" + np.array_str(curr_encode_y.numpy()) + "\n")

            encode_prob = tf.one_hot(tf.argmax(curr_encode_y), self.y_dim)
            encode_prob = tf.expand_dims(encode_prob, axis=0)
            curr_state = tf.expand_dims(curr_state, axis=0)
            curr_encode_y = tf.expand_dims(curr_encode_y, axis=0)

            _, _, z_k = self.cond_prior(curr_encode_y, k=10)
            encode_z_mu = tf.reduce_mean(z_k, axis=0)
            z_k = tf.squeeze(z_k, axis=1)  # (k, 1, z_dim) -> (k, z_dim)

            p_x_k = self.actor(tf.repeat(curr_state, repeats=10, axis=0), z_k)
            init_action_prob = tf.reduce_mean(p_x_k, axis=0, keepdims=True)
            init_action_dis = Categorical(probs=init_action_prob)
            action = init_action_dis.sample()
            action_log_prob = init_action_dis.log_prob(action)
            value = self.critic(curr_state, curr_encode_y)

            #  #################################################################################################### #
            #  ###################################### Unroll Trajectories ######################################### #
            #  #################################################################################################### #
            while not self.env.episode_ended:

                next_state, raw, done, obj_achieved = tf.numpy_function(func=self.env_step, inp=[action],
                                                                        Tout=(tf.float32, tf.float32,
                                                                              tf.float32, tf.float32))

                # Add to the stack
                one_hot_actions.append(tf.cast(tf.one_hot(action, self.a_dim), dtype=tf.float32))
                states.append(tf.cast(curr_state, dtype=tf.float32))
                encodes.append(tf.cast(curr_encode_y, dtype=tf.float32))
                encodes_z.append(tf.cast(encode_z_mu, dtype=tf.float32))
                actions.append(tf.cast(action, dtype=tf.float32))
                old_actions_logprob.append(tf.cast(action_log_prob, dtype=tf.float32))
                encodes_prob.append(tf.cast(encode_prob, dtype=tf.float32))
                values.append(tf.cast(value, dtype=tf.float32))
                raw_rewards.append(tf.cast(raw, dtype=tf.float32))

                # Add the value of terminal state
                if done:
                    if obj_achieved:
                        success_rate += 1
                        values.append(tf.cast([0.], dtype=tf.float32))
                    else:
                        values.append(values[-1])
                    break

                next_state = tf.expand_dims(next_state, axis=0)
                # Predict next encode by sampling k z^(i) from q(z|s_t, c_{t-1}) and averaging q(c_t|z^(i))
                next_onehot_encode, encode_prob, encode_z_mu = self.posterior(next_state, curr_encode_y,
                                                                              self.config['k'])

                # Computer action and value of the state
                action, action_log_prob, entropy = self.act(next_state, encode_z_mu)
                value = self.critic(next_state, next_onehot_encode)

                # Update the  curr_encode
                curr_state = next_state
                curr_encode_y = next_onehot_encode

            path = {
                'states': tf.concat(states, axis=0),
                'encodes': tf.concat(encodes, axis=0),
                'encodes_z': tf.concat(encodes_z, axis=0),
                'actions': tf.concat(actions, axis=0),
                'old_action_logprob': tf.concat(old_actions_logprob, axis=0),
                'encodes_prob': tf.concat(encodes_prob, axis=0),
                'values': tf.concat(values, axis=0),
                'one_hot_actions': tf.concat(one_hot_actions, axis=0),
                'raw_rewards': tf.concat(raw_rewards, axis=0)
            }
            trajectories.append(path)

            f.close()

        return trajectories, success_rate/num_episodes

    def train(self, expert_data_un: Dict, expert_data_ln: Dict, param_dir: str, fig_path: str, log_dir: str,
              exp_num: int = 0):

        if expert_data_un:
            num_un_expert_trans = expert_data_un['states'].shape[0]
            un_expert_gen = yield_batched_indexes(start=0, b_size=self.config['batch_size'],
                                                  n_samples=num_un_expert_trans)

        if expert_data_ln:
            num_ln_expert_trans = expert_data_ln['states'].shape[0]
            try:
                assert num_ln_expert_trans > self.config['batch_size']
            except AssertionError:
                print("Lower the batch size than ", num_ln_expert_trans)
                sys.exit(-1)
            ln_expert_gen = yield_batched_indexes(start=0, b_size=self.config['batch_size'],
                                                  n_samples=num_ln_expert_trans)

        total_un_disc_loss, total_ln_disc_loss, total_actor_loss, total_critic_loss, total_ent_loss, total_kl_loss = [], [], [], [], [], []
        train_avg_traj_len, success_rates = [], []
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
                        num_episodes = 2 * self.config['collect_episodes']
                    else:
                        num_episodes = self.config['collect_episodes']
                    trajectories, success_rate = self.collect_episodesv2(iter_num, num_episodes, log_dir)
                    success_rates.append(success_rate)
                    datac_time = round(time.time()-start, 3)

                    #  ############################################################################################### #
                    #  ########################################## Process Data ####################################### #
                    #  ############################################################################################### #
                    start = time.time()
                    trajectories = self.process_episodes(trajectories, iter_num, log_dir)

                    sampled_data, avg_traj_len = self.wrap_data(trajectories)
                    train_avg_traj_len.append(avg_traj_len)
                    datap_time = round(time.time() - start, 3)

                    num_gen_trans = sampled_data['states'].shape[0]

                    #  ############################################################################################### #
                    #  ###################################### Perform Optimisation ################################### #
                    #  ############################################################################################### #
                    start = time.time()
                    it_unDLoss, it_lnDLoss, it_ALoss, it_CLoss, it_ELoss, it_KLLoss = [], [], [], [], [], []

                    #  ####################################### Train Discriminator ################################### #
                    sampled_gen_D = yield_batched_indexes(start=0, b_size=self.config['batch_size'],
                                                          n_samples=num_gen_trans)
                    if iter_num <= 5:
                        num_diter = 120 - iter_num*20  # 120, 100, 80, 60, 40, 20
                    else:
                        num_diter = num_gen_trans // self.config['batch_size']  # ~ 500/50 = 10 iter
                    for i in range(num_diter):

                        # Supervised case: Minimise classification loss
                        if expert_data_ln:
                            ln_e_idxs = ln_expert_gen.__next__()
                            random.shuffle(ln_e_idxs)

                            d_loss = self.ln_disc_step(expert_data=[tf.gather(expert_data_ln[key],
                                                                              tf.constant(ln_e_idxs))
                                                                    for key in ['states', 'one_hot_actions']],
                                                       expert_labels=tf.gather(expert_data_ln["curr_latent_states"],
                                                                               tf.constant(ln_e_idxs)))

                            it_lnDLoss.append(d_loss)

                        # Unsupervised sub-case: Minimise both fake and real loss if unsupervised demos available
                        if expert_data_un:
                            un_e_idxs = un_expert_gen.__next__()
                            random.shuffle(un_e_idxs)

                            s_idxs = sampled_gen_D.__next__()
                            random.shuffle(s_idxs)

                            d_loss = self.un_disc_step(sampled_data=[tf.gather(sampled_data[key],
                                                                               tf.constant(s_idxs))
                                                                     for key in ['states', 'one_hot_actions']],
                                                       expert_data=[tf.gather(expert_data_un[key],
                                                                              tf.constant(un_e_idxs))
                                                                    for key in ['states', 'one_hot_actions']])
                            it_unDLoss.append(d_loss)
                        # Supervised sub-case: Minimise only fake loss if no unsupervised demos available
                        else:
                            s_idxs = sampled_gen_D.__next__()
                            random.shuffle(s_idxs)

                            d_loss = self.un_disc_step(sampled_data=[tf.gather(sampled_data[key],
                                                                               tf.constant(s_idxs))
                                                                     for key in ['states', 'one_hot_actions']],
                                                       expert_data=None)

                            it_unDLoss.append(d_loss)

                    #  ########################################### Train Critic ###################################### #
                    sampled_gen_C = yield_batched_indexes(start=0, b_size=self.config['batch_size'],
                                                          n_samples=num_gen_trans)
                    num_citer = num_gen_trans // self.config['batch_size']
                    for j in range(num_citer):
                        s_idxs = sampled_gen_C.__next__()
                        random.shuffle(s_idxs)
                        c_loss = self.critic_step({key: tf.gather(sampled_data[key], tf.constant(s_idxs))
                                                   for key in sampled_data.keys()}, clip_v=False)
                        it_CLoss.append(c_loss)

                    #  ########################################### Train Actor ####################################### #
                    # sampled_gen_A = yield_batched_indexes(start=0, b_size=2*self.config['batch_size'],
                    #                                       n_samples=num_gen_trans)
                    # num_aiter = num_gen_trans // (2*self.config['batch_size'])
                    # for j in range(num_aiter):
                    #     s_idxs = sampled_gen_A.__next__()
                    #     random.shuffle(s_idxs)
                    #     a_loss, ent_loss, kl_loss = self.actor_trpo_step({key: tf.gather(sampled_data[key],
                    #                                                                      tf.constant(s_idxs))
                    #                                                       for key in sampled_data.keys()})
                    a_loss, ent_loss, kl_loss = self.actor_trpo_step(sampled_data)
                    it_ALoss.append(a_loss)
                    it_ELoss.append(ent_loss)
                    it_KLLoss.append(kl_loss)

                    opt_time = round(time.time() - start, 3)

                    total_un_disc_loss.extend(it_unDLoss)
                    total_ln_disc_loss.extend(it_lnDLoss)
                    total_critic_loss.extend(it_CLoss)
                    total_actor_loss.extend(it_ALoss)
                    total_ent_loss.extend(it_ELoss)
                    total_kl_loss.extend(it_KLLoss)

                    pbar.refresh()
                    pbar.set_description("Cycle {}".format(iter_num))
                    pbar.set_postfix(LossDu=np.average(it_unDLoss), LossDl=np.average(it_lnDLoss),
                                     LossA=np.average(it_ALoss), LossC=np.average(it_CLoss),
                                     LossE=np.average(it_ELoss), LossKL=np.average(it_KLLoss),
                                     TimeDataC='{}s'.format(datac_time),
                                     TimeDataP='{}s'.format(datap_time), TimeOpt='{}s'.format(opt_time),
                                     AvgEpLen=avg_traj_len,)
                    pbar.update(1)

        plot_metric(train_avg_traj_len, fig_path, exp_num, y_label='AvgTrajLength')
        plot_metric(success_rates, fig_path, exp_num, y_label='SuccessRate')

        plot_metric(total_un_disc_loss, fig_path, exp_num, y_label='UnDiscLoss')
        plot_metric(total_ln_disc_loss, fig_path, exp_num, y_label='LnDiscLoss')
        plot_metric(total_actor_loss, fig_path, exp_num, y_label='ActorLoss')
        plot_metric(total_critic_loss, fig_path, exp_num, y_label='CriticLoss')
        plot_metric(total_ent_loss, fig_path, exp_num, y_label='Entropy')

        self.save_model(param_dir)


def run(env_name, exp_num=0, sup=0.1, random_transition=True):

    train_config = {'num_epochs': 10, 'num_cycles': 50, 'collect_episodes': 10, 'process_episodes': 5, 'batch_size': 50,
                    'w_size': 1, 'num_traj': 100, 'k': 10, 'perc_supervision': sup, 'gamma': 0.99, 'lambda': 0.95,
                    'd_iter': 10, 'ac_iter': 5, 'buffer_size': 1e4, 'd_lr': 5e-4, 'a_lr': 4e-4, 'c_lr': 3e-3,
                    'd_coeff': 2.0, 'p_coeff': 0.0, 'raw_coeff': 0.1, 'ent_coeff': 0.0, 'vf_coeff': 0.5,
                    'max_kl': 0.001, 'cg_damping': 0.1, 'cg_iters': 20, 'v_clip': 0.2,
                    'pretrained_model': 'CCVAE' if random_transition else 'fixedC_CCVAE', 'use_SSD': False,
                    'grad_clip_norm': 0.5}

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
        model_dir = os.path.join(root_dir, 'pretrained_params/Project', env_name, 'DirectedInfoGAIL_{}'.format(sup))
    else:
        train_data_path = os.path.join(data_dir, 'stateaction_fixed_latent_dynamic.pkl')
        test_data_path = os.path.join(data_dir, 'stateaction_fixed_latent_dynamic_test.pkl')
        model_dir = os.path.join(root_dir, 'pretrained_params/Project', env_name,
                                 'fixedC_DirectedInfoGAIL_{}'.format(sup))

    # Model Directory
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    param_dir = os.path.join(model_dir, "params{}".format(exp_num))
    fig_path = os.path.join(model_dir, '{}_loss'.format('DirectedInfoGAIL'))

    # Log Directory
    log_dir = os.path.join(model_dir, "logs{}".format(exp_num))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # ################################################ Load Data ################################################ #
    print("Loading data from - ",  train_data_path, file=open(file_txt_results_path, 'a'))
    with open(train_data_path, 'rb') as f:
        traj_sac = pkl.load(f)

    # ################################################ Parse Data ################################################ #
    demos_un, demos_ln = None, None
    expert_data_un, expert_data_ln = {}, {}

    # When using Semi-Supervised Discriminator, load sup. and unsup. samples separately
    if train_config['use_SSD']:
        demos_ln = causally_parse_dynamic_data_v2(traj_sac, lower_bound=0,
                                                  upper_bound=int(train_config["num_traj"] * sup),
                                                  window_size=train_config["w_size"])
        demos_un = causally_parse_dynamic_data_v2(traj_sac, lower_bound=int(train_config["num_traj"] * sup),
                                                  upper_bound=train_config["num_traj"],
                                                  window_size=train_config["w_size"])
    # When using Unsupervised Discriminator, no need to load supervised samples containing G.T. latent modes
    else:
        demos_un = causally_parse_dynamic_data_v2(traj_sac, lower_bound=0,
                                                  upper_bound=train_config["num_traj"],
                                                  window_size=train_config["w_size"])

    if demos_un:
        expert_data_un = {
            'states': tf.cast(demos_un['curr_states'], dtype=tf.float32),
            'one_hot_actions': tf.cast(demos_un['next_actions'], dtype=tf.float32),
        }

    if demos_ln:
        expert_data_ln = {
            'states': tf.cast(demos_ln['curr_states'], dtype=tf.float32),
            'one_hot_actions': tf.cast(demos_ln['next_actions'], dtype=tf.float32),
            'curr_latent_states': tf.cast(demos_ln['curr_latent_states'], dtype=tf.float32)
        }

    # ################################################ Train Model ###########################################
    agent = Agent(env_config['a_dim'], env_config['y_dim'], env_config['z_dim'], env_config['env'],
                  train_config)

    # CCVAE-Load Models
    pretrained_model_dir = os.path.join(root_dir, 'pretrained_params/Project', env_name,
                                        '{}_{}'.format(train_config['pretrained_model'], sup))
    pretrained_param_dir = os.path.join(pretrained_model_dir, "params{}".format(exp_num))
    print("Loading pre-trained model from - ", pretrained_param_dir, file=open(file_txt_results_path, 'a'))

    agent.load_pretrained_model(pretrained_param_dir, model_id='best')

    if not os.path.exists(param_dir):
        os.makedirs(param_dir)
    try:
        agent.train(expert_data_un, expert_data_ln, param_dir, fig_path, log_dir)
    except KeyboardInterrupt:
        agent.save_model(param_dir)
    print("Finish.", file=open(file_txt_results_path, 'a'))
    # else:
    #     print("Skipping Training", file=open(file_txt_results_path, 'a'))

    # ################################################ Test Model ################################################ #
    with open(test_data_path, 'rb') as f:
        test_traj_sac = pkl.load(f)

    print("\nTesting Data Results", file=open(file_txt_results_path, 'a'))
    agent = Agent(env_config['a_dim'], env_config['y_dim'], env_config['z_dim'], env_config['env'], train_config)
    agent.load_model(param_dir)
    evaluate_model_discrete(agent, "GAIL", test_traj_sac, train_config, file_txt_results_path)


if __name__ == "__main__":
    supervision_settings = [1.0]
    for perc_supervision in supervision_settings:
        run(env_name='StackBoxWorld', sup=perc_supervision, random_transition=False)






