import os
import random
import time
import gym
import numpy as np
import pickle as pkl
import tensorflow as tf
from tqdm import tqdm
from typing import Dict, List, Tuple
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from evaluation.eval import evaluate_sup_model_continuous
from utils.plot import plot_metric
from utils.env import get_env_params, preprocess_robotic_demos
from utils.trpo import linesearch, get_flat, conjugate_gradient, set_from_flat
from utils.misc import causally_parse_dynamic_data_v2, yield_batched_indexes, one_hot_encode
from utils.gail import ReplayBuffer
from keras.layers import Dense, Flatten, Add, Concatenate, LeakyReLU
from tensorflow_probability.python.distributions import Normal

file_txt_results_path = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/contGAIL_sup.txt'


class Actor(tf.keras.Model):
    def __init__(self, a_dim, actions_max):
        super(Actor, self).__init__()
        # Adding recommended Orthogonal initialization with scaling that varies from layer to layer
        relu_gain = tf.math.sqrt(2.0)
        relu_init = tf.initializers.Orthogonal(gain=relu_gain)

        self.max_actions = actions_max
        self.flatten = Flatten()
        self.fc1 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=relu_init)
        self.fc2 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=relu_init)
        self.fc3 = Dense(units=128, activation=tf.nn.relu, kernel_initializer=relu_init)
        self.fc4 = Dense(units=128, activation=tf.nn.relu, kernel_initializer=relu_init)
        self.add = Add()
        # self.leaky_relu = LeakyReLU()
        self.a_out = Dense(units=a_dim, activation=tf.nn.tanh,
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
        actions_prob = self.a_out(h) * self.max_actions
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


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)

        self.flatten = Flatten()
        self.concat = Concatenate()
        self.fc1 = Dense(units=256, activation=tf.nn.tanh, kernel_initializer=kernel_init)
        self.fc2 = Dense(units=256, activation=tf.nn.tanh, kernel_initializer=kernel_init)
        self.d_out = Dense(units=1, kernel_initializer=kernel_init)

    # @tf.function
    def call(self, curr_state, goal, curr_action):
        curr_state = self.flatten(curr_state)
        h = self.concat([curr_state, goal, curr_action])
        h = self.fc1(h)
        h = self.fc2(h)
        d_out = self.d_out(h)
        return d_out


class Agent(object):
    def __init__(self, a_dim: int, s_dim: int, g_dim: int, env: FetchPickAndPlaceEnv, train_config: Dict,
                 env_config: Dict):
        self.a_dim: int = a_dim
        self.s_dim: int = s_dim
        self.g_dim: int = g_dim
        self.config: Dict = train_config

        # Define the parameters of the network
        self.disc_reward_coeff = train_config['d_coeff']
        self.raw_reward_coeff = train_config['raw_coeff']
        self.ent_coeff = train_config['ent_coeff']
        self.vf_coeff = train_config['vf_coeff']

        # Declare Environment
        self.env: FetchPickAndPlaceEnv = env

        # Define the Buffer
        self.buffer = ReplayBuffer(buffer_size=train_config['buffer_size'])

        # Declare Networks
        self.actor = Actor(a_dim, env_config['action_max'])
        self.critic = Critic()
        self.discriminator = Discriminator()

        # Define Optimisers
        self.d_opt = tf.keras.optimizers.Adam(train_config['d_lr'])
        self.a_lr = tf.Variable(train_config['a_lr'])
        self.a_opt = tf.keras.optimizers.Adam(self.a_lr)
        self.c_lr = tf.Variable(train_config['c_lr'])
        self.c_opt = tf.keras.optimizers.Adam(self.c_lr)

        # Define Losses
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.cat_cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        # Define Action Std-dev
        self.action_scale = tf.Variable(np.array([[0.14631747, 0.15090476, 0.11856036, 0.02602926]]), dtype=tf.float32)

        self.debug = 0

    def set_learning_rate(self, actor_learning_rate=None, critic_learning_rate=None):
        """Update learning rate."""
        if actor_learning_rate:
            self.a_lr.assign(actor_learning_rate)
        if critic_learning_rate:
            self.c_lr.assign(critic_learning_rate)

    def load_pretrained_model(self, param_dir):
        # BUILD First
        _ = self.actor(np.ones([1, self.s_dim]), np.ones([1, self.g_dim]))

        # Load Models
        self.actor.load_weights(os.path.join(param_dir, "actor.h5"))

    def load_model(self, param_dir):
        # BUILD First
        _ = self.actor(np.ones(np.ones([1, self.s_dim]), np.ones([1, self.g_dim])))

        # Load Models
        self.actor.load_weights(os.path.join(param_dir, "actor.h5"))

    def save_model(self, param_dir):
        # Save weights
        self.discriminator.save_weights(os.path.join(param_dir, "discriminator.h5"), overwrite=True)
        self.actor.save_weights(os.path.join(param_dir, "actor.h5"), overwrite=True)
        self.critic.save_weights(os.path.join(param_dir, "critic.h5"), overwrite=True)

    def act(self, state, encode_z, action=None):
        action_mu = self.actor(state, encode_z)
        dist = Normal(loc=action_mu, scale=self.action_scale)
        if action is None:
            action = dist.sample()
        action_log_prob = dist.log_prob(action)

        # For multi-dim actions, the prob of overall action = multiply prob. of each dim since they are indep.
        # This translates to taking sum of log probabilities
        action_log_prob = tf.reduce_sum(action_log_prob, axis=-1)
        return action, action_log_prob, dist.entropy()

    def env_reset(self):
        init_obs = self.env.reset()
        init_state = np.concatenate([init_obs['observation'], init_obs['desired_goal']])
        init_latent_mode = one_hot_encode(np.array(0, dtype=int), dim=2)[0]
        return init_state.astype(np.float32), init_latent_mode.astype(np.float32)

    def env_step(self, action: np.ndarray):
        next_obs, reward, done, info = self.env.step(action[0])
        curr_state = np.concatenate([next_obs['observation'], next_obs['desired_goal']])
        object_rel_pos = next_obs['observation'][6:9]
        if np.linalg.norm(object_rel_pos) < 0.005:
            curr_encode_y = one_hot_encode(np.array(1, dtype=int), dim=2)[0]
            action_scale = np.array([[0.14631747, 0.15090476, 0.11856036, 0.02602926]])
        else:
            curr_encode_y = one_hot_encode(np.array(0, dtype=int), dim=2)[0]
            action_scale = np.array([[1.87254752e-01, 1.88611527e-01, 1.86052738e-01, 1.47017815e-18]])
        self.action_scale.assign(action_scale)

        return curr_state.astype(np.float32), curr_encode_y.astype(np.float32),\
               reward.astype(np.float32), np.array(done, np.int32)

    def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(func=self.env_step, inp=[action], Tout=[tf.float32, tf.float32, tf.float32, tf.int32])

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

    @tf.function
    def disc_step(self, sampled_data, expert_data):
        with tf.GradientTape() as disc_tape:

            # Probability that sampled data is from expert
            fake_output = self.discriminator(*sampled_data)
            fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)  # Prob=0: sampled data from expert

            # Prob that expert data is from expert
            real_output = self.discriminator(*expert_data)
            real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)  # Prob=1: expert data from expert

            # Compute gradient penalty for stable training
            epsilon = tf.random.uniform(shape=[], minval=0, maxval=1)
            inter = [epsilon * sampled_i + (1 - epsilon) * expert_i
                     for sampled_i, expert_i in zip(sampled_data, expert_data)]
            inter_output = self.discriminator(*inter)
            grad = tf.gradients(inter_output, [inter])[0]
            grad_penalty = tf.reduce_mean(tf.pow(tf.norm(grad, axis=-1) - 1, 2))

            # Total loss
            d_loss = fake_loss + real_loss + self.config['lambd'] * grad_penalty

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
            optim_gain = surr_loss + tf.cast(self.ent_coeff, dtype=tf.float32)*mean_ent

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
            return fvp + p * tf.cast(self.config['cg_damping'], dtype=tf.float32)  # Damping used for stability

        step_dir = conjugate_gradient(fisher_vector_product, policy_gradient,
                                      tf.cast(self.config['cg_iters'], dtype=tf.float32))

        # TRPO Step 5: Find the Approximate step-size (Delta_k)
        shs = 0.5 * tf.reduce_sum(step_dir*fisher_vector_product(step_dir))  # (0.5*X^T*Hessian*X) where X is step_dir
        lagrange_multiplier = tf.math.sqrt(shs / tf.cast(self.config['max_kl'], dtype=tf.float32))  # 1 / sqrt( (2*delta) / (X^T*Hessian*X) )
        full_step = step_dir / lagrange_multiplier  # Delta

        # TRPO Step 6 [POLICY UPDATE]: Perform back-tracking line search with expo. decay to update param.
        theta_prev = get_flat(self.actor.trainable_variables)  # Flatten the vector
        expected_improve_rate = tf.reduce_sum(policy_gradient*step_dir)/lagrange_multiplier
        success, theta = linesearch(self.actor, self.act, theta_prev, full_step, expected_improve_rate, data)

        if not tf.reduce_any(tf.math.is_nan(theta)):
        #     print("NaN detected. Skipping update...")
        # else:
            set_from_flat(self.actor, theta)

        # Compute the Generator losses based on current estimate of policy
        action, action_logprob, dist_entropy = self.act(data['states'], data['encodes_z'], data['actions'])
        ratio = tf.exp(action_logprob - data['old_action_logprob'])
        surr_loss = tf.reduce_mean(tf.math.multiply(ratio, data['advants']))
        mean_kl = tf.reduce_mean(
            tf.math.multiply(tf.exp(action_logprob), action_logprob - data['old_action_logprob']))
        mean_ent = tf.reduce_mean(dist_entropy)
        return surr_loss, mean_ent, mean_kl

    @tf.function
    def critic_step(self, data: Dict, clip_v=False):
        with tf.GradientTape() as critic_tape:

            # #################################### Critic Loss ####################################
            values = self.critic(data['states'], data['encodes'])
            # Need to multiply the mse with vf_coeff if actor and critic share param.
            # for balance else no need since critic loss will be minimised separately

            # c_loss = self.vf_coeff*self.mse(data['returns'], values)
            c_loss = self.vf_coeff * self.huber_loss(values, data['returns'])

            # # For clipped Critic loss
            if clip_v:
                value_pred_clipped = data['values'] + tf.clip_by_value(values - data['values'],
                                                                       - self.config['v_clip'],
                                                                       self.config['v_clip'])
                # c_loss_clipped = self.mse(data['returns'], value_pred_clipped)
                c_loss_clipped = self.huber_loss(value_pred_clipped, data['returns'])
                c_loss = self.vf_coeff*tf.maximum(c_loss, c_loss_clipped)

        critic_gradients = critic_tape.gradient(c_loss, self.critic.trainable_variables)
        # To perform global norm based gradient clipping
        # gradients, _ = tf.clip_by_global_norm(critic_gradients, self.config['grad_clip_norm'])
        self.c_opt.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        return c_loss

    @tf.function
    def tf_run_episode(self, curr_state: tf.Tensor, curr_encode_y: tf.Tensor, max_steps: int, standardize: bool = True):

        states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        encodes = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        old_actions_logprob = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        state_shape = curr_state.shape
        encode_shape = curr_encode_y.shape

        for t in tf.range(max_steps):

            # Convert state into a batched tensor (batch size = 1)
            curr_state = tf.expand_dims(curr_state, 0)
            curr_encode_y = tf.expand_dims(curr_encode_y, axis=0)
            states = states.write(t, tf.squeeze(curr_state))
            encodes = encodes.write(t, tf.squeeze(curr_encode_y))

            # Run the model and to get action probabilities and critic value
            value = self.critic(curr_state, curr_encode_y)

            action, action_log_prob, _ = self.act(curr_state, curr_encode_y, self.action_scale)

            # Apply action to the environment to get next state and reward
            curr_state, curr_encode_y, raw, done = self.tf_env_step(action)
            curr_state.set_shape(state_shape)
            curr_encode_y.set_shape(encode_shape)

            # Store data
            rewards = rewards.write(t, tf.squeeze(raw))
            values = values.write(t, tf.squeeze(value))
            actions = actions.write(t, tf.squeeze(action))
            old_actions_logprob = old_actions_logprob.write(t, tf.squeeze(action_log_prob))

            if tf.cast(done, tf.bool):
                break

        states = states.stack()
        encodes = encodes.stack()
        actions = actions.stack()
        old_actions_logprob = old_actions_logprob.stack()
        rewards = rewards.stack()
        values = values.stack()

        # Reward Augmentation
        rewards *= self.raw_reward_coeff
        d_pred = self.discriminator(states, encodes, actions)
        r_d = self.disc_reward_coeff * tf.math.log(tf.clip_by_value(tf.nn.sigmoid(d_pred), 1e-10, 1))
        # Alternatively use the following line to use the GAIL reward i.e. min the prob that gen samples are fake
        # r_d = self.disc_reward_coeff * -tf.math.log(1 - tf.clip_by_value(tf.nn.sigmoid(d_pred), 1e-10, 1))
        r_d = tf.reshape(r_d, shape=[-1, ])
        rewards += r_d
        orig_rewards = rewards

        # Compute Return
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + self.config['gamma'] * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) /
                       (tf.math.reduce_std(returns) + 1e-8))

        # Compute Advantage
        advantage = returns - values

        return states, encodes, actions, old_actions_logprob, orig_rewards, values, returns, advantage

    def train(self, expert_data: Dict, param_dir: str, fig_path: str, log_dir: str,
              exp_num: int = 0):

        num_expert_trans = expert_data['states'].shape[0]
        expert_gen = yield_batched_indexes(start=0, b_size=self.config['batch_size'], n_samples=num_expert_trans)

        total_disc_loss, total_actor_loss, total_critic_loss, total_ent_loss, total_kl_loss = [], [], [], [], []
        train_avg_traj_len = []
        with tqdm(total=self.config['num_epochs'] * self.config['num_cycles'], position=0, leave=True) as pbar:
            for epoch in range(self.config['num_epochs']):

                # # Shuffle Expert Data
                # idx_d = np.arange(num_expert_trans)
                # np.random.shuffle(idx_d)
                for cycle in range(self.config['num_cycles']):

                    iter_num = epoch * self.config['num_cycles'] + cycle

                    # ############################################################################################## #
                    # #################################### Collect/Process Data #################################### #
                    # ############################################################################################## #
                    start = time.time()
                    if iter_num <= 5:
                        # 20ep x 50s = 1000 (20 iter w batch size of 50)
                        num_episodes = 2 * self.config['collect_episodes']
                    else:
                        # 10ep x 50s = 500 (~10 iter w batch size of 50)
                        num_episodes = self.config['collect_episodes']

                    # Tensorflow Computation Graph compatible logic
                    max_steps_per_episode = self.env._max_episode_steps
                    trajectories = []
                    for ep in range(num_episodes):
                        file_path = os.path.join(log_dir, "iter_%d_path_%d.txt" % (iter_num, ep))
                        f = open(file_path, "w")
                        curr_state, curr_encode_y = tf.numpy_function(func=self.env_reset, inp=[],
                                                                      Tout=(tf.float32, tf.float32))
                        f.write("Init_State:\n" + np.array_str(curr_state.numpy()) + "\n")
                        f.write("Init_Encode:\n" + np.array_str(curr_encode_y.numpy()) + "\n")

                        states, encodes, actions, old_actions_logprob, rewards, values, returns, advantage = self.tf_run_episode(curr_state, curr_encode_y, max_steps_per_episode, standardize=False)

                        path = {'states': states, 'encodes': encodes, 'encodes_z': encodes, 'actions': actions,
                                'old_action_logprob': old_actions_logprob, 'values': values, 'rewards': rewards,
                                'returns': returns, 'advants': advantage}

                        # Book-keeping
                        f.write(
                            "\n#  ################################################################################ #\n")
                        f.write("Actions :\n" + np.array_str(path['actions'].numpy()) + "\n")
                        f.write(
                            "\n#  ################################################################################ #\n")
                        f.write("Latent Modes:\n" + np.array_str(path['encodes'].numpy()) + "\n")
                        f.write(
                            "\n#  ################################################################################ #\n")
                        f.write("Rewards ({}*logD(s,c,a) + {}*R(s):\n".format(self.config['d_coeff'],
                                                                            self.config['raw_coeff']) +
                                np.array_str(path['rewards'].numpy()) + "\n")
                        f.write("Returns:\n" + np.array_str(path['returns'].numpy()) + "\n")
                        f.write("Values:\n" + np.array_str(path['values'].numpy()) + "\n")
                        f.write("Advants:\n" + np.array_str(path['advants'].numpy()) + "\n")
                        f.close()

                        trajectories.append(path)
                    datac_time = round(time.time() - start, 3)

                    sampled_data, avg_traj_len = self.wrap_data(trajectories)
                    train_avg_traj_len.append(avg_traj_len)

                    num_gen_trans = sampled_data['states'].shape[0]

                    #  ############################################################################################### #
                    #  ###################################### Perform Optimisation ################################### #
                    #  ############################################################################################### #
                    start = time.time()
                    it_DLoss, it_ALoss, it_CLoss, it_ELoss, it_KLLoss = [], [], [], [], []

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

                        d_loss = self.disc_step(sampled_data=[tf.gather(sampled_data[key], tf.constant(s_idxs))
                                                              for key in ['states', 'encodes', 'actions']],
                                                expert_data=[tf.gather(expert_data[key], tf.constant(e_idxs))
                                                             for key in ['states', 'encodes', 'actions']])
                        it_DLoss.append(d_loss)

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

                    a_loss, ent_loss, kl_loss = self.actor_trpo_step(sampled_data)
                    it_ALoss.append(a_loss)
                    it_ELoss.append(ent_loss)
                    it_KLLoss.append(kl_loss)

                    opt_time = round(time.time() - start, 3)

                    total_disc_loss.extend(it_DLoss)
                    total_critic_loss.extend(it_CLoss)
                    total_actor_loss.extend(it_ALoss)
                    total_ent_loss.extend(it_ELoss)
                    total_kl_loss.extend(it_KLLoss)

                    pbar.refresh()
                    pbar.set_description("Cycle {}".format(iter_num))
                    pbar.set_postfix(LossD=np.average(it_DLoss),
                                     LossA=np.average(it_ALoss), LossC=np.average(it_CLoss),
                                     LossE=np.average(it_ELoss), LossKL=np.average(it_KLLoss),
                                     TimeDataC='{}s'.format(datac_time), TimeOpt='{}s'.format(opt_time),
                                     AvgEpLen=avg_traj_len,)
                    pbar.update(1)

        plot_metric(train_avg_traj_len, fig_path, exp_num, name='AvgTrajLength')
        plot_metric(total_disc_loss, fig_path, exp_num, name='DiscLoss')
        plot_metric(total_actor_loss, fig_path, exp_num, name='ActorLoss')
        plot_metric(total_critic_loss, fig_path, exp_num, name='CriticLoss')
        plot_metric(total_ent_loss, fig_path, exp_num, name='Entropy')

        self.save_model(param_dir)


def run(env_name, exp_num=0, sup=0.1):

    train_config = {'num_epochs': 10, 'num_cycles': 20, 'collect_episodes': 25, 'process_episodes': 5, 'batch_size': 50,
                    'w_size': 1, 'num_traj': 100, 'k': 10, 'perc_supervision': sup, 'gamma': 0.99, 'lambda': 0.95,
                    'd_iter': 10, 'ac_iter': 5, 'buffer_size': 1e4, 'd_lr': 5e-4, 'a_lr': 4e-4, 'c_lr': 3e-3,
                    'd_coeff': 1.0, 'raw_coeff': 0.0, 'ent_coeff': 0.0, 'vf_coeff': 1.0,
                    'max_kl': 0.001, 'cg_damping': 0.1, 'cg_iters': 20, 'v_clip': 0.2,
                    'pretrained_model': 'BC', 'grad_clip_norm': 0.5, 'lambd': 10, 'epsilon': 1.0,}

    print("\n\n---------------------------------------------------------------------------------------------",
          file=open(file_txt_results_path, 'a'))
    print("---------------------------------- Supervision {} : Exp {} ----------------------------------".format(
        sup, exp_num),
          file=open(file_txt_results_path, 'a'))
    print("---------------------------------------------------------------------------------------------",
          file=open(file_txt_results_path, 'a'))
    print("Train Config: ", train_config, file=open(file_txt_results_path, 'a'))

    # Get OpenAI gym params
    env = gym.make('FetchPickAndPlace-v1')
    env_config = get_env_params(env)

    # ################################################ Specify Paths ################################################ #
    # Root Directory
    root_dir = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/'

    # Data Directory
    data_dir = os.path.join(root_dir, 'training_data/{}'.format(env_name))
    train_data_path = os.path.join(data_dir, 'stateaction_latent_dynamic.pkl')
    test_data_path = os.path.join(data_dir, 'stateaction_latent_dynamic_test.pkl')

    # Model Directory
    model_dir = os.path.join(root_dir, 'pretrained_params/Project', env_name, 'GAIL_{}'.format(sup))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    param_dir = os.path.join(model_dir, "params{}".format(exp_num))
    fig_path = os.path.join(model_dir, '{}_loss'.format('GAIL'))

    # Log Directory
    log_dir = os.path.join(model_dir, "logs{}".format(exp_num))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    y_dim = env_config['latent_mode']
    s_dim = env_config['obs'] + env_config['goal']
    a_dim = env_config['action']

    # ################################################ Load Data ################################################ #
    print("Loading data from - ",  train_data_path, file=open(file_txt_results_path, 'a'))
    with open(train_data_path, 'rb') as f:
        traj_sac = pkl.load(f)

    # ################################################ Parse Data ################################################ #
    demos_train = causally_parse_dynamic_data_v2(traj_sac, lower_bound=0, upper_bound=train_config["num_traj"],
                                                 window_size=train_config["w_size"])
    demos_train = preprocess_robotic_demos(demos_train, env_config, window_size=train_config['w_size'], clip_range=5)

    expert_data = {
        'states': tf.cast(demos_train['curr_states'], dtype=tf.float32),
        'encodes': tf.cast(demos_train['curr_latent_states'], dtype=tf.float32),
        'actions': tf.cast(demos_train['next_actions'], dtype=tf.float32),
    }

    # ################################################ Train Model ###########################################
    agent = Agent(a_dim, s_dim, y_dim, env, train_config, env_config)

    # BC-Load Models
    pretrained_model_dir = os.path.join(root_dir, 'pretrained_params/Project', env_name,
                                        '{}_{}'.format(train_config['pretrained_model'], sup))
    pretrained_param_dir = os.path.join(pretrained_model_dir, "params{}".format(exp_num))
    print("Loading pre-trained model from - ", pretrained_param_dir, file=open(file_txt_results_path, 'a'))

    agent.load_pretrained_model(pretrained_param_dir)

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

    agent = Agent(a_dim, s_dim, y_dim, env, train_config, env_config)
    agent.load_model(param_dir)
    print("\nTesting Data Results", file=open(file_txt_results_path, 'a'))
    evaluate_sup_model_continuous(agent, "GAIL", test_traj_sac, train_config, file_txt_results_path)


if __name__ == "__main__":
    supervision_settings = [1.0]
    for perc_supervision in supervision_settings:
        run(env_name='OpenAIPickandPlace', sup=perc_supervision)






