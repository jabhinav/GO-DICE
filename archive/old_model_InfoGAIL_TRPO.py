import time
# from config import fig_path
from tqdm import tqdm
from archive.old_trpo import *
from config import dtype, fig_path
from utils.plot import *
from keras import backend as K
from keras.layers import Dense, LeakyReLU, Input, Lambda, Add, Concatenate
from keras import Model, models
from keras.optimizers import RMSprop, Adam

"""
Note that the original implementation assumes a gaussian distribution for the policy network whose predicted actions
(continuous) assumes the role of the mean of gaussian. Std. deviation is fixed manually. Thus we obtain the prob. 
distribution over continuous actions predicted by the policy
"""


class Agent(object):

    def __init__(self, env, state_dim, action_dim, encode_dim, config):
        """
        :param state_dim: Feature dimension i.e. 2 (in a 2D grid-world, specifying position)
        :param action_dim: Action dimension i.e. 5
        :param encode_dim: Latent code dimension i.e. 4 one-hot encoded of repr. {[0, 0], [0, 1], [1, 0], [1, 1]}
        :param
        """
        self.config = config
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.encode_dim = encode_dim
        print("State Dim:", state_dim, "Action Dim:", action_dim, "Encode Dim:", encode_dim)

        # Create tensors for the inputs to our network
        # self.states = states = tf.placeholder(dtype, shape=[None, state_dim])
        # self.encodes = encodes = tf.placeholder(dtype, shape=[None, encode_dim])
        # self.actions = actions = tf.placeholder(dtype, shape=[None, action_dim])
        # self.advants = advants = tf.placeholder(dtype, shape=[None])
        # self.oldaction_dist_discrete = oldaction_dist_discrete = tf.placeholder(dtype, shape=[None, action_dim])
        # self.states = states = tf.Variable(validate_shape=False, shape=tf.TensorShape([None, state_dim]),
        #                                    trainable=False, dtype=dtype)
        # self.encodes = encodes = tf.Variable(validate_shape=False, shape=tf.TensorShape([None, encode_dim]),
        #                                      trainable=False, dtype=dtype)
        # self.actions = actions = tf.Variable(validate_shape=False, shape=tf.TensorShape([None, action_dim]),
        #                                      trainable=False, dtype=dtype)
        # self.advants = tf.Variable(validate_shape=False, shape=tf.TensorShape([None]), trainable=False, dtype=dtype)
        # self.oldaction_dist_discrete = tf.Variable(validate_shape=False, shape=tf.TensorShape([None, action_dim]),
        #                                            trainable=False, dtype=dtype)

        self.states, self.encodes, self.actions, self.advants = None, None, None, None
        self.old_action_dist, self.action_dist = None, None

        # Create NN
        self.generator = self.create_generator()
        self.tmp_model = models.clone_model(self.generator)
        # print self.generator.summary()
        self.discriminator, self.discriminate = self.create_discriminator()
        # print self.discriminator.summary()
        self.posterior = self.create_posterior()
        # print self.posterior.summary()
        self.posterior_target = self.create_posterior()
        # print self.posterior_target.summary()

        self.demo_idx = 0
        self.baseline = NNBaseline(state_dim, encode_dim, self.config.lr_baseline, self.config.b_iter,
                                   self.config.batch_size)

        self.buffer = ReplayBuffer(self.config.buffer_size)
        # self.path_proc_pool = Pool(8)

    def create_generator(self):
        state = Input(shape=[self.state_dim])
        x = Dense(128)(state)
        x = LeakyReLU()(x)
        x = Dense(128)(x)
        x = LeakyReLU()(x)

        encodes = Input(shape=[self.encode_dim])
        c = Dense(128)(encodes)
        h = Add()([x, c])
        h = LeakyReLU()(h)

        actions = Dense(self.action_dim, activation='softmax')(h)
        model = Model(inputs=[state, encodes], outputs=actions)
        return model

    def create_discriminator(self):
        states = Input(shape=[self.state_dim])
        actions = Input(shape=[self.action_dim])
        h = Concatenate(axis=1)([states, actions])
        h = Dense(128)(h)
        h = LeakyReLU()(h)
        h = Dense(128)(h)
        h = LeakyReLU()(h)
        p = Dense(1)(h)
        discriminate = Model(inputs=[states, actions], outputs=p)

        states_n = Input(shape=[self.state_dim])
        actions_n = Input(shape=[self.action_dim])
        states_d = Input(shape=[self.state_dim])
        actions_d = Input(shape=[self.action_dim])
        p_n = discriminate([states_n, actions_n])
        p_d = discriminate([states_d, actions_d])
        p_d = Lambda(lambda x: -x)(p_d)
        p_output = Add()([p_n, p_d])
        model = Model(inputs=[states_n, actions_n, states_d, actions_d], outputs=p_output)

        rmsprop = RMSprop(lr=self.config.lr_discriminator)
        model.compile(loss=lambda y_pred, p_true: K.mean(y_pred * p_true), optimizer=rmsprop)

        return model, discriminate

    def create_posterior(self):
        states = Input(shape=[self.state_dim])
        actions = Input(shape=[self.action_dim])
        h = Concatenate(axis=1)([states, actions])
        h = Dense(128)(h)
        h = LeakyReLU()(h)
        h = Dense(128)(h)
        h = LeakyReLU()(h)

        # For flattened latent mode prediction
        latent_modes = Dense(self.encode_dim, activation='softmax')(h)

        # For (latent1, latent2) style prediction
        # latent_modes = Dense(encode_dim, activation='sigmoid',
        #                      init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h)
        # latent1 = Dense(1, activation='sigmoid', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h)
        # latent2 = Dense(1, activation='sigmoid', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h)
        # latent_modes = merge([latent1, latent2], mode='concat')

        model = Model(inputs=[states, actions], outputs=latent_modes)
        adam = Adam(lr=self.config.lr_posterior)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        return model

    def act(self, state, encodes):
        action_dis_mu = self.generator([state, encodes])

        # Pick the action by randomly sampling from the action_dis_mu distribution
        act = np.random.choice(self.action_dim, p=action_dis_mu.numpy()[0])

        act_one_hot = np.zeros((1, self.action_dim), dtype=np.float32)
        act_one_hot[0, act] = 1
        return act_one_hot

    def surr_loss(self, theta=None):
        if theta is None:
            model = self.generator
        else:
            model = self.tmp_model
            set_from_flat(self.tmp_model, theta)
        action_dist_discrete = model([self.states, self.encodes])  # The loss will be called from inside of GradientTape. Therefore action_dist needs to be inside or else use tape.watch()action_dist
        log_p_n = cat_log_prob(action_dist_discrete, self.actions)
        log_oldp_n = cat_log_prob(self.old_action_dist, self.actions)
        ratio_n = tf.exp(log_p_n - log_oldp_n)
        # -ve sign since we want to maximise; Add(subtract actually if surr is sent as -ve) self.ent_coeff * entropy
        surrogate_loss = -tf.reduce_mean(ratio_n * self.advants)
        return surrogate_loss

    def kl_constraint(self):
        N = tf.shape(self.states)[0]
        Nf = tf.cast(N, dtype)
        action_dist_discrete = self.generator([self.states, self.encodes])
        kl_constraint = selfKL_firstfixed(action_dist_discrete) / Nf
        return kl_constraint

    def take_trpo_step(self):
        config = self.config
        var_list = self.generator.trainable_weights

        # For Generator Loss
        # - Formula for surrogate loss function in TRPO. Need to exponentiate the log prob to get prob.
        # - Advantages have been computed earlier in the program which are discounted.
        # - compute ratio of current policy to the prev. iteration's policy.
        self.action_dist = self.generator([self.states, self.encodes])

        # TRPO Step 3: Find estimate of policy gradients i.e. g by differentiating surrogate advantage func.
        policy_gradient = flatgrad(loss_fn=self.surr_loss, var_list=var_list).numpy()

        # TRPO Step 4: Find the step direction i.e. X = inv(H) dot g using conjugate gradients that solves Hx=g
        def fisher_vector_product(p):
            """
            :param p: conjugate direction p_k to be used for computing A*p_k (flattened)
            :return:
            """
            # Compute the Second Gradient (of the product with p constant gives H*p)
            with tf.GradientTape() as tape:
                # Compute the H, Hessian which is the double derivative of KL_div(current policy || prev fixed policy)
                grad_kl = flatgrad(loss_fn=self.kl_constraint, var_list=var_list)

                # Computing the product between grad and tangents i.e first derivative of KL and p resp.
                kl_p = tf.reduce_sum(grad_kl*p)

            grad_grad_kl = tape.gradient(kl_p, var_list)
            flat_grad_grad_kl = tf.concat([tf.reshape(grad, [numel(v)])
                                           for (v, grad) in zip(var_list, grad_grad_kl)], axis=0).numpy()

            return flat_grad_grad_kl + p * config.cg_damping  # Damping used for stability

        # Negating policy_gradient again since it was computed as -ve of surr_loss but conj. grad requires Hx=g
        step_dir = conjugate_gradient(fisher_vector_product, -policy_gradient, cg_iters=config.cg_iters)

        # TRPO Step 5: Find the Approximate step-size (Delta_k)
        shs = .5 * step_dir.dot(fisher_vector_product(step_dir))  # (0.5*X^T*Hessian*X) where X is step_dir
        assert shs > 0
        lm = np.sqrt(shs / config.max_kl)  # 1 / sqrt( (2*delta) / (X^T*Hessian*X) )
        full_step = step_dir / lm  # Delta

        # TRPO Step 6[POLICY UPDATE]: Perform back-tracking line search with expo. decay to update param.
        # Flatten the vector
        theta_prev = get_flat(var_list).numpy()
        neg_g_dot_stepdir = -policy_gradient.dot(step_dir)
        success, theta = linesearch(self.surr_loss, theta_prev, full_step, neg_g_dot_stepdir / lm)
        if np.isnan(theta).any():
            print("NaN detected. Skipping update...")
        else:
            set_from_flat(self.generator, theta)

        # Compute the Generator losses based on current estimate of policy
        N = tf.shape(self.states)[0]
        Nf = tf.cast(N, dtype)
        action_dist_discrete = self.generator([self.states, self.encodes], training=False)
        surr_after = self.surr_loss()
        kl = KL_discrete(self.old_action_dist, action_dist_discrete) / Nf
        entropy = entropy_discrete(action_dist_discrete) / Nf
        return surr_after.numpy(), kl.numpy(), entropy.numpy()

    def compute_advant(self, path):
        """
        Args:
            path: Dictionary of states, encodes, action, rewards, action_dist
        In python3 and tensorflow 2.2+:-
        Computation is done in batches. model.predict method is designed for performance in
        large scale inputs. For small amount of inputs that fit in one batch,
        directly using `__call__` is recommended for faster execution, e.g.,
        `model(x)`, or `model(x, training=False)`

        """
        output_d = self.discriminate([path["states"], path["actions"]], training=False).numpy()
        output_p = self.posterior_target([path["states"], path["actions"]], training=False).numpy()

        # The following comes from E[D(s,a)+log(Q(c|s,a))] w.r.t the policy;
        # for Q(c|s,a), c is sampled from prior, that's why multiplication with path["encodes"]
        # The official InfoGAIL implementation although uses reward augmentation
        # but actually assigns np.ones to path['rewards'].
        path["rewards"] = output_d.flatten() + np.sum(np.log(output_p) * path["encodes"], axis=1)
        path["returns"] = discount(path["rewards"], self.config.gamma)

        # Compute the Advantage Function A(s,a,c) = Q(s,a,c)-V(s,c) <- baseline implements the V(s,c)
        path["baselines"] = self.baseline.predict(path)  # predict NN(state, encode) -> returns
        path_baselines = np.append(path["baselines"], 0 if path["obj_achieved"] else path["baselines"][-1])
        # For reaching the terminal state we append the value of terminal state which is 0 else we did not
        # complete the task because of early termination. The latter happens due to being stuck at the
        # current state, for which we append value of next state that is equal to value of current state.
        deltas = path["rewards"] + self.config.gamma * path_baselines[1:] - path_baselines[:-1]
        path["advants"] = discount(deltas, self.config.gamma * self.config.lam)
        return path

    def learn(self, demo, param_dir, exp_num, verbose=0):
        config = self.config
        num_ep_total = 0

        # Set up for training discriminator
        if verbose:
            print("Loading data ...")
        states_d, actions_d = demo["states"], demo["actions"]

        num_expert_demos = states_d.shape[0]  # Number of Demonstrations (state-action pairs)
        # if verbose:
        # print("Number of sample points from expert demos:", num_expert_demos)
        idx_d = np.arange(num_expert_demos)
        np.random.shuffle(idx_d)

        states_d = states_d[idx_d]
        actions_d = actions_d[idx_d]

        disc_loss, post_train_loss, post_val_loss, gen_surr_loss, gen_kl_div, gen_entropy = [], [], [], [], [], []
        baseline_train, baseline_val = [], []

        with tqdm(total=config.n_epochs, position=0, leave=True) as pbar:
            for i in range(0, config.n_epochs):
                if verbose:
                    print("\n********** Iteration {} **********".format(i))

                if i == 0:
                    paths_per_collect = 30
                else:
                    paths_per_collect = 10

                # #################################### #
                # #### STEP 1: Sample Trajectories ### #
                # #################################### #
                start_time = time.time()
                if verbose:
                    print("Rolling Out trajectories ...")
                paths = rollout_traj(self.env, self, self.state_dim, self.encode_dim,
                                     config.max_step_limit, paths_per_collect,
                                     self.discriminate, self.posterior, verbose)

                # No need to buffer the rollouts since in the given env. it is not expensive to sample them
                for path in paths:
                    self.buffer.add(path)
                paths = self.buffer.get_sample(config.sample_size)

                for path in paths:
                    action_dist = self.generator([path["states"], path["encodes"]])
                    path["action_dist"] = action_dist

                action_dist_n = np.concatenate([path["action_dist"] for path in paths])
                states_n = np.concatenate([path["states"] for path in paths])
                encodes_n = np.concatenate([path["encodes"] for path in paths])
                actions_n = np.concatenate([path["actions"] for path in paths])
                num_sampled_demos = states_n.shape[0]

                if verbose:
                    print("Epoch:", i, ", Total sampled data points:", num_sampled_demos)

                debug_time_taken(start_time, "Sample Trajectories")
                # #################################### #
                # #### STEP 2: Train discriminator ### #
                # #################################### #
                start_time = time.time()
                batch_size = config.batch_size

                # We will make sure all data points from expert trajectories are used by keeping a global index demo_idx
                start_d = self.demo_idx
                start_n = 0

                # Make sure Number of times Disc use data is more than number of times Gen does!
                # This is important for model trainings motivated by W-GAN
                for k in range(config.d_iter):

                    loss = self.discriminator.train_on_batch(
                        [states_n[start_n:start_n + batch_size],
                         actions_n[start_n:start_n + batch_size],
                         states_d[start_d:start_d + batch_size],
                         actions_d[start_d:start_d + batch_size]],
                        np.ones(batch_size)  # Gold Labels for y_pred = disc_pred_n (1) - disc_pred_d (0):-
                        # Disc predicts if action came from generated policy
                    )

                    # Weight Clipping between [-0.01, 0.01]
                    for layer in self.discriminator.layers:
                        weights = layer.get_weights()
                        weights = [np.clip(w, config.clamp_lower, config.clamp_upper) for w in weights]
                        layer.set_weights(weights)

                    start_d = self.demo_idx = self.demo_idx + batch_size
                    start_n = start_n + batch_size

                    if start_d + batch_size >= num_expert_demos:
                        start_d = self.demo_idx = (start_d + batch_size) % num_expert_demos
                        if start_d + batch_size >= num_expert_demos:
                            start_d = self.demo_idx = 0
                    if start_n + batch_size >= num_sampled_demos:
                        start_n = (start_n + batch_size) % num_sampled_demos
                        if start_n + batch_size >= num_sampled_demos:  # Even after we get over bound, reset start
                            start_n = 0

                    disc_loss.append(loss)
                    if verbose:
                        print("Discriminator step:", k, "loss:", loss)

                debug_time_taken(start_time, "Disc. Training")
                # #################################### #
                # ###### STEP 3: Train Posterior ##### #
                # #################################### #
                start_time = time.time()

                history_obj = self.posterior.fit(x=[states_n, actions_n], y=encodes_n, batch_size=batch_size,
                                                 epochs=5, validation_split=0.2, verbose=0, shuffle=True)
                # Train using posterior and predict on val data using posterior_target
                # (whose weight = 0.5 * its current weight +  0.5 * posterior's weight)
                posterior_weights = self.posterior.get_weights()
                posterior_target_weights = self.posterior_target.get_weights()
                for k in range(len(posterior_weights)):
                    posterior_target_weights[k] = 0.5 * posterior_weights[k] + \
                                                  0.5 * posterior_target_weights[k]
                self.posterior_target.set_weights(posterior_target_weights)
                post_train_loss.extend(history_obj.history['loss'])
                post_val_loss.extend(history_obj.history['val_loss'])

                debug_time_taken(start_time, "Posterior Training")
                # #################################### #
                # ###### STEP 4: Train Generator ##### #
                # #################################### #
                start_time = time.time()
                """
                We take policy steps using TRPO and update the generator param
                """
                # TRPO Step 2: Compute Return and use current estimate of value function to approx. advantage function
                paths = [self.compute_advant(path) for path in paths]

                # Standardize the advantage function to have mean=0 and std=1
                advants_n = np.concatenate([path["advants"] for path in paths])
                advants_n -= advants_n.mean()
                advants_n /= (advants_n.std() + 1e-8)
                # advants_n = (advants_n - advants_n.mean()) / (advants_n.std() + 1e-8)

                # TRPO Step 3: Fit value function by regression on MSE between Value func at current step and Return
                nn_baseline_history = self.baseline.fit(paths)
                baseline_train.extend(nn_baseline_history['loss'])
                baseline_val.extend(nn_baseline_history['val_loss'])

                # Update the variable values
                self.states, self.encodes, self.actions, self.advants, self.old_action_dist = states_n, encodes_n, \
                                                                                              actions_n, advants_n, \
                                                                                              action_dist_n
                # debug_time_taken(start_time, "Before TRPO")
                surr_after, kl_old_new, entropy = self.take_trpo_step()

                gen_surr_loss.append(surr_after)
                gen_kl_div.append(kl_old_new)
                gen_entropy.append(entropy)

                debug_time_taken(start_time, "Generator Training")
                # #################################### #
                #  ###### STEP 5: Book Keeping ##### #
                # #################################### #
                episode_rewards = np.array([path["rewards"].sum() for path in paths])
                stats = {}
                num_ep_total += len(episode_rewards)
                stats["Total number of episodes (cumulative)"] = num_ep_total
                stats["Average sum of rewards per episode"] = episode_rewards.mean()
                stats["Entropy"] = entropy
                stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
                stats["KL between old and new distribution"] = kl_old_new
                stats["Surrogate loss"] = surr_after

                if verbose:
                    for k, v in stats.items():
                        print(k + ": " + " " * (40 - len(k)) + str(v))

                if verbose:
                    print("Now we save model")
                self.generator.save_weights(os.path.join(param_dir, "generator_model_%d.h5" % i), overwrite=True)
                with open(os.path.join(param_dir, "generator_model_%d.json" % i), "w") as outfile:
                    json.dump(self.generator.to_json(), outfile)

                self.discriminator.save_weights(os.path.join(param_dir, "discriminator_model_%d.h5" % i),
                                                overwrite=True)
                with open(os.path.join(param_dir, "discriminator_model_%d.json" % i), "w") as outfile:
                    json.dump(self.discriminator.to_json(), outfile)

                self.baseline.model.save_weights(os.path.join(param_dir, "baseline_model_%d.h5" % i),
                                                 overwrite=True)
                with open(os.path.join(param_dir, "baseline_model_%d.json" % i), "w") as outfile:
                    json.dump(self.baseline.model.to_json(), outfile)

                self.posterior.save_weights(os.path.join(param_dir, "posterior_model_%d.h5" % i), overwrite=True)
                with open(os.path.join(param_dir, "posterior_model_%d.json" % i), "w") as outfile:
                    json.dump(self.posterior.to_json(), outfile)

                self.posterior_target.save_weights(os.path.join(param_dir, "posterior_target_model_%d.h5" % i),
                                                   overwrite=True)
                with open(os.path.join(param_dir, "posterior_target_model_%d.json" % i), "w") as outfile:
                    json.dump(self.posterior_target.to_json(), outfile)
                if verbose:
                    print("***********************************".format(i))

                pbar.refresh()
                pbar.set_description("Epoch {}".format(i + 1))
                pbar.set_postfix(sampled_points=num_sampled_demos, Entropy=entropy, Surrogate_loss=surr_after,
                                 KL_old_new=kl_old_new)
                pbar.update(1)

        plot_infoGAIL_loss(disc_loss, gen_surr_loss, post_train_loss, post_val_loss, baseline_train, baseline_val,
                           fig_path, exp_num)


class NNBaseline(object):

    def __init__(self, state_dim, encode_dim, lr_baseline, b_iter, batch_size):
        # print "Now we build baseline"
        self.model = self.create_net(state_dim, encode_dim, lr_baseline)
        # self.sess = sess
        self.b_iter = b_iter
        self.epochs = 2
        self.batch_size = batch_size
        self.first_time = True
        self.mixfrac = 0.1

    def create_net(self, state_dim, encode_dim, lr_baseline):

        state = Input(shape=[state_dim])
        x = Dense(128)(state)
        x = LeakyReLU()(x)
        x = Dense(128)(x)
        x = LeakyReLU()(x)

        encodes = Input(shape=[encode_dim])
        c = Dense(128)(encodes)
        h = Add()([x, c])
        h = LeakyReLU()(h)
        p = Dense(1)(h)  # indicates the expected accumulated future rewards.

        model = Model(inputs=[state, encodes], outputs=p)
        adam = Adam(lr=lr_baseline)
        model.compile(loss='mse', optimizer=adam)
        return model

    def fit(self, paths):
        feats = np.concatenate([path["states"] for path in paths])
        # auxs = np.concatenate([self._get_aux(path) for path in paths])
        encodes = np.concatenate([path["encodes"] for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])

        if self.first_time:
            self.first_time = False
            b_iter = 100
            epochs = 5
        else:
            returns_old = np.concatenate([self.predict(path) for path in paths])
            returns = returns * self.mixfrac + returns_old * (1 - self.mixfrac)
            b_iter = self.b_iter
            epochs = self.epochs

        history_obj = self.model.fit(x=[feats, encodes], y=returns, batch_size=self.batch_size, epochs=epochs,
                                     validation_split=0.2, verbose=0, shuffle=True)

        return history_obj.history

    def predict(self, path):
        if self.first_time:
            return np.zeros(pathlength(path))
        else:
            acc_return = self.model([path["states"], path["encodes"]], training=False)
        return np.reshape(acc_return, (acc_return.shape[0],))
