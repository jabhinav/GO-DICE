import os.path
import json
from utils import *
from config import fig_path
from tqdm import tqdm

"""
Note that the original implementation assumes a gaussian distribution for the policy network whose predicted actions
(continuous) assumes the role of the mean of gaussian. Std. deviation is fixed manually. Thus we obtain the prob. 
distribution over continuous actions predicted by the policy
"""


class Agent(object):

    def __init__(self, env, sess, state_dim, action_dim, encode_dim, config):
        """
        :param state_dim: Feature dimension i.e. 2 (in a 2D grid-world, specifying position)
        :param action_dim: Action dimension i.e. 5
        :param encode_dim: Latent code dimension i.e. 4 one-hot encoded of repr. {[0, 0], [0, 1], [1, 0], [1, 1]}
        :param
        """
        self.config = config
        self.env = env
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.encode_dim = encode_dim
        print "State Dim:", state_dim, "Action Dim:", action_dim, "Encode Dim:", encode_dim

        # Create tensors for the inputs to our network
        self.states = states = tf.placeholder(dtype, shape=[None, state_dim])
        self.encodes = encodes = tf.placeholder(dtype, shape=[None, encode_dim])
        self.actions = actions = tf.placeholder(dtype, shape=[None, action_dim])
        self.advants = advants = tf.placeholder(dtype, shape=[None])
        self.oldaction_dist_discrete = oldaction_dist_discrete = tf.placeholder(dtype, shape=[None, action_dim])

        # Create NN
        # print "Now we build generator"
        self.generator = self.create_generator(states, encodes)
        # print self.generator.summary()

        # print "Now we build discriminator"
        self.discriminator, self.discriminate = self.create_discriminator(state_dim, action_dim)
        # print self.discriminator.summary()
        # print "Now we build posterior"
        self.posterior = self.create_posterior(state_dim, action_dim, encode_dim)
        # print self.posterior.summary()
        self.posterior_target = self.create_posterior(state_dim, action_dim, encode_dim)
        # print self.posterior_target.summary()

        self.demo_idx = 0

        action_dist_discrete = self.generator.outputs[0]
        eps = 1e-8
        self.action_dist_discrete = action_dist_discrete
        N = tf.shape(states)[0]

        self.buffer = ReplayBuffer(self.config.buffer_size)

        # For Generator Loss
        # compute ratio of current policy to the prev. iteration's policy
        log_p_n = cat_log_prob(action_dist_discrete, actions)
        log_oldp_n = cat_log_prob(oldaction_dist_discrete, actions)
        ratio_n = tf.exp(log_p_n - log_oldp_n)
        # Formula for surrogate loss function in TRPO. Need to exponentiate the log prob to get prob.
        # Advantages have been computed earlier in the program which are discounted
        Nf = tf.cast(N, dtype)
        surr = -tf.reduce_mean(ratio_n * advants)

        var_list = self.generator.trainable_weights

        kl = KL_discrete(oldaction_dist_discrete, action_dist_discrete) / Nf
        ent = entropy_discrete(action_dist_discrete) / Nf

        self.losses = [surr, kl, ent]

        # Compute gradient of surrogate loss function
        self.policy_grad = flatgrad(loss=surr, var_list=var_list)

        self.flat_tangent = tf.placeholder(dtype, shape=[None])

        # Compute the H, Hessian which is the double derivative of KL_div(current policy || prev fixed policy)
        kl_constraint = selfKL_firstfixed(action_dist_discrete) / Nf
        # Compute the First Gradient
        grad_kl = tf.gradients(kl_constraint, var_list)

        # Split the tangent (or conj. direction i.e. p) params to each layer in the Generator network
        start = 0
        tangents = []
        shapes = map(var_shape, var_list)  # Get the shape of parameters in every generator layer eg: (128,2), (128, 4)
        for shape in shapes:
            layer_size = np.prod(shape)  # size will be number of params in each layer i.e. 128*2=256)
            param = tf.reshape(self.flat_tangent[start:(start + layer_size)], shape)
            tangents.append(param)
            start += layer_size
        # Computing the product between grad and tangents i.e first derivative of KL and p resp.
        kl_p = [tf.reduce_sum(g * t) for (g, t) in zip(grad_kl, tangents)]
        # Second Gradient of the product with p constant gives H*p
        self.fvp = flatgrad(loss=kl_p, var_list=var_list)  # Second Gradient of the product with p constant gives H*p

        self.gf = GetFlat(self.sess, var_list)
        self.sff = SetFromFlat(self.sess, var_list)

        self.baseline = NNBaseline(sess, state_dim, encode_dim, self.config.lr_baseline, self.config.b_iter,
                                   self.config.batch_size)

        self.sess.run(tf.global_variables_initializer())

    def create_generator(self, state, encodes):
        state = Input(tensor=state)
        x = Dense(128)(state)
        x = LeakyReLU()(x)
        x = Dense(128)(x)
        x = LeakyReLU()(x)

        encodes = Input(tensor=encodes)
        c = Dense(128)(encodes)
        h = merge([x, c], mode='sum')
        h = LeakyReLU()(h)

        # TODO: Check
        actions = Dense(self.action_dim, activation='softmax')(h)
        model = Model(input=[state, encodes], output=actions)
        return model

    def create_discriminator(self, state_action_dim, action_dim):
        states = Input(shape=[state_action_dim])
        actions = Input(shape=[action_dim])
        h = merge([states, actions], mode='concat')
        h = Dense(128)(h)
        h = LeakyReLU()(h)
        h = Dense(128)(h)
        h = LeakyReLU()(h)
        p = Dense(1)(h)
        discriminate = Model(input=[states, actions], output=p)

        states_n = Input(shape=[state_action_dim])
        actions_n = Input(shape=[action_dim])
        states_d = Input(shape=[state_action_dim])
        actions_d = Input(shape=[action_dim])
        p_n = discriminate([states_n, actions_n])
        p_d = discriminate([states_d, actions_d])
        p_d = Lambda(lambda x: -x)(p_d)
        p_output = merge([p_n, p_d], mode='sum')
        model = Model(input=[states_n, actions_n, states_d, actions_d], output=p_output)

        rmsprop = RMSprop(lr=self.config.lr_discriminator)
        model.compile(
            # little trick to use Keras predefined lambda loss function
            loss=lambda y_pred, p_true: K.mean(y_pred * p_true), optimizer=rmsprop
        )

        return model, discriminate

    def create_posterior(self, disc_ip_dim, action_dim, encode_dim):
        states = Input(shape=[disc_ip_dim])
        actions = Input(shape=[action_dim])
        h = merge([states, actions], mode='concat')
        h = Dense(128)(h)
        h = LeakyReLU()(h)
        h = Dense(128)(h)
        h = LeakyReLU()(h)

        # For flattened latent mode prediction
        latent_modes = Dense(encode_dim, activation='softmax')(h)

        # For (latent1, latent2) style prediction
        # latent_modes = Dense(encode_dim, activation='sigmoid',
        #                      init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h)
        # latent1 = Dense(1, activation='sigmoid', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h)
        # latent2 = Dense(1, activation='sigmoid', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h)
        # latent_modes = merge([latent1, latent2], mode='concat')

        model = Model(input=[states, actions], output=latent_modes)
        adam = Adam(lr=self.config.lr_posterior)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        return model

    def act(self, state, encodes):
        action_dis_mu = self.sess.run(self.action_dist_discrete, {self.states: state, self.encodes: encodes})

        # Pick the action by randomly sampling from the action_dis_mu distribution
        act = np.random.choice(self.action_dim, p=action_dis_mu[0])

        act_one_hot = np.zeros((1, self.action_dim), dtype=np.float32)
        act_one_hot[0, act] = 1
        return act_one_hot

    def learn(self, demo, param_dir, exp_num, verbose=0):
        config = self.config
        start_time = time.time()
        num_ep_total = 0

        # Set up for training discriminator
        if verbose:
            print "Loading data ..."
        states_d, actions_d = demo["states"], demo["actions"]

        num_expert_demos = states_d.shape[0]  # Number of Demonstrations (state-action pairs)
        # if verbose:
        # print "Number of sample points from expert demos:", num_expert_demos
        idx_d = np.arange(num_expert_demos)
        np.random.shuffle(idx_d)

        states_d = states_d[idx_d]
        actions_d = actions_d[idx_d]

        disc_loss = []
        post_train_loss = []
        post_val_loss = []
        gen_surr_loss = []
        gen_kl_div = []
        gen_entropy = []

        with tqdm(total=config.n_epochs, position=0, leave=True) as pbar:
            for i in xrange(0, config.n_epochs):
                if verbose:
                    print("\n********** Iteration {} **********".format(i))

                # if i == 99:
                #     print "Last Loop"

                if i == 0:
                    paths_per_collect = 30
                else:
                    paths_per_collect = 10

                # #################################### #
                # #### STEP 1: Sample Trajectories ### #
                # #################################### #
                if verbose:
                    print "Rolling Out trajectories ..."
                paths = rollout_traj(self.env, self, self.state_dim, self.encode_dim,
                                     config.max_step_limit, paths_per_collect,
                                     self.discriminate, self.posterior, verbose)

                # No need to buffer the rollouts since in the given env. it is not expensive to sample them
                for path in paths:
                    self.buffer.add(path)
                paths = self.buffer.get_sample(config.sample_size)

                for path in paths:
                    path["action_dist"] = self.sess.run(
                        self.action_dist_discrete, {self.states: path["states"], self.encodes: path["encodes"]}
                    )

                action_dist_n = np.concatenate([path["action_dist"] for path in paths])
                states_n = np.concatenate([path["states"] for path in paths])
                encodes_n = np.concatenate([path["encodes"] for path in paths])
                actions_n = np.concatenate([path["actions"] for path in paths])
                num_sampled_demos = states_n.shape[0]

                if verbose:
                    print "Epoch:", i, ", Total sampled data points:", num_sampled_demos

                # #################################### #
                # #### STEP 2: Train discriminator ### #
                # #################################### #
                batch_size = config.batch_size
                # if num_sampled_demos < batch_size:
                #     print "Skipping Epoch, since sampled data points {} from traj are less than reqd. batch size {}".\
                #         format(num_sampled_demos, batch_size)
                #     continue

                # We will make sure all data points from expert trajectories are used by keeping a global index demo_idx
                start_d = self.demo_idx
                start_n = 0

                # Set number of iterations for training the discriminator
                # if i <= 5:
                #     d_iter = 120 - i * 20
                # else:
                #     d_iter = 10
                for k in xrange(config.d_iter):

                    # try:
                    #     n_samples = states_n[start_n:start_n + batch_size].shape[0]
                    #     d_samples = states_d[start_d:start_d + batch_size].shape[0]
                    #     assert n_samples == batch_size and d_samples == batch_size
                    # except AssertionError:
                    #     print "Skipping Discriminator training, since sampled data points {} from traj are " \
                    #           "not equal to those from expert demo {}".format(
                    #         n_samples, d_samples)
                    #     disc_skipped = True
                    #     break
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
                        print "Discriminator step:", k, "loss:", loss

                # #################################### #
                # ###### STEP 3: Train Posterior ##### #
                # #################################### #

                idx = np.arange(num_sampled_demos)
                np.random.shuffle(idx)
                train_val_ratio = 0.8

                # Training data for posterior
                numno_train = int(num_sampled_demos * train_val_ratio)
                states_train = states_n[idx][:numno_train]
                actions_train = actions_n[idx][:numno_train]
                encodes_train = encodes_n[idx][:numno_train]

                # Validation data for posterior
                states_val = states_n[idx][numno_train:]
                actions_val = actions_n[idx][numno_train:]
                encodes_val = encodes_n[idx][numno_train:]

                # Train using posterior and predict on val data using posterior_target
                # (whose weight = 0.5 * its current weight +  0.5 * posterior's weight)
                start_n = 0
                for j in xrange(config.p_iter):
                    loss = self.posterior.train_on_batch(
                        [states_train[start_n:start_n + batch_size], actions_train[start_n:start_n + batch_size]],
                        encodes_train[start_n:start_n + batch_size]
                    )
                    start_n += batch_size
                    if start_n + batch_size >= numno_train:
                        start_n = (start_n + batch_size) % numno_train
                        if start_n + batch_size >= numno_train:
                            start_n = 0

                    posterior_weights = self.posterior.get_weights()
                    posterior_target_weights = self.posterior_target.get_weights()
                    for k in xrange(len(posterior_weights)):
                        posterior_target_weights[k] = 0.5 * posterior_weights[k] + \
                                                      0.5 * posterior_target_weights[k]
                    self.posterior_target.set_weights(posterior_target_weights)

                    output_p = self.posterior_target.predict([states_val, actions_val])
                    val_loss = -np.average(np.sum(np.log(output_p) * encodes_val, axis=1))
                    post_train_loss.append(loss)
                    post_val_loss.append(val_loss)
                    if verbose:
                        print "Posterior step:", j, "Val Loss:", loss, val_loss

                # #################################### #
                # ###### STEP 4: Train Generator ##### #
                # #################################### #
                """
                We take policy steps using TRPO and update the generator param
                """
                # TRPO Step 2: Compute Return and use current estomate of value function to approx. advantage function
                path_idx = 0
                for path in paths:
                    # file_path = log_dir + "/iter_%d_path_%d.txt" % (i, path_idx)
                    # f = open(file_path, "w")

                    # Compute accumulated return i.e. compute reward and discount it using gamma
                    output_d = self.discriminate.predict([path["states"], path["actions"]])
                    output_p = self.posterior_target.predict([path["states"], path["actions"]])

                    # The following comes from E[D(s,a)+log(Q(c|s,a))] w.r.t the policy;
                    # for Q(c|s,a), c is sampled from prior, that's why multiplication with path["encodes"]
                    # TODO: try with different weights to Disc and Post outputs to check convergence
                    # The official InfoGAIL implementation although uses reward augmentation
                    # but actually assigns np.ones to path['rewards'].
                    # TODO: Check if we actually have some rewards per (s,a) tuple
                    path["rewards"] = output_d.flatten() + np.sum(np.log(output_p) * path["encodes"], axis=1)
                    path["returns"] = discount(path["rewards"], config.gamma)

                    # Compute the Advantage Function A(s,a,c) = Q(s,a,c)-V(s,c) <- baseline implements the V(s,c)
                    path["baselines"] = self.baseline.predict(path)  # predict NN(state, encode) -> returns
                    path_baselines = np.append(path["baselines"], 0 if path["obj_achieved"] else path["baselines"][-1])
                    # For reaching the terminal state we append the value of terminal state which is 0 else we did not
                    # complete the task because of early termination. The latter happens due to being stuck at the
                    # current state, for which we append value of next state that is equal to value of current state.
                    deltas = path["rewards"] + config.gamma * path_baselines[1:] - path_baselines[:-1]
                    path["advants"] = discount(deltas, config.gamma * config.lam)

                    # f.write("Baseline:\n" + np.array_str(path_baselines) + "\n")
                    # f.write("Returns:\n" + np.array_str(path["returns"]) + "\n")
                    # f.write("Advants:\n" + np.array_str(path["advants"]) + "\n")
                    # f.write("Action_dist:\n" + np.array_str(path["action_dist"]) + "\n")
                    # f.write("Actions:\n" + np.array_str(path["actions"]) + "\n")
                    # f.write("Logstds:\n" + np.array_str(path["logstds"]) + "\n")
                    path_idx += 1

                # Standardize the advantage function to have mean=0 and std=1
                advants_n = np.concatenate([path["advants"] for path in paths])
                # advants_n -= advants_n.mean()
                advants_n /= (advants_n.std() + 1e-8)

                # TRPO Step 3: Fit value function by regression on MSE between Value func at current step and Return
                self.baseline.fit(paths)

                feed = {self.states: states_n,
                        self.encodes: encodes_n,
                        self.actions: actions_n,
                        self.advants: advants_n,
                        self.oldaction_dist_discrete: action_dist_n,
                        }

                # Flatten the vector
                theta_prev = self.gf()

                def fisher_vector_product(p):
                    feed[self.flat_tangent] = p  # Update the conjugate direction p_k to be used for computing A*p_k
                    flat_grad_grad_kl = self.sess.run(self.fvp, feed_dict=feed)
                    return flat_grad_grad_kl + p * config.cg_damping  # Damping used for stability

                """ Goal is to find step-size and direction using TRPO
                    Ref: https://www.telesens.co/2018/06/09/efficiently-computing-the-fisher-vector-product-in-trpo/ 
                """
                # TRPO Step 3: Find estimate of policy gradients i.e. g by differentiating surrogate advantage func.
                loss_grad = self.sess.run(self.policy_grad, feed_dict=feed)

                # TRPO Step 4: Find the step direction i.e. X = inv(H) dot g using conjugate gradients that solves Hx=g
                step_dir = conjugate_gradient(fisher_vector_product, -loss_grad, cg_iters=config.cg_iters)

                # TRPO Step 5: Find the Approximate step-size (Delta_k)
                shs = .5 * step_dir.dot(fisher_vector_product(step_dir))  # (0.5*X^T*Hessian*X) where X is step_dir
                try:
                    assert shs > 0
                    lm = np.sqrt(shs / config.max_kl)  # 1 / sqrt( (2*delta) / (X^T*Hessian*X) )
                    full_step = step_dir / lm  # Delta
                    neg_g_dot_stepdir = -loss_grad.dot(step_dir)

                    def loss(th):
                        self.sff(th)
                        return self.sess.run(self.losses[0], feed_dict=feed)

                    # TRPO Step 6 [POLICY UPDATE]: Perform back-tracking line search with exponential decay to update param.
                    success, theta = linesearch(loss, theta_prev, full_step, neg_g_dot_stepdir / lm)
                    self.sff(theta)

                    # Compute the Generator losses based on current estimate of policy
                    surr_after, kl_old_new, entropy = self.sess.run(self.losses, feed_dict=feed)
                    gen_surr_loss.append(surr_after)
                    gen_kl_div.append(kl_old_new)
                    gen_entropy.append(entropy)

                    episode_rewards = np.array([path["rewards"].sum() for path in paths])
                    stats = {}
                    num_ep_total += len(episode_rewards)
                    stats["Total number of episodes (cumulative)"] = num_ep_total
                    stats["Average sum of rewards per episode"] = episode_rewards.mean()
                    stats["Entropy"] = entropy
                    stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
                    stats["KL between old and new distribution"] = kl_old_new
                    stats["Surrogate loss"] = surr_after
                    # stats["shs"] = shs

                    if verbose:
                        for k, v in stats.iteritems():
                            print(k + ": " + " " * (40 - len(k)) + str(v))
                        # if entropy != entropy:
                        #     exit(-1)

                    # Save the model
                    if verbose:
                        print "Now we save model"
                    self.generator.save_weights(os.path.join(param_dir, "generator_model_%d.h5" % i), overwrite=True)
                    with open(os.path.join(param_dir, "generator_model_%d.json" % i), "w") as outfile:
                        json.dump(self.generator.to_json(), outfile)

                    self.discriminator.save_weights(os.path.join(param_dir, "discriminator_model_%d.h5" % i), overwrite=True)
                    with open(os.path.join(param_dir, "discriminator_model_%d.json" % i), "w") as outfile:
                        json.dump(self.discriminator.to_json(), outfile)

                    self.baseline.model.save_weights(os.path.join(param_dir, "baseline_model_%d.h5" % i), overwrite=True)
                    with open(os.path.join(param_dir, "baseline_model_%d.json" % i), "w") as outfile:
                        json.dump(self.baseline.model.to_json(), outfile)

                    self.posterior.save_weights(os.path.join(param_dir, "posterior_model_%d.h5" % i), overwrite=True)
                    with open(os.path.join(param_dir, "posterior_model_%d.json" % i), "w") as outfile:
                        json.dump(self.posterior.to_json(), outfile)

                    self.posterior_target.save_weights(os.path.join(param_dir, "posterior_target_model_%d.h5" % i), overwrite=True)
                    with open(os.path.join(param_dir, "posterior_target_model_%d.json" % i), "w") as outfile:
                        json.dump(self.posterior_target.to_json(), outfile)
                    if verbose:
                        print("***********************************".format(i))
                except AssertionError:
                    plot_infoGAIL_loss(disc_loss, gen_surr_loss, fig_path + "_{}.png".format(exp_num))
                    print "shs value is less than 0: ", shs, ". Exiting..."
                    exit(-1)

                pbar.refresh()
                pbar.set_description("Epoch {}".format(i + 1))
                pbar.set_postfix(sampled_points=num_sampled_demos, Entropy=entropy, Surrogate_loss=surr_after, KL_old_new=kl_old_new)
                pbar.update(1)

        plot_infoGAIL_loss(disc_loss, gen_surr_loss, fig_path + "_{}.png".format(exp_num))


class Generator(object):
    def __init__(self, sess, state_dim, encode_dim, action_dim):
        self.sess = sess
        self.lr = tf.placeholder(tf.float32, shape=[])

        K.set_session(sess)

        self.model, self.weights, self.state, self.encodes = self.create_generator(state_dim, encode_dim, action_dim)

        self.action_gradient = tf.placeholder(tf.float32, [None, action_dim])
        self.params_grad = tf.gradients(self.model.output, self.weights, self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)

        self.sess.run(tf.global_variables_initializer())

    def train(self, state, encodes, action_grads, lr):
        self.sess.run(self.optimize, feed_dict={
            self.state: state,
            self.encodes: encodes,
            self.lr: lr,
            self.action_gradient: action_grads,
            K.learning_phase(): 1
        })

    @staticmethod
    def create_generator(state_dim, encode_dim, action_dim):
        state = Input(shape=[state_dim])
        x = Dense(128)(state)
        x = LeakyReLU()(x)
        x = Dense(128)(x)
        x = LeakyReLU()(x)

        encodes = Input(shape=[encode_dim])
        c = Dense(128)(encodes)
        h = merge([x, c], mode='sum')
        h = LeakyReLU()(h)

        actions = Dense(action_dim, activation='softmax')(h)
        model = Model(input=[state, encodes], output=actions)
        return model, model.trainable_weights, state, encodes
