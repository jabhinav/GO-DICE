import os.path
import json
from utils import *
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import pickle as pkl
import os
import sys
from eval import pred_kl_div, get_data
from domains import RoadWorld, BoxWorld, TeamBoxWorld
from config import InfoGAIL_config as model_config
from config import state_action_path, latent_path, env_name, seed_values, supervision_setting, \
    latent_setting, param_dir, kl_figpath
from config import env_roadWorld_config, env_boxWorld_config, training_data_config, env_teamBoxWorld_config
from config import param_dir as p_dir

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


class RoadWorldResponse(object):
    def __init__(self, env, terminal_states):
        self.env = env
        self.stuck_count = 0
        self.terminal_states = terminal_states

    def update_stuck_count(self):
        self.stuck_count += 1

    def gen_response(self, curr_state, action, next_state):
        terminate = False
        reward = 0

        if next_state in self.terminal_states:
            terminate = True

        elif next_state == curr_state:
            neighs, out_of_space = self.env.neighbors(self.env.inv_states_dict[curr_state])
            # Terminate if action takes agent out of the grid
            if action in out_of_space.keys():
                terminate = True
                reward = -100

            self.update_stuck_count()
            # If the agent is stuck in a position for too long
            if self.stuck_count > model_config.stuck_count_limit:
                terminate = True
                reward = -100

        return terminate, reward


def rollout_traj_gail(env, agent, state_dim, max_step_limit, paths_per_collect, verbose=1):
    paths = []
    for p in xrange(paths_per_collect):
        if verbose:
            print "Rollout index:", p

        states, actions, raws = [], [], []

        state_idx, terminal_states_idx = env.get_initial_state()
        env_resp = RoadWorldResponse(env, terminal_states_idx)

        if not state_idx:
            print "No starting index. Skipping path"
            continue
        state = np.array(env.inv_states_dict[state_idx])
        state = np.expand_dims(state, axis=0)

        for i in range(max_step_limit):

            states.append(state)

            # Take action
            action = agent.act(state)  # Action here is not a prob dis. but the one-hot encoding of taken action
            actions.append(action)

            # reward_d += discriminate.predict([state, action])[0, 0] * 0.1
            # reward_p += np.sum(np.log(posterior.predict([state, action]))*encode)

            # Get action id
            action_id = np.where(action[0] == 1.0)[0][0]

            # Compute the next state using environment's transition dynamics
            current_state_idx = env.states_dict[tuple(state[0])]
            next_state_prob_dist = env.obstate_transition_matrix[current_state_idx, action_id, :]
            next_state_idx = np.random.choice(np.array(range(len(env.states_dict))), p=next_state_prob_dist)
            next_state = env.inv_states_dict[next_state_idx]
            next_state = np.expand_dims(np.array(next_state), axis=0)
            if verbose:
                print state[0], "->", env.inv_action_dict[action_id]

            terminate, reward = env_resp.gen_response(current_state_idx, action_id, next_state_idx)
            raws.append(reward)
            # Stop if number of steps overload or if we have reached the terminal state
            if terminate or i+1 == max_step_limit:
                path = dict2(
                    states=np.concatenate(states),
                    actions=np.concatenate(actions),
                    raws=np.array(raws),
                    obj_achieved=True if i+1 < max_step_limit and reward == 0 else False
                    )
                paths.append(path)
                break
            state = next_state

        # # Update the encode axis for the next path
        # encode_axis = (encode_axis + 1) % encode_dim
    return paths


class NNBaseline_GAIL(object):

    def __init__(self, sess, state_dim, lr_baseline, b_iter, batch_size):
        # print "Now we build baseline"
        self.model = self.create_net(state_dim, lr_baseline)
        self.sess = sess
        self.b_iter = b_iter
        self.batch_size = batch_size
        self.first_time = True
        self.mixfrac = 0.1

    def create_net(self, state_dim, lr_baseline):

        state = Input(shape=[state_dim])
        x = Dense(128)(state)
        x = LeakyReLU()(x)
        x = Dense(128)(x)
        h = LeakyReLU()(x)
        p = Dense(1)(h)  # indicates the expected accumulated future rewards.

        model = Model(input=[state], output=p)
        adam = Adam(lr=lr_baseline)
        model.compile(loss='mse', optimizer=adam)
        return model

    def fit(self, paths):
        feats = np.concatenate([path["states"] for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])

        if self.first_time:
            self.first_time = False
            b_iter = 100
        else:
            returns_old = np.concatenate([self.predict(path) for path in paths])
            returns = returns * self.mixfrac + returns_old * (1 - self.mixfrac)
            b_iter = self.b_iter

        num_data = feats.shape[0]
        idx = np.arange(num_data)
        np.random.shuffle(idx)
        train_val_ratio = 0.7
        num_train = int(num_data * train_val_ratio)
        feats_train = feats[idx][:num_train]
        returns_train = returns[idx][:num_train]

        feats_val = feats[idx][num_train:]
        returns_val = returns[idx][num_train:]

        start = 0
        batch_size = self.batch_size
        for i in xrange(b_iter):
            loss = self.model.train_on_batch(
                [feats_train[start:start + batch_size]],
                returns_train[start:start + batch_size]
            )
            start += batch_size
            if start >= num_train:
                start = (start + batch_size) % num_train
            val_loss = np.average(np.square(self.model.predict(
                [feats_val]).flatten() - returns_val))
            # print "Baseline step:", i, "loss:", loss, "val:", val_loss

    def predict(self, path):
        if self.first_time:
            return np.zeros(pathlength(path))
        else:
            acc_return = self.model.predict(
                [path["states"]])
        return np.reshape(acc_return, (acc_return.shape[0], ))


class Agent(object):

    def __init__(self, _env, sess, state_dim, action_dim, config=None):
        self.config = config
        self.env = _env
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Create tensors for the inputs to our network
        self.states = states = tf.placeholder(dtype, shape=[None, state_dim])
        self.actions = actions = tf.placeholder(dtype, shape=[None, action_dim])
        self.advants = advants = tf.placeholder(dtype, shape=[None])
        self.oldaction_dist_discrete = oldaction_dist_discrete = tf.placeholder(dtype, shape=[None, action_dim])

        self.generator = self.create_generator(states)
        self.discriminator, self.discriminate = self.create_discriminator(state_dim, action_dim)

        self.demo_idx = 0

        action_dist_discrete = self.generator.outputs[0]
        self.action_dist_discrete = action_dist_discrete
        N = tf.shape(states)[0]

        self.buffer = ReplayBuffer(self.config.buffer_size)

        log_p_n = cat_log_prob(action_dist_discrete, actions)
        log_oldp_n = cat_log_prob(oldaction_dist_discrete, actions)
        ratio_n = tf.exp(log_p_n - log_oldp_n)
        Nf = tf.cast(N, dtype)
        surr = -tf.reduce_mean(ratio_n * advants)

        var_list = self.generator.trainable_weights

        kl = KL_discrete(oldaction_dist_discrete, action_dist_discrete) / Nf
        ent = entropy_discrete(action_dist_discrete) / Nf

        self.losses = [surr, kl, ent]

        self.policy_grad = flatgrad(loss=surr, var_list=var_list)
        self.flat_tangent = tf.placeholder(dtype, shape=[None])

        kl_constraint = selfKL_firstfixed(action_dist_discrete) / Nf
        grad_kl = tf.gradients(kl_constraint, var_list)

        start = 0
        tangents = []
        shapes = map(var_shape, var_list)
        for shape in shapes:
            layer_size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + layer_size)], shape)
            tangents.append(param)
            start += layer_size
        kl_p = [tf.reduce_sum(g * t) for (g, t) in zip(grad_kl, tangents)]
        self.fvp = flatgrad(loss=kl_p, var_list=var_list)

        self.gf = GetFlat(self.sess, var_list)
        self.sff = SetFromFlat(self.sess, var_list)

        self.baseline = NNBaseline_GAIL(sess, state_dim, self.config.lr_baseline,
                                        self.config.b_iter, self.config.batch_size)

        self.sess.run(tf.global_variables_initializer())

    def create_generator(self, state):
        state = Input(tensor=state)
        x = Dense(128)(state)
        x = LeakyReLU()(x)
        x = Dense(128)(x)
        h = LeakyReLU()(x)

        # TODO: Check
        actions = Dense(self.action_dim, activation='softmax')(h)
        model = Model(input=[state], output=actions)
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

    def act(self, state):
        action_dis_mu = self.sess.run(self.action_dist_discrete, {self.states: state})

        # Pick the action by randomly sampling from the action_dis_mu distribution
        act = np.random.choice(self.action_dim, p=action_dis_mu[0])

        act_one_hot = np.zeros((1, self.action_dim), dtype=np.float32)
        act_one_hot[0, act] = 1
        return act_one_hot

    def learn(self, demo, save_param_dir, exp_num, verbose=0):
        config = self.config
        num_ep_total = 0

        # Set up for training discriminator
        if verbose:
            print "Loading data ..."
        states_d, actions_d = demo["states"], demo["actions"]

        num_expert_demos = states_d.shape[0]  # Number of Demonstrations (state-action pairs)
        if verbose:
            print "Number expert sample points:", num_expert_demos
        idx_d = np.arange(num_expert_demos)
        np.random.shuffle(idx_d)

        states_d = states_d[idx_d]
        actions_d = actions_d[idx_d]

        disc_loss = []
        gen_surr_loss = []
        gen_kl_div = []
        gen_entropy = []

        with tqdm(total=config.n_epochs, position=0, leave=True) as pbar:
            for i in xrange(0, config.n_epochs):
                if verbose:
                    print("\n********** Iteration {} **********".format(i))

                if i == 99:
                    print "Last Loop"

                if i == 0:
                    paths_per_collect = 30
                else:
                    paths_per_collect = 10

                # #################################### #
                # #### STEP 1: Sample Trajectories ### #
                # #################################### #
                if verbose:
                    print "Rolling Out trajectories ..."
                paths = rollout_traj_gail(self.env, self, self.state_dim,
                                          config.max_step_limit, paths_per_collect, verbose)

                # No need to buffer the rollouts since in the given env. it is not expensive to sample them
                for path in paths:
                    self.buffer.add(path)
                paths = self.buffer.get_sample(config.sample_size)

                for path in paths:
                    path["action_dist"] = self.sess.run(
                        self.action_dist_discrete, {self.states: path["states"]}
                    )

                action_dist_n = np.concatenate([path["action_dist"] for path in paths])
                states_n = np.concatenate([path["states"] for path in paths])
                actions_n = np.concatenate([path["actions"] for path in paths])

                if verbose:
                    print "Epoch:", i, ", Total sampled data points:", states_n.shape[0]

                # #################################### #
                # #### STEP 2: Train discriminator ### #
                # #################################### #
                num_sampled_demos = states_n.shape[0]
                batch_size = config.batch_size

                start_d = self.demo_idx
                start_n = 0

                if i <= 5:
                    d_iter = 120 - i * 20
                else:
                    d_iter = 10
                for k in xrange(d_iter):

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
                    if start_n + batch_size >= num_sampled_demos:
                        start_n = (start_n + batch_size) % num_sampled_demos

                    disc_loss.append(loss)
                    if verbose:
                        print "Discriminator step:", k, "loss:", loss

                # #################################### #
                # ###### STEP 4: Train Generator ##### #
                # #################################### #
                """
                We take policy steps using TRPO and update the generator param
                """
                path_idx = 0
                for path in paths:
                    output_d = self.discriminate.predict([path["states"], path["actions"]])
                    path["rewards"] = output_d.flatten()*0.1 + np.ones(path["raws"].shape[0]) * 2
                    path["returns"] = discount(path["rewards"], config.gamma)
                    path["baselines"] = self.baseline.predict(path)
                    path_baselines = np.append(path["baselines"], 0 if path["obj_achieved"] else path["baselines"][-1])
                    # For reaching the terminal state we append the value of terminal state which is 0 else we did not
                    # complete the task because of early termination. The latter happens due to being stuck at the
                    # current state, for which we append value of next state that is equal to value of current state.
                    deltas = path["rewards"] + config.gamma * path_baselines[1:] - path_baselines[:-1]
                    path["advants"] = discount(deltas, config.gamma * config.lam)
                    path_idx += 1

                advants_n = np.concatenate([path["advants"] for path in paths])
                # advants_n -= advants_n.mean()
                advants_n /= (advants_n.std() + 1e-8)

                self.baseline.fit(paths)

                feed = {self.states: states_n,
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

                loss_grad = self.sess.run(self.policy_grad, feed_dict=feed)
                step_dir = conjugate_gradient(fisher_vector_product, -loss_grad, cg_iters=config.cg_iters)
                shs = .5 * step_dir.dot(fisher_vector_product(step_dir))  # (0.5*X^T*Hessian*X) where X is step_dir
                try:
                    assert shs > 0
                    lm = np.sqrt(shs / config.max_kl)
                    full_step = step_dir / lm
                    neg_g_dot_stepdir = -loss_grad.dot(step_dir)

                    def loss(th):
                        self.sff(th)
                        return self.sess.run(self.losses[0], feed_dict=feed)

                    success, theta = linesearch(loss, theta_prev, full_step, neg_g_dot_stepdir / lm)
                    self.sff(theta)

                    surr_after, kl_old_new, entropy = self.sess.run(self.losses, feed_dict=feed)
                    gen_surr_loss.append(surr_after)
                    gen_kl_div.append(kl_old_new)
                    gen_entropy.append(entropy)

                    episode_rewards = np.array([path["rewards"].sum() for path in paths])
                    num_ep_total += len(episode_rewards)

                    if verbose:
                        print "Now we save model"
                    self.generator.save_weights(os.path.join(save_param_dir, "generator_model_%d.h5" % i),
                                                overwrite=True)
                    with open(os.path.join(save_param_dir, "generator_model_%d.json" % i), "w") as outfile:
                        json.dump(self.generator.to_json(), outfile)

                    if verbose:
                        print("***********************************".format(i))
                except AssertionError:
                    print "shs value is less than 0: ", shs, ". Exiting..."
                    exit(-1)

                pbar.refresh()
                pbar.set_description("Epoch {}".format(i + 1))
                pbar.set_postfix(sampled_points=num_sampled_demos, Entropy=entropy, Surrogate_loss=surr_after,
                                 KL_old_new=kl_old_new)
                pbar.update(1)


def run(exp_num, env, env_config, param_dir):
    print("\n********** Experiment {} **********".format(exp_num))
    state_dim = env_config.state_dim
    action_dim = env_config.action_dim

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    # define the model
    agent = Agent(env, sess, state_dim, action_dim, model_config)

    # Load expert (state, action) pairs and G.T latent modes
    print "Loading data"
    with open(state_action_path, 'rb') as f:
        trajectories = pkl.load(f)
    with open(latent_path, 'rb') as f:
        latent_modes_traj = pkl.load(f)

    demo = parse_data(env, trajectories, latent_modes_traj, lower_bound=0, upper_bound=training_data_config.num_traj)
    sub_param_dir = os.path.join(param_dir, "params{}".format(exp_num))
    if not os.path.exists(sub_param_dir):
        os.makedirs(sub_param_dir)
    print("Now we load the weights")
    try:
        GAIL_pretrained_model_dir = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/pretrained_params/{}/' \
                                    'params/{}/{}/{}/'.format(env_name, supervision_setting, latent_setting, 'BC')
        pre_trained_model_path = GAIL_pretrained_model_dir + "generator_bc_model_{}.h5".format(exp_num)
        print "Picking Pre-trained BC model from", pre_trained_model_path
        agent.generator.load_weights(pre_trained_model_path)
        print("Weight loaded successfully")
    except:
        print("Cannot find the weight")

    agent.learn(demo, sub_param_dir, exp_num, verbose=0)

    print("Finish.")


def run_multiple_runs():

    # For averaging across different runs
    for exp_num, seed_value in enumerate(seed_values[:1]):

        # if exp_num == 0:
        #     continue
        # Set a seed value
        np.random.seed(seed_value)
        tf.set_random_seed(seed_value)

        # Run multiple experiments
        run(exp_num, env, env_config, param_dir=p_dir)


def measure_KL_with_expert(exp_num=0):
    data = get_data()
    state_dim = env_config.state_dim
    action_dim = env_config.action_dim
    encode_dim = env_config.encode_dim

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    # define the model
    agent = Agent(env, sess, state_dim, action_dim, model_config)
    kl_models, w_kl_models = [], []
    model_dir = param_dir + "params{}/".format(exp_num)
    for model_id in range(model_config.n_epochs):
        model_path = os.path.join(model_dir, 'generator_model_' + str(model_id) + ".h5")
        agent.generator.load_weights(model_path)
        act_pred = agent.generator.predict([data['states_d']])
        w_kl = pred_kl_div(data['actions_d'], act_pred, weights=data['weights'])
        w_kl_models.append(w_kl[0])

    plot_infogail_results(w_kl_models, kl_figpath + "perf{}.png".format(exp_num))


if __name__ == "__main__":
    run_multiple_runs()
    measure_KL_with_expert()