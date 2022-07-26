import os
import sys
import gym
import numpy as np
import pickle as pkl
import tensorflow as tf
from tqdm import tqdm

from utils.env import get_env_params, preprocess_robotic_demos
from utils.misc import causally_parse_dynamic_data_v2
from evaluation.eval import evaluate_model_continuous
from keras.layers import Dense, Flatten, Add
from utils.vae import get_transn_loss, kl_divergence_gaussian
from utils.plot import plot_vae_loss
from tensorflow_probability.python.distributions import Normal

global file_txt_results_path
file_txt_results_path = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/OpenAIPickandPlace_results.txt'


class Encoder(tf.keras.Model):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
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

        # it is better to model std_dev as log(std_dev) as it is more numerically stable to take exponent compared to
        # computing log. Hence, our final KL divergence term is:
        scale = self.std_out(h)
        scale = tf.clip_by_value(scale, clip_value_min=1e-3, clip_value_max=1e3)
        # Note if returning sampled z as well, decorate sample_fn with @tf.function so that it gets saved in the
        # computation graph and there is no need to implement it while testing or loading the model
        return locs, scale


class Decoder(tf.keras.Model):
    def __init__(self, a_dim, actions_max):
        super(Decoder, self).__init__()

        self.max_actions = actions_max
        self.flatten = Flatten()
        self.fc1 = Dense(units=256, activation=tf.nn.relu)
        self.fc2 = Dense(units=256, activation=tf.nn.relu)
        self.fc3 = Dense(units=128, activation=tf.nn.relu)
        self.fc4 = Dense(units=128, activation=tf.nn.relu)
        self.add = Add()
        self.action_out = Dense(units=a_dim, activation=tf.nn.tanh)

    def call(self, curr_state, prev_encode_z):
        s = self.flatten(curr_state)
        s = self.fc1(s)
        s = self.fc2(s)
        s = self.fc3(s)

        c = self.fc4(prev_encode_z)
        h = self.add([s, c])
        actions_pi = self.action_out(h) * self.max_actions
        return actions_pi


class Classifier(tf.keras.Model):
    def __init__(self, y_dim):
        super(Classifier, self).__init__()
        self.out_prob_y = Dense(units=y_dim, activation=tf.nn.softmax)

    def call(self, encodes_z):
        prob_y = self.out_prob_y(encodes_z)
        return prob_y


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


class CCVAE(tf.keras.Model):
    def __init__(self, z_dim, y_dim, a_dim, actions_max):
        super(CCVAE, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.a_dim = a_dim
        # # ------------------------------------------ Model Declaration ------------------------------------------ # #
        # Encode current latent mode using stack of states (upto a window size) and previous latent mode (i/of c_1:t-1)
        self.encoder = Encoder(z_dim)
        # Decode action using current state (i/of s_1:t) and current latent mode (i/of c_1:t)
        self.decoder = Decoder(a_dim, actions_max)
        # Classifier to classify encoded input, z into one of the labels, y
        self.classifier = Classifier(y_dim)
        # Conditional Prior to obtain Prior probability of z given y (Originally it was N(0, I))
        self.cond_prior = Conditional_Prior(z_dim)

    def sample_gumbel_tf(self, shape, eps=1e-20):
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    def sample_gumbel_softmax_tf(self, logits, temperature):
        y = logits + self.sample_gumbel_tf(tf.shape(logits))
        return tf.nn.softmax(y / temperature)

    def gumbel_softmax_tf(self, logits, temperature, latent_dim, is_prob=False):
        """
        Returns: Discrete Sampler which returns a prob. distribution over discrete states. The returned prob dist.
        starts to peak at 1 class with decrease in temp. Case Study: For uniformly distributed logits(& say if temp
        is 0.1) then all the classes are equally likely to be sampled with softmax distribution peaking at one of them.

        General Case: Returned Samples are more likely the ones that carry high prob in the input distribution when
        temp starts going down.
        """
        # If input is a prob distribution, convert into logits
        if is_prob:
            logits = tf.math.log(logits)
        # While testing, we get the exact one-hot vector else we go with softmaxed vector to allow diff. during training
        y = self.sample_gumbel_softmax_tf(logits, temperature)
        return tf.reshape(y, shape=[-1, latent_dim])

    def sample_unit_normal(self, shape):
        epsilon = tf.random.normal(shape, mean=0.0, stddev=1.0, )
        return epsilon

    def sample_normal(self, mu, std, latent_dim):
        z = mu + tf.math.multiply(std, self.sample_unit_normal(tf.shape(std)))
        return tf.reshape(z, shape=[-1, latent_dim])

    def multi_sample_normal(self, mu, std, latent_dim, k=100):
        samples = []
        for _ in range(k):
            z = self.sample_normal(mu, std, latent_dim)
            samples.append(z)
        return samples


class Agent:
    def __init__(self, s_dim, stack_s_dim, a_dim, z_dim, y_dim, num_samples, supervision, train_config, env_config):
        self.train_config = train_config
        self.env_config = env_config
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        print("State Dim:", s_dim, " Action Dim:", a_dim, " Latent Dim: ", z_dim, " Label Dim:", y_dim)
        self.stack_s_dim = stack_s_dim

        # Parameters
        self.supervision = supervision
        self.eps = 1e-20
        self.lr = train_config['lr']

        # Classification q(y|x) weight
        self.alpha = 0.1*num_samples

        # Other Param
        # self.latent_sampler_temp = latent_sampler_temp = self.model_config.init_temp
        self.latent_sampler_temp = self.train_config['init_temp']

        # To avoid unintended behaviours when variables declared outside the scope of tf.function are modified because
        # their inital value will be used when tf.function traces the computation graph, in order to make sure that its
        # updated value is used either pass it as an arg to the function or declare it as a variable whose value can
        # then be changed outside using var.assign which will reflect automatically inside the computation graph
        self.p_Y = tf.Variable(1/y_dim, dtype=tf.float32)

        self.model = CCVAE(z_dim, y_dim, a_dim, env_config['action_max'])
        self.optimiser = tf.keras.optimizers.Adam(self.lr)

    def load_model(self,  param_dir, model_id):
        model = CCVAE(self.z_dim, self.y_dim, self.a_dim, self.env_config['action_max'])

        # BUILD First
        _ = model.encoder(np.ones([1, self.train_config['w_size'] * self.s_dim]),
                          np.ones([1, self.y_dim]))  # (s, y) -> z
        _ = model.decoder(np.ones([1, self.s_dim]), np.ones([1, self.z_dim]))  # (s, z) -> a
        _ = model.classifier(np.ones([1, self.z_dim]))  # (z) -> y
        _ = model.cond_prior(np.ones([1, self.y_dim]))

        model.encoder.load_weights(os.path.join(param_dir, "encoder_model_{}.h5".format(model_id)))
        model.decoder.load_weights(os.path.join(param_dir, "decoder_model_{}.h5".format(model_id)))
        model.classifier.load_weights(os.path.join(param_dir, "classifier_{}.h5".format(model_id)))
        model.cond_prior.load_weights(os.path.join(param_dir, "cond_prior_{}.h5".format(model_id)))

    def classifier_loss(self, s_curr, c_prev, c_curr, k=100):
        [post_locs, post_scales] = self.model.encoder(s_curr, c_prev)
        # Draw k samples from q(z|x) and compute log(q(y_curr|z_k)) = log(q(y|z_k))*y_curr for each.
        qy_z_k = [(lambda _z:  tf.reduce_sum(tf.math.multiply(self.model.classifier(_z), c_curr), axis=-1))(_z)
                  for _z in self.model.multi_sample_normal(post_locs, post_scales, self.z_dim, k)]
        qy_z_k = tf.concat(values=[tf.expand_dims(qy_z, axis=0) for qy_z in qy_z_k], axis=0)
        lqy_x = tf.math.log(tf.reduce_sum(qy_z_k, axis=0) + self.eps) - tf.cast(tf.math.log(float(k)), dtype=tf.float32)
        return tf.reshape(lqy_x, shape=[tf.shape(s_curr)[0], ])

    def unsup_loss(self, stack_states, curr_state, prev_encode, next_action):
        # INFERENCE: Compute the approximate posterior q(z|x)
        [post_locs, post_scales] = self.model.encoder(stack_states, prev_encode)
        z = self.model.sample_normal(post_locs, post_scales, self.z_dim)

        # INFERENCE: Compute the classification prob q(y|z)
        qy_z = self.model.classifier(z)
        # Sample y
        sampled_curr_encode_y = self.model.gumbel_softmax_tf(qy_z, self.latent_sampler_temp, self.y_dim, is_prob=True)
        log_qy_z = tf.reduce_sum(tf.math.log(qy_z + self.eps) * sampled_curr_encode_y, axis=-1)

        # GENERATION: Compute the Prior p(y)
        log_py = tf.reduce_sum(tf.math.log(self.p_Y + self.eps) * sampled_curr_encode_y, axis=-1)

        # GENERATION: Compute the Conditional prior p(z|y)
        [prior_locs, prior_scales, sampled_z_k] = self.model.cond_prior(sampled_curr_encode_y)

        # GENERATION: Compute the log-likelihood of actions i.e. p(x|z)
        mu_actions = self.model.decoder(curr_state, z)

        # ELBO
        ll = -tf.reduce_sum(tf.math.squared_difference(mu_actions, next_action), axis=-1)
        kl = kl_divergence_gaussian(mu1=post_locs, log_sigma_sq1=tf.math.log(post_scales + self.eps),
                                    mu2=prior_locs, log_sigma_sq2=tf.math.log(prior_scales + self.eps),
                                    mean_batch=False)
        elbo = ll + log_py - kl - log_qy_z

        transn_loss = get_transn_loss(prev_encode, sampled_curr_encode_y)
        loss = tf.reduce_mean(-elbo)
        return loss

    def sup_loss(self, stack_states, curr_state, prev_encode, curr_encode, next_action):

        # INFERENCE: Compute the approximate posterior q(z|x)
        [post_locs, post_scales] = self.model.encoder(stack_states, prev_encode)
        z = self.model.sample_normal(post_locs, post_scales, self.z_dim)

        # INFERENCE: Compute the classification prob q(y|z)
        qy_z = self.model.classifier(z)
        log_qy_z = tf.reduce_sum(tf.math.log(qy_z + self.eps) * curr_encode, axis=-1)

        # INFERENCE: Compute label classification q(y|x) <- Sum_z(q(y|z)*q(z|x)) ~ 1/k * Sum(q(y|z_k))
        log_qy_x = self.classifier_loss(stack_states, prev_encode, curr_encode)

        # GENERATION: Compute the Prior p(y)
        log_py = tf.reduce_sum(tf.math.log(self.p_Y + self.eps) * curr_encode, axis=-1)

        # GENERATION: Compute the Conditional prior p(z|y)
        [prior_locs, prior_scales, sampled_z_k] = self.model.cond_prior(curr_encode)

        # GENERATION: Compute the log-likelihood of actions i.e. p(x|z)
        mu_actions = self.model.decoder(curr_state, z)

        # We only want gradients wrt to params of qyz, so stop them propagating to qzx! Why? Ref. Appendix C.3.1
        # In short, to reduce the variance in the gradients of classifier param! To a certain extent these gradients can
        # be viewed as redundant, as there is already gradients to update the predictive distribution due to the
        # log q(y|x) term anyway

        # Note: PYTORCH Detach stops the tensor from being tracked in the subsequent operations involving the tensor:
        # The original implementation is detaching the tensor z
        # log_qy_z_ = tf.stop_gradient(tf.reduce_sum(tf.log(self.classifier([z]) + self.eps) * ln_curr_encode_y_ip,
        #                                            axis=-1))
        # Compute weighted ratio
        # w = tf.exp(log_qy_z_ - log_qy_x)
        w = tf.exp(log_qy_z - log_qy_x)

        # ELBO
        ll = -tf.reduce_sum(tf.math.squared_difference(mu_actions, next_action), axis=-1)
        kl = kl_divergence_gaussian(mu1=post_locs, log_sigma_sq1=tf.math.log(post_scales + self.eps),
                                    mu2=prior_locs, log_sigma_sq2=tf.math.log(prior_scales + self.eps),
                                    mean_batch=False)
        elbo_term1 = tf.math.multiply(w, ll - kl - log_qy_z)
        elbo = elbo_term1 + log_py + log_qy_x*self.alpha

        # Transition Loss does not make any sense since curr latent mode is already known
        # transn_loss = get_transn_loss(ln_prev_encode_y_ip, ln_curr_encode_y_ip)
        loss = tf.reduce_mean(-elbo)

        # For Debugging
        # self.obs_var1 = tf.reduce_mean(elbo_term1)
        # self.obs_var2 = tf.reduce_mean(log_qy_x*self.alpha)
        return loss

    # # # IMPORTANT
    # tf.function wraps the function for tf's graph computation [efficient, fast, portable].
    # It applies to a function and all other functions it calls.
    # No need for decorating fns that are called from inside train_step
    # # TIP
    # Include as much computation as possible under a tf.function to maximize the performance gain.
    # For example, decorate a whole training step or the entire training loop.
    @tf.function
    def train_step(self, data, supervised):
        with tf.GradientTape() as tape:
            if supervised:
                loss = self.sup_loss(data['curr_stack_states'], data['curr_states'], data['prev_encodes_y'],
                                     data['curr_encodes_y'], data['next_actions'])
            else:
                loss = self.unsup_loss(data['curr_stack_states'], data['curr_states'], data['prev_encodes_y'],
                                       data['next_actions'])
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, demos_un, demos_ln, param_dir, fig_path, exp_num):

        batch_size = self.train_config['batch_size']

        train_unsup, train_sup = None, None
        # Get Unsupervised Data [Convert the dtype to match with tensorflow's processed dtype]
        if self.supervision != 1.0:
            unsupervised_data = {
                'curr_stack_states': demos_un['curr_stack_states'].astype('float32'),
                'curr_states': demos_un['curr_states'].astype('float32'),
                'prev_actions': demos_un['prev_actions'].astype('float32'),
                'next_actions': demos_un['next_actions'].astype('float32'),
                'prev_encodes_y': demos_un['prev_latent_states'].astype('float32'),
                'curr_encodes_y': demos_un['curr_latent_states'].astype('float32'),
            }
            num_un_expert_demos = demos_un['curr_states'].shape[0]
            try:
                assert num_un_expert_demos > batch_size
            except AssertionError:
                print("Num of Supervised Samples {} less than batch size {}".format(num_un_expert_demos, batch_size),
                      file=open(file_txt_results_path, 'a'))
                sys.exit(-1)
            train_unsup = tf.data.Dataset.from_tensor_slices(unsupervised_data).batch(batch_size)
            print("Number of Unsupervised Samples: ", num_un_expert_demos, file=open(file_txt_results_path, 'a'))

        # Get Supervised Data [Convert the dtype to match with tensorflow's processed dtype]
        if self.supervision != 0.0:
            supervised_data = {
                'curr_stack_states': demos_ln['curr_stack_states'].astype('float32'),
                'curr_states': demos_ln['curr_states'].astype('float32'),
                'prev_actions': demos_ln['prev_actions'].astype('float32'),
                'next_actions': demos_ln['next_actions'].astype('float32'),
                'prev_encodes_y': demos_ln['prev_latent_states'].astype('float32'),
                'curr_encodes_y': demos_ln['curr_latent_states'].astype('float32'),
            }
            num_ln_expert_demos = demos_ln['curr_states'].shape[0]
            try:
                assert num_ln_expert_demos > batch_size
            except AssertionError:
                print("Num of Supervised Samples {} less than batch size {}".format(num_ln_expert_demos, batch_size),
                      file=open(file_txt_results_path, 'a'))
                sys.exit(-1)
            train_sup = tf.data.Dataset.from_tensor_slices(supervised_data).batch(batch_size)
            print("Number of Supervised Samples: ", num_ln_expert_demos, file=open(file_txt_results_path, 'a'))

        # Declare Parameters
        _lr = self.train_config['lr']
        sup_loss, unsup_loss = [], []
        max_loss = np.inf

        # Train the Model
        with tqdm(total=self.train_config['n_epochs'], position=0, leave=True) as pbar:
            for i in range(0, self.train_config['n_epochs']):
                epoch_loss = []

                # Unsupervised Case
                if self.supervision == 0.0:
                    for unsupervised_batched_data in train_unsup:
                        loss = self.train_step(unsupervised_batched_data, supervised=False)
                        sup_loss.append(0.)
                        unsup_loss.append(loss.numpy())
                        epoch_loss.append(loss.numpy())

                # Semi-Supervised Case
                elif self.supervision < 1.0:

                    for unsupervised_batched_data in train_unsup:
                        loss = self.train_step(unsupervised_batched_data, supervised=False)
                        unsup_loss.append(loss.numpy())
                        epoch_loss.append(loss.numpy())

                    for supervised_batched_data in train_sup:
                        loss = self.train_step(supervised_batched_data, supervised=True)
                        sup_loss.append(loss.numpy())
                        epoch_loss.append(loss.numpy())

                # Fully Supervised Case
                else:
                    for supervised_batched_data in train_sup:
                        loss = self.train_step(supervised_batched_data, supervised=True)
                        sup_loss.append(loss.numpy())
                        unsup_loss.append(0.)
                        epoch_loss.append(loss.numpy())

                # Save the model
                avg_epoch_loss = np.average(np.array(epoch_loss))
                if avg_epoch_loss < max_loss:
                    max_loss = avg_epoch_loss
                    self.model.encoder.save_weights(os.path.join(param_dir, "encoder_model_best.h5"), overwrite=True)
                    self.model.decoder.save_weights(os.path.join(param_dir, "decoder_model_best.h5"), overwrite=True)
                    self.model.cond_prior.save_weights(os.path.join(param_dir, "cond_prior_best.h5"), overwrite=True)
                    self.model.classifier.save_weights(os.path.join(param_dir, "classifier_best.h5"), overwrite=True)

                if i == 0 or i == self.train_config['n_epochs']-1:
                    self.model.encoder.save_weights(os.path.join(param_dir, "encoder_model_%d.h5" % i), overwrite=True)
                    self.model.decoder.save_weights(os.path.join(param_dir, "decoder_model_%d.h5" % i), overwrite=True)
                    self.model.cond_prior.save_weights(os.path.join(param_dir, "cond_prior_%d.h5" % i), overwrite=True)
                    self.model.classifier.save_weights(os.path.join(param_dir, "classifier_%d.h5" % i), overwrite=True)

                pbar.refresh()
                pbar.set_description("Epoch {}".format(i + 1))
                pbar.set_postfix(Loss=avg_epoch_loss)
                pbar.update(1)

        plot_vae_loss(sup_loss, fig_path + "_SupLoss", exp_num)
        plot_vae_loss(unsup_loss, fig_path + "_UnsupLoss", exp_num)


def run(env_name, exp_num=0, sup=0.1):
    print("\n\n---------------------------------- Supervision {} : Exp {} ----------------------------------".format(
        sup, exp_num),
          file=open(file_txt_results_path, 'a'))

    train_config = {
        "n_epochs": 1000,
        "batch_size": 100,
        "num_iters": 10,
        "lr": 0.001,
        "init_temp": 0.1,
        "anneal_rate": 0.00003,
        "w_size": 1,
        "num_traj": 100,
        'perc_supervision': sup,
    }

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
    model_dir = os.path.join(root_dir, 'pretrained_params/Project', env_name, 'CCVAE_{}'.format(sup))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    param_dir = os.path.join(model_dir, "params{}".format(exp_num))
    fig_path = os.path.join(model_dir, '{}_loss'.format('CCVAE'))

    z_dim = 6
    y_dim = env_config['latent_mode']
    state_dim = env_config['obs'] + env_config['goal']
    stack_state_dim = train_config["w_size"]*state_dim
    action_dim = env_config['action']

    # ################################################ Load Data ################################################ #
    print("Loading data", file=open(file_txt_results_path, 'a'))
    with open(train_data_path, 'rb') as f:
        traj_sac = pkl.load(f)

    # ################################################ Parse Data ################################################ #
    demos_train = causally_parse_dynamic_data_v2(traj_sac, lower_bound=0, upper_bound=train_config['num_traj'],
                                                 window_size=train_config["w_size"])
    demos_train = preprocess_robotic_demos(demos_train, env_config, window_size=train_config['w_size'], clip_range=5)
    num_samples = demos_train['curr_states'].shape[0]

    # Get Supervised/Unsupervised Data
    demos_ln, demos_un = None, None
    if sup == 0.0:
        demos_un = causally_parse_dynamic_data_v2(traj_sac, lower_bound=0, upper_bound=train_config['num_traj'],
                                                  window_size=train_config["w_size"])
        demos_un = preprocess_robotic_demos(demos_un, env_config, window_size=train_config['w_size'], clip_range=5)
    elif sup < 1.0:
        demos_ln = causally_parse_dynamic_data_v2(traj_sac, lower_bound=0, upper_bound=int(train_config['num_traj'] * sup),
                                                  window_size=train_config["w_size"])
        demos_ln = preprocess_robotic_demos(demos_ln, env_config, window_size=train_config['w_size'], clip_range=5)

        demos_un = causally_parse_dynamic_data_v2(traj_sac, lower_bound=int(train_config['num_traj'] * sup),
                                                  upper_bound=train_config['num_traj'],
                                                  window_size=train_config["w_size"])
        demos_un = preprocess_robotic_demos(demos_un, env_config, window_size=train_config['w_size'], clip_range=5)
    else:
        demos_ln = causally_parse_dynamic_data_v2(traj_sac, lower_bound=0, upper_bound=train_config['num_traj'],
                                                  window_size=train_config["w_size"])
        demos_ln = preprocess_robotic_demos(demos_ln, env_config, window_size=train_config['w_size'], clip_range=5)

    # ################################################ Train Model ################################################ #
    ssvae_learner = Agent(state_dim, stack_state_dim, action_dim, z_dim, y_dim, num_samples, sup,
                          train_config, env_config)
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)
        ssvae_learner.train(demos_un, demos_ln, param_dir, fig_path, exp_num)
        print("Finish.", file=open(file_txt_results_path, 'a'))
    else:
        print("Model already exists! Skipping training", file=open(file_txt_results_path, 'a'))

    # ################################################ Test Model ################################################ #
    model_id = 'best'  # best, 999
    with open(test_data_path, 'rb') as f:
        test_traj_sac = pkl.load(f)

    print("\nTesting Data Results", file=open(file_txt_results_path, 'a'))
    ssvae_learner = Agent(state_dim, stack_state_dim, action_dim, z_dim, y_dim, num_samples, sup,
                          train_config, env_config)
    ssvae_learner.load_model(param_dir, model_id)
    evaluate_model_continuous(ssvae_learner.model, "vae", test_traj_sac, train_config, file_txt_results_path)


if __name__ == "__main__":
    for sup in [0.0, 0.1, 0.2, 0.5, 1.0]:
        run(env_name='OpenAIPickandPlace', sup=sup)
