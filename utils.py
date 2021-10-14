from collections import deque

import numpy as np
import cv2
import random
import time
import tensorflow as tf
import scipy.signal
import matplotlib.pyplot as plt
import keras.backend as K
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Activation, Convolution2D, MaxPooling2D, Flatten, Input, merge, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
from config import training_data_config
from config import InfoGAIL_config as model_config

dtype = tf.float32


def one_hot_encode(vector, dim=None):
    if dim is None:
        one_hot_encoded = np.zeros((vector.size, vector.max() + 1))
        one_hot_encoded[np.arange(vector.size), vector] = 1
    else:
        one_hot_encoded = np.zeros((vector.size, dim))
        one_hot_encoded[np.arange(vector.size), vector] = 1
    return one_hot_encoded


def parse_data(env, trajectories, latent_modes_traj, lower_bound=0, upper_bound=training_data_config.num_traj):
    total_traj = len(trajectories)
    expert_demos = trajectories[lower_bound:upper_bound]
    latent_modes_traj = latent_modes_traj[lower_bound:upper_bound]

    states = []
    actions = []
    latent_modes = []
    for traj_id in xrange(upper_bound-lower_bound):
        demo = expert_demos[traj_id]
        for sa_pair_id in xrange(len(demo)):
            sa_pair = demo[sa_pair_id]
            states.append(env.inv_states_dict[sa_pair[0]])
            actions.append(sa_pair[1])
            latent_modes.append(
                env.latent_state_dict[tuple(latent_modes_traj[traj_id][sa_pair_id])]
            )
    demo_data = {
        'states': np.array(states),
        'actions': np.array(one_hot_encode(np.array(actions), dim=len(env.action_dict))),
        'latent_states': np.array(one_hot_encode(np.array(latent_modes), dim=len(env.latent_dict)))
    }
    return demo_data


def linesearch(f, theta_k, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(theta_k)
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):  # alpha=0.5
        theta_new = theta_k + stepfrac * fullstep
        newfval = f(theta_new)  # Surr Loss at new theta
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return True, theta_new
    return False, theta_k


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):  # Do not play with residual_tol
    x = np.zeros_like(b)  # Initialise X0 <- can be zeros
    r = b.copy()  # Residual, r0 = b - A*x0 which equals b given x0 = 0
    p = b.copy()  # Initialise Conjugate direction P0 same as r0=b
    rdotr = r.dot(r)
    for i in xrange(cg_iters):  # Theoretically, the method converges in n=dim(b) iterations
        Ap = f_Ax(p)  # Compute vector product AxP
        alpha = rdotr / p.dot(Ap)
        x += alpha * p  # Update approx of x* that solves Ax=b
        r -= alpha * Ap  # Update residuals
        newrdotr = r.dot(r)  # Compute <r_(k+1), r_(k+1)>
        beta = newrdotr / rdotr
        p = r + beta * p  # Get the next conjugate direction to move along
        rdotr = newrdotr
        if rdotr < residual_tol:  # If r = b-Ax -> 0 we are close to x*
            break
    return x


# Filter data along 1D with an IIR or FIR filter i.e. discount accumulated rewards to compute overall return gamma^t*R_t
def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter(b=[1], a=[1, -gamma], x=x[::-1], axis=0)[::-1]


def pathlength(path):
    return len(path["actions"])


def cat_log_prob(dist, action, epsilon=10e-9):
    return tf.log(tf.reduce_sum(tf.mul(dist, action), axis=1) + tf.convert_to_tensor(epsilon))  # Add epsilon to account for zero prob


def selfKL_firstfixed(p):
    p1 = map(tf.stop_gradient, [p])  # Detach the policy parameters from kth iter to compute param at iter. k+1 in TRPO
    p2 = p
    return KL_discrete(p1, p2)  # KL(pi_k, pi)


def KL_discrete(p, q, epsilon=10e-9):
    """Calculates the KL divergence of two distributions.
        Arguments:
            p    : Distribution p.
            q    : Distribution q.
        Returns:
            the divergence value.
            :param epsilon:
        """
    kl = tf.reduce_sum(tf.mul(p, tf.log(p + tf.convert_to_tensor(epsilon)))
                       - tf.mul(p, tf.log(q + tf.convert_to_tensor(epsilon))))
    return kl


def entropy_discrete(p, epsilon=10e-9):
    # h = tf.reduce_sum(logstd + tf.constant(0.5*np.log(2*np.pi*np.e), tf.float32))
    h = -tf.reduce_sum(tf.mul(p, tf.log(p+tf.convert_to_tensor(epsilon))))
    return h


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
            "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))


# Take the gradients of the loss w.r.t to weights and flatten the grad of each layer and concat across layers
def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    # Flattening the gradients into K*1, where K is the total number of parameters in the policy net.
    return tf.concat(0, [tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)])


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_paths = 0
        self.buffer = deque()

    def get_sample(self, sample_size):
        if self.num_paths < sample_size:
            return random.sample(self.buffer, self.num_paths)
        else:
            return random.sample(self.buffer, sample_size)

    def size(self):
        return self.buffer_size

    def add(self, path):
        if self.num_paths < self.buffer_size:
            self.buffer.append(path)
            self.num_paths += 1
        else:
            self.buffer.popleft()
            self.buffer.append(path)

    def count(self):
        return self.num_paths

    def erase(self):
        self.buffer = deque()
        self.num_paths = 0


class RoadWorldResponse(object):
    def __init__(self, env, terminal_states):
        self.env = env
        self.stuck_count = 0
        self.terminal_states = terminal_states

    def reset_stuck_count(self):
        self.stuck_count = 0

    def update_stuck_count(self):
        self.stuck_count += 1

    def gen_response(self, curr_state, action, next_state):
        terminate = False
        reward = 0

        if next_state in self.terminal_states:
            terminate = True

        elif next_state == curr_state:
            # neighs, out_of_space = self.env.neighbors(self.env.inv_states_dict[curr_state])
            # # Terminate if action takes agent out of the grid
            # if action in out_of_space.keys():
            #     terminate = True
            #     reward = -100

            self.update_stuck_count()
            # If the agent is stuck in a position for too long
            if self.stuck_count > model_config.stuck_count_limit:
                terminate = True
                reward = -100

        else:
            self.reset_stuck_count()

        return terminate, reward


def rollout_traj(env, agent, state_dim, encode_dim,
                 max_step_limit, paths_per_collect,
                 discriminate, posterior=None, verbose=0):
    paths = []
    encode_axis = 0
    for p in xrange(paths_per_collect):
        if verbose:
            print "Rollout index:", p

        states, encodes, actions, raws = [], [], [], []

        # Sample the latent mode for each rollout (fixed), Encoding in one-hot dimension
        encode = np.zeros((1, encode_dim), dtype=np.float32)
        encode[0, encode_axis] = 1
        (latent1, latent2) = env.inv_latent_dict[encode_axis]
        env.set_latent1(latent1)
        env.set_latent1(latent2)

        # Get the initial position of the agent. In some envs, latent_mode fixes the terminal state like RoadWorld
        # while in some they are pre-fixed(BoxWorld). We take the generic route and obtain terminal states for each path
        state_idx, terminal_states_idx = env.get_initial_state()
        env_resp = RoadWorldResponse(env, terminal_states_idx)

        if not state_idx:
            print "No starting index. Skipping path"
            continue
        state = np.array(env.inv_states_dict[state_idx])
        state = np.expand_dims(state, axis=0)

        # state = get_initial_state_SD(env, encode)
        reward_d = 0
        reward_p = 0

        for i in range(max_step_limit):

            states.append(state)
            encodes.append(encode)

            # Take action
            action = agent.act(state, encode)  # Action here is not a prob dis. but the one-hot encoding of taken action
            actions.append(action)

            # reward_d += discriminate.predict([state, action])[0, 0] * 0.1
            # # Log prob of the latent code that was used in the prior
            # reward_p += np.sum(np.log(posterior.predict([state, action]))*encode)

            # Get action id
            action_id = np.where(action[0] == 1.0)[0][0]

            # Compute the next state using environment's transition dynamics
            current_state_idx = env.states_dict[tuple(state[0])]
            next_state_prob_dist = env.obstate_transition_matrix[env.states_dict[tuple(state[0])], action_id, :]
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
                    encodes=np.concatenate(encodes),
                    actions=np.concatenate(actions),
                    raws=np.array(raws),
                    obj_achieved=True if i + 1 < max_step_limit and reward == 0 else False,
                    )
                paths.append(path)
                break
            state = next_state

        # Update the encode axis for the next path
        encode_axis = (encode_axis + 1) % encode_dim
    return paths


class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


class Dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


class GetFlat(object):
    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat(0, [tf.reshape(v, [numel(v)]) for v in var_list])

    def __call__(self):
        return self.op.eval(session=self.session)


class SetFromFlat(object):
    def __init__(self, session, var_list):
        self.session = session
        assigns = []
        shapes = map(var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(
                tf.assign(v, tf.reshape(theta[start:start + size], shape))
            )
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})


# # Baseline Network that predicts the accumulated return i.e. the Value Func
class NNBaseline(object):

    def __init__(self, sess, state_dim, encode_dim, lr_baseline, b_iter, batch_size):
        # print "Now we build baseline"
        self.model = self.create_net(state_dim, encode_dim, lr_baseline)
        self.sess = sess
        self.b_iter = b_iter
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
        h = merge([x, c], mode='sum')
        h = LeakyReLU()(h)
        p = Dense(1)(h)  # indicates the expected accumulated future rewards.

        model = Model(input=[state, encodes], output=p)
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
        # auxs_train = auxs[idx][:num_train]
        encodes_train = encodes[idx][:num_train]
        returns_train = returns[idx][:num_train]

        feats_val = feats[idx][num_train:]
        # auxs_val = auxs[idx][num_train:]
        encodes_val = encodes[idx][num_train:]
        returns_val = returns[idx][num_train:]

        start = 0
        batch_size = self.batch_size
        for i in xrange(b_iter):
            loss = self.model.train_on_batch(
                [feats_train[start:start + batch_size],
                 # auxs_train[start:start + batch_size],
                 encodes_train[start:start + batch_size]],
                returns_train[start:start + batch_size]
            )
            start += batch_size
            if start >= num_train:
                start = (start + batch_size) % num_train
            val_loss = np.average(np.square(self.model.predict(
                [feats_val, encodes_val]).flatten() - returns_val))
            # print "Baseline step:", i, "loss:", loss, "val:", val_loss

    def predict(self, path):
        if self.first_time:
            return np.zeros(pathlength(path))
        else:
            acc_return = self.model.predict(
                [path["states"], path["encodes"]])
        return np.reshape(acc_return, (acc_return.shape[0], ))


def plot_BC_loss(epochs, train_loss, val_loss, fig_path):
    fig, ax = plt.subplots()
    ax.semilogy(train_loss, 'b', label='Train')
    ax.semilogy(val_loss, 'r', label='Val')
    ax.legend(loc='upper right')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    # plt.show()
    plt.savefig(fig_path)


def plot_infoGAIL_loss(disc_loss, gen_surr_loss, fig_path):
    figure, axis = plt.subplots(2)

    # For Discriminator
    axis[0].plot(disc_loss)
    axis[0].set_title("Discriminator Loss")
    axis[0].set_xlabel('Epochs')
    axis[0].set_ylabel('Loss')
    # axis.plot(disc_loss)
    # axis.set_title("Discriminator Loss")
    # axis.set_xlabel('Epochs')
    # axis.set_ylabel('Loss')

    # For Posterior
    # axis[0, 1].plot(post_train_loss, 'b')
    # # axis[0, 1].legend(['Training Loss', 'Val Loss'])
    # axis[0, 1].set_title("Posterior Train Loss")
    # axis[0, 1].set_xlabel('Epochs')
    # axis[0, 1].set_ylabel('Loss')
    #
    # axis[1, 1].plot(post_val_loss, 'r')
    # # axis[0, 1].legend(['Training Loss', 'Val Loss'])
    # axis[1, 1].set_title("Posterior Val Loss")
    # axis[1, 1].set_xlabel('Epochs')
    # axis[1, 1].set_ylabel('Loss')

    # For Generator Surrogate
    axis[1].plot(gen_surr_loss)
    axis[1].set_title("Generator Surrogate")
    axis[1].set_xlabel('Epochs')
    axis[1].set_ylabel('Surr Loss')

    # For Generator KL Div
    # axis[1, 0].plot(gen_kl_div)
    # axis[1, 0].set_title("Generator KL Div")
    # axis[1, 0].set_xlabel('Epochs')
    # axis[1, 0].set_ylabel('KL Div')

    # For Generator Entropy
    # axis[1, 1].plot(gen_entropy)
    # axis[1, 1].set_title("Generator Entropy")
    # axis[1, 1].set_xlabel('Epochs')
    # axis[1, 1].set_ylabel('Entropy')

    figure.tight_layout()
    # plt.show()
    plt.savefig(fig_path)


def plot_infogail_results(w_kl, fig_path):
    figure, axis = plt.subplots()

    # axis[0, 0].plot(kl)
    # axis[0, 0].set_title("KL Div")
    # axis[0, 0].set_xlabel('Epochs')
    # axis[0, 0].set_ylabel('Divergence')

    # axis[0, 1].plot(acc)
    # axis[0, 1].set_title("Accuracy")
    # axis[0, 1].set_xlabel('Epochs')
    # axis[0, 1].set_ylabel('Accuracy')

    axis.plot(w_kl)
    axis.set_title("Weighted KL Div")
    axis.set_xlabel('Epochs')
    axis.set_ylabel('Divergence')

    # axis[1, 1].plot(w_acc)
    # axis[1, 1].set_title("Weighted Accuracy")
    # axis[1, 1].set_xlabel('Epochs')
    # axis[1, 1].set_ylabel('Accuracy')

    figure.tight_layout()
    plt.savefig(fig_path)
