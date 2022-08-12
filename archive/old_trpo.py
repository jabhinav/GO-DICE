import numpy as np
import tensorflow as tf
import scipy.signal


def get_flat(var_list):
    op = tf.concat([tf.reshape(v, [numel(v)]) for v in var_list], axis=0)
    return op


def set_from_flat(model, theta):
    """
    Create the process of assigning updated vars
    """
    shapes = [v.shape.as_list() for v in model.trainable_variables]
    size_theta = np.sum([np.prod(shape) for shape in shapes])
    # self.assign_weights_op = tf.assign(self.flat_weights, self.flat_wieghts_ph)
    start = 0
    for i, shape in enumerate(shapes):
        size = np.prod(shape)
        param = tf.reshape(theta[start:start + size], shape)
        model.trainable_variables[i].assign(param)
        start += size
    assert start == size_theta, "messy shapes"


def linesearch(f, theta_k, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(theta_k)
    for (_n_backtracks, step_size) in enumerate(.5**np.arange(max_backtracks)):  # alpha=0.5
        theta_new = theta_k + step_size * fullstep
        newfval = f(theta_new)  # Surr Loss at new theta
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * step_size
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return True, theta_new
    return False, theta_k


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):  # Do not play with residual_tol
    x = np.zeros_like(b)  # Initialise X0 <- can be zeros
    r = b.copy()  # Residual, r0 = b - A*x0 which equals b given x0 = 0
    p = b.copy()  # Initialise Conjugate direction P0 same as r0=b
    rdotr = r.dot(r)
    for i in range(cg_iters):  # Theoretically, the method converges in n=dim(b) iterations
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
    return tf.math.log(tf.reduce_sum(tf.math.multiply(dist, action), axis=1) + tf.convert_to_tensor(epsilon))
    # Add epsilon to account for zero prob


def selfKL_firstfixed(p):
    p1 = tf.stop_gradient(p)  # Detach the policy parameters from kth iter to compute param at iter. k+1 in TRPO
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
    kl = tf.reduce_sum(tf.math.multiply(p, tf.math.log(p + tf.convert_to_tensor(epsilon)))
                       - tf.math.multiply(p, tf.math.log(q + tf.convert_to_tensor(epsilon))))
    return kl


def entropy_discrete(p, epsilon=10e-9):
    # h = tf.reduce_sum(logstd + tf.constant(0.5*np.log(2*np.pi*np.e), tf.float32))
    h = -tf.reduce_sum(tf.math.multiply(p, tf.math.log(p+tf.convert_to_tensor(epsilon))))
    return h


def var_shape(x):
    out = [k for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))


# Take the gradients of the loss w.r.t to weights and flatten the grad of each layer and concat across layers
def flatgrad(loss_fn, var_list):
    with tf.GradientTape() as tape:
        loss = loss_fn()
    grads = tape.gradient(loss, var_list)
    # Flattening the gradients into K*1, where K is the total number of parameters in the policy net.
    return tf.concat([tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)], axis=0)  # tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_paths = 0
        self.buffer = deque()

    def get_sample(self, sample_size):
        if self.num_paths < sample_size:
            return random.sample_transitions(self.buffer, self.num_paths)
        else:
            return random.sample_transitions(self.buffer, sample_size)

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
    for p in range(paths_per_collect):
        if verbose:
            print("Rollout index:", p)

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
            print("No starting index. Skipping path")
            continue
        state = np.array(env.inv_states_dict[state_idx])
        state = np.expand_dims(state, axis=0)

        # state = get_initial_state_SD(env, encode)
        reward_d = 0
        reward_p = 0

        for i in range(max_step_limit):

            # Take action
            action = agent.act(state, encode)  # Action here is not a prob dis. but the one-hot encoding of taken action

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
                print(state[0], "->", env.inv_action_dict[action_id])

            terminate, reward = env_resp.gen_response(current_state_idx, action_id, next_state_idx)

            states.append(state)
            encodes.append(encode)
            actions.append(action)
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