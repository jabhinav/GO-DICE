import os
import sys
import time
import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from typing import Dict, List, Tuple, Union
from utils.env import get_PnP_env, get_config_env
from keras.layers import Dense, Flatten, Add, Concatenate, LeakyReLU
import tensorflow_probability as tfp
from domains.PnP import PnPEnv, MyPnPEnvWrapperForGoalGAIL, PnPExpert, PnPExpertTwoObj
from her.rollout import RolloutWorker
from her.replay_buffer import ReplayBuffer
from her.transitions import make_sample_her_transitions

logger = logging.getLogger(__name__)


class Actor(tf.keras.Model):
    def __init__(self, a_dim, actions_max):
        super(Actor, self).__init__()

        self.max_actions = actions_max
        self.fc1 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.fc2 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.a_out = Dense(units=a_dim, activation=tf.nn.tanh, kernel_initializer=tf.keras.initializers.GlorotUniform())

    def call(self, curr_state, goal_state):
        ip = tf.concat([curr_state, goal_state], axis=1)
        h = self.fc1(ip)
        h = self.fc2(h)
        actions = self.a_out(h) * self.max_actions
        return actions


class Critic(tf.keras.Model):
    def __init__(self, actions_max):
        super(Critic, self).__init__()

        self.max_actions = actions_max
        self.fc1 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.fc2 = Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.q_out = Dense(units=1, activation=None, kernel_initializer=tf.keras.initializers.GlorotUniform())

    def call(self, state, goal, actions):
        ip = tf.concat([state, goal, actions / self.max_actions], axis=1)
        h = self.fc1(ip)
        h = self.fc2(h)
        q = self.q_out(h)
        return q


class Discriminator(tf.keras.Model):
    def __init__(self, rew_type: str = 'negative'):
        super(Discriminator, self).__init__()
        kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)
        self.concat = Concatenate()
        self.fc1 = Dense(units=256, activation=tf.nn.tanh, kernel_initializer=kernel_init)
        self.fc2 = Dense(units=256, activation=tf.nn.tanh, kernel_initializer=kernel_init)
        self.d_out = Dense(units=1, kernel_initializer=kernel_init)

        self.rew_type: str = rew_type

    def call(self, state, goal, action):
        ip = self.concat([state, goal, action])
        h = self.fc1(ip)
        h = self.fc2(h)
        d_out = self.d_out(h)
        return d_out

    def get_reward(self, state, goal, action):
        # Compute the Discriminator Output
        ip = self.concat([state, goal, action])
        h = self.fc1(ip)
        h = self.fc2(h)
        d_out = self.d_out(h)

        # Convert the output into reward
        if self.rew_type == 'airl':
            return d_out
        elif self.rew_type == 'gail':
            return -tf.math.log(1 - tf.nn.sigmoid(d_out) + 1e-8)
        elif self.rew_type == 'normalized':
            return tf.nn.sigmoid(d_out)
        elif self.rew_type == 'negative':
            return tf.math.log(tf.nn.sigmoid(d_out) + 1e-8)
        else:
            print("Specify the correct reward type")
            raise NotImplementedError


class DDPG(object):
    def __init__(self, env: MyPnPEnvWrapperForGoalGAIL, args):
        self.env = env
        self.args = args
        self.a_dim: int = args.a_dim
        self.s_dim: int = args.s_dim
        self.g_dim: int = args.g_dim

        # Declare Networks
        self.actor_main = Actor(self.a_dim, args.action_max)
        self.actor_target = Actor(self.a_dim, args.action_max)
        self.critic_main = Critic(args.action_max)
        self.critic_target = Critic(args.action_max)

        # Define Optimisers
        self.a_lr = tf.Variable(args.a_lr, trainable=False)
        self.a_opt = tf.keras.optimizers.Adam(args.a_lr)
        self.c_lr = tf.Variable(args.c_lr, trainable=False)
        self.c_opt = tf.keras.optimizers.Adam(args.c_lr)

        # Compile the target networks so that no errors are thrown while copying weights
        self.actor_target.compile(optimizer=self.a_opt)
        self.critic_target.compile(optimizer=self.c_opt)

    def set_learning_rate(self, actor_learning_rate=None, critic_learning_rate=None):
        """Update learning rate."""
        if actor_learning_rate:
            self.a_lr.assign(actor_learning_rate)
        if critic_learning_rate:
            self.c_lr.assign(critic_learning_rate)

    def compute_Qsga(self, state, goal, action_mu, use_target_network=False):
        """
            This will be called after action is computed and state, goal are processed
        """
        if use_target_network:
            Q = self.critic_target(state, goal, action_mu)
        else:
            Q = self.critic_main(state, goal, action_mu)
        return Q

    def act(self, state, achieved_goal, goal, noise_eps=0., random_eps=0.,
            use_target_net=False, compute_Q=False, **kwargs):
        # Pre-process the state and goal
        if self.args.relative_goals:
            goal = goal - achieved_goal
        state = tf.clip_by_value(state, -self.args.clip_obs, self.args.clip_obs)
        goal = tf.clip_by_value(goal, -self.args.clip_obs, self.args.clip_obs)

        # Predict action
        if use_target_net:
            action_mu = self.actor_target(state, goal)
        else:
            action_mu = self.actor_main(state, goal)

        # # Action Post-Processing
        # First add gaussian noise and clip
        noise = noise_eps * self.args.action_max * tf.experimental.numpy.random.rand(*action_mu.shape)
        action = action_mu + tf.cast(noise, tf.float32)
        action = tf.clip_by_value(action, -self.args.action_max, self.args.action_max)

        # Take epsilon greedy action
        random_action = self._random_action(action.shape)

        action_sampler = tfp.distributions.Binomial(1, random_eps)
        choose = action_sampler.sample(sample_shape=(action.shape[0], 1))
        action += choose * (random_action - action)

        if compute_Q:
            Q = self.compute_Qsga(state, goal, action_mu, use_target_net)
            return action, Q
        return action

    def _random_action(self, shape):
        return tf.random.uniform(shape, -self.args.action_max, self.args.action_max)

    def update_targets(self, tau=None):
        if tau is None:
            tau = self.args.tau

        # Update target networks using Polyak averaging.
        self.actor_target.set_weights(
            tau * self.actor_main.get_weights() + (1 - tau) * self.actor_target.get_weights())
        self.critic_target.set_weights(
            tau * self.critic_main.get_weights() + (1 - tau) * self.critic_target.get_weights())

    def compute_loss(self, data):
        target_actions = self.actor_target(data['next_states'], data['goal_states'])  # a' <- Pi(s', g)
        next_state_quality = self.critic_target(data['next_states'], data['goal_states'],
                                                target_actions)  # Q(s', g, a')
        next_state_quality = tf.squeeze(next_state_quality, axis=1)
        critic_values = self.critic_main(data['states'], data['goal_states'], data['actions'])  # Q(s, g, a)
        critic_values = tf.squeeze(critic_values, axis=1)
        target_values = data['rewards'] + self.args.gamma * next_state_quality * data[
            'dones']  # y = r + gamma * Q(s', g, a') * done
        critic_loss = self.args.vf_coeff * tf.keras.losses.MSE(target_values,
                                                               critic_values)  # J(Q, y) = (Q(s, g, a) - y)^2

        # Actor loss: -Q(s,g,a)
        curr_actions = self.actor_main(data['states'], data['goal_states'])
        actor_loss = -tf.reduce_mean(self.critic_main(data['states'], data['goal_states'], curr_actions))

        return actor_loss, critic_loss

    # @tf.function
    def train(self, data: Dict, annealing_factor=1., q_annealing=1.):

        # Compute DDPG losses
        with tf.GradientTape as actor_tape, tf.GradientTape as critic_tape:
            actor_loss, critic_loss = self.compute_loss(data)

        gradients_actor = actor_tape.gradient(actor_loss, self.actor_main.trainable_variables)
        gradients_critic = critic_tape.gradient(critic_loss, self.critic_main.trainable_variables)
        self.a_opt.apply_gradients(zip(gradients_actor, self.actor_main.trainable_variables))
        self.c_opt.apply_gradients(zip(gradients_critic, self.critic_main.trainable_variables))

    def load_model(self, param_dir):
        # BUILD First
        _ = self.actor_main(np.ones(np.ones([1, self.s_dim]), np.ones([1, self.g_dim])))

        # Load Models
        self.actor_main.load_weights(os.path.join(param_dir, "actor_main.h5"))

    def save_model(self, param_dir):
        # Save weights
        self.actor_main.save_weights(os.path.join(param_dir, "actor_main.h5"), overwrite=True)
        self.critic_main.save_weights(os.path.join(param_dir, "critic_main.h5"), overwrite=True)
        self.actor_target.save_weights(os.path.join(param_dir, "actor_target.h5"), overwrite=True)
        self.critic_target.save_weights(os.path.join(param_dir, "critic_target.h5"), overwrite=True)


class Agent(object):
    def __init__(self,
                 args,
                 buffer_shape: Dict[str, Tuple],
                 expert_buffer: ReplayBuffer = None,
                 gail_discriminator: Discriminator = None):

        self.args = args
        self.buffer_shape = buffer_shape

        # Declare Environment for Policy
        env = get_PnP_env(args)
        self.env: MyPnPEnvWrapperForGoalGAIL = env

        # Define Policy
        self.policy = DDPG(env, args)

        # Declare Discriminator
        self.discriminator = gail_discriminator

        # Define the Transition function to map episodic data to transitional data (passed env will be used)
        sample_her_transitions_pol = make_sample_her_transitions(args.replay_strategy, args.replay_k, env.reward_fn,
                                                                 env,
                                                                 discriminator=gail_discriminator,
                                                                 gail_weight=args.gail_weight,
                                                                 two_rs=args.two_rs and args.anneal_disc,
                                                                 with_termination=True)

        # Define the Buffers
        self.init_state = None
        self.on_policy_buffer = ReplayBuffer(buffer_shape, args.buffer_size, args.horizon, sample_her_transitions_pol)
        self.expert_buffer = expert_buffer

        # ROLLOUT WORKER
        self.policy_rollout_worker = RolloutWorker(self.env, self.policy, T=args.horizon, rollout_terminate=False,
                                                   exploit=False, noise_eps=args.noise_eps, random_eps=args.random_eps,
                                                   use_target_net=True, render=False)

        # Define Optimisers
        self.d_lr = tf.Variable(args.d_lr, trainable=False)
        self.d_opt = tf.keras.optimizers.Adam(args.d_lr)

        # # Define Losses
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def set_learning_rate(self, disc_learning_rate=None, actor_learning_rate=None, critic_learning_rate=None):
        """Update learning rate."""
        if disc_learning_rate:
            self.d_lr.assign(disc_learning_rate)
        self.policy.set_learning_rate(actor_learning_rate, critic_learning_rate)

    def load_model(self, param_dir):
        self.policy.load_model(param_dir)

    def save_model(self, param_dir):
        # Save weights
        self.discriminator.save_weights(os.path.join(param_dir, "discriminator.h5"), overwrite=True)
        self.policy.save_model(param_dir)

    # @tf.function
    def disc_train(self, sampled_data, expert_data):
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
            d_loss = fake_loss + real_loss + self.args.lambd * grad_penalty

        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return d_loss

    def _preprocess_og(self, states, achieved_goals, goals):
        if self.args.relative_goals:
            goals = goals - achieved_goals
        states = tf.clip_by_value(states, -self.args.clip_obs, self.args.clip_obs)
        goals = tf.clip_by_value(goals, -self.args.clip_obs, self.args.clip_obs)
        return states, goals

    def process_transitions(self, transitions, expert=False, annealing_factor=1., w_q2=1.):
        states, achieved_goals, goals = transitions['states'], transitions['achieved_goals'], transitions['goals']
        states_2, achieved_goals_2 = transitions['states_2'], transitions['achieved_goals_2']

        transitions['states'], transitions['goals'] = self._preprocess_og(states, achieved_goals, goals)
        transitions['states_2'], transitions['goals_2'] = self._preprocess_og(states_2, achieved_goals_2, goals)
        transitions['is_demo'] = tf.cast(int(expert) * tf.ones_like(transitions['r']), tf.float32)
        transitions['annealing_factor'] = tf.cast(annealing_factor * tf.ones_like(transitions['r']), tf.float32)
        if self.args.anneal_disc:
            transitions['r'] = transitions['r'] + w_q2 * transitions['rd']
        transitions_batch = [transitions[key] for key in transitions.keys()]
        return transitions_batch

    def sample_data(self, batch_size, expert_batch_size, annealing_factor=1., w_q2=1.):

        transitions_batch, expert_transitions_batch = None, None

        # Sample Policy Transitions
        if batch_size > 0:
            policy_transitions = self.on_policy_buffer.sample(batch_size)
            transitions_batch = self.process_transitions(policy_transitions, w_q2=w_q2)

        # Sample Expert Transitions
        if self.expert_buffer and expert_batch_size > 0:
            expert_transitions = self.expert_buffer.sample(batch_size)
            expert_transitions_batch = self.process_transitions(expert_transitions, expert=True,
                                                                annealing_factor=annealing_factor, w_q2=w_q2)

        return transitions_batch, expert_transitions_batch

    def train(self):
        args = self.args

        for outer_iter in range(1, args.outer_iters):
            annealing_factor = args.annealing_coeff ** outer_iter
            q_annealing = args.q_annealing ** (outer_iter - 1)

            with tqdm(total=args.num_epochs * args.num_cycles, position=0, leave=True) as pbar:

                for epoch in range(args.num_epochs):

                    self.policy_rollout_worker.clear_history()
                    for cycle in range(args.num_cycles):

                        iter_num = epoch * args.num_cycles + cycle

                        # ############################################################################################ #
                        # #################################### Collect/Process Data ################################## #
                        # ############################################################################################ #
                        start = time.time()

                        # Collect and store episodes
                        for ep_num in range(args.collect_episodes):
                            episode = self.policy_rollout_worker.generate_rollout(slice_goal=(3, 6)
                            if args.full_space_as_goal else None,
                                                                                  compute_Q=True)
                            self.on_policy_buffer.store_episode(episode)

                        for _ in range(args.n_batches):
                            data_pol, data_exp = self.sample_data(batch_size=args.batch_size,
                                                                  expert_batch_size=args.batch_size,
                                                                  annealing_factor=annealing_factor, w_q2=q_annealing)
                            # Train Policy on each batch

                        # Update Policy Target Network

                        # To train Discriminator per cycle of Policy Training
                        if args.train_dis_per_rollout and args.n_batches_disc > 0 and not (
                                epoch == args.n_epochs - 1 and cycle == args.n_cycles - 1):
                            # Empty the on policy buffer
                            pass

                        pbar.update(1)

                    # To train Discriminator per few cycles of Policy Training
                    if not args.train_dis_per_rollout and args.n_batches_disc > 0 and epoch != args.n_epochs - 1:
                        # Empty the on policy buffer
                        pass


def run(args, store_data_path=None):
    # Load the environment config
    args = get_config_env(args)

    logger.info("\n\n---------------------------------------------------------------------------------------------")
    logger.info(args)

    # # Root Directory
    # root_dir = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/'

    # Two Object Env: target_in_the_air activates only for 2-object case
    exp_env = get_PnP_env(args)

    buffer_shape = {
        'states': (args.horizon + 1, args.s_dim),
        'achieved_goals': (args.horizon + 1, args.g_dim),
        'goals': (args.horizon, args.g_dim),
        'actions': (args.horizon, args.a_dim),
        'successes': (args.horizon,)
    }

    # Define and Load the Discriminator Network - GAIL
    gail_discriminator = Discriminator(rew_type=args.rew_type)

    # Define the Transition function to map episodic data to transitional data
    sample_her_transitions_exp = make_sample_her_transitions(args.replay_strategy, args.replay_k,
                                                             exp_env.reward_fn, exp_env,
                                                             discriminator=gail_discriminator,
                                                             gail_weight=args.gail_weight,
                                                             two_rs=args.two_rs and args.anneal_disc,
                                                             with_termination=True)

    # ###################################################### Expert ################################################# #
    logger.info("Generating Expert Data")
    # Load Expert Policy
    if args.two_object:
        expert_policy = PnPExpertTwoObj(exp_env, args.full_space_as_goal, expert_behaviour=args.expert_behaviour)
    else:
        expert_policy = PnPExpert(exp_env, args.full_space_as_goal)
    # Load Buffer to store expert data
    expert_buffer = ReplayBuffer(buffer_shape, args.buffer_size, args.horizon, sample_her_transitions_exp)
    # Initiate a worker to generate expert rollouts
    expert_worker = RolloutWorker(exp_env, expert_policy, T=args.horizon, rollout_terminate=True,
                                  exploit=True, noise_eps=0, random_eps=0, use_target_net=False,
                                  render=False)
    # Generate and store expert data
    for i in range(args.num_demos):
        print("\nGenerating demo:", i + 1)
        expert_worker.policy.reset()
        _episode = expert_worker.generate_rollout(slice_goal=(3, 6) if args.full_space_as_goal else None)
        expert_buffer.store_episode(_episode)

    # ################################################### Store Data ############################################## #
    if store_data_path:
        expert_buffer.save_buffer_data(path=store_data_path)

    # ################################################### Train Model ############################################## #
    else:
        agent = Agent(args, buffer_shape, expert_buffer, gail_discriminator)
        agent.train()

    sys.exit(-1)
