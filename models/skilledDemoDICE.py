import copy
import logging
import time
from abc import ABC
from argparse import Namespace

import numpy as np
import tensorflow as tf
from tensorflow_gan.python.losses import losses_impl as tfgan_losses
from tqdm import tqdm

from her.replay_buffer import ReplayBufferTf
from networks.general import SkilledActors, Critic, Discriminator
from utils.env import get_expert
from .Base import AgentBase

logger = logging.getLogger(__name__)


class skilledDemoDICE(tf.keras.Model, ABC):
    def __init__(self, args: Namespace):
        super(skilledDemoDICE, self).__init__()
        self.args = args

        self.args.EPS = np.finfo(np.float32).eps  # Small value = 1.192e-07 to avoid division by zero in grad penalty
        self.args.EPS2 = 1e-3

        # Define Networks
        self.skilled_actors = SkilledActors(args.a_dim, args.c_dim)
        self.critic = Critic()
        self.disc = Discriminator()

        # Define Target Networks [Target Actors and Directors are already defined in SkilledActors]
        self.critic_target = Critic()
        self.critic_target.trainable = False

        # Define Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=args.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=args.critic_lr)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=args.disc_lr)

        self.build_model()

        self.act_w_expert_skill = False
        self.act_w_expert_action = False
        self.expert = get_expert(args.num_objs, args)

        # For HER
        self.use_her = False
        logger.info('[[[ Using HER ? ]]]: {}'.format(self.use_her))

    @tf.function(experimental_relax_shapes=True)
    def train(self, data_exp, data_rb):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(self.skilled_actors.variables)
            tape.watch(self.critic.variables)
            tape.watch(self.disc.variables)

            # Form Inputs
            init_rb_ip = tf.concat([data_rb['prev_skills'], data_rb['init_states'], data_rb['goals']], 1)
            curr_exp_ip = tf.concat([data_exp['prev_skills'], data_exp['states'], data_exp['goals']], 1)
            next_exp_ip = tf.concat([data_exp['curr_skills'], data_exp['states_2'], data_exp['goals']], 1)
            curr_rb_ip = tf.concat([data_rb['prev_skills'], data_rb['states'], data_rb['goals']], 1)
            next_rb_ip = tf.concat([data_rb['curr_skills'], data_rb['states_2'], data_rb['goals']], 1)

            disc_expert_inputs = tf.concat([curr_exp_ip, data_exp['curr_skills'], data_exp['actions']], 1)
            disc_rb_inputs = tf.concat([curr_rb_ip, data_rb['curr_skills'], data_rb['actions']], 1)

            # Compute cost of (c'_E, s_E, c_E, a_E ; g_E)
            cost_expert = self.disc(disc_expert_inputs)
            # Compute cost of (c'_R, s_R, c_R, a_R ; g_R)
            cost_rb = self.disc(disc_rb_inputs)
            # Get Reward from Discriminator = log-dist ratio between expert and rb.
            # This is the reward = - log (1/c(c',s,c,a;g) - 1)
            reward = - tf.math.log(1 / (tf.nn.sigmoid(cost_rb) + self.args.EPS2) - 1 + self.args.EPS2)

            # Compute Discriminator loss
            cost_loss = tfgan_losses.modified_discriminator_loss(cost_expert, cost_rb, label_smoothing=0.)

            # Compute gradient penalty for Discriminator
            alpha = tf.random.uniform(shape=(disc_expert_inputs.shape[0], 1))
            interpolates_1 = alpha * disc_expert_inputs + (1 - alpha) * disc_rb_inputs
            interpolates_2 = alpha * tf.random.shuffle(disc_rb_inputs) + (1 - alpha) * disc_rb_inputs
            interpolates = tf.concat([interpolates_1, interpolates_2], axis=0)
            with tf.GradientTape() as tape2:
                tape2.watch(interpolates)
                cost_interpolates = self.disc(interpolates)
                cost_interpolates = tf.math.log(
                    1 / (tf.nn.sigmoid(cost_interpolates) + self.args.EPS2) - 1 + self.args.EPS2)
            cost_grads = tape2.gradient(cost_interpolates, [interpolates])[0] + self.args.EPS
            cost_grad_penalty = tf.reduce_mean(tf.square(tf.norm(cost_grads, axis=1, keepdims=True) - 1))
            cost_loss_w_pen = cost_loss + self.args.cost_grad_penalty_coeff * cost_grad_penalty

            # Compute the value function
            init_nu = self.critic(init_rb_ip)
            expert_nu = self.critic(curr_exp_ip)  # not used in loss calc
            rb_nu = self.critic(curr_rb_ip)
            rb_nu_next = self.critic(next_rb_ip)

            # Compute the Advantage function (on replay buffer)
            rb_adv = tf.stop_gradient(reward) + self.args.discount * rb_nu_next - rb_nu

            # Linear Loss = (1 - gamma) * E[init_nu]
            linear_loss = (1 - self.args.discount) * tf.reduce_mean(init_nu)
            # Non-Linear Loss = (1 + alpha) * E[exp(Adv_nu / (1 + alpha))]
            non_linear_loss = (1 + self.args.replay_regularization) * tf.reduce_logsumexp(
                rb_adv / (1 + self.args.replay_regularization))
            nu_loss = linear_loss + non_linear_loss

            # Compute gradient penalty for nu
            beta = tf.random.uniform(shape=(disc_expert_inputs.shape[0], 1))
            nu_inter = beta * curr_exp_ip + (1 - beta) * curr_rb_ip
            nu_next_inter = beta * next_exp_ip + (1 - beta) * next_rb_ip
            nu_input = tf.concat([curr_exp_ip, nu_inter, nu_next_inter], 0)
            with tf.GradientTape(watch_accessed_variables=False) as tape3:
                tape3.watch(nu_input)
                nu_output = self.critic(nu_input)
            nu_grad = tape3.gradient(nu_output, [nu_input])[0] + self.args.EPS
            nu_grad_penalty = tf.reduce_mean(tf.square(tf.norm(nu_grad, axis=-1, keepdims=True) - 1))
            nu_loss_w_pen = nu_loss + self.args.nu_grad_penalty_coeff * nu_grad_penalty

            # Compute Actor and Director Loss : Weighted BC Loss with the Advantage function
            weight = tf.math.exp(rb_adv / (1 + self.args.replay_regularization))
            weight = tf.reshape(weight, (-1, 1))
            weight = tf.expand_dims(weight, -1)
            weight = weight / tf.reduce_mean(weight)  # Normalise weight using self-normalised importance sampling

            # Compute the log probs of current skill using the director
            curr_skill_log_prob = self.skilled_actors.get_director_log_probs(
                tf.concat([data_rb['states'], data_rb['goals']], 1),
                data_rb['prev_skills'],
                data_rb['curr_skills']
            )

            # Compute the log probs of current action using the actor
            curr_action_log_prob = self.skilled_actors.get_actor_log_probs(
                tf.concat([data_rb['states'], data_rb['goals']], 1),
                data_rb['curr_skills'],
                data_rb['actions']
            )

            # The current skill index will give the actor network to update
            pi_loss = - tf.reduce_mean(tf.stop_gradient(weight) * (curr_skill_log_prob + curr_action_log_prob))

        nu_grads = tape.gradient(nu_loss_w_pen, self.critic.variables)
        pi_grads = tape.gradient(pi_loss, self.skilled_actors.variables)
        cost_grads = tape.gradient(cost_loss_w_pen, self.disc.variables)

        self.critic_optimizer.apply_gradients(zip(nu_grads, self.critic.variables))
        self.actor_optimizer.apply_gradients(zip(pi_grads, self.skilled_actors.variables))
        self.disc_optimizer.apply_gradients(zip(cost_grads, self.disc.variables))

        return {
            'loss/cost': cost_loss,
            'loss/linear': linear_loss,
            'loss/non-linear': non_linear_loss,
            'loss/nu': nu_loss,
            'loss/pi': pi_loss,

            'penalty/cost_grad_penalty': self.args.cost_grad_penalty_coeff * cost_grad_penalty,
            'penalty/nu_grad_penalty': self.args.nu_grad_penalty_coeff * nu_grad_penalty,

            'avg_nu/expert': tf.reduce_mean(expert_nu),
            'avg_nu/rb': tf.reduce_mean(rb_nu),
            'avg_nu/init': tf.reduce_mean(init_nu),
            'avg/rb_adv': tf.reduce_mean(rb_adv),
        }

    @tf.function(experimental_relax_shapes=True)
    def pretrain(self, data_exp):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(self.skilled_actors.variables)

            # Compute the log probs of current skill using the director
            curr_skill_log_prob = self.skilled_actors.get_director_log_probs(
                tf.concat([data_exp['states'], data_exp['goals']], 1),
                data_exp['prev_skills'],
                data_exp['curr_skills']
            )
            loss_directors = - tf.reduce_mean(curr_skill_log_prob)

            # Compute the log probs of current action using the actor
            curr_action_log_prob = self.skilled_actors.get_actor_log_probs(
                tf.concat([data_exp['states'], data_exp['goals']], 1),
                data_exp['curr_skills'],
                data_exp['actions']
            )
            loss_actors = - tf.reduce_mean(curr_action_log_prob)

            total_loss = loss_directors + loss_actors

            # [For Debugging], extract log_prob for actions corresponding to each curr skill
            per_skill_loss = {}
            for c in range(self.args.c_dim):
                t = tf.where(tf.equal(tf.argmax(data_exp['curr_skills'], axis=-1), c))
                per_skill_loss[f'loss/skill{c}'] = - tf.reduce_mean(tf.gather(curr_action_log_prob, t))

        grads = tape.gradient(total_loss, self.skilled_actors.variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.skilled_actors.variables))

        loss_dict = {
            'loss/pi': total_loss,
            'loss/actors': loss_actors,
            'loss/directors': loss_directors
        }
        loss_dict.update(per_skill_loss)

        return loss_dict

    @tf.function(experimental_relax_shapes=True)  # This is needed to avoid shape errors
    def act(self, state, env_goal, prev_goal, prev_skill, epsilon, stddev):
        state = tf.clip_by_value(state, -self.args.clip_obs, self.args.clip_obs)
        env_goal = tf.clip_by_value(env_goal, -self.args.clip_obs, self.args.clip_obs)
        prev_goal = tf.clip_by_value(prev_goal, -self.args.clip_obs, self.args.clip_obs)

        # ###################################### Current Goal ####################################### #
        curr_goal = env_goal

        # ###################################### Current Skill ###################################### #
        if self.act_w_expert_skill:
            curr_skill = tf.numpy_function(self.expert.sample_curr_skill, [state[0], env_goal[0], prev_skill[0]],
                                           tf.float32)
            curr_skill = tf.expand_dims(curr_skill, axis=0)
        else:
            # Get the director corresponding to the previous skill and obtain the current skill
            _, curr_skill, _ = self.skilled_actors.call_director(tf.concat([state, curr_goal], axis=1), prev_skill)

        # ########################################## Action ######################################### #
        # Explore
        if tf.random.uniform(()) < epsilon:
            action = tf.random.uniform((1, self.args.a_dim), -self.args.action_max, self.args.action_max)

        # Exploit
        else:
            # Get the actor corresponding to the current skill and obtain the action
            action_mu, _, _ = self.skilled_actors.call_actor(tf.concat([state, curr_goal], axis=1), curr_skill)
            # Add noise to action
            action_dev = tf.random.normal(tf.shape(action_mu), mean=0.0, stddev=stddev)
            action = action_mu + action_dev

            if self.act_w_expert_action:
                action = tf.numpy_function(self.expert.sample_action, [state[0], env_goal[0], curr_skill[0], action[0]],
                                           tf.float32)
                action = tf.expand_dims(action, axis=0)

            action = tf.clip_by_value(action, -self.args.action_max, self.args.action_max)

        # Safety check for action, should not be nan or inf
        has_nan = tf.math.reduce_any(tf.math.is_nan(action))
        has_inf = tf.math.reduce_any(tf.math.is_inf(action))
        if has_nan or has_inf:
            logger.warning('Action has nan or inf. Setting action to zero. Action: {}'.format(action))
            action = tf.zeros_like(action)

        return curr_goal, curr_skill, action

    def get_init_skill(self):
        """
        One-Obj: Pick Object 0 i.e. [1, 0, 0]
        Two-Obj: Pick Object 0 i.e. [1, 0, 0, 0, 0, 0] if expert_behaviour = 0
        Two-Obj: Pick Object 0 i.e. [1, 0, 0, 0, 0, 0] or [0, 0, 0, 1, 0, 0] if expert_behaviour = 1
        Three-Obj with stacking: Pick Object 0 i.e. [1, 0, 0, 0, 0, 0, 0, 0, 0]
        """
        if self.args.num_objs == 1:
            skill = tf.one_hot(0, self.args.c_dim, dtype=tf.float32)

        elif self.args.num_objs == 2:
            if self.args.expert_behaviour == '0':
                skill = tf.one_hot(0, self.args.c_dim, dtype=tf.float32)  # Pick Object 0 first (default)
            #
            # elif self.args.expert_behaviour == 1:
            #
            # 	if tf.random.uniform(shape=()) < 0.5:
            # 		skill = tf.one_hot(0, self.args.c_dim, dtype=tf.float32)
            # 	else:
            # 		skill = tf.one_hot(3, self.args.c_dim, dtype=tf.float32)

            else:
                raise ValueError("Invalid expert behaviour to determine init skill in two-object environment: " + str(
                    self.args.expert_behaviour))

        elif self.args.num_objs == 3:
            skill = tf.one_hot(0, self.args.c_dim, dtype=tf.float32)

        else:
            raise ValueError("Invalid number of objects: " + str(self.args.num_objs))

        skill = tf.reshape(skill, shape=(1, -1))
        return skill

    @staticmethod
    def get_init_goal(init_state, g_env):
        return g_env

    def save_(self, dir_param):
        for i in range(len(self.skilled_actors.directors)):
            self.skilled_actors.directors[i].save_weights(dir_param + "/director_" + str(i) + ".h5")
        for i in range(len(self.skilled_actors.actors)):
            self.skilled_actors.actors[i].save_weights(dir_param + "/policy_" + str(i) + ".h5")
        self.critic.save_weights(dir_param + "/nu_net.h5")
        self.disc.save_weights(dir_param + "/cost_net.h5")

    def load_(self, dir_param):
        for i in range(len(self.skilled_actors.directors)):
            self.skilled_actors.directors[i].load_weights(dir_param + "/director_" + str(i) + ".h5")
        for i in range(len(self.skilled_actors.actors)):
            self.skilled_actors.actors[i].load_weights(dir_param + "/policy_" + str(i) + ".h5")
        self.critic.load_weights(dir_param + "/nu_net.h5")
        self.disc.load_weights(dir_param + "/cost_net.h5")

    def build_model(self):

        # Define the Directors (for each previous skill, there is a director to determine the current skill)
        for director in self.skilled_actors.directors:
            _ = director(tf.concat([tf.ones([1, self.args.s_dim]), tf.ones([1, self.args.g_dim])], 1))

        # Define the Actors (for each current skill, there is an actor to determine the action)
        for actor in self.skilled_actors.actors:
            _ = actor(tf.concat([tf.ones([1, self.args.s_dim]), tf.ones([1, self.args.g_dim])], 1))

        # Define the Directors (for each previous skill, there is a director to determine the current skill)
        for director in self.skilled_actors.target_directors:
            _ = director(tf.concat([tf.ones([1, self.args.s_dim]), tf.ones([1, self.args.g_dim])], 1))

        # Define the Actors (for each current skill, there is an actor to determine the action)
        for actor in self.skilled_actors.target_actors:
            _ = actor(tf.concat([tf.ones([1, self.args.s_dim]), tf.ones([1, self.args.g_dim])], 1))

        # Define the Critic to estimate the value of (prev_skill, state, goal)
        _ = self.critic(
            np.ones([1, self.args.c_dim]),
            np.ones([1, self.args.s_dim]),
            np.ones([1, self.args.g_dim]),
        )

        _ = self.critic_target(
            np.ones([1, self.args.c_dim]),
            np.ones([1, self.args.s_dim]),
            np.ones([1, self.args.g_dim]),
        )

        # Define the Discriminator to estimate distribution of (prev_skill, state, goal, current_skill, action)
        _ = self.disc(
            np.ones([1, self.args.c_dim]),
            np.ones([1, self.args.s_dim]),
            np.ones([1, self.args.g_dim]),
            np.ones([1, self.args.c_dim]),
            np.ones([1, self.args.a_dim]),
        )

    def change_training_mode(self, training_mode: bool):
        self.skilled_actors.change_training_mode(training_mode)

    def update_target_networks(self):

        # Update the weights of the target actors and directors using Polyak averaging
        for i in range(self.args.c_dim):

            # Update the reference actor
            trg_actor_weights = self.skilled_actors.target_actors[i].get_weights()
            actor_weights = self.skilled_actors.actors[i].get_weights()
            for j in range(len(actor_weights)):
                trg_actor_weights[j] = self.args.actor_polyak * trg_actor_weights[j] + \
                                       (1 - self.args.actor_polyak) * actor_weights[j]
            self.skilled_actors.target_actors[i].set_weights(trg_actor_weights)

            # Update the reference director
            trg_director_weights = self.skilled_actors.target_directors[i].get_weights()
            director_weights = self.skilled_actors.directors[i].get_weights()
            for j in range(len(director_weights)):
                trg_director_weights[j] = self.args.director_polyak * trg_director_weights[j] + \
                                          (1 - self.args.director_polyak) * director_weights[j]
            self.skilled_actors.target_directors[i].set_weights(trg_director_weights)

    # Update the weights of the target critic using Polyak averaging


class Agent(AgentBase):
    def __init__(self, args,
                 expert_buffer: ReplayBufferTf = None,
                 offline_buffer: ReplayBufferTf = None):

        super().__init__(args, skilledDemoDICE(args), 'skilledDemoDICE', expert_buffer, offline_buffer)

    def load_actor(self, dir_param):
        self.model.skilled_actors.load_weights(dir_param + "/policy.h5")

    @tf.function
    def infer_skills(self, buffered_data, gt_curr_skill, compute_viterbi_acc=False):

        avg_viterbi_acc_per_step = 0.0
        avg_viterbi_acc_per_ep = 0.0
        log_probs = []

        # # Do viterbi decoding to get the best skill sequence for each episode and store it in the buffer
        new_prev_skills = []
        new_curr_skills = []
        num_episodes = len(buffered_data['prev_skills'])
        for ep_idx in range(0, num_episodes, 1):
            # Get the init skill for the episode
            init_skill = buffered_data['prev_skills'][ep_idx][0]

            # Collect the states [0:T-1] from given [0:T], i.e. exclude terminal state
            states = buffered_data['states'][ep_idx]
            states = tf.gather(states, tf.range(0, tf.shape(states)[0] - 1))
            env_goals = buffered_data['env_goals'][ep_idx]
            env_goals = tf.gather(env_goals, tf.range(0, tf.shape(env_goals)[0] - 1))
            states = tf.concat([states, env_goals], axis=1)

            # Collect the actions [0:T-1]
            actions = buffered_data['actions'][ep_idx]

            # Get the skill sequence for the episode
            skill_seq, log_prob = self.model.skilled_actors.viterbi_decode(states=states,
                                                                           actions=actions,
                                                                           init_skill=init_skill,
                                                                           use_ref=True)
            # Convert the skill sequence (T+1, 1) to one-hot encoding
            skill_seq = tf.one_hot(tf.squeeze(skill_seq, axis=-1), depth=self.args.c_dim)

            # Store the log_prob of the viterbi decoded skill sequence
            log_probs.append(log_prob)
            # Update the buffer with the viterbi decoded skill sequence
            new_prev_skills.append(tf.gather(skill_seq, tf.range(0, tf.shape(skill_seq)[0] - 1)))
            new_curr_skills.append(tf.gather(skill_seq, tf.range(1, tf.shape(skill_seq)[0])))

            # Viterbi's accuracy: avg. no. of correct skills in the viterbi decoded skill sequence
            if compute_viterbi_acc:
                per_timestep_acc = tf.equal(tf.argmax(gt_curr_skill[ep_idx], axis=-1),
                                            tf.argmax(skill_seq[1:], axis=-1))
                avg_viterbi_acc_per_step += tf.reduce_mean(tf.cast(per_timestep_acc, dtype=tf.float32))
                avg_viterbi_acc_per_ep += tf.cast(tf.reduce_all(per_timestep_acc), dtype=tf.float32)

        new_prev_skills = tf.stack(new_prev_skills, axis=0)
        new_curr_skills = tf.stack(new_curr_skills, axis=0)

        result = {
            'per_step_viterbi_acc': avg_viterbi_acc_per_step / num_episodes,
            'per_ep_viterbi_acc': avg_viterbi_acc_per_ep / num_episodes,
            'avg_log_prob': tf.reduce_mean(log_probs),
        }

        return result, new_prev_skills, new_curr_skills

    @tf.function
    def pretrain(self):

        # Pretrain the actors and directors with the expert data
        data_expert = self.sample_data(self.expert_buffer, self.args.batch_size)
        loss_dict = self.model.pretrain(data_expert)
        return loss_dict

    def learn(self):
        args = self.args
        max_return, max_return_with_exp_assist = None, None
        log_step = 0

        # Saving G.T. skill sequences for verifying Viterbi Decoding
        data_exp = self.expert_buffer.sample_episodes()
        c_exp_curr_gt = copy.deepcopy(data_exp['curr_skills'])
        data_off = self.offline_buffer.sample_episodes()
        c_off_curr_gt = copy.deepcopy(data_off['curr_skills'])

        # Load the expert data into the expert buffer
        self.expert_buffer.load_data_into_buffer(buffered_data=data_exp, clear_buffer=True)

        # Load the expert data and offline data into the offline buffer
        self.offline_buffer.load_data_into_buffer(buffered_data=data_exp, clear_buffer=True)
        self.offline_buffer.load_data_into_buffer(buffered_data=data_off, clear_buffer=False)

        # Viterbi Accuracy Initialization
        vit_res_exp = {
            'per_step_viterbi_acc': 1.0,
            'per_ep_viterbi_acc': 1.0,
        }
        vit_res_off = {
            'per_step_viterbi_acc': 1.0,
            'per_ep_viterbi_acc': 1.0,
        }

        # # Get the indices of the expert and offline episodes (offline data has expert data as well)
        # ep_idxs_exp = list(range(0, self.expert_buffer.get_current_size_ep().numpy()))
        # ep_idxs_off = list(set(range(0, self.offline_buffer.get_current_size_ep().numpy())) -
        # 					   set(range(0, self.expert_buffer.get_current_size_ep().numpy())))

        # Pretraining skilled actors on expert data for informed viterbi decoding
        # if self.args.max_pretrain_time_steps > 0 and not self.args.skill_supervision == 'none':
        # 	tf.print("Pretraining the actors and directors with expert data")
        # 	logger.info("Pretraining the actors and directors with expert data")
        #
        # 	self.model.skilled_actors.change_training_mode(training_mode=True)
        # 	with tqdm(total=self.args.max_pretrain_time_steps, desc='Pretraining', leave=False) as pbar:
        #
        # 		for curr_t in range(0, self.args.max_pretrain_time_steps):
        #
        # 			# Pretrain the actors and directors with the expert data
        # 			loss_dict = self.pretrain()
        #
        # 			# Update the reference actors and directors at every update interval
        # 			# (skills from now on are sampled by the reference actors and directors)
        # 			self.model.update_target_networks()
        #
        # 			vit_dec_result, _, _ = self.infer_skills(
        # 				data_exp, c_exp_curr_gt, compute_viterbi_acc=True if args.num_skills is None else False
        # 			)
        #
        # 			# Log
        # 			if self.args.log_wandb:
        # 				dict_to_log = {
        # 					f'pretrain_{key}': loss_dict[key] for key in loss_dict.keys()
        # 				}
        # 				dict_to_log.update({
        # 					'pretrain_expert_viterbi_acc': vit_dec_result['viterbi_acc'],
        # 				})
        # 				self.wandb_logger.log(
        # 					dict_to_log, step=log_step)
        #
        # 			# Update the progress bar with loss and time steps
        # 			pbar.set_postfix({'loss': loss_dict['loss/pi'].numpy(), 'time_steps': curr_t})
        # 			pbar.update(1)
        # 			log_step += 1
        #
        #
        # 	# Save the model
        # 	self.save_model(args.dir_param)

        with tqdm(total=args.max_time_steps, leave=False) as pbar:
            for curr_t in range(0, args.max_time_steps):

                # [Evaluate] the policy
                if curr_t % args.eval_interval == 0 and self.args.eval_demos > 0:
                    pbar.set_description('Evaluating')

                    max_return, max_return_with_exp_assist = self.evaluate(
                        max_return=max_return,
                        max_return_with_exp_assist=max_return_with_exp_assist,
                        log_step=log_step
                    )

                # Update the reference actors and directors using polyak averaging
                if curr_t % args.update_target_interval == 0:
                    tf.print("Updating the target actors and directors at train step {}".format(curr_t))
                    self.model.update_target_networks()

                    # # Update the offline skills [Must do at time step 0]
                    # if curr_t % args.update_skills_interval == 0:

                    # # Full supervision of latent skills: No update of expert and offline skills
                    # if self.args.skill_supervision == 'full':
                    #
                    # 	# vit_res_exp, _, _ = self.infer_skills(data_exp, c_exp_curr_gt)
                    # 	# vit_res_off, _, _ = self.infer_skills(data_off, c_off_curr_gt)

                    # # Update the offline skills [Using Target Networks for inference, no need to update at time steps
                    # # other than when target networks are updated]
                    # Semi-supervision of latent skills: Update offline skills
                    if 'semi' in self.args.skill_supervision:

                        vit_res_off, c_off_prev, c_off_curr = self.infer_skills(
                            data_off, c_off_curr_gt, compute_viterbi_acc=True if args.num_skills is None else False
                        )

                        # Update the offline buffer with the viterbi decoded skill sequence
                        data_off['prev_skills'] = c_off_prev
                        data_off['curr_skills'] = c_off_curr

                    # Unsupervised latent skills: Update expert and offline skills
                    elif self.args.skill_supervision == 'none':
                        decoding_start_time = time.time()
                        vit_res_exp, c_exp_prev, c_exp_curr = self.infer_skills(
                            data_exp, c_exp_curr_gt, compute_viterbi_acc=True if args.num_skills is None else False
                        )
                        vit_res_off, c_off_prev, c_off_curr = self.infer_skills(
                            data_off, c_off_curr_gt, compute_viterbi_acc=True if args.num_skills is None else False
                        )
                        tf.print("Decoding time: {}".format(time.time() - decoding_start_time))

                        # Update the expert buffer with the viterbi decoded skill sequence
                        data_exp['prev_skills'] = c_exp_prev
                        data_exp['curr_skills'] = c_exp_curr

                        # Update the offline buffer with the viterbi decoded skill sequence
                        data_off['prev_skills'] = c_off_prev
                        data_off['curr_skills'] = c_off_curr

                    # Load the updated expert data into the expert buffer
                    self.expert_buffer.load_data_into_buffer(buffered_data=data_exp, clear_buffer=True)

                    # Load the updated expert data and offline data into the offline buffer
                    self.offline_buffer.load_data_into_buffer(buffered_data=data_exp, clear_buffer=True)
                    self.offline_buffer.load_data_into_buffer(buffered_data=data_off, clear_buffer=False)

                    if self.args.log_wandb:
                        self.wandb_logger.log({
                            'expert_viterbi_acc': vit_res_exp['per_step_viterbi_acc'],
                            'offline_viterbi_acc': vit_res_off['per_step_viterbi_acc'],
                            'expert_viterbi_acc_ep': vit_res_exp['per_ep_viterbi_acc'],
                            'offline_viterbi_acc_ep': vit_res_off['per_ep_viterbi_acc'],
                        }, step=log_step)

                # [Train] the policy
                pbar.set_description('Training')
                avg_loss_dict = self.train()

                for key in avg_loss_dict.keys():
                    avg_loss_dict[key] = avg_loss_dict[key].numpy().item()

                # Log
                if self.args.log_wandb:
                    avg_loss_dict.update({
                        'policy_buffer_size': self.offline_buffer.get_current_size_trans(),
                        'expert_buffer_size': self.expert_buffer.get_current_size_trans(),
                    })
                    self.wandb_logger.log(
                        avg_loss_dict,
                        step=log_step
                    )

                # Update
                pbar.update(1)
                log_step += 1

        # Save the model
        self.save_model(args.dir_param)

        if args.test_demos > 0:
            self.visualise(use_expert_skill=False)
