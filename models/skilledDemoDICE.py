import copy
import logging
from abc import ABC
from argparse import Namespace

import numpy as np
import tensorflow as tf
from tensorflow_gan.python.losses import losses_impl as tfgan_losses
from tqdm import tqdm

from domains.PnPExpert import PnPExpert, PnPExpertTwoObj, PnPExpertTwoObjImitator
from her.replay_buffer import ReplayBufferTf
from networks.general import SkilledActors, Critic, Discriminator
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
		
		# Define Optimizers
		self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=args.actor_lr)
		self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=args.critic_lr)
		self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=args.disc_lr)
		
		self.build_model()
		
		self.act_w_expert_skill = False
		self.act_w_expert_action = False
		if args.two_object:
			# self.expert = PnPExpertTwoObj(expert_behaviour=args.expert_behaviour, wrap_skill_id=args.wrap_skill_id)
			self.expert = PnPExpertTwoObjImitator(wrap_skill_id=args.wrap_skill_id)
		else:
			self.expert = PnPExpert()
		
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
			weight = tf.expand_dims(tf.math.exp(rb_adv / (1 + self.args.replay_regularization)), 1)
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
	
	def pretrain(self, data_exp):
		with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
			tape.watch(self.skilled_actors.variables)
			
			# Compute the log probs of current skill using the director
			curr_skill_log_prob = self.skilled_actors.get_director_log_probs(
				tf.concat([data_exp['states'], data_exp['goals']], 1),
				data_exp['prev_skills'],
				data_exp['curr_skills']
			)
			
			# Compute the log probs of current action using the actor
			curr_action_log_prob = self.skilled_actors.get_actor_log_probs(
				tf.concat([data_exp['states'], data_exp['goals']], 1),
				data_exp['curr_skills'],
				data_exp['actions']
			)
			
			total_loss = - tf.reduce_mean(curr_skill_log_prob + curr_action_log_prob)
			
		grads = tape.gradient(total_loss, self.skilled_actors.variables)
		self.actor_optimizer.apply_gradients(zip(grads, self.skilled_actors.variables))
		
		return {
			'loss/pi': total_loss,
		}
	
	@tf.function(experimental_relax_shapes=True)  # This is needed to avoid shape errors
	def act(self, state, env_goal, prev_goal, prev_skill, epsilon, stddev):
		state = tf.clip_by_value(state, -self.args.clip_obs, self.args.clip_obs)
		env_goal = tf.clip_by_value(env_goal, -self.args.clip_obs, self.args.clip_obs)
		prev_goal = tf.clip_by_value(prev_goal, -self.args.clip_obs, self.args.clip_obs)
		
		# ###################################### Current Goal ###################################### #
		curr_goal = env_goal
		
		# ###################################### Current Skill ###################################### #
		if self.act_w_expert_skill:
			curr_skill = tf.numpy_function(self.expert.sample_curr_skill, [state[0], env_goal[0], prev_skill[0]], tf.float32)
			curr_skill = tf.expand_dims(curr_skill, axis=0)
		else:
			# Get the director corresponding to the previous skill and obtain the current skill
			_, curr_skill, _ = self.skilled_actors.call_director(tf.concat([state, curr_goal], axis=1), prev_skill)
		
		# ###################################### Action ###################################### #
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
				action = tf.numpy_function(self.expert.sample_action, [state[0], env_goal[0], prev_skill[0], action[0]], tf.float32)
				action = tf.expand_dims(action, axis=0)
			
			action = tf.clip_by_value(action, -self.args.action_max, self.args.action_max)
		
		return curr_goal, curr_skill, action
	
	def expert_skill_transition(self, state, env_goal, prev_skill, env_goal_thresh=0.01):
		
		assert self.args.wrap_skill_id == '1'
		
		gripper_pos = state[:3]
		delta_obj1 = state[3:6] - gripper_pos
		delta_obj2 = state[6:9] - gripper_pos
		delta_goal1 = state[3:6] - env_goal[:3]
		delta_goal2 = state[6:9] - env_goal[3:]
		
		# If previous skill was pick object 1
		if np.argmax(prev_skill) == 0:
			# If gripper is not close to object 1, then keep picking object 1
			if np.linalg.norm(delta_obj1) <= 0.1 or np.linalg.norm(delta_obj1 - np.array([0, 0, 0.08])) <= env_goal_thresh:
				# Transition to grab object 1
				curr_skill = np.array([0, 1, 0, 0, 0, 0])
			else:
				# Keep picking object 1
				curr_skill = np.array([1, 0, 0, 0, 0, 0])
				
		# If previous skill was grab object 1
		elif np.argmax(prev_skill) == 1:
			# If gripper is not close to object 1, then keep grabbing object 1
			if np.linalg.norm(delta_obj1) <= env_goal_thresh:
				# Grab skill should be terminated -> Transition to drop skill
				curr_skill = np.array([0, 0, 1, 0, 0, 0])
			else:
				curr_skill = np.array([0, 1, 0, 0, 0, 0])
				
		# If previous skill was drop object 1
		elif np.argmax(prev_skill) == 2:
			# If gripper is not close to goal 1, then keep dropping object 1
			if np.linalg.norm(delta_goal1) <= env_goal_thresh:
				# Drop skill should be terminated -> Transition to pick skill of object 2
				curr_skill = np.array([0, 0, 0, 1, 0, 0])
			else:
				curr_skill = np.array([0, 0, 1, 0, 0, 0])
				
		# If previous skill was pick object 2
		elif np.argmax(prev_skill) == 3:
			# If gripper is not close to object 2, then keep picking object 2
			if np.linalg.norm(delta_obj2) <= 0.1 or np.linalg.norm(delta_obj2 - np.array([0, 0, 0.08])) <= env_goal_thresh:
				# Transition to grab object 2
				curr_skill = np.array([0, 0, 0, 0, 1, 0])
			else:
				# Keep picking object 2
				curr_skill = np.array([0, 0, 0, 1, 0, 0])
		
		# If previous skill was grab object 2
		elif np.argmax(prev_skill) == 4:
			# If gripper is not close to object 2, then keep grabbing object 2
			if np.linalg.norm(delta_obj2) <= env_goal_thresh:
				# Grab skill should be terminated -> Transition to drop skill
				curr_skill = np.array([0, 0, 0, 0, 0, 1])
			else:
				curr_skill = np.array([0, 0, 0, 0, 1, 0])
				
		# If previous skill was drop object 2
		else:
			# Stay in drop skill
			curr_skill = np.array([0, 0, 0, 0, 0, 1])
			
		return np.cast[np.float32](curr_skill)
	
	def get_init_skill(self):
		if self.args.wrap_skill_id == '0':
			skill = tf.one_hot(0, self.args.c_dim, dtype=tf.float32)  # Can only be used with expert_behaviour = 0
			skill = tf.reshape(skill, shape=(1, -1))
		elif self.args.wrap_skill_id == '1':
			# TODO: Must learn from expert data when expert_behaviour = 1
			skill = tf.one_hot(0, self.args.c_dim, dtype=tf.float32)  # Pick Object 0 first (default)
			skill = tf.reshape(skill, shape=(1, -1))
		else:
			raise NotImplementedError
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

		# Define the Critic to estimate the value of (prev_skill, state, goal)
		_ = self.critic(
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
		

class Agent(AgentBase):
	def __init__(self, args,
				 expert_buffer_unseg: ReplayBufferTf = None,
				 policy_buffer_unseg: ReplayBufferTf = None):
		
		super().__init__(args, skilledDemoDICE(args), 'skilledDemoDICE', expert_buffer_unseg, policy_buffer_unseg)
	
	def load_actor(self, dir_param):
		self.model.skilled_actors.load_weights(dir_param + "/policy.h5")
	
	@tf.function
	def compute_skills(self, buffered_data, gt_curr_skill):
		
		viterbi_acc = 0.0
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
																		   init_skill=init_skill)
			# Convert the skill sequence (T+1, 1) to one-hot encoding
			skill_seq = tf.one_hot(tf.squeeze(skill_seq, axis=-1), depth=self.args.c_dim)
			
			# Store the log_prob of the viterbi decoded skill sequence
			log_probs.append(log_prob)
			# Update the buffer with the viterbi decoded skill sequence
			new_prev_skills.append(tf.gather(skill_seq, tf.range(0, tf.shape(skill_seq)[0] - 1)))
			new_curr_skills.append(tf.gather(skill_seq, tf.range(1, tf.shape(skill_seq)[0])))
			
			# Compute the accuracy of the viterbi decoded skill sequence (use self.offline_curr_skill)
			viterbi_acc += tf.reduce_mean(tf.cast(tf.equal(tf.argmax(gt_curr_skill[ep_idx], axis=-1),
														   tf.argmax(skill_seq[1:], axis=-1)),
												  dtype=tf.float32))
			
		
		new_prev_skills = tf.stack(new_prev_skills, axis=0)
		new_curr_skills = tf.stack(new_curr_skills, axis=0)
		
		result = {
				'viterbi_acc': viterbi_acc/num_episodes,
				'avg_log_prob': tf.reduce_mean(log_probs),
		}
		
		return result, new_prev_skills, new_curr_skills
	
	def pretrain(self):
		
		log_step = 0
		# For Verifying Viterbi Decoding
		buffered_data = self.expert_buffer_unseg.sample_episodes()
		expert_gt_curr_skill = buffered_data['curr_skills']
		
		self.model.skilled_actors.change_training_mode(training_mode=True)
		with tqdm(total=self.args.max_pretrain_time_steps, desc='Pretraining', leave=False) as pbar:
			
			for total_time_steps in range(0, self.args.max_pretrain_time_steps, self.args.horizon):
				
				# Pretrain the actors and directors with the expert data
				data_expert = self.sample_data(self.expert_buffer_unseg, self.args.batch_size)
				loss_dict = self.model.pretrain(data_expert)
				
				vit_dec_result, _, _ = self.compute_skills(buffered_data, expert_gt_curr_skill)
				
				# Log
				if self.args.log_wandb:
					self.wandb_logger.log(
						{
							'pretrain_loss': loss_dict['loss/pi'],
							'pretrain_expert_viterbi_acc': vit_dec_result['viterbi_acc'],
						}, step=log_step)

				# Update the progress bar with loss and time steps
				pbar.set_postfix({'loss': loss_dict['loss/pi'].numpy(), 'time_steps': total_time_steps})
				pbar.update(self.args.horizon)
				log_step += 1
		
		return log_step

	def learn(self):
		args = self.args
		max_return, max_return_with_exp_assist = None, None
		log_step = 0
		
		# Pretraining skilled actor on expert data for informed viterbi decoding
		if self.args.max_pretrain_time_steps > 0:
			tf.print("Pretraining the actors and directors with expert data")
			logger.info("Pretraining the actors and directors with expert data")
			log_step = self.pretrain()
			
		# For Viterbi Decoding
		buffered_data = self.policy_buffer_unseg.sample_episodes()
		offline_gt_curr_skill = copy.deepcopy(buffered_data['curr_skills'])
		
		with tqdm(total=args.max_time_steps, leave=False) as pbar:
			for total_time_steps in range(0, args.max_time_steps, args.horizon):
				
				# [Evaluate] the policy
				if total_time_steps % args.eval_interval == 0 and self.args.eval_demos > 0:
					pbar.set_description('Evaluating')
					max_return, max_return_with_exp_assist = self.evaluate(max_return=max_return,
																		   max_return_with_exp_assist=max_return_with_exp_assist,
																		   log_step=log_step)
					
				# Update the offline skills [Must do at time step 0]
				if total_time_steps % args.update_offline_skills_interval == 0:
					vit_dec_result, new_prev_skills, new_curr_skills = self.compute_skills(buffered_data,
																						   offline_gt_curr_skill)
					
					# # Step-2) Update the policy buffer with the viterbi decoded skill sequence [Semi-Supervised]
					if not self.args.use_offline_gt_skills:
						logger.info('Updating the policy buffer with the viterbi decoded skill sequence at train step {}'.format(total_time_steps))
						buffered_data['prev_skills'] = new_prev_skills
						buffered_data['curr_skills'] = new_curr_skills
						self.policy_buffer_unseg.load_data_into_buffer(buffered_data=buffered_data)
					
					if self.args.log_wandb:
						self.wandb_logger.log(vit_dec_result, step=log_step)
				
				# [Train] the policy
				pbar.set_description('Training')
				avg_loss_dict = self.train()
				
				for key in avg_loss_dict.keys():
					avg_loss_dict[key] = avg_loss_dict[key].numpy().item()
				
				# Log
				if self.args.log_wandb:
					self.wandb_logger.log(avg_loss_dict, step=log_step)
					self.wandb_logger.log({
						'policy_buffer_size': self.policy_buffer_unseg.get_current_size_trans(),
						'expert_buffer_size': self.expert_buffer_unseg.get_current_size_trans(),
					}, step=log_step)
				
				# Update
				pbar.update(args.horizon)
				log_step += 1
		
		# Save the model
		self.save_model(args.dir_param)
		
		if args.test_demos > 0:
			self.visualise(use_expert_skill=False)
			self.visualise(use_expert_skill=True)
