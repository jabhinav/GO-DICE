import datetime
import logging
import os
from typing import Dict, Union, List, Optional

import tensorflow as tf
import wandb
import random
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from her.replay_buffer import ReplayBufferTf
from her.rollout import RolloutWorker
from evaluation.eval import evaluate_worker, get_trajectory_option_activation
from utils.env import get_PnP_env, get_PointMass_env
from domains.PnP import MyPnPEnvWrapper
from domains.PointMassDropNReach import MyPointMassDropNReachEnvWrapper
from collections import Counter
from utils.env import save_env_img

logger = logging.getLogger(__name__)


class AgentBase(object):
	def __init__(
			self,
			args,
			model,
			algo: str,
			expert_buffer: ReplayBufferTf,
			offline_buffer: ReplayBufferTf
	):
		
		self.args = args
		self.model = model
		
		# Define the Buffers
		self.expert_buffer = expert_buffer
		self.offline_buffer = offline_buffer
		
		self.offline_gt_prev_skill = None
		self.offline_gt_curr_skill = None
		
		# Define Tensorboard for logging Losses and Other Metrics
		if not os.path.exists(args.dir_summary):
			os.makedirs(args.dir_summary)
		
		if not os.path.exists(args.dir_plot):
			os.makedirs(args.dir_plot)
		self.summary_writer = tf.summary.create_file_writer(args.dir_summary)
		
		# Environments
		if args.env_name == 'OpenAIPickandPlace':
			self.env: MyPnPEnvWrapper = get_PnP_env(args)
			self.eval_env: MyPnPEnvWrapper = get_PnP_env(args)
		elif args.env_name == 'MyPointMassDropNReach':
			self.env: MyPointMassDropNReachEnvWrapper = get_PointMass_env(args)
			self.eval_env: MyPointMassDropNReachEnvWrapper = get_PointMass_env(args)
		else:
			raise NotImplementedError('Env %s not implemented' % args.env_name)
		
		# Define the Rollout Workers
		# Worker 1: For offline data collection
		self.policy_worker = RolloutWorker(
			self.env, self.model, T=args.horizon, rollout_terminate=False, render=False,
			is_expert_worker=False
		)
		# Worker 2: For evaluation
		self.eval_worker = RolloutWorker(
			self.eval_env, self.model, T=args.horizon, rollout_terminate=True, render=False,
			is_expert_worker=False
		)
		# Worker 3: For visualisation
		self.visualise_worker = RolloutWorker(
			self.eval_env, self.model, T=args.horizon, rollout_terminate=False, render=args.visualise_test,
			is_expert_worker=False
		)
		
		# Define wandb logging
		if self.args.log_wandb:
			current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
			self.wandb_logger = wandb.init(
				project=args.wandb_project,
				config=vars(args),
				id='{}_{}'.format(algo, current_time),
				reinit=True,  # Allow multiple wandb.init() calls in the same process.
			)
			# Clear tensorflow graph and cache
			tf.keras.backend.clear_session()
			tf.compat.v1.reset_default_graph()
	
	def preprocess_in_state_space(self, item):
		item = tf.clip_by_value(item, -self.args.clip_obs, self.args.clip_obs)
		return item
	
	def save_model(self, dir_param):
		if not os.path.exists(dir_param):
			os.makedirs(dir_param)
		self.model.save_(dir_param)
	
	def load_model(self, dir_param):
		self.model.load_(dir_param)
	
	def process_data(self, transitions, expert=False, is_supervised=False):
		
		trans = transitions.copy()
		
		# Process the states and goals
		trans['states'] = self.preprocess_in_state_space(trans['states'])
		trans['states_2'] = self.preprocess_in_state_space(trans['states_2'])
		trans['env_goals'] = self.preprocess_in_state_space(trans['env_goals'])
		trans['init_states'] = self.preprocess_in_state_space(trans['init_states'])
		trans['her_goals'] = self.preprocess_in_state_space(trans['her_goals'])
		trans['achieved_goals'] = self.preprocess_in_state_space(trans['achieved_goals'])
		
		# # Make sure her goals has the same shape as env_goals
		# try:
		# 	assert trans['env_goals'].shape == trans['her_goals'].shape
		# except AssertionError:
		# 	tf.print("Shapes of env goals {} and her goals {} are not the same.".format(trans['env_goals'].shape,
		# 																				trans['her_goals'].shape))
		# 	raise AssertionError
		
		if self.model.use_her:
			trans['goals'] = trans['her_goals']
		else:
			trans['goals'] = trans['env_goals']
		
		# Define if the transitions are from expert or not/are supervised or not
		trans['is_demo'] = tf.cast(expert, dtype=tf.int32) * tf.ones_like(trans['successes'], dtype=tf.int32)
		trans['is_sup'] = tf.cast(is_supervised, dtype=tf.int32) * tf.ones_like(trans['successes'], dtype=tf.int32)
		
		# Compute terminate skills i.e. if prev_skill != curr_skill then terminate_skill = 1 else 0
		trans['terminate_skills'] = tf.cast(tf.not_equal(tf.argmax(trans['prev_skills'], axis=-1),
														 tf.argmax(trans['curr_skills'], axis=-1)),
											dtype=tf.int32)
		# reshape the terminate_skills to be of shape (batch_size, 1)
		trans['terminate_skills'] = tf.reshape(trans['terminate_skills'], shape=(-1, 1))
		
		# Make sure the data is of type tf.float32
		for key in trans.keys():
			trans[key] = tf.cast(trans[key], dtype=tf.float32)
		
		return trans
	
	def sample_data(self, buffer, batch_size):
		
		# Sample Transitions
		transitions: Union[Dict[int, dict], dict] = buffer.sample_transitions(batch_size)
		
		# Process the transitions
		keys = None
		if all(isinstance(v, dict) for v in transitions.values()):
			for option in transitions.keys():
				
				# For skills whose transition data is not None
				if transitions[option] is not None:
					transitions[option] = self.process_data(
						transitions[option], tf.constant(True, dtype=tf.bool), tf.constant(True, dtype=tf.bool)
					)
					
					keys = transitions[option].keys()
			
			# If keys is None, No transitions were sampled
			if keys is None:
				raise ValueError("No transitions were sampled")
			
			# Concatenate the transitions from different skills
			combined_transitions = {key: [] for key in keys}
			
			for option in transitions.keys():
				
				if transitions[option] is not None:
					for key in keys:
						combined_transitions[key].append(transitions[option][key])
			
			for key in keys:
				combined_transitions[key] = tf.concat(combined_transitions[key], axis=0)
			
			transitions = combined_transitions
		
		elif isinstance(transitions, dict):
			transitions = self.process_data(
				transitions, tf.constant(True, dtype=tf.bool), tf.constant(True, dtype=tf.bool)
			)
		
		else:
			raise ValueError("Invalid type of transitions")
		
		return transitions
	
	@tf.function
	def train(self):
		
		self.model.change_training_mode(training_mode=True)
		
		data_expert = self.sample_data(self.expert_buffer, self.args.batch_size)
		data_policy = self.sample_data(self.offline_buffer, self.args.batch_size)
		loss_dict = self.model.train(data_expert, data_policy)
		
		# Average the losses
		avg_loss_dict = {}
		for key in loss_dict.keys():
			if key not in avg_loss_dict.keys():
				avg_loss_dict[key] = []
			avg_loss_dict[key].append(loss_dict[key])
		for key in avg_loss_dict.keys():
			avg_loss_dict[key] = tf.reduce_mean(avg_loss_dict[key])
		
		return avg_loss_dict
	
	def learn(self):
		# This is a base class method, inherited classes must implement this method
		raise NotImplementedError
	
	def evaluate(self, max_return=None, max_return_with_exp_assist=None, log_step=None, debug_with_exp_assist=False):
		
		self.model.change_training_mode(training_mode=False)
		
		# ############################### Evaluate the model ############################################## #
		self.model.act_w_expert_skill = False
		self.model.act_w_expert_action = False
		
		avg_return, avg_time, avg_goal_dist, avg_perc_dec, avg_subgoal_achievement_reward, \
			avg_subgoals_achieved, avg_subgoals_dist = evaluate_worker(self.eval_worker, self.args.eval_demos,
																	   subgoal_reward=self.args.subgoal_reward)
		
		# avg_return = convert_tf_to_numpy(avg_return)
		# avg_time = convert_tf_to_numpy(avg_time)
		# avg_goal_dist = convert_tf_to_numpy(avg_goal_dist)
		# avg_perc_dec = convert_tf_to_numpy(avg_perc_dec)
		# avg_subgoal_achievement_reward = convert_tf_to_numpy(avg_subgoal_achievement_reward)
		# avg_subgoals_achieved = convert_tf_to_numpy(avg_subgoals_achieved)
		# avg_subgoals_dist = convert_tf_to_numpy(avg_subgoals_dist)
		
		if max_return is None:
			max_return = avg_return
		elif avg_return > max_return:
			max_return = avg_return
			self.save_model(os.path.join(self.args.dir_param, 'best_model'))
		
		# Log the data
		if self.args.log_wandb:
			log_metrics = {
				'stats/success_rate': avg_return,
				'stats/eval_max_return': max_return,
				'stats/eval_avg_time': avg_time,
				'stats/eval_avg_goal_dist': avg_goal_dist,
				'stats/eval_avg_perc_dec': avg_perc_dec,
				'stats/eval_avg_subgoal_achievement_reward': avg_subgoal_achievement_reward,
			}
			for key, value in avg_subgoals_achieved.items():
				log_metrics[f'stats/{key.replace("/", "_")}'] = value
			for key, value in avg_subgoals_dist.items():
				log_metrics[f'stats/distances/{key.replace("/", "_")}'] = value
			
			self.wandb_logger.log(log_metrics, step=log_step)
		
		tf.print(f"\nEval Return: {avg_return}, "
				 f"\nEval Avg Time: {avg_time}, "
				 f"\nEval Avg Goal Dist: {avg_goal_dist}",
				 f"\nEval Avg Perc Dec: {avg_perc_dec}",
				 f"\nEval Avg Subgoal Achievement Reward: {avg_subgoal_achievement_reward}",)
		
		# ############################### Evaluate the model (Expert Assist) ################################### #
		if debug_with_exp_assist:
			
			# # TODO: max_return_with_exp_assist with expert goals won't run for multi-object case as of now,
			# #  the expert goals are in 3D while the model conditions on the env goals which are in num_objects x 3D
			
			# We use expert skill to assist the policy
			self.model.act_w_expert_skill = True  # Define here to use expert skill transitions
			self.model.act_w_expert_action = False  # Define here to use expert policy for action selection
			avg_return, avg_time, avg_goal_dist, avg_perc_dec, avg_subgoal_achievement_reward, \
				avg_subgoals_achieved, avg_subgoals_dist = evaluate_worker(self.eval_worker, self.args.eval_demos,
																		   subgoal_reward=self.args.subgoal_reward)
			
			if max_return_with_exp_assist is None:
				max_return_with_exp_assist = avg_return
			
			elif avg_return > max_return_with_exp_assist:
				max_return_with_exp_assist = avg_return
				self.save_model(os.path.join(self.args.dir_param, 'best_model_with_exp_assist'))
				
			# Log the data
			if self.args.log_wandb:
				log_metrics = {
					'statsExpertAssist/success_rate': avg_return,
					'statsExpertAssist/eval_max_return': max_return,
					'statsExpertAssist/eval_avg_time': avg_time,
					'statsExpertAssist/eval_avg_goal_dist': avg_goal_dist,
					'statsExpertAssist/eval_avg_perc_dec': avg_perc_dec,
					'statsExpertAssist/eval_avg_subgoal_achievement_reward': avg_subgoal_achievement_reward,
				}
				for key, value in avg_subgoals_achieved.items():
					log_metrics[f'statsExpertAssist/{key.replace("/", "_")}'] = value
				for key, value in avg_subgoals_dist.items():
					log_metrics[f'statsExpertAssist/distances/{key.replace("/", "_")}'] = value
				
				self.wandb_logger.log(log_metrics, step=log_step)
			
			tf.print(f"\nEval Return (ExpertAssist): {avg_return}, "
					 f"\nEval Avg Time (ExpertAssist): {avg_time}, "
					 f"\nEval Avg Goal Dist (ExpertAssist): {avg_goal_dist}",
					 f"\nEval Avg Perc Dec (ExpertAssist): {avg_perc_dec}",
					 f"\nEval Avg Subgoal Achievement Reward (ExpertAssist): {avg_subgoal_achievement_reward}", )
		
		return max_return, max_return_with_exp_assist
	
	def visualise(self,
				  use_expert_options=False,
				  use_expert_action=False,
				  resume_states: Optional[List[dict]] = None,
				  num_episodes=None):
		logger.info("Visualising the policy with expert Options: {}".format(use_expert_options))
		logger.info("Visualising the policy with expert action: {}".format(use_expert_action))
		
		tf.config.run_functions_eagerly(True)
		
		self.model.change_training_mode(training_mode=False)
		
		self.model.act_w_expert_skill = use_expert_options
		self.model.act_w_expert_action = use_expert_action
		
		i = 0
		exception_count = 0
		pbar = tqdm(total=num_episodes, position=0, leave=True, desc="Visualising ")
		while i < self.args.test_demos:
			if resume_states is None:
				# logger.info("No resume init state provided. Randomly initializing the env.")
				resume_state_dict = None
			else:
				logger.info(f"Resume init state provided! Check if you intended to do so.")
				# sample a random state from the list of resume states
				resume_state_dict = random.choice(resume_states)
			
			try:
				episode, stats = self.visualise_worker.generate_rollout(resume_state_dict=resume_state_dict)
			except Exception as e:
				exception_count += 1
				logger.info(f"Exception occurred: {e}")
				if exception_count < 10:
					continue
				else:
					raise e
					
			# Save the last state of the episode as image
			save_env_img(self.visualise_worker.env, path_to_save=os.path.join(self.args.log_dir,
																			  f"episode_{i}_last_state.png"))
			
			i += 1
			pbar.update(1)
		
		pbar.close()
		
	
	def compute_avg_return(self, eval_demos=100, subgoal_reward=1, avg_of_reward=False):
		"""
		:param eval_demos: Number of evaluation episodes
		:param subgoal_reward: Reward for achieving subgoals
		:param avg_of_reward: If True, then compute the mean and standard deviation of the return over the evaluation
		Just computes the average return over the evaluation episodes without logging anything in wandb
		"""
		success, time, goal_dist, perc_dec, subgoal_achievement_reward, subgoals_achieved, subgoals_dist = \
			evaluate_worker(
				self.eval_worker,
				num_episodes=eval_demos,
				subgoal_reward=subgoal_reward,
				return_avg_stats=False,
				show_progress=True
			)
		
		if avg_of_reward:
			# Compute the mean and standard deviation of the return over the evaluation episodes
			avg_return = np.mean(subgoal_achievement_reward)
			std_dev_return = np.std(subgoal_achievement_reward)
		else:
			avg_return = np.mean(success)
			std_dev_return = np.std(success)
		
		logger.info("Avg. Return: {} +- {}".format(avg_return, std_dev_return))
		print("Avg. Return: {} +- {}".format(avg_return, std_dev_return))
		
		return avg_return, std_dev_return
	
	def evaluate_trajectory_option_activation(self, eval_demos):
		i = 0
		option_activation_frac_ep = {}
		exception_count = 0
		pbar = tqdm(total=eval_demos, position=0, leave=True, desc="Evaluating Option Activation ")
		
		while i < eval_demos:
			try:
				episode, stats = self.visualise_worker.generate_rollout(resume_state_dict=None)
			except Exception as e:
				exception_count += 1
				logger.info(f"Exception occurred: {e}")
				if exception_count < 10:
					continue
				else:
					raise e
			
			prev_options, curr_options = get_trajectory_option_activation(episode)
			trajectory_len = len(prev_options)
			
			# 1) Compute the frequency of each Option and Log other stats
			options_freq = Counter(curr_options)
			success = stats['ep_success'].numpy() if isinstance(stats['ep_success'], tf.Tensor) else stats['ep_success']
			length = stats['ep_length'].numpy() if isinstance(stats['ep_length'], tf.Tensor) else stats['ep_length']
			logger.info(f"\n\nEpisode {i}.\n[Options Freq]: {options_freq}")
			logger.info(f"[Success]: {success}, [Length]: {length}")
			
			# 2) Let's compute the average activation of each option over the trajectory
			for option in options_freq.keys():
				if option not in option_activation_frac_ep.keys():
					option_activation_frac_ep[option] = []
				option_activation_frac_ep[option].append(options_freq[option] / trajectory_len)
			
			
			# 3) Let's plot the activation of each option over the trajectory i.e. time v/s 1/0 (active/inactive)
			ep_dir_plot = os.path.join(self.args.dir_plot, f"episode_{i}")
			if not os.path.exists(ep_dir_plot):
				os.makedirs(ep_dir_plot)
			
			# Per Option Plots
			for option in options_freq.keys():
				activation = [1 if curr_options[t] == option else 0 for t in range(len(curr_options))]
				plt.figure()
				plt.plot(activation)
				plt.xlabel("Time")
				plt.ylabel("Activation")
				plt.title(f"Option {option}")
				plt.savefig(os.path.join(ep_dir_plot, f"option_{option}_activation.pdf"))
				plt.close()
				
			# 4) Let's plot the activation of each option over the trajectory i.e. time v/s 1/0 (active/inactive)
			# All options in one plot with each option having a different color
			plt.figure()
			# distinct_options = list(options_freq.keys())
			# colors = plt.cm.rainbow(np.linspace(0, 1, len(distinct_options)))
			# for option, color in zip(distinct_options, colors):
			# 	activation = [1 if curr_options[t] == option else 0 for t in range(len(curr_options))]
			# 	plt.plot(activation, color=color, label=f"Option {option}")
			
			# Let's use the same color for all options and plot option activated at each time step like a step function
			activation = [0 for _ in range(len(curr_options))]
			for t in range(len(curr_options)):
				activation[t] = curr_options[t]
			plt.plot(activation, color='black', label=f"Option Activation")
			plt.yticks(np.arange(0, max(options_freq.keys()) + 1, 1.0))  # max + 1: since option #ID starts from 0
			
			plt.xlabel("Time")
			plt.ylabel("Option Number")
			plt.title(f"Options Activation")
			plt.legend()
			plt.savefig(os.path.join(ep_dir_plot, f"options_activation.pdf"))
			plt.close()
			
			i += 1
			pbar.update(1)
		
		pbar.close()
		option_activation_avg_ep = {key: np.mean(value) for key, value in option_activation_frac_ep.items()}
		option_activation_std_ep = {key: np.std(value) for key, value in option_activation_frac_ep.items()}
		for key in sorted(option_activation_avg_ep.keys(), key=lambda x: int(x)):
			logger.info(f"Option {key}: {option_activation_avg_ep[key]} +- {option_activation_std_ep[key]}")
