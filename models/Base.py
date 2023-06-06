import datetime
import logging
import os
from typing import Dict, Union, List, Optional

import tensorflow as tf
import wandb

from domains.PnP import MyPnPEnvWrapper
from her.replay_buffer import ReplayBufferTf
from her.rollout import RolloutWorker
from utils.custom import evaluate_worker
from utils.env import get_PnP_env

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
		self.env: MyPnPEnvWrapper = get_PnP_env(args)
		self.eval_env: MyPnPEnvWrapper = get_PnP_env(args)
		
		# Define the Rollout Workers
		self.policy_worker = RolloutWorker(
			self.env, self.model, T=args.horizon, rollout_terminate=False, render=False,
			is_expert_worker=False
		)
		self.eval_worker = RolloutWorker(
			self.eval_env, self.model, T=args.horizon, rollout_terminate=True, render=False,
			is_expert_worker=False
		)
		self.visualise_worker = RolloutWorker(
			self.eval_env, self.model, T=args.horizon, rollout_terminate=False, render=True,
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
			for skill in transitions.keys():
				
				# For skills whose transition data is not None
				if transitions[skill] is not None:
					transitions[skill] = self.process_data(
						transitions[skill], tf.constant(True, dtype=tf.bool), tf.constant(True, dtype=tf.bool)
					)
					
					keys = transitions[skill].keys()
			
			# If keys is None, No transitions were sampled
			if keys is None:
				raise ValueError("No transitions were sampled")
			
			# Concatenate the transitions from different skills
			combined_transitions = {key: [] for key in keys}
			
			for skill in transitions.keys():
				
				if transitions[skill] is not None:
					for key in keys:
						combined_transitions[key].append(transitions[skill][key])
			
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
	
	def evaluate(self, max_return=None, max_return_with_exp_assist=None, log_step=None):
		
		self.model.change_training_mode(training_mode=False)
		self.model.act_w_expert_skill = False
		self.model.act_w_expert_action = False
		
		avg_return, avg_time, avg_goal_dist, avg_perc_dec = evaluate_worker(self.eval_worker, self.args.eval_demos)
		if max_return is None:
			max_return = avg_return
		elif avg_return > max_return:
			max_return = avg_return
			self.save_model(os.path.join(self.args.dir_param, 'best_model'))
		
		# Log the data
		if self.args.log_wandb:
			self.wandb_logger.log({
				'stats/success_rate': avg_return,
				'stats/eval_max_return': max_return,
				'stats/eval_avg_time': avg_time,
				'stats/eval_avg_goal_dist': avg_goal_dist,
				'stats/eval_avg_perc_dec': avg_perc_dec,
			}, step=log_step)
		tf.print(f"Eval Return: {avg_return}, "
				 f"Eval Avg Time: {avg_time}, "
				 f"Eval Avg Goal Dist: {avg_goal_dist}",
				 f"Eval Avg Perc Dec: {avg_perc_dec}")
		
		# # TODO: max_return_with_exp_assist with expert goals won't run for multi-object case as of now,
		# #  the expert goals are in 3D while the model conditions on the env goals which are in num_objects x 3D
		
		# # We use expert skill to assist the policy
		# self.model.act_w_expert_skill = True
		# self.model.act_w_expert_action = False
		# avg_return, avg_time, avg_goal_dist, avg_perc_dec = evaluate_worker(self.eval_worker, self.args.eval_demos)
		# if max_return_with_exp_assist is None:
		# 	max_return_with_exp_assist = avg_return
		# elif avg_return > max_return_with_exp_assist:
		# 	max_return_with_exp_assist = avg_return
		# 	self.save_model(os.path.join(self.args.dir_param, 'best_model_with_exp_assist'))
		#
		# # Log the data
		# if self.args.log_wandb:
		# 	self.wandb_logger.log({
		# 		'stats/exp_assist/success_rate': avg_return,
		# 		'stats/exp_assist/eval_max_return': max_return_with_exp_assist,
		# 		'stats/exp_assist/eval_avg_time': avg_time,
		# 		'stats/exp_assist/eval_avg_goal_dist': avg_goal_dist,
		# 		'stats/exp_assist/eval_avg_perc_dec': avg_perc_dec,
		# 	}, step=log_step)
		# tf.print(f"Eval Return (Exp Assist): {avg_return}, "
		# 		 f"Eval Avg Time (Exp Assist): {avg_time}, "
		# 		 f"Eval Avg Goal Dist (Exp Assist): {avg_goal_dist}, "
		# 		 f"Eval Avg Perc Dec (Exp Assist): {avg_perc_dec}")
		
		return max_return, max_return_with_exp_assist
	
	def visualise(self,
				  use_expert_skill=False,
				  use_expert_action=False,
				  resume_states: Optional[List[dict]] = None,
				  num_episodes=None):
		logger.info("Visualising the policy with expert skill: {}".format(use_expert_skill))
		logger.info("Visualising the policy with expert action: {}".format(use_expert_action))
		
		tf.config.run_functions_eagerly(True)
		
		self.model.change_training_mode(training_mode=False)
		
		self.model.act_w_expert_skill = use_expert_skill
		self.model.act_w_expert_action = use_expert_action
		_ = evaluate_worker(
			worker=self.visualise_worker,
			num_episodes=self.args.test_demos if num_episodes is None else num_episodes,
			log_traj=True,
			resume_states=resume_states
		)
