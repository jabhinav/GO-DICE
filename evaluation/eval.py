import os
import random
from typing import Optional, List

import numpy as np
import tensorflow as tf

from her.rollout import RolloutWorker
from utils.custom import logger
from utils.env import save_env_img
from tqdm import tqdm


def get_index_of_all_max(preds):
	sample_max_indexes = []
	for _i in range(preds.shape[0]):
		sample_max_indexes.append(set(np.where(preds[_i] == np.max(preds[_i]))[0]))
	return sample_max_indexes


def pred_binary_error(_gt, _pred, weights=None):
	#  Choose maximum prob output which matches GT if multiple actions are predicted with high and equal prob
	_gt_max_indexes = get_index_of_all_max(_gt)
	_pred_max_indexes = get_index_of_all_max(_pred)
	num_t = len(_gt)
	count = 0
	if weights is not None:
		for i in range(num_t):
			if _gt_max_indexes[i] & _pred_max_indexes[i]:
				count += 1. * weights[i]
		return 1. - count
	else:
		for i in range(num_t):
			if _gt_max_indexes[i] & _pred_max_indexes[i]:
				count += 1.
		return 1. - (count / num_t)


def evaluate(actor, env, num_episodes=100):
	"""Evaluates the policy.
	
	Args:
	  actor: A policy to evaluate.
	  env: Environment to evaluate the policy on.
	  num_episodes: A number of episodes to average the policy on.
	
	Returns:
	  Averaged reward and a total number of steps.
	"""
	total_timesteps = 0
	total_returns = 0
	
	for _ in range(num_episodes):
		state = env.reset()
		done = False
		while not done:
			action, _, _ = actor(np.array([state]))
			action = action[0].numpy()
			
			next_state, reward, done, _ = env.step(action)
			
			total_returns += reward
			total_timesteps += 1
			state = next_state
	
	return total_returns / num_episodes, total_timesteps / num_episodes


# @tf.function
# def tf_evaluate_worker(
# 		worker: RolloutWorker,
# 		num_episodes,
# 		log_traj: bool = False,
# 		log_dir: Optional[str] = None,
# 		resume_states: Optional[List[dict]] = None,
# 		subgoal_reward: int = 0,
# 		num_objs: int = 3,
# ):
# 	"""Evaluates the policy.
#
# 		Args:
# 		  worker: Rollout worker.
# 		  num_episodes: A number of episodes to average the policy on.
# 		  log_traj: Whether to log the skills or not (default: False).
# 		  log_dir: Directory to log (default: None).
# 		  resume_states: List of Resume states (default: None).
# 		  subgoal_reward: Subgoal reward (default: None).
# 		  num_objs: Number of objects (default: 3).
# 		Returns:
# 		  Bunch of metrics.
# 	"""
#
# 	total_timesteps = []
# 	total_returns = []
# 	avg_final_goal_dist = []
# 	avg_perc_decrease = []
# 	avg_subgoal_reward = []
# 	avg_subgoals_achieved = {
# 		'subgoals/{}/{}'.format(key, i): [] for key in ['pick', 'place'] for i in range(num_objs)
# 	}
# 	avg_subgoal_distances = {
# 		'subgoals/{}/{}'.format(key, i): [] for key in ['place'] for i in range(num_objs)
# 	}
#
# 	i = 0
# 	exception_count = 0
# 	while i < num_episodes:
#
# 		if resume_states is None:
# 			# logger.info("No resume init state provided. Randomly initializing the env.")
# 			resume_state_dict = None
# 		else:
# 			logger.info(f"Resume init state provided! Check if you intended to do so.")
# 			# sample a random state from the list of resume states
# 			resume_state_dict = random.choice(resume_states)
#
# 		try:
# 			episode, stats = worker.generate_rollout(resume_state_dict=resume_state_dict)
# 		except Exception as e:
# 			exception_count += 1
# 			logger.info(f"Exception occurred: {e}")
# 			if exception_count < 10:
# 				continue
# 			else:
# 				raise e
#
# 		# Metric 1: Success Rate
# 		# success = stats['ep_success'].numpy() if isinstance(stats['ep_success'], tf.Tensor) else stats['ep_success']
# 		success = stats['ep_success']
# 		total_returns.append(success)
#
# 		# Metric 2: Episode Length
# 		# length = stats['ep_length'].numpy() if isinstance(stats['ep_length'], tf.Tensor) else stats['ep_length']
# 		length = stats['ep_length']
# 		total_timesteps.append(length)
#
# 		# Metric 3: Final Goal Distance
# 		final_goal_dist = episode['distances'][0][-1]
# 		# final_goal_dist = final_goal_dist.numpy() if isinstance(final_goal_dist, tf.Tensor) else final_goal_dist
# 		avg_final_goal_dist.append(final_goal_dist)
#
# 		# Metric 4: Percentage Decrease in Goal Distance
# 		init_goal_dist = episode['distances'][0][0]
# 		# init_goal_dist = init_goal_dist.numpy() if isinstance(init_goal_dist, tf.Tensor) else init_goal_dist
# 		perc_decrease = (init_goal_dist - final_goal_dist) / init_goal_dist
# 		avg_perc_decrease.append(perc_decrease)
#
# 		# Metric 5: SubGoal Achievement Indicators
# 		# Define the types of returned subgoal information based on number of objects
# 		Tout = [tf.float32 for _ in range(3 * num_objs)]
# 		all_info = tf.numpy_function(func=worker.env.get_subgoal_info_listed,
# 																 inp=[],
# 																 Tout=Tout)
#
# 		rew = 0
# 		for i in range(num_objs):
# 			avg_subgoals_achieved['subgoals/pick/{}'.format(i)].append(all_info[i * 3])
# 			avg_subgoals_achieved['subgoals/place/{}'.format(i)].append(all_info[i * 3 + 1])
# 			avg_subgoal_distances['subgoals/place/{}'.format(i)].append(all_info[i * 3 + 2])
# 			rew += all_info[i * 3] * subgoal_reward + all_info[i * 3 + 1] * subgoal_reward
#
# 		avg_subgoal_reward.append(rew)
#
# 		# for subgoal, achieved in subgoals_achieved.items():
# 		# 	if subgoal not in avg_subgoals_achieved:
# 		# 		avg_subgoals_achieved[subgoal] = []
# 		#
# 		# 	avg_subgoals_achieved[subgoal].append(achieved)
# 		#
# 		# for subgoal, dist in subgoal_distances.items():
# 		# 	if subgoal not in avg_subgoal_distances:
# 		# 		avg_subgoal_distances[subgoal] = []
# 		#
# 		# 	avg_subgoal_distances[subgoal].append(dist)
# 		#
# 		# avg_subgoal_reward.append(
# 		# 	tf.reduce_sum(tf.cast(list(subgoals_achieved.values()), dtype=tf.float32)) * subgoal_reward
# 		# )
#
# 		if log_traj:
#
# 			# Save the last state of the episode as image
# 			save_env_img(worker.env, path_to_save=os.path.join(log_dir, f"episode_{i}_last_state.png"))
#
# 			prev_skills = episode['prev_skills'].numpy() \
# 				if isinstance(episode['prev_skills'], tf.Tensor) else episode['prev_skills']
# 			prev_skills = np.argmax(prev_skills[0], axis=1).tolist()
# 			prev_skills: List[int] = [skill for skill in prev_skills]
#
# 			curr_skills = episode['curr_skills'].numpy() \
# 				if isinstance(episode['curr_skills'], tf.Tensor) else episode['curr_skills']
# 			curr_skills = np.argmax(curr_skills[0], axis=1).tolist()
# 			curr_skills: List[int] = [skill for skill in curr_skills]
#
# 			actions = episode['actions'].numpy() if isinstance(episode['actions'], tf.Tensor) else episode['actions']
# 			actions = actions[0]
# 			logger.info(f"\n[Episode Num]: {i}")
#
# 			# Log the trajectory in the form <time_step, prev_skill, curr_skill, action>
# 			for t in range(len(prev_skills)):
# 				logger.info(f"<{t}: {prev_skills[t]} -> {curr_skills[t]} -> {actions[t]}>")
#
# 			# Compute the frequency of each skill
# 			skill_freq = Counter(prev_skills)
#
# 			# Log the success and length of the episode
# 			logger.info(f"[Success]: {success}, [Length]: {length}, [Skill Freq]: {skill_freq}")
#
# 		i += 1
#
# 	# avg_subgoals_achieved = {k: np.mean(v) for k, v in avg_subgoals_achieved.items()}
# 	# avg_subgoal_distances = {k: np.mean(v) for k, v in avg_subgoal_distances.items()}
#
# 	avg_subgoals_achieved = {k: tf.reduce_mean(tf.cast(v, dtype=tf.float32)) for k, v in avg_subgoals_achieved.items()}
# 	avg_subgoal_distances = {k: tf.reduce_mean(tf.cast(v, dtype=tf.float32)) for k, v in avg_subgoal_distances.items()}
#
# 	# return np.mean(total_returns), np.mean(total_timesteps), np.mean(avg_final_goal_dist), np.mean(avg_perc_decrease), \
# 	# 	np.mean(avg_subgoal_reward), avg_subgoals_achieved, avg_subgoal_distances
#
# 	return tf.reduce_mean(tf.cast(total_returns, dtype=tf.float32)), \
# 		tf.reduce_mean(tf.cast(total_timesteps, dtype=tf.float32)), \
# 		tf.reduce_mean(tf.cast(avg_final_goal_dist, dtype=tf.float32)), \
# 		tf.reduce_mean(tf.cast(avg_perc_decrease, dtype=tf.float32)), \
# 		tf.reduce_mean(tf.cast(avg_subgoal_reward, dtype=tf.float32)), \
# 		avg_subgoals_achieved, \
# 		avg_subgoal_distances


def evaluate_worker(
		worker: RolloutWorker,
		num_episodes,
		resume_states: Optional[List[dict]] = None,
		subgoal_reward: int = 0,
		return_avg_stats: bool = True,
		show_progress: bool = False,
):
	"""Evaluates the policy.

		Args:
		  worker: Rollout worker.
		  num_episodes: A number of episodes to average the policy on.
		  resume_states: List of Resume states (default: None).
		  subgoal_reward: Subgoal reward (default: None).
		  return_avg_stats: Whether to return avg. of stats or not (default: False).
		  show_progress: Whether to show progress bar or not (default: False).
		Returns:
		  Bunch of metrics.
	"""
	
	total_timesteps = []
	total_returns = []
	avg_final_goal_dist = []
	avg_perc_decrease = []
	avg_subgoal_achievement_reward = []
	subgoals_avg_info = {}
	subgoals_dist_info = {}
	
	i = 0
	exception_count = 0
	pbar = tqdm(total=num_episodes, position=0, leave=True, desc="Evaluating ") if show_progress else None
	while i < num_episodes:
		
		if resume_states is None:
			# logger.info("No resume init state provided. Randomly initializing the env.")
			resume_state_dict = None
		else:
			logger.info(f"Resume init state provided! Check if you intended to do so.")
			# sample a random state from the list of resume states
			resume_state_dict = random.choice(resume_states)
		
		try:
			episode, stats = worker.generate_rollout(resume_state_dict=resume_state_dict)
		except Exception as e:
			exception_count += 1
			logger.info(f"Exception occurred: {e}")
			if exception_count < 10:
				continue
			else:
				raise e
		
		# Metric 1: Success Rate
		success = stats['ep_success'].numpy() if isinstance(stats['ep_success'], tf.Tensor) else stats['ep_success']
		total_returns.append(success)
		
		# Metric 2: Episode Length
		length = stats['ep_length'].numpy() if isinstance(stats['ep_length'], tf.Tensor) else stats['ep_length']
		total_timesteps.append(length)
		
		# Metric 3: Final Goal Distance
		final_goal_dist = episode['distances'][0][-1]
		final_goal_dist = final_goal_dist.numpy() if isinstance(final_goal_dist, tf.Tensor) else final_goal_dist
		avg_final_goal_dist.append(final_goal_dist)
		
		# Metric 4: Percentage Decrease in Goal Distance
		init_goal_dist = episode['distances'][0][0]
		init_goal_dist = init_goal_dist.numpy() if isinstance(init_goal_dist, tf.Tensor) else init_goal_dist
		perc_decrease = (init_goal_dist - final_goal_dist) / init_goal_dist
		avg_perc_decrease.append(perc_decrease)
		
		# Metric 5: SubGoal Achievement Indicators
		subgoal_info, subgoal_distances = worker.env.get_subgoal_info()  # Do after the episode is done and before the reset
		for subgoal, info in subgoal_info.items():
			if subgoal not in subgoals_avg_info:
				subgoals_avg_info[subgoal] = []
			subgoals_avg_info[subgoal].append(info)
		
		for subgoal, dist in subgoal_distances.items():
			if subgoal not in subgoals_dist_info:
				subgoals_dist_info[subgoal] = []
			subgoals_dist_info[subgoal].append(dist)
		
		avg_subgoal_achievement_reward.append(len([k for k, v in subgoal_info.items() if v]) * subgoal_reward)
			
		# Log the success and length of the episode
		logger.info(f"\n[Episode Num]: {i}")
		logger.info(f"[Success]: {success}, [Length]: {length}")
		
		i += 1
		if pbar is not None:
			pbar.update(1)
	
	if return_avg_stats:
		subgoals_avg_info = {k: np.mean(v) for k, v in subgoals_avg_info.items()}
		subgoals_dist_info = {k: np.mean(v) for k, v in subgoals_dist_info.items()}
		
		return np.mean(total_returns), np.mean(total_timesteps), np.mean(avg_final_goal_dist), \
			np.mean(avg_perc_decrease), np.mean(avg_subgoal_achievement_reward), subgoals_avg_info, subgoals_dist_info
	else:
		return total_returns, total_timesteps, avg_final_goal_dist, avg_perc_decrease, avg_subgoal_achievement_reward, \
			subgoals_avg_info, subgoals_dist_info


def get_trajectory_option_activation(episode):
	prev_options = episode['prev_skills'].numpy() \
		if isinstance(episode['prev_skills'], tf.Tensor) else episode['prev_skills']
	curr_options = episode['curr_skills'].numpy() \
		if isinstance(episode['curr_skills'], tf.Tensor) else episode['curr_skills']
	
	# noinspection PyTypeChecker
	prev_options = np.argmax(prev_options[0], axis=1).tolist()
	# noinspection PyTypeChecker
	curr_options = np.argmax(curr_options[0], axis=1).tolist()
	
	prev_options: List[int] = [skill for skill in prev_options]
	curr_options: List[int] = [skill for skill in curr_options]
	
	actions = episode['actions'].numpy() if isinstance(episode['actions'], tf.Tensor) else episode['actions']
	actions = actions[0]
	
	# # Log the trajectory in the form <time_step, prev_skill, curr_skill, action>
	# for t in range(len(prev_options)):
	# 	logger.info(f"<{t}: {prev_options[t]} -> {curr_options[t]} -> {actions[t]}>")
	
	return prev_options, curr_options
	

