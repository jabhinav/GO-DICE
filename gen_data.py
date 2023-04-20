import argparse
import json
import os
import datetime
import pickle
import logging
import tensorflow as tf
import numpy as np
from typing import Dict
from utils.env import get_PnP_env
from utils.buffer import get_buffer_shape
from domains.PnPExpert import PnPExpert, PnPExpertTwoObj, PnPExpertTwoObjImitator
from her.rollout import RolloutWorker
from her.replay_buffer import ReplayBufferTf
from utils.plot import plot_metrics
from utils.env import get_config_env
from collections import OrderedDict

# # WARNING: For generating data, tf must be run eagerly
# else the init_state/goals generated during tracing will be saved for all rollouts
tf.config.run_functions_eagerly(True)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join('./logging', "dataGen_" + current_time)
if not os.path.exists(log_dir):
	os.makedirs(log_dir)
logging.basicConfig(filename=os.path.join(log_dir, 'logs.txt'), filemode='w',
					format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt='%m/%d/%Y %H:%M:%S',
					level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_pick_drop(args, episode: Dict[str, tf.Tensor]):
	# Each Tensor in episode is of shape (num_episodes, horizon, dim)
	# Merge pick and drop into one skill and retain drop
	
	# Case: 1
	if args.wrap_skill_id == '0':
		# combine pick and grab [1, 0, 0] -> [1] and [0, 1, 0] -> [1]
		temp = episode['curr_skills'][:, :, 0] + episode['curr_skills'][:, :, 1]
		episode['curr_skills'] = tf.stack([temp, episode['curr_skills'][:, :, 2]], axis=-1)
		
		temp = episode['prev_skills'][:, :, 0] + episode['prev_skills'][:, :, 1]
		episode['prev_skills'] = tf.stack([temp, episode['prev_skills'][:, :, 2]], axis=-1)
	
	else:
		raise NotImplementedError
	
	return episode


def run(args):
	exp_env = get_PnP_env(args)
	buffer_shape = get_buffer_shape(args)
	
	if args.two_object:
		data_type = f'two_obj_{args.expert_behaviour}_{args.wrap_skill_id}_{args.split_tag}'
	else:
		data_type = f'single_obj_{args.split_tag}'
	data_path = os.path.join(args.dir_data, data_type + '.pkl')
	env_state_dir = os.path.join(args.dir_data, f'{data_type}_env_states_{args.split_tag}')
	if not os.path.exists(env_state_dir):
		os.makedirs(env_state_dir)
		
	# ############################################# EXPERT POLICY ############################################# #
	if args.two_object:
		num_skills = 5 if args.wrap_skill_id == '2' else 6 if args.wrap_skill_id == '1' else 3
		buffer_shape['prev_skills'] = (args.horizon, num_skills)
		buffer_shape['curr_skills'] = (args.horizon, num_skills)
		# expert_policy = PnPExpertTwoObj(args.expert_behaviour, wrap_skill_id=args.wrap_skill_id)
		expert_policy = PnPExpertTwoObjImitator(wrap_skill_id=args.wrap_skill_id)
	else:
		expert_policy = PnPExpert()
	
	# Initiate a worker to generate expert rollouts
	expert_worker = RolloutWorker(
		exp_env, expert_policy, T=args.horizon, rollout_terminate=args.rollout_terminate, render=args.render,
		is_expert_worker=True
	)
	
	# Load Buffer to store expert data
	expert_buffer = ReplayBufferTf(buffer_shape, args.buffer_size, args.horizon)
	
	# ############################################# Generate DATA ############################################# #
	
	logger.info("Generating {} Expert Demos for {}".format(args.split_tag, args.expert_demos))
	
	n = 0
	while n < args.expert_demos:
		tf.print("Generating {} Episode {}/{}".format(args.split_tag, n + 1, args.expert_demos))
		
		try:
			_episode, exp_stats = expert_worker.generate_rollout()
		except ValueError as e:
			tf.print("Episode Error! Generating another Episode.")
			continue
		
		if args.two_object:
			dist = np.linalg.norm(_episode['states'][0][-1][3:9] - _episode['env_goals'][0][-1][:])
		else:
			dist = np.linalg.norm(_episode['states'][0][-1][3:6] - _episode['env_goals'][0][-1][:])
		
		logger.info(f"({n}/{args.expert_demos}) dist. to goal achieved = {round(dist, 4)}")
		
		# Check if episode is successful
		if tf.math.equal(tf.argmax(_episode['successes'].numpy()[0]), 0):
			# Check the distance between the goal and the object
			tf.print("Episode Unsuccessful! Generating another Episode.")
			continue
		else:
			expert_buffer.store_episode(_episode)
			plot_metrics(
				[_episode['distances'].numpy()[0]], labels=['|G_env - AG_curr|'],
				fig_path=os.path.join(args.dir_root_log, 'Expert_{}_{}.png'.format(
					'two_obj_{}'.format(args.expert_behaviour) if args.two_object else 'single_obj', n)),
				y_label='Metrics', x_label='Steps'
			)
			path_to_init_state_dict = tf.constant(os.path.join(env_state_dir, 'env_{}.pkl'.format(n)))
			with open(path_to_init_state_dict.numpy(), 'wb') as handle:
				pickle.dump(exp_stats['init_state_dict'], handle, protocol=pickle.HIGHEST_PROTOCOL)
			n += 1
	
	expert_buffer.save_buffer_data(data_path)
	logger.info("Saved Expert Demos at {} training.".format(data_path))
	logger.info("Expert Policy Success Rate (Train Data): {}".format(expert_worker.current_success_rate()))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--expert_demos', type=int, default=5, help='(GOAL GAIL usage: 10)')
	parser.add_argument('--split_tag', type=str, default='train')
	parser.add_argument('--render', type=bool, default=True)
	
	# Specify Environment Configuration
	parser.add_argument('--env_name', type=str, default='OpenAIPickandPlace')
	parser.add_argument('--full_space_as_goal', type=bool, default=False)
	parser.add_argument('--two_object', type=bool, default=True)
	parser.add_argument('--expert_behaviour', type=str, default='0', choices=['0', '1', '2'],
						help='Expert behaviour in two_object env')
	parser.add_argument('--stacking', type=bool, default=False)
	parser.add_argument('--target_in_the_air', type=bool, default=False,
						help='Is only valid in two object task')
	parser.add_argument('--fix_goal', type=bool, default=False,
						help='Fix the goal position for one object task')
	parser.add_argument('--fix_object', type=bool, default=False,
						help='Fix the object position for one object task')
	
	# Specify Rollout Data Configuration
	parser.add_argument('--horizon', type=int, default=150,
						help='Set 100 for one_obj, 150 for two_obj:0, two_obj:1 and 150 for two_obj:2')
	parser.add_argument('--rollout_terminate', type=bool, default=False,
						help='We retain the success flag=1 for states which satisfy goal condition,')
	parser.add_argument('--buffer_size', type=int, default=int(1e6),
						help='--')
	
	parser.add_argument('--wrap_skill_id', type=str, default='1', choices=['0', '1', '2'],
						help='consumed by multi-object expert to determine how to wrap effective skills')
	
	parser.add_argument('--dir_root_log', type=str, default=log_dir)
	parser.add_argument('--dir_data', type=str, default='./pnp_data/study')
	_args = parser.parse_args()
	
	# Load the environment config
	_args = get_config_env(_args)
	
	logger.info("---------------------------------------------------------------------------------------------")
	config: dict = vars(_args)
	config = {key: str(value) for key, value in config.items()}
	config = OrderedDict(sorted(config.items()))
	logger.info(json.dumps(config, indent=4))
	
	run(_args)
