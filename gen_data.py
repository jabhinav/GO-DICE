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
from domains.PnPExpert import PnPExpertOneObjImitator, PnPExpertTwoObjImitator, PnPExpertMultiObjImitator
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


def merge_pick_grab(args, episode: Dict[str, tf.Tensor]):
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
	num_objs = int(args.num_objs)
	
	if num_objs == 3:
		data_type = f'three_obj_{args.expert_behaviour}_{args.split_tag}'
	elif num_objs == 2:
		data_type = f'two_obj_{args.expert_behaviour}_{args.split_tag}'
	elif num_objs == 1:
		data_type = f'single_obj_{args.split_tag}'
	
	data_path = os.path.join(args.dir_data, data_type + '.pkl')
	env_state_dir = os.path.join(args.dir_data, f'{data_type}_env_states')
	if not os.path.exists(env_state_dir):
		os.makedirs(env_state_dir)
		
	# ############################################# EXPERT POLICY ############################################# #
	num_skills = args.num_objs * 3
	buffer_shape['prev_skills'] = (args.horizon, num_skills)
	buffer_shape['curr_skills'] = (args.horizon, num_skills)
	if num_objs >= 3:
		expert_policy = PnPExpertMultiObjImitator(args)
	elif num_objs == 2:
		expert_policy = PnPExpertTwoObjImitator(args)
	elif num_objs == 1:
		expert_policy = PnPExpertOneObjImitator(args)
	else:
		raise NotImplementedError
	
	# Initiate a worker to generate expert rollouts
	expert_worker = RolloutWorker(
		exp_env, expert_policy, T=args.horizon, rollout_terminate=args.rollout_terminate, render=args.render,
		is_expert_worker=True
	)
	
	# Load Buffer to store expert data
	expert_buffer = ReplayBufferTf(buffer_shape, args.buffer_size, args.horizon)
	
	# ############################################# Generate DATA ############################################# #
	
	logger.info("Generating {} Expert Demos for {}".format(args.split_tag, args.num_demos))
	
	n = 0
	while n < args.num_demos:
		tf.print("Generating {} Episode {}/{}".format(args.split_tag, n + 1, args.num_demos))
		try:
			_episode, exp_stats = expert_worker.generate_rollout(epsilon=args.random_eps, stddev=args.noise_eps)
		except ValueError as e:
			tf.print("Episode Error! Generating another Episode.")
			continue
		
		dist = np.linalg.norm(_episode['states'][0][-1][3:3+3*num_objs] - _episode['env_goals'][0][-1][:])
		logger.info(f"({n}/{args.num_demos}) dist. to goal achieved = {round(dist, 4)}")
		
		# Check if episode is successful
		if tf.math.equal(tf.argmax(_episode['successes'].numpy()[0]), 0) \
				and args.random_eps == 0.0 and args.noise_eps == 0.0:  # If No noise or randomization
			# Check the distance between the goal and the object
			tf.print("Episode Unsuccessful! Generating another Episode.")
			continue
		else:
			expert_buffer.store_episode(_episode)
			plot_metrics(
				[_episode['distances'].numpy()[0]], labels=['|G_env - AG_curr|'],
				fig_path=os.path.join(args.dir_root_log, 'Expert_{}_{}.png'.format(data_type, n)),
				y_label='Metrics', x_label='Steps'
			)
			path_to_init_state_dict = tf.constant(os.path.join(env_state_dir, 'env_{}.pkl'.format(n)))
			with open(path_to_init_state_dict.numpy(), 'wb') as handle:
				pickle.dump(exp_stats['init_state_dict'], handle, protocol=pickle.HIGHEST_PROTOCOL)
			n += 1
	
	expert_buffer.save_buffer_data(data_path)
	logger.info("Saved Expert Demos at {} training.".format(data_path))
	logger.info("Expert Policy Success Rate: {}".format(expert_worker.current_success_rate()))
	tf.print("Expert Policy Success Rate: {}".format(expert_worker.current_success_rate()))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_demos', type=int, default=100)
	parser.add_argument('--split_tag', type=str, default='train')
	parser.add_argument('--render', type=bool, default=False)
	
	# Specify Environment Configuration
	parser.add_argument('--env_name', type=str, default='OpenAIPickandPlace')
	parser.add_argument('--full_space_as_goal', type=bool, default=False)
	parser.add_argument('--num_objs', type=int, default=1)
	parser.add_argument('--expert_behaviour', type=str, default='0', choices=['0', '1'],
						help='Expert behaviour in multi-obj env. Note that 1 is the randomised behaviour which '
							 'wont work in the stacking env because goals are suspended in the air and reaching them '
							 'is impossible without stacking objects below. FIX: order of goal for the objects should '
							 'be implemented in the env first, which then must be observed by the agent.')
	parser.add_argument('--stacking', type=bool, default=False)
	parser.add_argument('--fix_goal', type=bool, default=False,
						help='[Debug] Fix the goal position for one object task')
	parser.add_argument('--fix_object', type=bool, default=False,
						help='[Debug] Fix the object position for one object task')
	
	# Specify Rollout Data Configuration
	parser.add_argument('--horizon', type=int, default=100,
						help='Set 100 for one_obj, 150 for two_obj and 200 for three_obj')
	parser.add_argument('--rollout_terminate', type=bool, default=False,
						help='We retain the success flag=1 for states which satisfy goal condition,')
	parser.add_argument('--buffer_size', type=int, default=int(1e6),
						help='--')
	parser.add_argument('--random_eps', type=float, default=0.0, help='random eps = 0.05')
	parser.add_argument('--noise_eps', type=float, default=0.0, help='noise eps = 0.01')
	
	parser.add_argument('--dir_root_log', type=str, default=log_dir)
	parser.add_argument('--dir_data', type=str, default='./pnp_data/study')
	_args = parser.parse_args()
	
	# Load the environment config
	_args = get_config_env(_args, ag_in_env_goal=True)
	
	logger.info("---------------------------------------------------------------------------------------------")
	config: dict = vars(_args)
	config = {key: str(value) for key, value in config.items()}
	config = OrderedDict(sorted(config.items()))
	logger.info(json.dumps(config, indent=4))
	
	run(_args)
