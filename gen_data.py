import argparse
import json
import os
import datetime
import pickle
import logging
import tensorflow as tf
import numpy as np
from utils.env import get_PnP_env
from utils.buffer import get_buffer_shape
from domains.PnPExpert import PnPExpert, PnPExpertTwoObj
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


def run(args):
	exp_env = get_PnP_env(args)
	
	# ############################################# EXPERT POLICY ############################################# #
	if args.two_object:
		expert_policy = PnPExpertTwoObj(exp_env.latent_dim, args.expert_behaviour)
	else:
		expert_policy = PnPExpert(exp_env.latent_dim)
	
	# Initiate a worker to generate expert rollouts
	expert_worker = RolloutWorker(
		exp_env, expert_policy, T=args.horizon, rollout_terminate=args.rollout_terminate, render=False,
		is_expert_worker=True
	)
	# ############################################# DATA TRAINING ############################################# #
	# Load Buffer to store expert data
	num_train_episodes = int(args.expert_demos * args.perc_train)
	if num_train_episodes:
		expert_buffer = ReplayBufferTf(get_buffer_shape(args), args.buffer_size, args.horizon)
		
		train_data_path = os.path.join(args.dir_data, '{}_train.pkl'.format(
			'two_obj_{}'.format(args.expert_behaviour) if args.two_object else 'single_obj'))
		env_state_dir = os.path.join(
			args.dir_data,
			'{}_env_states_train'.format(
				'two_obj_{}'.format(args.expert_behaviour) if args.two_object else 'single_obj'))
		
		if not os.path.exists(env_state_dir):
			os.makedirs(env_state_dir)
		
		exp_stats = {'success_rate': 0.}
		logger.info("Generating {} Expert Demos for training.".format(num_train_episodes))
		
		n = 0
		while n < num_train_episodes:
			tf.print("Generating Train Episode {}/{}".format(n + 1, num_train_episodes))
			try:
				_episode, exp_stats = expert_worker.generate_rollout()
			except ValueError as e:
				tf.print("Episode Error! Generating another Episode.")
				continue
			
			if args.two_object:
				dist = np.linalg.norm(_episode['states'][0][-1][3:9] - _episode['env_goals'][0][-1][:])
			else:
				dist = np.linalg.norm(_episode['states'][0][-1][3:6] - _episode['env_goals'][0][-1][:])
			logger.info(f"({n}/{num_train_episodes}) dist. to goal achieved = {round(dist, 4)}")
			
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
		
		expert_buffer.save_buffer_data(train_data_path)
		logger.info("Saved Expert Demos at {} training.".format(train_data_path))
		logger.info("Expert Policy Success Rate (Train Data): {}".format(expert_worker.current_success_rate()))
	
	# ############################################# DATA VALIDATION ############################################# #
	expert_worker.clear_history()
	num_val_episodes = args.expert_demos - num_train_episodes
	
	if num_val_episodes:
		val_buffer = ReplayBufferTf(get_buffer_shape(args), args.buffer_size, args.horizon)
		
		val_data_path = os.path.join(args.dir_data, '{}_val.pkl'.format(
			'two_obj_{}'.format(args.expert_behaviour) if args.two_object else 'single_obj'))
		env_state_dir = os.path.join(
			args.dir_data,
			'{}_env_states_val'.format('two_obj_{}'.format(args.expert_behaviour) if args.two_object else 'single_obj'))
		
		if not os.path.exists(env_state_dir):
			os.makedirs(env_state_dir)
		
		logger.info("Generating {} Expert Demos for validation.".format(num_val_episodes))
		exp_stats = {'success_rate': 0.}
		
		# Generate and store expert validation data
		n = 0
		while n < num_val_episodes:
			tf.print("Generating Val Episode {}/{}".format(n + 1, num_val_episodes))
			try:
				_episode, exp_stats = expert_worker.generate_rollout()
			except ValueError as e:
				tf.print("Episode Error! Generating another Episode.")
				continue
			
			if args.two_object:
				dist = np.linalg.norm(_episode['states'][0][-1][3:9] - _episode['env_goals'][0][-1][:])
			else:
				dist = np.linalg.norm(_episode['states'][0][-1][3:6] - _episode['env_goals'][0][-1][:])
			logger.info(f"({n}/{num_val_episodes}) dist. to goal achieved = {round(dist, 4)}")
			
			# Check if episode is successful
			if tf.math.equal(tf.argmax(_episode['successes'].numpy()[0]), 0):
				tf.print("Episode Unsuccessful! Generating another Episode.")
				continue
			else:
				val_buffer.store_episode(_episode)
				path_to_init_state_dict = tf.constant(os.path.join(env_state_dir, 'env_{}.pkl'.format(n)))
				with open(path_to_init_state_dict.numpy(), 'wb') as handle:
					pickle.dump(exp_stats['init_state_dict'], handle, protocol=pickle.HIGHEST_PROTOCOL)
				n += 1
		
		val_buffer.save_buffer_data(val_data_path)
		logger.info("Saved Expert Demos at {} for validation.".format(val_data_path))
		logger.info("Expert Policy Success Rate (Val Data): {}".format(exp_stats['success_rate']))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--expert_demos', type=int, default=100, help='Use 100 (GOAL GAIL usage)')
	parser.add_argument('--perc_train', type=int, default=1.0,
						help='Percentage of expert demos to use for training. Use 0.9')
	
	# Specify Environment Configuration
	parser.add_argument('--env_name', type=str, default='OpenAIPickandPlace')
	parser.add_argument('--full_space_as_goal', type=bool, default=False)
	parser.add_argument('--two_object', type=bool, default=False)
	parser.add_argument('--expert_behaviour', type=str, default='0', choices=['0', '1', '2'],
						help='Expert behaviour in two_object env')
	parser.add_argument('--stacking', type=bool, default=False)
	parser.add_argument('--target_in_the_air', type=bool, default=False,
						help='Is only valid in two object task')
	parser.add_argument('--fix_goal', type=bool, default=False,
						help='Fix the goal position for one object task')
	parser.add_argument('--fix_object', type=bool, default=False,
						help='Fix the object position for one object task')
	
	parser.add_argument('--horizon', type=int, default=100,
						help='Set 50 for one_obj, 125 for two_obj:0, two_obj:1 and 150 for two_obj:2')
	parser.add_argument('--rollout_terminate', type=bool, default=False,
						help='We retain the success flag=1 for states which satisfy goal condition,')
	parser.add_argument('--buffer_size', type=int, default=int(1e6),
						help='--')
	
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
