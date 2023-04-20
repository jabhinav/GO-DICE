import datetime
import json
import os
import sys
import time
from collections import OrderedDict

import logging
import tensorflow as tf

from configs.DICE import get_DICE_args
from her.replay_buffer import ReplayBufferTf
from her.transitions import sample_transitions
from models.demoDICE import Agent as Agent_demoDICE
from models.skilledDemoDICE import Agent as Agent_skilledDemoDICE
from modelsBaseline.BC import Agent as Agent_BC
from utils.buffer import get_buffer_shape
from utils.custom import state_to_goal
from verify import run_verify


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

get_config = {
	'BC': get_DICE_args,
	# 'goalOptionBC': get_goalGuidedOptionBC_args,
	# 'ValueDICE': get_DICE_args,
	'DemoDICE': get_DICE_args,
	'skilledDemoDICE': get_DICE_args,
}

Agents = {
	'BC': Agent_BC,
	# 'goalOptionBC': run_goalOptionBC,
	# 'ValueDICE': run_valueDICE,
	'DemoDICE': Agent_demoDICE,
	'skilledDemoDICE': Agent_skilledDemoDICE,
}


def verify():
	tf.config.run_functions_eagerly(True)
	log_dir = os.path.join('./logging', 'verify' + current_time)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir, exist_ok=True)
	
	logging.basicConfig(filename=os.path.join(log_dir, 'logs.txt'), filemode='w',
						format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S',
						level=logging.INFO)
	logger = logging.getLogger(__name__)
	
	model = 'skilledDemoDICE'
	logger.info("# ################# Verifying ################# #")

	args = get_config[model](log_dir, db=True)
	args.model = model
	
	logger.info("---------------------------------------------------------------------------------------------")
	config: dict = vars(args)
	config = {key: str(value) for key, value in config.items()}
	config = OrderedDict(sorted(config.items()))
	logger.info(json.dumps(config, indent=4))
	
	run_verify(args)


def run(db=False):
	
	if db:
		print("Running in Debug Mode. (db=True)")
	
	tf.config.run_functions_eagerly(db)
	
	log_dir = os.path.join('./logging', '{}'.format('debug' if db else 'run') + current_time)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir, exist_ok=True)
	
	logging.basicConfig(filename=os.path.join(log_dir, 'logs.txt'), filemode='w',
						format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S',
						level=logging.INFO)
	logger = logging.getLogger(__name__)
	
	model = 'skilledDemoDICE'
	logger.info("# ################# Working on Model: \"{}\" ################# #".format(model))
	
	args = get_config[model](log_dir, db=db)
	args.model = model
	
	logger.info("---------------------------------------------------------------------------------------------")
	config: dict = vars(args)
	config = {key: str(value) for key, value in config.items()}
	config = OrderedDict(sorted(config.items()))
	logger.info(json.dumps(config, indent=4))
	
	# # For Debugging [#1]
	# verify(args)
	
	# # For Debugging [#2]
	# if args.fix_goal and args.fix_object:
	# 	data_prefix = 'fOfG_'
	# elif args.fix_goal and not args.fix_object:
	# 	data_prefix = 'dOfG_'
	# elif args.fix_object and not args.fix_goal:
	# 	data_prefix = 'fOdG_'
	# else:
	# 	data_prefix = 'dOdG_'
	
	# Clear tensorflow graph and cache
	tf.keras.backend.clear_session()
	tf.compat.v1.reset_default_graph()
	
	# ######################################################################################################## #
	# ############################################# DATA LOADING ############################################# #
	# ######################################################################################################## #
	# Load Buffer to store expert data
	n_objs = 2 if args.two_object else 1
	buffer_shape = get_buffer_shape(args)
	
	expert_buffer_unseg = ReplayBufferTf(
		buffer_shape, args.buffer_size, args.horizon,
		sample_transitions('random_segmented', state_to_goal=state_to_goal(n_objs), num_options=args.c_dim),
	)
	policy_buffer_unseg = ReplayBufferTf(
		buffer_shape, args.buffer_size, args.horizon,
		sample_transitions('random_segmented', state_to_goal=state_to_goal(n_objs), num_options=args.c_dim)
	)
	
	if args.two_object:
		data_file = 'two_obj_{}_{}_train.pkl'.format(args.expert_behaviour, args.wrap_skill_id)
	else:
		data_file = 'single_obj_train.pkl'
	train_data_path = os.path.join(args.dir_data, data_file)
	
	if not os.path.exists(train_data_path):
		logger.error("Train data not found at {}. Please run the data generation script first.".format(train_data_path))
		sys.exit(-1)
	else:
		logger.info("Loading Expert Demos from {} into TrainBuffer for training.".format(train_data_path))
		# Store the expert data in the expert buffer -> D_E
		expert_buffer_unseg.load_data_into_buffer(train_data_path, num_demos_to_load=args.expert_demos)
		
		# # Store the expert data in the policy buffer for DemoDICE -> D_U = D_E + D_I
		# policy_buffer_unseg.load_data_into_buffer(train_data_path,
		# 										  num_demos_to_load=args.expert_demos)
		# # # BC offline data
		# policy_buffer_unseg.load_data_into_buffer(f'./pnp_data/BC_{data_prefix}offline_data.pkl',
		# 										  num_demos_to_load=args.imperfect_demos,
		# 										  clear_buffer=False)
		
		# Store the expert data in the policy buffer for DemoDICE -> D_U = D_E + D_I
		policy_buffer_unseg.load_data_into_buffer(train_data_path)
	
	# ########################################################################################################### #
	# ############################################# TRAINING #################################################### #
	# ########################################################################################################### #
	start = time.time()
	
	agent = Agents[args.model](args, expert_buffer_unseg, policy_buffer_unseg)
	
	# logger.info("Load Actor Policy from {}".format(args.dir_pre))
	# agent.load_actor(dir_param=args.dir_pre)
	# print("Actor Loaded")
	
	logger.info("Training .......")
	agent.learn()
	logger.info("Done Training in {}".format(str(datetime.timedelta(seconds=time.time() - start))))


if __name__ == "__main__":
	# run(db=False)
	verify()

