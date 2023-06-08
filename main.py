import datetime
import json
import os
import sys
import time
import numpy as np
from collections import OrderedDict
from typing import Dict, Tuple

import logging
import tensorflow as tf

from configs.DICE import get_DICE_args
from her.replay_buffer import ReplayBufferTf
from her.transitions import sample_transitions
from models.demoDICE import Agent as Agent_demoDICE
from models.skilledDemoDICE import Agent as Agent_skilledDemoDICE
from models.GoFar import Agent as Agent_GoFar
from models.BC import Agent as Agent_BC
from utils.buffer import get_buffer_shape
from utils.custom import state_to_goal
from verify import run_verify


get_config = {
	'BC': get_DICE_args,
	# 'goalOptionBC': get_goalGuidedOptionBC_args,
	# 'ValueDICE': get_DICE_args,
	'DemoDICE': get_DICE_args,
	'GoFar': get_DICE_args,
	'SkilledDemoDICE': get_DICE_args,
}

Agents = {
	'BC': Agent_BC,
	# 'goalOptionBC': run_goalOptionBC,
	# 'ValueDICE': run_valueDICE,
	'DemoDICE': Agent_demoDICE,
	'GoFar': Agent_GoFar,
	'SkilledDemoDICE': Agent_skilledDemoDICE,
}


def verify(algo: str):
	current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	tf.config.run_functions_eagerly(True)
	log_dir = os.path.join('./logging', 'verify' + current_time)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir, exist_ok=True)
	
	logging.basicConfig(filename=os.path.join(log_dir, 'logs.txt'), filemode='w',
						format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S',
						level=logging.INFO)
	logger = logging.getLogger(__name__)
	
	logger.info("# ################# Verifying ################# #")

	args = get_config[algo](log_dir, db=True)
	args.algo = algo
	
	logger.info("---------------------------------------------------------------------------------------------")
	config: dict = vars(args)
	config = {key: str(value) for key, value in config.items()}
	config = OrderedDict(sorted(config.items()))
	logger.info(json.dumps(config, indent=4))
	
	run_verify(args, model_dir='./logging/TwoObj_skilledDemoDICE_semi_unseg_polyak/models')


def run(db: bool, algo: str):
	current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	
	# Set the random seeds randomly for more randomness
	np.random.seed(int(time.time()))
	tf.random.set_seed(int(time.time()))
	
	if db:
		print("Running in Debug Mode. (db=True)")
	
	tf.config.run_functions_eagerly(db)
	
	log_dir = os.path.join('./logging', algo, '{}'.format('debug' if db else 'run') + current_time)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir, exist_ok=True)
	
	logging.basicConfig(filename=os.path.join(log_dir, 'logs.txt'), filemode='w',
						format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S',
						level=logging.INFO)
	logger = logging.getLogger(__name__)
	
	logger.info("# ################# Working on Model: \"{}\" ################# #".format(algo))
	
	args = get_config[algo](log_dir, db=db)
	args.algo = algo
	
	logger.info("---------------------------------------------------------------------------------------------")
	config: dict = vars(args)
	config = {key: str(value) for key, value in config.items()}
	config = OrderedDict(sorted(config.items()))
	logger.info(json.dumps(config, indent=4))
	
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
	n_objs = args.num_objs
	buffer_shape: Dict[str, Tuple[int, ...]] = get_buffer_shape(args)
	
	expert_buffer = ReplayBufferTf(
		buffer_shape, args.buffer_size, args.horizon,
		sample_transitions(args.trans_style, state_to_goal=state_to_goal(n_objs), num_options=args.c_dim),
	)
	offline_buffer = ReplayBufferTf(
		buffer_shape, args.buffer_size, args.horizon,
		sample_transitions(args.trans_style, state_to_goal=state_to_goal(n_objs), num_options=args.c_dim)
	)
	if n_objs == 3:
		expert_data_file = 'three_obj_{}_{}_train.pkl'.format(args.expert_behaviour, args.wrap_skill_id)
		offline_data_file = 'three_obj_{}_{}_offline.pkl'.format(args.expert_behaviour, args.wrap_skill_id)
	elif n_objs == 2:
		expert_data_file = 'two_obj_{}_{}_train.pkl'.format(args.expert_behaviour, args.wrap_skill_id)
		offline_data_file = 'two_obj_{}_{}_offline.pkl'.format(args.expert_behaviour, args.wrap_skill_id)
	elif n_objs == 1:
		expert_data_file = 'single_obj_train.pkl'
		offline_data_file = 'single_obj_offline.pkl'
	else:
		raise NotImplementedError
	
	expert_data_path = os.path.join(args.dir_data, expert_data_file)
	if args.wandb_project.endswith('Exp'):
		offline_data_path = os.path.join(args.dir_data, expert_data_file)  # For EXP
	else:
		offline_data_path = os.path.join(args.dir_data, offline_data_file)
	
	if not os.path.exists(expert_data_path):
		logger.error("Expert data not found at {}. Please run the data generation script first.".format(expert_data_path))
		sys.exit(-1)
	
	if not os.path.exists(offline_data_path):
		logger.error("Offline data not found at {}. Please run the data generation script first.".format(offline_data_path))
		sys.exit(-1)
	
	# Store the expert data in the expert buffer -> D_E
	logger.info("Loading Expert Demos from {} into Expert Buffer for training.".format(expert_data_path))
	expert_buffer.load_data_into_buffer(path_to_data=expert_data_path, num_demos_to_load=args.expert_demos)
	
	# Store the expert and offline data in the policy buffer for DemoDICE -> D_O
	logger.info("Loading Offline Demos from {} into Offline Buffer for training.".format(offline_data_path))
	offline_buffer.load_data_into_buffer(path_to_data=offline_data_path, num_demos_to_load=args.offline_demos)

	# ########################################################################################################### #
	# ############################################# TRAINING #################################################### #
	# ########################################################################################################### #
	start = time.time()
	
	agent = Agents[args.algo](args, expert_buffer, offline_buffer)
	
	# logger.info("Load Actor Policy from {}".format(args.dir_pre))
	# agent.load_actor(dir_param=args.dir_pre)
	# print("Actor Loaded")
	
	logger.info("Training .......")
	agent.learn()
	logger.info("Done Training in {}".format(str(datetime.timedelta(seconds=time.time() - start))))


if __name__ == "__main__":
	num_runs = 1
	for i in range(num_runs):
		run(db=True, algo='SkilledDemoDICE')
	# verify(algo='GoFar')

