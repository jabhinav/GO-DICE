import datetime
import json
import os
from collections import OrderedDict

import tensorflow as tf

import logging
from configs.goalGuidedOptionBC import get_goalGuidedOptionBC_args
from configs.DICE import get_DICE_args
from models.goalGuidedOptionBC import run as run_goalOptionBC
from models.valueDICE import run as run_valueDICE
from models.demoDICE import run as run_demoDICE
from models.BC import run as run_BC

from verify import run_gpred as run_verify

tf.config.run_functions_eagerly(False)
# tf.config.run_functions_eagerly(True)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join('./logging', 'run' + current_time)
if not os.path.exists(log_dir):
	os.makedirs(log_dir)

logging.basicConfig(filename=os.path.join(log_dir, 'logs.txt'), filemode='w',
					format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt='%m/%d/%Y %H:%M:%S',
					level=logging.INFO)
logger = logging.getLogger(__name__)

get_config = {
	'BC': get_DICE_args,
	'goalOptionBC': get_goalGuidedOptionBC_args,
	'ValueDICE': get_DICE_args,
	'DemoDICE': get_DICE_args,
	# 'gDemoDICE': get_DICE_args,
}

run_model = {
	'BC': run_BC,
	'goalOptionBC': run_goalOptionBC,
	'ValueDICE': run_valueDICE,
	'DemoDICE': run_demoDICE,
	# 'gDemoDICE': run_gdemoDICE,
}


def run(_args):
	run_model[_args.model](_args)


def verify(_args):
	logger.info("# ################# Verifying ################# #")
	run_verify(_args)


if __name__ == "__main__":
	model = 'DemoDICE'
	logger.info("# ################# Working on Model: \"{}\" ################# #".format(model))
	
	# store_data_at = os.path.join(os.getcwd(), 'pnp_data/two_obj_fickle_start.pkl')
	args = get_config[model](log_dir)
	args.model = model
	
	logger.info("---------------------------------------------------------------------------------------------")
	config: dict = vars(args)
	config = {key: str(value) for key, value in config.items()}
	config = OrderedDict(sorted(config.items()))
	logger.info(json.dumps(config, indent=4))
	
	run(args)
	# verify(args)
