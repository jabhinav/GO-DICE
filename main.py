import json
import os
from collections import OrderedDict

import logging
import datetime
import tensorflow as tf
from configs.GoalGAIL import get_GoalGAIL_args
from configs.ClassicVAE import get_ClassicVAE_args
from models.GoalGAIL import run as run_GoalGAIL
from models.VAE import run as run_classicVAE

# tf.config.run_functions_eagerly(True)
tf.config.run_functions_eagerly(False)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join('./logging', current_time)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(filename=os.path.join(log_dir, 'logs'), filemode='w',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


get_config = {
    'ClassicVAE': get_ClassicVAE_args,
    'GoalGAIL': get_GoalGAIL_args
}

run_model = {
    'ClassicVAE': run_classicVAE,
    'GoalGAIL': run_GoalGAIL
}


if __name__ == "__main__":
    model = 'ClassicVAE'
    logger.info("################## Working on Model: \"{}\" ##################".format(model))

    # store_data_at = os.path.join(os.getcwd(), 'pnp_data/two_obj_fickle_start.pkl')
    args = get_config[model](log_dir)

    logger.info("---------------------------------------------------------------------------------------------")
    config: dict = vars(args)
    config = {key: str(value) for key, value in config.items()}
    config = OrderedDict(sorted(config.items()))
    logger.info(json.dumps(config, indent=4))
    
    run_model[model](args, store_data_path=None)
