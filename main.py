import json
import os
import logging
import datetime
import tensorflow as tf
from collections import OrderedDict

from configs.GoalGAIL import get_GoalGAIL_args
from configs.Classic import get_ClassicVAE_args
from configs.gCCVAE import get_gCCVAE_args
from configs.lCCVAE import get_lCCVAE_args

from models.BC import run as run_BC
from models.gBC import run as run_gBC
from models.GoalGAIL import run as run_GoalGAIL
from models.ClassicVAE import run as run_classicVAE
from models.gCCVAE import run as run_gCCVAE
from models.lCCVAE import run as run_lCCVAE

# tf.config.run_functions_eagerly(False)
tf.config.run_functions_eagerly(True)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join('./logging', current_time)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(filename=os.path.join(log_dir, 'logs.txt'), filemode='w',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


get_config = {
    'BC': get_ClassicVAE_args,
    'gBC': get_ClassicVAE_args,
    'ClassicVAE': get_ClassicVAE_args,
    'gCCVAE': get_gCCVAE_args,
    'lCCVAE': get_lCCVAE_args,
    'GoalGAIL': get_GoalGAIL_args
}

run_model = {
    'BC': run_BC,
    'gBC': run_gBC,
    'ClassicVAE': run_classicVAE,
    'gCCVAE': run_gCCVAE,
    'lCCVAE': run_lCCVAE,
    'GoalGAIL': run_GoalGAIL
}


if __name__ == "__main__":
    model = 'gBC'
    logger.info("################## Working on Model: \"{}\" ##################".format(model))

    # store_data_at = os.path.join(os.getcwd(), 'pnp_data/two_obj_fickle_start.pkl')
    args = get_config[model](log_dir)

    logger.info("---------------------------------------------------------------------------------------------")
    config: dict = vars(args)
    config = {key: str(value) for key, value in config.items()}
    config = OrderedDict(sorted(config.items()))
    logger.info(json.dumps(config, indent=4))
    
    run_model[model](args)
