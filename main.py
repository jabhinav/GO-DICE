import json
import os
import logging
import datetime
import tensorflow as tf
from collections import OrderedDict
from configs.GoalGAIL import get_GoalGAIL_args
from configs.ClassicVAE import get_ClassicVAE_args
from configs.CCVAE import get_ccVAE_args
from models.GoalGAIL import run as run_GoalGAIL
from models.VAE import run as run_classicVAE
from models.CCVAE import run as run_ccVAE

tf.config.run_functions_eagerly(False)
# tf.config.run_functions_eagerly(True)

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
    'CCVAE': get_ccVAE_args,
    'GoalGAIL': get_GoalGAIL_args
}

run_model = {
    'ClassicVAE': run_classicVAE,
    'CCVAE': run_ccVAE,
    'GoalGAIL': run_GoalGAIL
}


if __name__ == "__main__":
    model = 'CCVAE'
    logger.info("################## Working on Model: \"{}\" ##################".format(model))

    # store_data_at = os.path.join(os.getcwd(), 'pnp_data/two_obj_fickle_start.pkl')
    args = get_config[model](log_dir)

    logger.info("---------------------------------------------------------------------------------------------")
    config: dict = vars(args)
    config = {key: str(value) for key, value in config.items()}
    config = OrderedDict(sorted(config.items()))
    logger.info(json.dumps(config, indent=4))
    
    run_model[model](args)
