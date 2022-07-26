import numpy as np
import tensorflow as tf
import pickle as pkl
import os
import sys
from models import Agent
from domains.teamboxworld import TeamBoxWorld
from domains.boxworld import BoxWorld
from domains.roadworld import RoadWorld
from config import InfoGAIL_config as model_config
from config import state_action_path, latent_path, env_name, seed_values, InfoGAIL_pretrained_model_dir
from config import env_roadWorld_config, env_boxWorld_config, training_data_config, env_teamBoxWorld_config
from utils.misc import parse_data
from config import param_dir as p_dir


def run(exp_num, env, env_config, param_dir, finetune=0):
    print("\n********** Experiment {} **********".format(exp_num))
    state_dim = env_config.state_dim
    encode_dim = env_config.encode_dim
    action_dim = env_config.action_dim

    # define the model
    agent = Agent(env, state_dim, action_dim, encode_dim, model_config)

    # Load expert (state, action) pairs and G.T latent modes
    print("Loading data")
    with open(state_action_path, 'rb') as f:
        trajectories = pkl.load(f)
    with open(latent_path, 'rb') as f:
        latent_modes_traj = pkl.load(f)

    # Retrieve demo for InfoGAIL (Get training data)
    demo = parse_data(env, trajectories, latent_modes_traj, lower_bound=0, upper_bound=training_data_config.num_traj)

    param_dir = os.path.join(param_dir, "params{}".format(exp_num))
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)
    # Now load the weight
    print("Now we load the weights")
    try:
        if finetune:
            agent.generator.load_weights(param_dir + "/generator_model_37.h5")
            agent.discriminator.load_weights(param_dir + "/discriminator_model_37.h5")
            agent.baseline.model.load_weights(param_dir + "/baseline_model_37.h5")
            agent.posterior.load_weights(param_dir + "/posterior_model_37.h5")
            agent.posterior_target.load_weights(param_dir + "/posterior_target_model_37.h5")
        else:
            print("Picking Pre-trained BC model from", InfoGAIL_pretrained_model_dir + "generator_bc_model_{}.h5".format(exp_num))
            agent.generator.load_weights(InfoGAIL_pretrained_model_dir + "generator_bc_model_{}.h5".format(exp_num))
        print("Weight loaded successfully")
    except:
        print("Cannot find the weight")

    print("Number of sample points from expert demos:", demo['states'].shape[0])
    agent.train(demo, param_dir, exp_num, verbose=0)

    print("Finish.")


def run_multiple_runs(specified_runs=1):

    if env_name == "BoxWorld":
        env_config = env_boxWorld_config
        env = BoxWorld(grid_length_x=env_config.grid_length_x, grid_length_y=env_config.grid_length_y)
    elif env_name == "RoadWorld":
        env_config = env_roadWorld_config
        env = RoadWorld(grid_length_x=env_config.grid_length_x, grid_length_y=env_config.grid_length_y)
    elif env_name == "TeamBoxWorld":
        env_config = env_teamBoxWorld_config
        env = TeamBoxWorld(grid_length_x=env_config.grid_length_x, grid_length_y=env_config.grid_length_y)
    else:
        print("Specify a valid environment name in the config file")
        sys.exit(-1)

    # For averaging across different runs
    for exp_num, seed_value in enumerate(seed_values[:specified_runs]):

        # Set a seed value
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)

        # Run multiple experiments
        run(exp_num, env, env_config, param_dir=p_dir)


if __name__ == "__main__":
    run_multiple_runs()
