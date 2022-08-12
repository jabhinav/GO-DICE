import os
import sys
from domains.teamboxworld import TeamBoxWorld
from domains.boxworld import BoxWorld
from domains.roadworld import RoadWorld
from AAMAS_config import env_roadWorld_config, env_boxWorld_config, env_teamBoxWorld_config, env_pickBoxWorld_config

env_name = 'BoxWorld'
latent_setting = 'dynamic'


if latent_setting == 'static':
    static = True
else:
    static = False
# temperatures = [0.01, 0.01, 0.01]
temperatures = None
num_demos_to_generate = 100
verbose = 1

root_dir = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/'
# Settings numX1_numX2 = 1_1, 1_2, 3_2 : for ablation experiments, specify latent modes to be used for generating data
# latent_restriction = '3_2'
# Path to Data
# data_dir = root_dir + 'training_data/{}_{}'.format(env_name, latent_restriction)
data_dir = root_dir + 'training_data/{}'.format(env_name)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
state_action_path = data_dir + '/stateaction_{}_test.pkl'.format(latent_setting)
latent_path = data_dir + '/latent_{}_test.pkl'.format(latent_setting)

if env_name == "BoxWorld":
    env_config = env_boxWorld_config
    env = BoxWorld(grid_length_x=env_config.grid_length_x, grid_length_y=env_config.grid_length_y, static=static,
                   temperatures=temperatures)
elif env_name == "RoadWorld":
    env_config = env_roadWorld_config
    env = RoadWorld(grid_length_x=env_config.grid_length_x, grid_length_y=env_config.grid_length_y)
elif env_name == "TeamBoxWorld":
    env_config = env_teamBoxWorld_config
    env = TeamBoxWorld(grid_length_x=env_config.grid_length_x, grid_length_y=env_config.grid_length_y, static=static,
                       temperatures=temperatures)
# elif env_name == "PickBoxWorld":
#     env_config = env_pickBoxWorld_config
#     env = PickBoxWorld(grid_length_x=env_config.grid_length_x, grid_length_y=env_config.grid_length_y, root_dir=root_dir)
else:
    print("Specify a valid environment name in the config file")
    sys.exit(-1)

# Demo: (state_id, action_id)
# Latents: (x1, x2)
demos, latents = env.get_demonstrations(quantity=num_demos_to_generate)

# # Save the data in python-2 compatible format
# with open(state_action_path, 'wb') as o:
#     pkl.dump(np.array(demos), o, protocol=2)
# with open(latent_path, 'wb') as o:
#     pkl.dump(np.array(latents), o, protocol=2)

print("Done!")
if verbose:
    for demo, latent in zip(demos, latents):
        print("------- Demo -------")
        for index in range(len(demo)):
            state_id = demo[index][0]
            action_id = demo[index][1]
            curr_latent_mode = latent[index]
            state = env.inv_states_dict[state_id]
            action = env.inv_action_dict[action_id]
            print("Latent [State - Action]: {} [{} - {}]".format(curr_latent_mode, state, action))
