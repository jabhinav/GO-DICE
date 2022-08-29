import tensorflow as tf


class Dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


seed_values = [2410, 1230, 10, 500, 781, 81, 52, 1567, 1890, 1]

training_data_config = Dict2(
    num_traj=100
)

dtype = tf.float32

# Paths
env_name = 'BoxWorld'  # RoadWorld, BoxWorld, TeamBoxWorld
supervision_setting = "unsupervised"  # unsupervised, supervised
latent_setting = 'static'  # static, dynamic
model_name = 'BC'  # BC(unsupervised, supervised), BC_InfoGAIL(unsupervised InfoGAIL), InfoGAIL, GAIL

# print("Env: {}, Latent_Supervision: {}, Latent_setting: {}, model: {}".format(env_name, supervision_setting,
#                                                                               latent_setting, model_name))

# Provide path to Data
data_dir = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/training_data/{}'.format(env_name)
state_action_path = data_dir + '/{}_stateaction_{}.pkl'.format(env_name, latent_setting)
latent_path = data_dir + '/{}_latent_{}.pkl'.format(env_name, latent_setting)

InfoGAIL_pretrained_model_dir = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/pretrained_params/' \
                                '{}/params/{}/{}/{}/'.format(env_name, supervision_setting, latent_setting,
                                                             'BC_InfoGAIL')
param_dir = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/pretrained_params/' \
            '{}/params/{}/{}/{}/'.format(env_name, supervision_setting, latent_setting, model_name)
# log_dir = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/pretrained_params/{}/logs/'.format(env_name)
fig_name = '{}_loss'.format(model_name)
fig_path = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/pretrained_params/{}/params/' \
           '{}/{}/{}/{}'.format(env_name, supervision_setting, latent_setting, model_name, fig_name)

kl_figpath = '/Users/apple/Desktop/PhDCoursework/COMP641/InfoGAIL/pretrained_params/{}/params/' \
             '{}/{}/{}/'.format(env_name, supervision_setting, latent_setting, model_name)


"""
cg_damping: Artifact for numerical stability, should be smallish. Adjusts Hessian-vector product calculation: 
            Hv -> (alpha I + H)v where alpha is the damping coefficient. Probably don’t play with this hyperparameter.
cg_iters: Number of iterations of conjugate gradient to perform. Increasing this will lead to a more accurate approx. 
          to H^{-1} g, and possibly slightly-improved performance, but at the cost of slowing things down. 
          Also probably don’t play with this hyper-parameter.
b_iters: Maximum number of steps allowed in the backtracking line search. Since the line search usually doesn’t 
         backtrack, and usually only steps back once when it does, this hyperparameter doesn’t often matter.
"""

# Overall Model Configuration
BC_config = Dict2(
    batch_size=50,
    train_val_ratio=0.8,
    initial_lr=0.001,
    lr_decay_factor=.99,
    n_epochs=2500
)

BC_InfoGAIL_config = Dict2(
    batch_size=50,
    train_val_ratio=0.8,
    initial_lr=0.001,
    lr_decay_factor=.99,
    n_epochs=2500
)

InfoGAIL_config = Dict2(
    n_epochs=500,
    buffer_size=50,  # Maximum buffer size
    paths_per_collect=10,  # Paths to collect in each epoch (1st epoch we collect 30 paths)
    max_step_limit=30,  # Enough for grid-sizes of 30, 25
    min_step_limit=5,  # Set it based on performance
    sample_size=20,  # Paths from buffer used to train the model
    batch_size=25,  # Batch size = num(s,a,c) tuples for training
    d_iter=10,  # Number of Discriminator iterations (not used from here, it is defined locally)
    lr_discriminator=5e-5,
    p_iter=50,  # Number of Posterior iterations
    lr_posterior=1e-4,
    # pre_step=0,
    gamma=0.95,  # Discount factor for future rewards. (Always between 0 and 1.)
    lam=0.97,
    max_kl=0.001,  # KL-divergence limit for TRPO / NPG update. (Should be small for stability. Values like 0.01, 0.05.)
    cg_damping=0.1,
    cg_iters=20,  # Number of iterations to approx. solution of Hx=g
    clamp_lower=-0.01,
    clamp_upper=0.01,
    lr_baseline=1e-4,
    b_iter=25,  # Number of NN baseline iterations approximating value function
    stuck_count_limit=3
)

"""
Note that state space is fully observable to the agent
"""
# Define the Env Road World
env_roadWorld_config = Dict2(
    state_dim=2,  # Grid Position
    encode_dim=2*2,  # Goal (Top-Left, Top-Right), Direction (Go upwards, Go downwards)
    action_dim=5,  # Up, Down, Right, Left, Stay
    grid_length_x=3,
    grid_length_y=10
)

# Define the Env Box World
env_boxWorld_config = Dict2(
    state_dim=2+2,  # Grid Position (x, y) and
    # Binary indicator for whether M(=2) objects are collected i.e 1 for not collected
    encode_dim=2*2,  # Urgency (0, 1) and Fatigue (0, 1, 2)
    action_dim=4,  # Up, Down, Right, Left
    grid_length_x=5,
    grid_length_y=5
)

# Define the Team Box World
"""
S = (x, y, g_R, O_1, ...., O_M) i.e. Total states = N*N*(M+1)*(2^M)
Each Object O_i can be 0 or 1 indicating whether it has been collected or not resp.
g_R represents the goal of the robot which essentially is either to collect specific object out of M or not collect
"""
env_teamBoxWorld_config = Dict2(
    state_dim=2+2+1,  # > Grid Position (x, y),  > Binary indicator for whether M(=2) objects are collected and
    # > the goal of the robot g_R in [-1:M] i.e. which object to proceed towards and collect (obj ids start from 0).
    # If g_R=-1, then robot is asked to stay wherever it is.
    encode_dim=2*3,  # Trust (low, high) and Fatigue (low, med, high)
    action_dim=4+2,  # (Up, Down, Right, Left) for agent, M(=1) actions to direct the robot towards an object.
    # NOTE FOR STAY, NOT CONSIDERING ADDITIONAL ACTION (ASK)
    grid_length_x=5,
    grid_length_y=5
)


env_bigTeamBoxWorld_config = Dict2(
    state_dim=2+4+1,  # Grid Position (x, y),  Binary indicator for whether M(=4) objects are collected and
    # the goal of the robot g_R in [0, M] i.e. which object to proceed towards and collect.
    # If g_R=0, then robot is asked to stay wherever it is.
    encode_dim=2*3,  # Trust (low, high) and Fatigue (low, med, high)
    action_dim=4+4,  # (Up, Down, Right, Left) for agent, M(=4) actions to direct the robot towards an object or stay
    grid_length_x=10,
    grid_length_y=10
)


env_pickBoxWorld_config = Dict2(
    n_objs=3,
    state_dim=4+2*3,
    encode_dim=3,
    action_dim=4,  # Up, Down, Right, Left
    grid_length_x=5,
    grid_length_y=5
)
