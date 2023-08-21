# Installation Instructions

## Compatibility

- Compatible with TensorFlow 2.x
- Compatible with arm64 /x86_64 architecture

## TensorFlow Installation on M1/M2 Macs with native GPU support

### Step 1: Install conda

Download the conda installer from [here](https://www.anaconda.com/download). Make sure to download the arch. specific conda can from [here](https://www.anaconda.com/download).

### Step 2: Create a conda environment

```bash
conda create -n tf-gpu python=3.8
```

### Step 3: Activate the conda environment

```bash
conda activate tf-gpu
```

### Step 4: Install TensorFlow

```bash
pip install tensorflow && pip install tensorflow-metal
```

Note: tensorflow-metal is required for GPU support on M1/M2 Macs. Details [here](https://developer.apple.com/metal/tensorflow-plugin/).

### Step 5: Verify the installation

```bash
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

## Install Required Packages

```bash
pip install -r requirements.txt
```

Note: The `latest` branch uses `gymansium` and `mujoco` for the environments. The `dev` branch uses `gym` and `mujoco-py` for the environments.

## Resources

- [Gymnasium](https://gymnasium.farama.org/content/basic_usage): Gymnasium (maintained by Farama) is the now-supported version of OpenAI Gym.
- [Gymnasium-Robotics](https://robotics.farama.org): Robotics environments for Gymnasium provided separately by Farama.
- [Mujoco](https://mujoco.readthedocs.io/en/stable/python.html): MuJoCo is a physics engine for detailed, efficient rigid body simulations with contacts. Starting with version 2.1.2, MuJoCo comes with native Python bindings. Earlier they needed to be installed separately via `mujoco-py`.


# Data
The PickAndPlace data (Fetch Tasks) is located in `pnp_data/OpenAIPickandPlace` . We have 4 datasets, each named based on the environment and how a given expert completes the task. For each I have generated 100 expert demonstrations with following horizon lengths (set to comfortably complete the task):-
- Single Object PnP
- Two Object PnP
- Three Object PnP
- Stack Three Object

Each dataset is in the form of a dictionary saved as a pickle file. The dictionary has following `keys = states, actions, goals, achieved_goals, successes`. Note that goals are not part of the observed states and may need to be concatenated for algorithms requiring goal-conditioning. Also, the terminal state is included in the `states` because of which its size is equal to `horizon_length+1`.

# Environment
The PnP Environments and Expert Policies are located in `domains/PnP.py` and `domains/PnPExpert.py`.
How to access these environments:-
from domains.PnP import MyPnPEnvWrapperForGoalGAIL

`one_obj_env = MyPnPEnvWrapperForGoalGAIL(full_space_as_goal=False, two_obj=False,
                                         stacking=False, target_in_the_air=True)`

`two_obj_env = MyPnPEnvWrapperForGoalGAIL(full_space_as_goal=False, two_obj=True,
                                         stacking=False, target_in_the_air=False)`
                                         
Stick to this environment initialisation for now! You can look at the code of `MyPnPEnvWrapperForGoalGAIL` to see what the methods `reset()` and `step()` returns since I have used a wrapper around off-the-shelf Gym environments.
