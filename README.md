# GO-DICE: Goal-Conditioned Option-Aware Offline Imitation Learning via Stationary Distribution Correction Estimation

This repository contains the official implementation of the paper titled "GO-DICE: Goal-Conditioned Option-Aware Offline Imitation Learning via Stationary Distribution Correction Estimation" accepted at AAAI-24. 

## Abstract
Offline imitation learning (IL) refers to learning expert behavior solely from demonstrations, without any additional interaction with the environment. Despite significant advances in offline IL, existing techniques find it challenging to learn policies for long-horizon tasks and require significant re-training when task specifications change. Towards addressing these limitations, we present GO-DICE an offline IL technique for goal-conditioned long-horizon sequential tasks. GO-DICE discerns a hierarchy of sub-tasks from demonstrations and uses these to learn separate policies for sub-task transitions and action execution, respectively; this hierarchical policy learning facilitates long-horizon reasoning. Inspired by the expansive DICE-family of techniques, policy learning at both the levels transpires within the space of stationary distributions. Further, both policies are learnt with goal conditioning to minimize need for retraining when task goals change. Experimental results substantiate that GO-DICE outperforms recent baselines, as evidenced by a marked improvement in the completion rate of increasingly challenging pick-and-place Mujoco robotic tasks. GO-DICE is also capable of leveraging imperfect demonstration and partial task segmentation when available, both of which boost task performance relative to learning from expert demonstrations alone.

## Paper Link
The extended version of the paper associated with this code is available [here](https://arxiv.org/abs/2312.10802).

## Authors
- Abhinav Jain, Rice University
- Prof. Vaibhav Unhelkar, Rice University

## Acknowledgments
This research was supported in part by NSF award #2205454, the Army Research Office through Cooperative Agreement Number W911NF-20-2-0214, and Rice University funds.

# Requirements
Install the dependencies using `pip install -r requirements.txt`.

Note that this code requires `mujoco-py==2.1.2.14`, `gym==0.21.0` and `tensorflow==2.11.0`


# Data
The PickAndPlace data (Fetch Tasks) is located in `pnp_data/OpenAIPickandPlace` . We have 4 datasets, each named based on the environment and how a given expert completes the task. For each I have generated 100 expert demonstrations with following horizon lengths (set to comfortably complete the task):-
- Single Object PnP
- Two Object PnP
- Three Object PnP
- Stack Three Object

Each dataset is in the form of a dictionary saved as a pickle file. The dictionary has following `keys = states, actions, goals, achieved_goals, successes`. Note that goals are not part of the observed states and may need to be concatenated for algorithms requiring goal-conditioning. Also, the terminal state is included in the `states` because of which its size is equal to `horizon_length+1`.

## Resources

- [Gymnasium](https://gymnasium.farama.org/content/basic_usage): Gymnasium (maintained by Farama) is the now-supported version of OpenAI Gym.
- [Gymnasium-Robotics](https://robotics.farama.org): Robotics environments for Gymnasium provided separately by Farama.
- [Mujoco](https://mujoco.readthedocs.io/en/stable/python.html): MuJoCo is a physics engine for detailed, efficient rigid body simulations with contacts. Starting with version 2.1.2, MuJoCo comes with native Python bindings. Earlier they needed to be installed separately via `mujoco-py`.

# Environment
The PnP Environments and Expert Policies are located in `domains/PnP.py` and `domains/PnPExpert.py`.
How to access these environments:-

```python
from domains.PnP import MyPnPEnvWrapper
# Single Object PnP
one_obj_env = MyPnPEnvWrapper(full_space_as_goal=False, num_objs=1, stacking=False, target_in_the_air=True)
# Two Object PnP
two_obj_env = MyPnPEnvWrapper(full_space_as_goal=False, num_objs=2, stacking=False, target_in_the_air=False)
# Three Object PnP
three_obj_env = MyPnPEnvWrapper(full_space_as_goal=False, num_objs=3, stacking=False, target_in_the_air=False)
# Stack Three Object
stack_three_obj_env = MyPnPEnvWrapper(full_space_as_goal=False, num_objs=3, stacking=True, target_in_the_air=False)
```


Stick to this environment initialisation for now! You can look at the code of `MyPnPEnvWrapper` to see what the methods `reset()` and `step()` returns since I have used a wrapper around off-the-shelf Gym environments.

## How to run?

### [main.py](main.py)
Following functions constitute the main.py file:-
- `run()`: To train the policy using specified algorithm `algo`
- `evaluate(algo: str, path_to_models: str, num_eval_demos: int)`: To evaluate the policies learned using `algo` saved at `path_to_models` using `num_eval_demos` number of demos. 
- `verify(algo: str, path_to_models: str, num_test_demos: int)`: To visualise the policies learned using `algo` saved at `path_to_models` using `num_test_demos` number of demos. 
- `record(algo: str, path_to_models: str, num_record_demos: int)`: To record the policies-in-action (as multiple .png) learned using `algo` saved at `path_to_models` using `num_record_demos` number of demos.

### [save_env_img.py](save_env_img.py)
Simply saves the (initial state) environment as an image. Useful for debugging.

### [zero_shot_option_transfer.py](zero_shot_option_transfer.py)
Contains the code for Zero-Shot Option Transfer (one-object low-level policy juxtaposed with multi-object high-level policy).

## Citation
If you find this work useful, please consider citing:
```
@article{jain2023go,
  title={GO-DICE: Goal-Conditioned Option-Aware Offline Imitation Learning via Stationary Distribution Correction Estimation},
  author={Jain, Abhinav and Unhelkar, Vaibhav},
  journal={arXiv preprint arXiv:2312.10802},
  year={2023}
}
```

## Contact
If you have any questions or feedback regarding the code, feel free to contact the authors:
- [aj70@rice.edu](mailto:aj70@rice.edu)


