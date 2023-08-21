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

