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
from domains.PnP import MyPnPEnvWrapperForGoalGAIL

`one_obj_env = MyPnPEnvWrapperForGoalGAIL(full_space_as_goal=False, two_obj=False,
                                         stacking=False, target_in_the_air=True)`

`two_obj_env = MyPnPEnvWrapperForGoalGAIL(full_space_as_goal=False, two_obj=True,
                                         stacking=False, target_in_the_air=False)`
                                         
Stick to this environment initialisation for now! You can look at the code of `MyPnPEnvWrapperForGoalGAIL` to see what the methods `reset()` and `step()` returns since I have used a wrapper around off-the-shelf Gym environments.
