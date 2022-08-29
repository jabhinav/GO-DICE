from domains.PnP import MyPnPEnvWrapperForGoalGAIL
from utils.env import save_env_img

exp_env = MyPnPEnvWrapperForGoalGAIL(full_space_as_goal=False, two_obj=False,
                                     stacking=False, target_in_the_air=False)

old_obs = exp_env._env.reset()
save_env_img(exp_env, path_to_save='./old_env.png')
old_state = exp_env._env.sim.get_state()
old_goal = exp_env._env.goal

new_obs = exp_env._env.reset()
save_env_img(exp_env, path_to_save='./new_env.png')
new_state = exp_env._env.sim.get_state()

# Reset
exp_env._env.sim.set_state(old_state)
exp_env._env.sim.forward()
exp_env._env.goal = old_goal
reset_obs = exp_env._env._get_obs()
save_env_img(exp_env, path_to_save='./old_env_reset.png')

print("Old Obs: {}\nNew Obs: {}\nReset Obs: {}".format(old_obs, new_obs, reset_obs))
print("Reset==New: {}, Reset==Old: {}".format(reset_obs['observation'] == new_obs['observation'], reset_obs['observation'] == old_obs['observation']))
