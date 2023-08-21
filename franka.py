import gymnasium as gym
env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave'])
dataset = env.get_dataset()