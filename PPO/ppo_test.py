"""Test an RL agent on the OpenAI Gym Hopper environment"""
import os
import random
import time

from PPO.ppo_utils import TrainTestCallback, create_agent, train
import torch
import gym

from env.custom_hopper import *
from stable_baselines3 import PPO
import matplotlib.pyplot as plt


def train_and_test_policy (
	train_env: str = 'CustomHopper-target-v0',
	test_env: str = 'CustomHopper-target-v0',
	episodes: int = 14000,
	timesteps: int = 300,
	output_folder: str = './PPO_output/target-target/',
	clip_range: float = -1,
	learning_rate: float = 1e-3,
	gamma: float = 0.99,
	print_std_deviation: bool = False,
	seed: int = 10
):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
 
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
  
	agent = create_agent(
		env=train_env,
		verbose=0,
		clip_range=clip_range,
		learning_rate=learning_rate,
		gamma=gamma,
	)

	callbacks = [
		TrainTestCallback(
			agent,
			output_folder,
			test_env=test_env,
			print_test_std=print_std_deviation
		)
	]
  
	train(agent, callbacks=callbacks, total_timestep=episodes*timesteps, model_output_path=None)

	


def test_ppo_policy(
	test_env: str = 'CustomHopper-target-v0',
	model: str = "ppo_model.mdl",
	episodes: int = 1000,
	render: bool = False,
	seed: int = 10
):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	env = gym.make(test_env)

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	
	agent = PPO.load(model, env)
	state = env.reset()
	test_rewards = []
	episode_lengths = [] 

	for episode in range(episodes):
		done = False
		test_reward = 0
		episode_length = 0
		state = env.reset()

		while not done:

			action, _states = agent.predict(state, deterministic=True)

			state, reward, done, info = env.step(action)



			if render:
				time.sleep(0.02)
				env.render()

			test_reward += reward
			episode_length += 1
		
		test_rewards.append(test_reward)
		episode_lengths.append(episode_length)

	return test_rewards, episode_lengths
