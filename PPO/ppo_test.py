"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym

from env.custom_hopper import *
from stable_baselines3 import PPO
import matplotlib.pyplot as plt


def test_ppo_policy(
    test_env: str = 'CustomHopper-target-v0',
    model: str = "ppo_model.mdl",
    episodes: int = 1000,
    render: bool = False
):
	env = gym.make(test_env)

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	
	agent = PPO.load(model)
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
				env.render()

			test_reward += reward
			episode_length += 1
		
		test_rewards.append(test_reward)
		episode_lengths.append(episode_length)

	return test_rewards, episode_lengths
