"""Train an RL agent on the OpenAI Gym Hopper environment using
	REINFORCE 
"""
import argparse

import torch
import gym

from env.custom_hopper import *
from reinforce.reinforce_agent import ReinforcePolicy, ReinforceAgent
import matplotlib.pyplot as plt
import random
import numpy as np
from utils.plot import plotAvgTxtFiles, plotTrainRewards

import time


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n-episodes', default=14000, type=int, help='Number of training episodes')
	parser.add_argument('--print-every', default=2000, type=int, help='Print info every <> episodes')
	parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
	parser.add_argument('--plot', default=True, type=bool, help='enable the creation of rewards plot')

	return parser.parse_args()

args = parse_args()


def main(baseline):
	random.seed(1)
	np.random.seed(1)
	torch.manual_seed(1)

	env = gym.make('CustomHopper-source-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	print('Dynamics env.sim.model.body_names:', env.sim.model.body_names)
	

	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = ReinforcePolicy(observation_space_dim, action_space_dim)
	agent = ReinforceAgent(policy, device=args.device, baseline=baseline)

	#
	# TASK 2: interleave data collection to policy updates
	#
	train_rewards = []
	timesteps = []
	time_consumings_per_episodes = []

	for episode in range(args.n_episodes):
		start = time.perf_counter()
		done = False
		train_reward = 0
		timestep = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over
			action, action_probabilities = agent.get_action(state)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done)

			train_reward += reward
			timestep = timestep + 1

		agent.update_policy()
  
		end = time.perf_counter()
		time_consuming = end - start
		time_consumings_per_episodes.append(time_consuming)
		train_rewards.append(train_reward)
		timesteps.append(timestep)

		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)
			print('Time consuming: ', time_consuming)
			 
	plotTrainRewards(train_rewards, f"reinforce_b{baseline}", 100)
	plotTrainRewards(timesteps, f"timestep_reinforce_b{baseline}", 100, create_txt=False)
	plotTrainRewards(time_consumings_per_episodes, f"time_consuming_reinforce_b{baseline}", 100, create_txt=False)

	print("Total training time: ", sum(time_consumings_per_episodes))
	torch.save(agent.policy.state_dict(), f"model_reinforce_b{baseline}.mdl")  #riga modificata

	

if __name__ == '__main__':
	print("baseline 0")
	main(0)
	print("baseline 5")
	main(5)
	print("baseline 10")
	main(10)
	print("baseline 20")
	main(20)
	print("baseline 50")
	main(50)
	print("baseline 85")
	main(85)
	files = [
		'train_rewards_means_b0.txt', 
		'train_rewards_means_b5.txt', 
		'train_rewards_means_b10.txt', 
		'train_rewards_means_b20.txt', 
		'train_rewards_means_b50.txt', 
		'train_rewards_means_b85.txt'
	]
	plotAvgTxtFiles(files,  "reinforce")


	""" 
		TODO: time consumption for episode and fot all the training
	"""
