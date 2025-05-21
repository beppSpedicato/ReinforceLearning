"""Train an RL agent on the OpenAI Gym Hopper environment using
	REINFORCE 
"""
import argparse
import os
import time

import torch
import gym
import random
import numpy as np

from env.custom_hopper import *
from actor_critic.actor_critic_agent import ActorCriticAgent, ActorCriticPolicy
import matplotlib.pyplot as plt
from utils.plot import plotAvgTxtFiles
from utils.plot import plotTrainRewards

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n-episodes', default=14000, type=int, help='Number of training episodes')
	parser.add_argument('--print-every', default=2000, type=int, help='Print info every <> episodes')
	parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
	parser.add_argument('--plot', default=True, type=bool, help='enable the creation of rewards plot')
	parser.add_argument('--alpha1', default=0.013396063146884157, type=float, help='weight for actor loss')
	parser.add_argument('--alpha2', default=0.009748205129440874, type=float, help='weight for critic loss')

	return parser.parse_args()

args = parse_args()
print(args)

def main():
	random.seed(10)
	np.random.seed(10)
	torch.manual_seed(10)
	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = ActorCriticPolicy(observation_space_dim, action_space_dim)
	agent = ActorCriticAgent(policy, device=args.device, alpha1=args.alpha1, alpha2=args.alpha2)

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
			timestep += 1
   
		end = time.perf_counter()
		time_consuming = end - start
		time_consumings_per_episodes.append(time_consuming)
		train_rewards.append(train_reward)
		timesteps.append(timestep)
		agent.update_policy()
		
		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)

	outputFolder = f"./trained-models/actorcritic_a1:{args.alpha1}_a2:{args.alpha2}"
	if not os.path.exists(outputFolder):
		os.mkdir(outputFolder)
	print("Total training time: ", sum(time_consumings_per_episodes))


	plotTrainRewards(train_rewards, f"rewards_actorcritic", 100, y_label="Rewards per episode", outputFolder=outputFolder)
	plotTrainRewards(
	 	timesteps, 
	 	f"timestep_actorcritic", 
	  	100, 
	   	chart_title="Timesteps per episode", 
		create_txt=False,
		outputFolder=outputFolder, 
		label="Timestep per episodes",
		y_label='Number of timesteps'
	)
	plotTrainRewards(
		time_consumings_per_episodes, 
	 	f"time_consuming_actorcritic",
	  	100, 
	   	create_txt=False,
		chart_title="Time consuming in seconds",
		y_label="Time (seconds)",
		outputFolder=outputFolder,
		label="Time consuming per episode"
	) 
	torch.save(agent.policy.state_dict(), f"{outputFolder}/model_actorcritic.mdl")

if __name__ == '__main__':
	main()
