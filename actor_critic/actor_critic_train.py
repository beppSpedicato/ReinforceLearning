"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE 
"""
import argparse

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
    parser.add_argument('--baseline', default=0, type=int, help='baseline for reinforce update policy')
    parser.add_argument('--plot', default=True, type=bool, help='enable the creation of rewards plot')

    return parser.parse_args()

args = parse_args()


def main():
	random.seed(1)
	np.random.seed(1)
	torch.manual_seed(1)
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
	agent = ActorCriticAgent(policy, device=args.device, baseline=args.baseline)

	train_rewards = []
	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over
			action, action_probabilities = agent.get_action(state)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done)

			train_reward += reward
		
		train_rewards.append(train_reward)
		agent.update_policy()
		
		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)

	plotTrainRewards(train_rewards, "actorcritic", 100)
	plotAvgTxtFiles(["train_rewards_means_actorcritic.txt"], "actorcritic_avg")
	torch.save(agent.policy.state_dict(), f"model_actorcritic_b{args.baseline}.mdl")  #riga modificata

if __name__ == '__main__':
	main()
