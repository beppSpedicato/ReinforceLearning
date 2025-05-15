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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=1000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=200, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--baseline', default=0, type=int, help='baseline for reinforce update policy')
    parser.add_argument('--plot', default=True, type=bool, help='enable the creation of rewards plot')

    return parser.parse_args()

def plotRewards (train_rewards, baseline):
	plt.figure(figsize=(10, 5))
	plt.plot(train_rewards, label='Train reward per episode')
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	plt.title('Training Rewards')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(f"train_rewards_reinforce_b{baseline}.png")
	plt.close()

def plotTimesteps (train_rewards, baseline):
	plt.figure(figsize=(10, 5))
	plt.plot(train_rewards, label='timestepper episode')
	plt.xlabel('Episode')
	plt.ylabel('Timestep')
	plt.title('Training timestep')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(f"train_timestep_reinforce_b{baseline}.png")
	plt.close()

args = parse_args()


def main():
	random.seed(1)
	np.random.seed(1)
	torch.manual_seed(1)

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	#env.seed(32)

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = ReinforcePolicy(observation_space_dim, action_space_dim)
	agent = ReinforceAgent(policy, device=args.device, baseline=args.baseline)

    #
    # TASK 2: interleave data collection to policy updates
    #
	train_rewards = []
	timesteps = []
	max_train_reward = False

	for episode in range(args.n_episodes):
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
		train_rewards.append(train_reward)
		timesteps.append(timestep)

		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)

	plotTimesteps(timesteps, args.baseline)
	plotRewards(train_rewards, args.baseline)
	torch.save(agent.policy.state_dict(), f"model_reinforce_b{args.baseline}.mdl")  #riga modificata

	

if __name__ == '__main__':
	main()
