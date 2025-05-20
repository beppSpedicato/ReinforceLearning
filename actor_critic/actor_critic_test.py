"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym

from env.custom_hopper import *
from actor_critic.actor_critic_agent import ActorCriticAgent, ActorCriticPolicy

import matplotlib.pyplot as plt

def plotRewards (train_rewards):
	plt.figure(figsize=(10, 5))
	plt.plot(train_rewards, label=f'TEST reward per episode for model actor critic')
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	plt.title('Test Rewards')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(f"test_rewards_for_actor_critic.png")
	plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=1000, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = ActorCriticPolicy(observation_space_dim, action_space_dim)
	policy.load_state_dict(torch.load(args.model), strict=True)

	agent = ActorCriticAgent(policy, device=args.device)

	test_rewards = []

	for episode in range(args.episodes):
		done = False
		test_reward = 0
		state = env.reset()

		while not done:

			action, _ = agent.get_action(state, evaluation=True)

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			if args.render:
				env.render()

			test_reward += reward
		
		test_rewards.append(test_reward)

		print(f"Episode: {episode} | Return: {test_reward}")
	
	plotRewards(test_rewards)
	

if __name__ == '__main__':
	main()