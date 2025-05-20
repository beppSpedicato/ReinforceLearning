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

def plot_multiple_baselines(file_list):
    plt.figure(figsize=(12, 6))

    for filename in file_list:
        with open(filename, 'r') as f:
            lines = f.readlines()
            # La prima riga contiene la baseline
            baseline = lines[0].strip()
            # Le righe successive contengono le medie (convertite a float)
            means = [float(line.strip()) for line in lines[1:]]

        # Asse x: indice delle finestre di 500 episodi (0, 1, 2, ...)
        x = [i for i in range(len(means))]

        # Plot
        plt.plot(x, means, marker='o', label=baseline)

    plt.xlabel('Window index (ogni 500 episodi)')
    plt.ylabel('Media reward')
    plt.title('Confronto medie reward per baseline')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"train_rewards_reinforce.png")
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=14000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=2000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--baseline', default=0, type=int, help='baseline for reinforce update policy')
    parser.add_argument('--plot', default=True, type=bool, help='enable the creation of rewards plot')

    return parser.parse_args()

def plotRewards(train_rewards, baseline):
    plt.figure(figsize=(10, 5))
    plt.plot(train_rewards, label='Train reward per episode')

    # Calcolare la media ogni 500 episodi
    window_size = 500
    means = []
    positions = []
    for i in range(0, len(train_rewards), window_size):
        window = train_rewards[i:i+window_size]
        mean_value = np.mean(window)
        means.append(mean_value)
        positions.append(i + window_size//2)  # centro della finestra

    # Tracciare la linea delle medie
    plt.plot(positions, means, color='red', label=f'Media ogni {window_size} episodi', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"train_rewards_reinforce_b{baseline}.png")
    plt.close()

	# Creazione file testo con le medie
    filename = f"train_rewards_means_b{baseline}.txt"
    with open(filename, 'w') as f:
        f.write(f"baseline{baseline}\n")
        for mean_value in means:
            f.write(f"{mean_value}\n")

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


def main(baseline):
	random.seed(1)
	np.random.seed(1)
	torch.manual_seed(1)

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	#env.seed(32)

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

	# plotTimesteps(timesteps, args.baseline)
	plotRewards(train_rewards, baseline)
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
    plot_multiple_baselines(files)


    """ 
        TODO: time consumption for episode and fot all the training
        
    """
