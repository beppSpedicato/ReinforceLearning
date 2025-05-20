"""Test a random policy on the OpenAI Gym Hopper environment.

    
    TASK 1: Play around with this code to get familiar with the
            Hopper environment.

            For example:
                - What is the state space in the Hopper environment? Is it discrete or continuous?
                - What is the action space in the Hopper environment? Is it discrete or continuous?
                - What is the mass value of each link of the Hopper environment, in the source and target variants respectively?
                - what happens if you don't reset the environment even after the episode is over?
                - When exactly is the episode over?
                - What is an action here?
"""
import pdb

import gym

from env.custom_hopper import *

def main():

	env = gym.make('CustomHopper-v0')
	# env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('State space:', env.observation_space)  # state-space
	print('Action space:', env.action_space)      # action-space
	print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper

	n_episodes = 10  # Ridotto per debug
	render = True

	for episode in range(n_episodes):
		done = False
		state = env.reset()
		step_count = 0

		while not done:
			action = env.action_space.sample()
			state, reward, done, info = env.step(action)
			
			step_count += 1

			print(f"Step {step_count} | Done: {done} | Reward: {reward:.2f}")

			if render:
				env.render()

			# Verifica causa di fine episodio
			if done:
				print(f"\n🔴 Episodio terminato al passo {step_count}")
				if "TimeLimit.truncated" in info and info["TimeLimit.truncated"]:
					print("⏱️  Causa: Raggiunto limite massimo di passi (probabilmente 1000).")
				else:
					print("💥 Causa: Il robot è caduto (altezza sotto soglia o instabilità).\n")

if __name__ == '__main__':
	main()