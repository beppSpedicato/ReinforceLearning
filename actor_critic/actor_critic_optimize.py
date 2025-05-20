""" Optimize alpha1 and alpha2 using optuna """

import argparse
import torch
import gym
import random
import numpy as np
import optuna

from env.custom_hopper import *
from actor_critic.actor_critic_agent import ActorCriticAgent, ActorCriticPolicy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=14000, type=int, help='Number of training episodes')
    parser.add_argument('--start-window', default=5000, type=int, help='Start window for mean calculation')
    parser.add_argument('--end-window', default=9000, type=int, help='End window for mean calculation')
    parser.add_argument('--n-trials', default=9000, type=int, help='Number of optimization trials')

    return parser.parse_args()

args = parse_args()

def train(alpha1, alpha2, n_episodes):
    # seed
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
	
    env = gym.make('CustomHopper-source-v0')
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = ActorCriticPolicy(observation_space_dim, action_space_dim)
    agent = ActorCriticAgent(policy, device='cpu', alpha1=alpha1, alpha2=alpha2)

    train_rewards = []
    for episode in range(n_episodes):
        done = False
        train_reward = 0
        state = env.reset()

        while not done:
            action, action_probabilities = agent.get_action(state)
            previous_state = state

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            agent.store_outcome(previous_state, state, action_probabilities, reward, done)

            train_reward += reward
		
        train_rewards.append(train_reward)
        agent.update_policy()

    return train_rewards


def optimize_window_mean(alpha1, alpha2, n_episodes, start=0, end=-1):
    if (end == -1):
        end = len(n_episodes)

    train_rewards = train(alpha1, alpha2, n_episodes)
    window = train_rewards[start:end]
    return sum(window) / len(window)
    


def objective(trial):
    alpha1 = trial.suggest_float("alpha1", 1e-3, 1.0, log=True)
    alpha2 = trial.suggest_float("alpha2", 1e-3, 1.0, log=True)
    
    norm = alpha1 + alpha2
    alpha1 /= norm
    alpha2 /= norm

    total_reward = optimize_window_mean(alpha1, alpha2, args.n_episodes, start=args.start_window, end=args.end_window)

    return total_reward
	

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    print("Best alpha1:", study.best_params["alpha1"])
    print("Best alpha2:", study.best_params["alpha2"])
    print("Best reward:", study.best_value)

if __name__ == '__main__':
    print(args)
    main()
