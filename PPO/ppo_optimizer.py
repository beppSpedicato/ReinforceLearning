import argparse
import torch
import gym
import random
import numpy as np
import optuna

from env.custom_hopper import *

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from PPO.ppo_utils import create_agent, train

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=1000, type=int, help='Number of training episodes')
    parser.add_argument('--mean-timestep', default=100, type=int, help='Mean number of timestep per episode')
    parser.add_argument('--n-trials', default=50, type=int, help='Number of optimization trials')
    parser.add_argument('--n-eval-episodes', default=1000, type=int, help='Number of eval episodes')

    return parser.parse_args()

args = parse_args()

class TrainOptimizeCallback(BaseCallback):
    def __init__(
        self,
        model: PPO,
        output_folder: str,
        test_env_path: str = None,
        test_every: int = None,
        max_episode: int = None,
        verbose: int = 0
    ):
        super().__init__(verbose)
        if test_env_path is not None:
            self.test_env = gym.make(test_env_path)
        else:
            self.test_env = None
        self.output_folder = output_folder
        self.test_every = test_every
        self.max_episode = max_episode

        self.init_callback(model)


    def _init_callback(self) -> None:
        
        self.train_rewards = []
        self.train_episode_length = []

        self.current_episode_length = 0
        self.current_reward = 0
    

    def _on_step(self):
        """Action to be done at each step."""

        # Retrieve the data
        reward = self.locals['rewards'][0] # float
        done = self.locals['dones'][0] # bool

        # Store them
        self.current_reward += reward
        self.current_episode_length += 1

        if (done):
            self.train_rewards.append(self.current_reward)
            self.current_reward = 0
            self.train_episode_length.append(self.current_episode_length)
            self.current_episode_length = 0

        return self.max_episode is None or len(self.train_rewards) < self.max_episode
        
    def _on_training_end(self):
        pass


def optimize_call(clip_range, n_episodes, n_eval_episodes):
    env = "CustomHopper-source-v0"
    train_env = gym.make(env)
    
    agent = create_agent(clip_range=clip_range, verbose=0)
    train(agent, total_timestep=n_episodes) # todo: set appropriate number
    
    mean_reward, _ = evaluate_policy(agent, train_env, n_eval_episodes=n_eval_episodes)
    return mean_reward


def objective(trial):
    clip_range = trial.suggest_float("clip_range", 0.01, 0.3, log=True)
    total_reward = optimize_call(clip_range, args.n_episodes*args.mean_timestep, args.n_eval_episodes)

    return total_reward
	
def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials, n_jobs=4)

    print("Best clip range:", study.best_params["clip_range"])
    print("Best reward:", study.best_value)

if __name__ == '__main__':
    print(args)
    main()
