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

def optimize_call(clip_range, n_episodes, n_eval_episodes, env: str = "CustomHopper-source-v0"):
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
