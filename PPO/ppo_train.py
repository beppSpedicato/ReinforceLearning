"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
import argparse
import torch
import random
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import json
from utils.plot import plotTrainRewards
from utils.plot import plotAvgTxtFiles
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=14000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()
class TrainTestCallback(BaseCallback):

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
        
        self._rolling_number = 0

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
        print(sum(self.train_episode_length) / len(self.train_episode_length))
        plotTrainRewards(self.train_rewards, "ppo", 10)
        plotAvgTxtFiles(["train_rewards_means_ppo.txt"], "PPO_AVG_100_episodes")
        
def main():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    mean_timestep_for_episodes = 300


    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": args.n_episodes*mean_timestep_for_episodes,
        "env_name": "CustomHopper-source-v0",
    }
    
    run = wandb.init(
        project="sb3",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    
    train_env = gym.make(config['env_name'])

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    output_folder = f"./PPO_output"
    agent = PPO(config['policy_type'], train_env, verbose=1, tensorboard_log=f"PPO_output/PPO_runs/{run.id}")
    
    train_test_callback = TrainTestCallback(model=agent, output_folder=output_folder, verbose=0)
    wandb_callback = WandbCallback(gradient_save_freq=100, model_save_path=f"PPO_output/PPO_models/{run.id}", verbose=2)
    callbacks = CallbackList([train_test_callback, wandb_callback])

    agent.learn(total_timesteps=config['total_timesteps'], callback=callbacks)
    agent.save("PPO_output/ppo_model.mdl")


if __name__ == '__main__':
    main()



""" 
TODO:
step: 
1. try to tune clippings using optuna and compare different results (negative and positive advantage)
2. vedere parte di test e rispondere alle domande
3. creare introduzione per spiegazione modello (con PPO-CLip e PPO-Penalty)
4. (optional): create WanDB instance

"""