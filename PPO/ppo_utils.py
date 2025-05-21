import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
import argparse
import torch
import random
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import json
from utils.plot import plotTrainRewards
from utils.plot import plotAvgTxtFiles
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList

"""
Create and initialize a PPO agent for a given environment.
Parameters
----------
policy_type : str, optional
    The type of policy architecture to use (e.g., "MlpPolicy", "CnnPolicy"). Default is "MlpPolicy".
env : str, optional
    The Gym environment ID to instantiate (must be registered). Default is "CustomHopper-source-v0".
tensorboard_log : str or None, optional
    Path to a directory for TensorBoard logging. If None, logging is disabled. Default is None.
logEnv : bool, optional
    If True, prints environment details including observation space, action space, and dynamics parameters. Default is False.

Returns
-------
PPO
    A PPO agent initialized with the specified environment and policy.
"""
def create_agent(
    policy_type: str="MlpPolicy",
    env: str="CustomHopper-source-v0",
    tensorboard_log: str=None,
    logEnv: bool=False,
    clip_range: float=0.5 ,
    verbose: int=1
):
    train_env = gym.make(env)
    if logEnv:
        print('State space:', train_env.observation_space)  # state-space
        print('Action space:', train_env.action_space)  # action-space
        print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    return PPO(policy_type, train_env, verbose=verbose, tensorboard_log=tensorboard_log, clip_range=clip_range)

"""
Create and initialize a PPO agent for a given environment.
Parameters
----------
policy_type : str, optional
    The type of policy architecture to use (e.g., "MlpPolicy", "CnnPolicy"). Default is "MlpPolicy".
env : str, optional
    The Gym environment ID to instantiate (must be registered). Default is "CustomHopper-source-v0".
tensorboard_log : str or None, optional
    Path to a directory for TensorBoard logging. If None, logging is disabled. Default is None.
logEnv : bool, optional
    If True, prints environment details including observation space, action space, and dynamics parameters. Default is False.

Returns
-------
PPO
    A PPO agent initialized with the specified environment and policy.
"""
def train(
    agent: PPO,
    callbacks: list=[],
    total_timestep: int=1000000,
    model_output_path: str=None
):
    callbacks = CallbackList(callbacks)
    learn = agent.learn(total_timesteps=total_timestep, callback=callbacks)

    if model_output_path != None:
        agent.save(model_output_path)

    return learn.get_parameters()


class TrainTestCallback(BaseCallback):
    """
    A custom callback for monitoring training rewards and episode lengths during PPO training.
    
    This callback tracks the cumulative rewards and episode lengths for each training episode.
    At the end of training, it computes and logs average episode statistics, and generates plots
    using external utilities.

    Parameters
    ----------
    model : PPO
        The PPO agent being trained.
    output_folder : str
        Directory path to store plots and output statistics.
    max_episode : int, optional
        Maximum number of episodes to track. If None, tracking continues until training ends. Default is None.
    verbose : int, optional
        Verbosity level. Default is 0.
    """
    def __init__(
        self,
        model: PPO,
        output_folder: str,
        max_episode: int = None,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.output_folder = output_folder
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
        print(sum(self.train_episode_length) / len(self.train_episode_length))
        plotTrainRewards(self.train_rewards, "ppo", 10, outputFolder=self.output_folder)
        plotAvgTxtFiles([f"{self.output_folder}/train_means_ppo.txt"], "PPO_AVG_100_episodes")