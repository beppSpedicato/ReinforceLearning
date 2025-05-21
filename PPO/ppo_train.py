"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
from PPO.ppo_utils import TrainTestCallback, create_agent, train
import gym
from env.custom_hopper import *
import argparse
import torch
import random
import json
from utils.plot import plotTrainRewards
from utils.plot import plotAvgTxtFiles
import wandb
from wandb.integration.sb3 import WandbCallback

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=14000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--use-wandb', default=False, type=bool, help='use or not wandb')
    parser.add_argument('--mean-timestep', default=300, type=int, help="mean number of timestep per episode")
    parser.add_argument('--output-folder', default="./PPO_output", type=str, help="Output path for models and charts")

    return parser.parse_args()

args = parse_args()
    
    
def main():
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)

    config = {
        "policy_type": "MlpPolicy",
        # "total_timesteps": args.n_episodes*args.mean_timestep,
        "total_timesteps": 100,
        "env_name": "CustomHopper-source-v0",
    }

    # define agent
    tensorboard_log = None
    if args.use_wandb:
        run = wandb.init(
            project="sb3",
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        tensorboard_log=f"{args.output_folder}/PPO_runs/{run.id}"
        
    agent = create_agent(
        policy_type=config['policy_type'],
        env=config['env_name'],
        tensorboard_log=tensorboard_log
    )

    # define callback instances
    callbacks = [TrainTestCallback(model=agent, output_folder=args.output_folder, verbose=0)]
    if args.use_wandb:
        wandb_callback = WandbCallback(gradient_save_freq=100, model_save_path=f"{args.output_folder}/PPO_models/{run.id}", verbose=2)
        callbacks.append(wandb_callback)
        
    learn = train(agent, callbacks=callbacks, total_timestep=config['total_timesteps'], model_output_path=f"{args.output_folder}/ppo_model.mdl")

    print(learn.get_parameters())


if __name__ == '__main__':
    main()
