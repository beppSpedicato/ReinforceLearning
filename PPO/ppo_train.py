"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

	Read the stable-baselines3 documentation and implement a training
	pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import os
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
	parser.add_argument('--print-every', default=2000, type=int, help='Print info every <> episodes')
	parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
	parser.add_argument('--use-wandb', default=True, type=bool, help='use or not wandb')
	parser.add_argument('--mean-timestep', default=300, type=int, help="mean number of timestep per episode")
	parser.add_argument('--output-folder', default="./PPO_output", type=str, help="Output path for models and charts")
	parser.add_argument('--clip-range', default=-1, type=float, help="Clip range (if negative it is not setted)")
	parser.add_argument('--seed', default=10, type=int, help="select the starting seed")
	parser.add_argument('--learning_rate', default=1e-3, type=int, help="learning rate")
	parser.add_argument('--gamma', default=0.99, type=int, help="discount factor")
	
	return parser.parse_args()

args = parse_args()
	
	
def main():
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	config = {
		"policy_type": "MlpPolicy",
		"total_timesteps": args.n_episodes*args.mean_timestep,
		"env_name": "CustomHopper-source-v0",
	}

	if args.clip_range > 0:
		output_folder = f"{args.output_folder}/clip-range:{args.clip_range}"
		project = f"PPO-{config['env_name']}-clip-range-{args.clip_range}"
	else:
		output_folder = f"{args.output_folder}/no-clip-range"
		project = f"PPO-{config['env_name']}-no-clip-range"

	if not os.path.exists(output_folder):
		os.mkdir(output_folder)

	# define agent
	tensorboard_log = None
	if args.use_wandb:
		run = wandb.init(
			project=project,
			config=config,
			sync_tensorboard=True,
			monitor_gym=True,
			save_code=True,
		)
		tensorboard_log=f"{args.output_folder}/PPO_runs/{run.id}"
		
	agent = create_agent(
		policy_type=config['policy_type'],
		env=config['env_name'],
		tensorboard_log=tensorboard_log,
		clip_range=args.clip_range
		learning_rate=args.learning_rate,
		gamma=args.gamma,
	)

	# define callback instances
	callbacks = [TrainTestCallback(model=agent, output_folder=output_folder, verbose=0, test_window=None)]
	if args.use_wandb:
		wandb_callback = WandbCallback(gradient_save_freq=100, model_save_path=f"{output_folder}/PPO_models/{run.id}", verbose=2)
		callbacks.append(wandb_callback)
		
	train(agent, callbacks=callbacks, total_timestep=config['total_timesteps'], model_output_path=f"{output_folder}/ppo_model.mdl")


if __name__ == '__main__':
	main()
