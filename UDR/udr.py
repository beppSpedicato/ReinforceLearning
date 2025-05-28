import random
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from PPO.ppo_utils import TrainTestCallback, create_agent, train
import numpy as np
import torch
import os
import gym
from stable_baselines3.common.evaluation import evaluate_policy
import optuna

class UDRCallback(BaseCallback):
	def __init__(
		self,
		model: PPO,
		verbose: int = 0,
		delta: float = 1
	):
		self.delta = delta
		super().__init__(verbose)
		self.init_callback(model)

	def _on_step(self):
		done = self.locals['dones'][0] 
		if done:
			self.training_env.envs[0].udr_sample_parameters(delta=self.delta, log=(self.verbose==1))

		return True

def train_test_ppo_with_udr (
	output_folder: str = "./udr_output",
	train_env: str = "CustomHopper-source-v0",
	test_env: str = "CustomHopper-target-v0",
	clip_range: float = -1,
	episodes: int = 8000,
	timesteps: int = 300,
	delta: int = 1,
	print_std_deviation: bool = False
):
	random.seed(10)
	np.random.seed(10)
	torch.manual_seed(10)
 
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

  
	agent = create_agent(
		env=train_env,
		verbose=0,
		clip_range=clip_range
	)

	callbacks = [
		TrainTestCallback(
			agent,
			output_folder,
			test_env=test_env,
			print_test_std=print_std_deviation
		),
		UDRCallback(agent, delta=delta)
	]
  
	train(agent, callbacks=callbacks, total_timestep=episodes*timesteps, model_output_path=None)


def optimize_call(clip_range, n_episodes, n_eval_episodes, delta, env: str = "CustomHopper-source-v0"):
    train_env = gym.make(env)
    
    agent = create_agent(clip_range=clip_range, verbose=0)
    callbacks = [
        UDRCallback(agent, delta=delta)
    ]
    train(agent, total_timestep=n_episodes, callbacks=callbacks)
    
    mean_reward, _ = evaluate_policy(agent, train_env, n_eval_episodes=n_eval_episodes)
    return mean_reward


def objective(trial, delta):
    clip_range = trial.suggest_float("clip_range", 0.01, 0.3, log=True)
    total_reward = optimize_call(clip_range, 5000*300, 1000, delta)

    return total_reward

	
def optimize_udr_policy(delta):
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda train: objective(train, delta), 
        n_trials=10,
        n_jobs=4
    )

    return study.best_params["clip_range"]