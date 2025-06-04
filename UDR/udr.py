import random
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from PPO.ppo_utils import TrainTestCallback, create_agent, train
import numpy as np
import torch
import os

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
    gamma: float = 0.99,
    learning_rate=1e-3,
	delta: int = 1,
	print_std_deviation: bool = False,
	seed: int = 10
):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
 
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

  
	agent = create_agent(
		env=train_env,
		verbose=0,
		clip_range=clip_range,
		learning_rate=learning_rate,
		gamma=gamma
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

