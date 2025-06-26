import random
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from PPO.ppo_utils import TrainTestCallback, create_agent, train
import numpy as np
import torch
import os

class UDRCallback(BaseCallback):
	"""
	Uniform Domain Randomization (UDR) Callback
	
	This callback implements basic uniform domain randomization for reinforcement
	learning environments.
	
	The callback triggers domain randomization at the end of each episode,
	ensuring that the agent experiences diverse environment conditions during training.
	This helps improve generalization to unseen domains but doesn't adapt based on
	the agent's performance.
	"""
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
			# Apply Uniform distribution to environment masses
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
	seed: int = 10,
	model_output_path: str = None
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
  
	train(agent, callbacks=callbacks, total_timestep=episodes*timesteps, model_output_path=model_output_path)

