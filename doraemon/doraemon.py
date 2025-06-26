import random
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from scipy.stats import beta
from PPO.ppo_utils import TrainTestCallback, create_agent, train
import numpy as np
import torch
import os
from scipy.special import kl_div

class DoraemonCallback(BaseCallback):
	"""
	Doraemon Domain Randomization Callback
	
	This callback implements the Doraemon algorithm for dynamic domain randomization.
	It maintains beta distributions for environment parameters and updates them based
	on the agent's performance using importance sampling and KL divergence constraints.
	
	The algorithm works by:
	1. Collecting trajectories and their corresponding environment parameters
	2. Estimating success rates using importance sampling
	3. Updating parameter distributions when success rate is below threshold
	"""
	
	def __init__(
		self,
		model: PPO,
		verbose: int = 0,
		delta: float = 0.5,
		a: list = [100, 100, 100],  # Alpha parameters for beta distributions
        b: list = [100, 100, 100],  # Beta parameters for beta distributions
        K: int = 20,                # Number of trajectories to collect before updating
        alpha: float = 0.5,         # Success rate threshold
        epsilon: float = 0.05,      # KL divergence constraint
        step: float = 2.0,          # Step size for parameter updates
        success_threeshold: int = 1600,  # Minimum reward to consider trajectory successful
	):
		self.delta = delta
		self.a = a
		self.b = b
		self.trajectories = []
		self.dynamics_params = []
		self.current_reward = 0
		self.K = K
		self.alpha = alpha
		self.success_threeshold = success_threeshold
		self.epsilon = epsilon
		self.step = step
  
		super().__init__(verbose)
		self.init_callback(model)

	def _on_step(self):
		done = self.locals['dones'][0]
		reward = self.locals['rewards'][0]
		self.current_reward += reward
  
		if done:
			# Apply distribution to environment masses
			params = self.training_env.envs[0].beta_sample_parameters(self.a, self.b, self.delta, log=(self.verbose==1))

			# Collect Trajectories and Environment Parameters
			self.trajectories.append(self.current_reward)
			self.dynamics_params.append(params)

			# When K trajectories was collected:
			if len(self.trajectories) >= self.K:
				G_ha = self.estimate_success_rate(self.a, self.b, self.a, self.b)
				if (G_ha > 0 and self.verbose == 1):
					print(G_ha)

				if (G_ha < self.alpha):
					self.update_dist()

				self.trajectories = []
				self.dynamics_params = []

			self.current_reward = 0

		return True

	def estimate_success_rate(self, old_a, old_b, new_a, new_b):
		numeratore = 0.0
		denominatore = 0.0

		for k in range(self.K):
			e_k = self.dynamics_params[k]
			t_k = self.trajectories[k]
			
			success = 1 if t_k >= self.success_threeshold else 0
			w_k = self.nu_phi(new_a, new_b, e_k) / self.nu_phi(old_a, old_b, e_k)
			
			numeratore += w_k * success
			denominatore += w_k

		return numeratore / (denominatore + 1e-8)  


	def nu_phi(self, a_list, b_list, xi, eps=1e-10):
		masses = self.training_env.envs[0].get_masses_ranges(self.delta)

		total_pdf = 1.0
		for i, x in enumerate(xi):
			low, high = masses[i]
			scaled_x = (x - low) / (high - low)
   
			if not (0 <= scaled_x <= 1):
				return eps 

			p = beta.pdf(scaled_x, a_list[i], b_list[i]) / (high - low)
			total_pdf *= max(p, eps)

		return total_pdf


	def kl_divergence(self, a, b, new_a, new_b, samples=100):
		total_kl = 0.0
		for a_old, b_old, a_new, b_new in zip(a, b, new_a, new_b):
			x = np.linspace(0.01, 0.99, samples)
			p = beta.pdf(x, a_old, b_old)
			q = beta.pdf(x, a_new, b_new)
			p = np.clip(p, 1e-10, None)
			q = np.clip(q, 1e-10, None)
			kl = np.sum(kl_div(p, q)) / samples
			total_kl += kl
		return total_kl

	def update_dist(self):
		step = self.step
		candidates = []
		second_candidates = []

		new_a = self.a[:].copy()
		new_b = self.b[:].copy()
  
		for s in [-step, step]:
			for i in range(len(self.a)):
				new_a[i] = max(1.0, self.a[i] + s)
				kl = self.kl_divergence(self.a, self.b, new_a, new_b)
				if kl <= self.epsilon:
					G_ha =  self.estimate_success_rate(self.a, self.b, new_a, new_b)
					if G_ha >= self.alpha:
						entropy = sum(beta(a, b).entropy() for a, b in zip(new_a, new_b))
						candidates.append((entropy, new_a, new_b))
					else:
						second_candidates.append((G_ha, new_a, new_b))

				new_b[i] = max(1.0, self.b[i] + s)
				kl = self.kl_divergence(self.a, self.b, new_a, new_b)
				if kl <= self.epsilon:
					G_ha =  self.estimate_success_rate(self.a, self.b, new_a, new_b)
					if G_ha >= self.alpha:
						entropy = sum(beta(a, b).entropy() for a, b in zip(new_a, new_b))
						candidates.append((entropy, new_a, new_b))
					else:
						second_candidates.append((G_ha, new_a, new_b))

		if candidates:
			best = max(candidates, key=lambda x: x[0])
			if (self.verbose == 1):
				print(best)

			self.a = best[1]
			self.b = best[2]
			
		else if second_candidates:
			best = max(second_candidates, key=lambda x: x[0])
			if (self.verbose == 1):
				print(best)

			self.a = best[1]
			self.b = best[2]
		

def train_test_ppo_with_doraemon (
	output_folder: str = "./udr_output",
	train_env: str = "CustomHopper-source-v0",
	test_env: str = "CustomHopper-target-v0",
	clip_range: float = -1,
	episodes: int = 8000,
	timesteps: int = 300,
	gamma: float = 0.99,
	learning_rate: float = 1e-3,
	print_std_deviation: bool = False,
	seed: int = 10,
	verbose: int = 0,

	# doraemon parameters
 	epsilon: float = 0.05,
	step: float = 2.0,
	delta: float = 0.5,
	alpha: float = 0.5,

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
		DoraemonCallback(
	  		agent,
		 	verbose=verbose,

			# doraemon parameters
			epsilon=epsilon,
			step=step,
			alpha=alpha,
			delta=delta, 
		)
	]
  
	train(agent, callbacks=callbacks, total_timestep=episodes*timesteps, model_output_path=model_output_path)


