""" Optimize alpha1 and alpha2 using optuna """

import torch
import gym
import random
import numpy as np
import optuna

from env.custom_hopper import *
from actor_critic.actor_critic_agent import ActorCriticAgent, ActorCriticPolicy

""" 
Volendo si può provare ad ottimizzare facendo la media su una determinata window (5000-12000)
 """

def train(alpha1, alpha2, n_episodes):
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
	
    env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

    """
		Training
	"""
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = ActorCriticPolicy(observation_space_dim, action_space_dim)
    agent = ActorCriticAgent(policy, device='cpu', alpha1=alpha1, alpha2=alpha2)

    train_rewards = []
    for episode in range(n_episodes):
        done = False
        train_reward = 0
        state = env.reset()  # Reset the environment and observe the initial state

        while not done:  # Loop until the episode is over
            action, action_probabilities = agent.get_action(state)
            previous_state = state

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            agent.store_outcome(previous_state, state, action_probabilities, reward, done)

            train_reward += reward
		
        train_rewards.append(train_reward)
        agent.update_policy()

    print(sum(train_rewards) / len(train_rewards))
    return sum(train_rewards) / len(train_rewards)


def objective(trial):
    # Suggest values for alpha1 and alpha2
    alpha1 = trial.suggest_float("alpha1", 1e-3, 1.0, log=True)
    alpha2 = trial.suggest_float("alpha2", 1e-3, 1.0, log=True)
    
    # Normalize alphas (optional but recommended to keep them on the same scale)
    norm = alpha1 + alpha2
    alpha1 /= norm
    alpha2 /= norm

    # Train your actor-critic model with these alphas
    num_episodes = 9000
    total_reward = train(alpha1, alpha2, num_episodes)

    # We want to maximize reward, so we return negative loss or positive reward
    return total_reward
	

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best alpha1:", study.best_params["alpha1"])
    print("Best alpha2:", study.best_params["alpha2"])
    print("Best reward:", study.best_value)

if __name__ == '__main__':
    main()


""" 
Trial: 50 * 9000 episodi

[I 2025-05-20 16:58:48,059] A new study created in memory with name: no-name-bab95711-93b8-48a8-8c0c-6b3fda75fe39
102.09619377417724
[I 2025-05-20 17:02:39,695] Trial 0 finished with value: 102.09619377417724 and parameters: {'alpha1': 0.011194683632032857, 'alpha2': 0.001983725328795775}. Best is trial 0 with value: 102.09619377417724.
114.59209389526258
[I 2025-05-20 17:06:27,161] Trial 1 finished with value: 114.59209389526258 and parameters: {'alpha1': 0.008593362210509608, 'alpha2': 0.06161320147757416}. Best is trial 1 with value: 114.59209389526258.
127.94132966896092
[I 2025-05-20 17:10:34,435] Trial 2 finished with value: 127.94132966896092 and parameters: {'alpha1': 0.1358242860561165, 'alpha2': 0.11065875531028595}. Best is trial 2 with value: 127.94132966896092.
113.84295738240337
[I 2025-05-20 17:14:17,166] Trial 3 finished with value: 113.84295738240337 and parameters: {'alpha1': 0.0011144508657729044, 'alpha2': 0.18689142928269015}. Best is trial 2 with value: 127.94132966896092.
180.055013468003
[I 2025-05-20 17:20:10,423] Trial 4 finished with value: 180.055013468003 and parameters: {'alpha1': 0.010854093017509482, 'alpha2': 0.17697633760134854}. Best is trial 4 with value: 180.055013468003.
91.63112092824242
[I 2025-05-20 17:23:13,470] Trial 5 finished with value: 91.63112092824242 and parameters: {'alpha1': 0.044068072301217134, 'alpha2': 0.0021426375757310667}. Best is trial 4 with value: 180.055013468003.
129.09818884950843
[I 2025-05-20 17:27:17,055] Trial 6 finished with value: 129.09818884950843 and parameters: {'alpha1': 0.06158024604926831, 'alpha2': 0.029042418581097275}. Best is trial 4 with value: 180.055013468003.
110.44219543360202
[I 2025-05-20 17:31:12,910] Trial 7 finished with value: 110.44219543360202 and parameters: {'alpha1': 0.001147910946319665, 'alpha2': 0.3874331276317051}. Best is trial 4 with value: 180.055013468003.
150.5733218492027
[I 2025-05-20 17:35:44,393] Trial 8 finished with value: 150.5733218492027 and parameters: {'alpha1': 0.15786272635456128, 'alpha2': 0.0016010924350612073}. Best is trial 4 with value: 180.055013468003.
117.69630621479232
[I 2025-05-20 17:39:57,398] Trial 9 finished with value: 117.69630621479232 and parameters: {'alpha1': 0.0028392315517932506, 'alpha2': 0.008240140721531983}. Best is trial 4 with value: 180.055013468003.
161.9136081687513
[I 2025-05-20 17:46:28,819] Trial 10 finished with value: 161.9136081687513 and parameters: {'alpha1': 0.4900161069707557, 'alpha2': 0.5739870487791664}. Best is trial 4 with value: 180.055013468003.


"""