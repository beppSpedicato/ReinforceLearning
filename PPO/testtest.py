from PPO.ppo_test import test_ppo_policy


test_ppo_policy(
    model='./trained-models/ppo/target-env-train/clip-range:0.11258122567530965/ppo_model.mdl',
    render=True
)