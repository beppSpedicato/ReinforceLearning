from PPO.ppo_test import test_ppo_policy


test_ppo_policy(
    model='./trained-models/doraemon/e0.05-a0.5-d0.5-s2.0/seed-10/model.mdl',
    render=True,
    test_env='CustomHopper-target-v0'
)