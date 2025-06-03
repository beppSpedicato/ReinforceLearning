from PPO.ppo_test import train_and_test_policy

target_env = "CustomHopper-target-v0"
source_env = "CustomHopper-source-v0"

optimized_clip_range = 0.19877024509129543
optimized_learning_rate = 0.0008
optimized_gamma = 0.992

print("Start experiments: ")

print('\nsource->source')
train_and_test_policy(
    train_env=source_env,
    test_env=source_env,
    output_folder="./PPO_output/source-source/",
    clip_range=optimized_clip_range,
    learning_rate=optimized_learning_rate,
    gamma=optimized_gamma,
    episodes=8000,
    seed=20
)

print('\ntarget->target')
train_and_test_policy(
    train_env=target_env,
    test_env=target_env,
    output_folder="./PPO_output/target-target/",
    clip_range=optimized_clip_range,
    learning_rate=optimized_learning_rate,
    gamma=optimized_gamma,
    episodes=8000,
    seed=20
)

print('\nsource->target')
train_and_test_policy(
    train_env=source_env,
    test_env=target_env,
    output_folder="./PPO_output/source-target/",
    clip_range=optimized_clip_range,
    learning_rate=optimized_learning_rate,
    gamma=optimized_gamma,
    episodes=8000,
    seed=20
)

