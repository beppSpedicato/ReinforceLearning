import matplotlib.pyplot as plt
import numpy as np

def plotTrainRewards(
    train_rewards: list,
    title: str,
    window_size: int
): 
    plt.figure(figsize=(10, 5))
    plt.plot(train_rewards, label='Train reward per episode')

    means = []
    positions = []
    for i in range(0, len(train_rewards), window_size):
        window = train_rewards[i:i+window_size]
        mean_value = np.mean(window)
        means.append(mean_value)
        positions.append(i + window_size//2)  # centro della finestra

    # Tracciare la linea delle medie
    plt.plot(positions, means, color='red', label=f'Average every {window_size} episodes', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"train_rewards_{title}.png")
    plt.close()

	# Creazione file testo con le medie
    filename = f"train_rewards_means_{title}.txt"
    with open(filename, 'w') as f:
        f.write(f"{title}\n")
        for mean_value in means:
            f.write(f"{mean_value}\n")

def plotAvgTxtFiles(file_list, merge_title):
    plt.figure(figsize=(12, 6))

    for filename in file_list:
        with open(filename, 'r') as f:
            lines = f.readlines()
            baseline = lines[0].strip()
            means = [float(line.strip()) for line in lines[1:]]

        x = [i for i in range(len(means))]
        plt.plot(x, means, marker='o', label=baseline)

    plt.xlabel('Window index (ogni 500 episodi)')
    plt.ylabel('Media reward')
    plt.title('Confronto medie reward per baseline')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{merge_title}.png")
    plt.close()