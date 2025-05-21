import matplotlib.pyplot as plt
import numpy as np


def plotTrainRewards(
	train_rewards: list,
	title: str,
	window_size: int,
	create_txt: bool = True,
	chart_title: str = "Training rewards",
	x_label: str = "Episodes",
	y_label: str = "Reward",
	outputFolder: str = "./",
	label: str = "Train reward per episode"
): 
	plt.figure(figsize=(10, 5))
	plt.plot(train_rewards, label=label)

	means = []
	positions = []
	for i in range(0, len(train_rewards), window_size):
		window = train_rewards[i:i+window_size]
		mean_value = np.mean(window)
		means.append(mean_value)
		positions.append(i + window_size//2)
  
	plt.plot(positions, means, color='red', label=f'Average every {window_size} episodes', linewidth=2)

	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(chart_title)
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(f"{outputFolder}/train_{title}.png")
	plt.close()

	if (create_txt):
		filename = f"{outputFolder}/train_means_{title}.txt"
		with open(filename, 'w') as f:
			f.write(f"{title}\n")
			for mean_value in means:
				f.write(f"{mean_value}\n")

def plotAvgTxtFiles(
    file_list, 
    merge_title,
	x_label: str = 'Window index (every 500 episodes)',
	y_label: str = 'Mean reward',
	title: str = 'Mean rewards for each baselines'
):
	plt.figure(figsize=(12, 6))

	for filename in file_list:
		with open(filename, 'r') as f:
			lines = f.readlines()
			baseline = lines[0].strip()
			means = [float(line.strip()) for line in lines[1:]]

		x = [i for i in range(len(means))]
		plt.plot(x, means, marker='o', label=baseline)

	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(f"{merge_title}.png")
	plt.close()