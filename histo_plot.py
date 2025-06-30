import matplotlib.pyplot as plt
import numpy as np




def plot_histogram(data, name):
    positions = np.arange(len(data))  # X-axis: positions (priorities)
    plt.bar(positions, data)

    plt.xlabel("Priority Rank (the higher the index, the higher the priority)", fontsize=16)
    plt.ylabel("Number of Selections / Number of TXOPs", fontsize=16)
    # plt.title("Priority Selection Frequency")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.legend([name], loc='upper right')
    plt.show()
