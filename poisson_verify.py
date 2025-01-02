import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, poisson
import os
import h5py

def verify_poisson(STAs_arrivals_matrix, interval=1e-3):
    """
    Verify that traffic follows a Poisson distribution for each STA.
    Args:
        STAs_arrivals_matrix: List of arrival times for each STA.
        interval: Time interval to count arrivals (in seconds).
    """
    for i, arrivals in enumerate(STAs_arrivals_matrix):
        # Calculate inter-arrival times
        inter_arrival_times = np.diff(arrivals)
        
        # Plot histogram of inter-arrival times
        plt.figure(figsize=(10, 4))
        plt.hist(inter_arrival_times, bins=50, density=True, alpha=0.75, label="Observed")
        
        # Fit and plot exponential distribution
        rate = 1 / np.mean(inter_arrival_times)  # Rate parameter λ
        x = np.linspace(0, np.max(inter_arrival_times), 1000)
        plt.plot(x, rate * np.exp(-rate * x), 'r-', lw=2, label="Exponential Fit")
        plt.title(f"STA {i+1} - Inter-Arrival Times")
        plt.xlabel("Inter-Arrival Time (s)")
        plt.ylabel("Density")
        plt.legend()
        plt.show()
        
        # Count arrivals in fixed intervals
        time_bins = np.arange(0, arrivals[-1] + interval, interval)
        arrival_counts, _ = np.histogram(arrivals, bins=time_bins)
        
        # Plot histogram of arrival counts
        plt.figure(figsize=(10, 4))
        plt.hist(arrival_counts, bins=range(arrival_counts.max() + 1), density=True, alpha=0.75, label="Observed")
        
        # Fit and plot Poisson distribution
        mu = np.mean(arrival_counts)  # Mean count in interval
        k = np.arange(0, arrival_counts.max() + 1)
        plt.plot(k, poisson.pmf(k, mu), 'r-', lw=2, label="Poisson Fit")
        plt.title(f"STA {i+1} - Arrival Counts per Interval")
        plt.xlabel("Arrival Count")
        plt.ylabel("Probability")
        plt.legend()
        plt.show()


# Define the path to the HDF5 file
sim = '30metros-16STAs'
traffic_type = 'Poisson'
traffic_load = 'low'
iteration = 1
output_dir = os.path.join(os.getcwd(), 'traffic datasets')
h5_file_path = os.path.join(output_dir, sim, traffic_type, traffic_load, f"STAs_arrivals_matrix{iteration}.h5")

# Open and load the dataset
with h5py.File(h5_file_path, 'r') as h5file:
    STAs_arrivals_matrix = []
    for key in h5file.keys():
        STAs_arrivals_matrix.append(h5file[key][:])

# STAs_arrivals_matrix is now a list of arrays, one per STA
print(f"Loaded data for iteration {iteration}:\n", STAs_arrivals_matrix)

verify_poisson(STAs_arrivals_matrix, interval=1e-3)