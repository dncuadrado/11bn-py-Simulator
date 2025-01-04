import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Import tqdm for progress bar
from Utils import *
import h5py
import os

################## Deployment Generator ###########################


# Helper function to simulate one iteration
def simulate_iteration(i):
    stop = 0
    while stop == 0:
        # Deployment-dependent data
        AP_matrix, STA_matrix = AP_STA_coordinates(AP_number, STA_number, scenario_type, grid_value)

        association = AP_STA_Association(AP_number, STA_number, scenario_type)

        # Call the function to plot
        # PlotDeployment(AP_matrix, STA_matrix, association, grid_value, walls)

        channelMatrix, RSSI_dB_vector_to_export = GetChannelMatrix(MaxTxPower, Cca, AP_matrix, STA_matrix, scenario_type, walls, checkSegmentIntersection, Getloss)

        # Overheads
        _, _, DCFoverheads, _ = OverheadsCalc(EDCAaccessCategory)
                                     
        per_STA_DCF_throughput_bianchi = Throughput_DCF_bianchi(AP_number, STA_number, association, RSSI_dB_vector_to_export, Pn_dBm, Nsc, Nss, TXOP_duration, 
                                                                DCFoverheads, EDCAaccessCategory)
        
        if 0.9*np.min(per_STA_DCF_throughput_bianchi) > 30:   # Compare against the VR bitrate
            stop = 1

    return STA_matrix, channelMatrix, RSSI_dB_vector_to_export

# Start Timer
start_time = time.time()

###### Input parameters

# Simulation parameters
traffic_type = 'VR'
traffic_load = '30-60'
EDCAaccessCategory = 'VI'

# Scenario-related
AP_number = 4
STA_number = 16
grid_value = 40
scenario_type = 'grid'
sim = '20metros-16STAs'
walls = np.array([[0, grid_value, grid_value/2, grid_value/2], 
                  [grid_value/2, grid_value/2, 0, grid_value]])

# System-related parameters
TXOP_duration = 5E-3
Pn_dBm = -95
Cca = -82
BW = 80
Nss = 2
L = 12E3

# Channel-related parameters
MaxTxPower, Nsc = TXpowerCalc(BW, Nss)

# Seed for reproducibility
rndGeneration = {
    '20metros-8STAs': 1, 
    '20metros-16STAs': 2,
    '30metros-16STAs': 3,
}
np.random.seed(rndGeneration[sim])

# Simulation parameters for parallel processing
iterations = 100

# Pre-allocate variables
STA_matrix_save = np.empty((STA_number, 2, iterations))        
channelMatrix_save = np.empty((STA_number, AP_number, iterations))
RSSI_dB_vector_to_export_save = np.empty((STA_number, AP_number, iterations))


# # Result arrays for throughput
per_STA_DCF_throughput_bianchi = np.zeros((iterations, STA_number))
DL_throughput_CSR_bianchi = np.zeros((iterations, STA_number))

# Pre-allocate result arrays
per_STA_DCF_throughput_bianchi = np.zeros((iterations, STA_number))
DL_throughput_CSR_bianchi = np.zeros((iterations, STA_number))


EDCAaccessCategory = 'VI' if traffic_type == 'VR' else 'BE'


# Run simulations with progress bar
max_workers = 8  # Adjust the number of workers as needed
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    for i, (STA_matrix, channelMatrix, RSSI_dB_vector_to_export) in enumerate(
        tqdm(executor.map(simulate_iteration, range(iterations), chunksize=10), 
             total=iterations, desc="Simulating", unit=" iteration")
    ):
        STA_matrix_save[:, :, i] = STA_matrix
        channelMatrix_save[:, :, i] = channelMatrix
        RSSI_dB_vector_to_export_save[:, :, i] = RSSI_dB_vector_to_export


# Define output folder
output_folder = os.path.join(os.getcwd(), 'deployments datasets' , sim)
os.makedirs(output_folder, exist_ok=True)

# Define file path
h5file_path = os.path.join(output_folder, 'deployment_datasets.h5')

# Save data to HDF5 file
with h5py.File(h5file_path, 'w') as f:
    f.create_dataset('STA_matrix_save', data=STA_matrix_save)
    f.create_dataset('channelMatrix_save', data=channelMatrix_save)
    f.create_dataset('RSSI_dB_vector_to_export_save', data=RSSI_dB_vector_to_export_save)

# End Timer and print elapsed time
end_time = time.time()
print(f"Simulation took {end_time - start_time:.2f} seconds")