import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Import tqdm for progress bar
from Utils import *
import h5py
import os


# Helper function to simulate one iteration
def simulate_iteration(i):
    # Deployment-dependent data
    AP_matrix, _ = AP_STA_coordinates(AP_NUMBER, STA_NUMBER, SCENARIO_TYPE, GRID_VALUE)
    STA_matrix = STA_matrix_save[:,:,i]

    association = AP_STA_Association(AP_NUMBER, STA_NUMBER, SCENARIO_TYPE)

    # Call the function to plot
    # PlotDeployment(AP_matrix, STA_matrix, association, GRID_VALUE, walls)

    # channelMatrix, RSSI_dB_vector_to_export = GetChannelMatrix(MaxTxPower, CCA, AP_matrix, STA_matrix, SCENARIO_TYPE, walls, checkSegmentIntersection, Getloss)
    channelMatrix = channelMatrix_save[:,:,i]
    RSSI_dB_vector_to_export = RSSI_dB_vector_to_export_save[:,:,i]

    # Overheads
    preTX_overheadsDCF, preTX_overheadsCSR, DCFoverheads, CSRoverheads = OverheadsCalc(EDCAaccessCategory)

    # Compute Throughput DCF and CSR
    CGs_STAs, TxPowerMatrix = CG_creationTPC(AP_NUMBER, STA_NUMBER, PN_DBM, NSC, NSS, 
                                             association, channelMatrix, MaxTxPower, 
                                             CG_filter='on', TPC_method='PSO')    # TPC Optimization method: None, 'PSO', 'IPOPT', 'DE'
    
    per_STA_DCF_throughput_bianchi = Throughput_DCF_bianchi(AP_NUMBER, STA_NUMBER, association, RSSI_dB_vector_to_export, PN_DBM, NSC, NSS, TXOP_DURATION, 
                                                            DCFoverheads, EDCAaccessCategory)

    DL_throughput_CSR_bianchi = Throughput_CSR_bianchi(AP_NUMBER, STA_NUMBER, CGs_STAs, TxPowerMatrix, channelMatrix, PN_DBM, NSC, NSS, TXOP_DURATION,
                                                       CSRoverheads, EDCAaccessCategory)
                                                  
    return per_STA_DCF_throughput_bianchi, DL_throughput_CSR_bianchi

# Start Timer
start_time = time.time()

# Simulation parameters
EDCAaccessCategory = 'VI'

# Scenario-related
AP_NUMBER = 4
STA_NUMBER = 8
GRID_VALUE = 40
SCENARIO_TYPE = 'grid'

sim = '20metros-8STAs'
walls = np.array([[0, GRID_VALUE, GRID_VALUE/2, GRID_VALUE/2], 
                  [GRID_VALUE/2, GRID_VALUE/2, 0, GRID_VALUE]])

# System-related parameters
TXOP_DURATION = 5E-3
PN_DBM = -95
CCA = -82
BW = 80
NSS = 2
L = 12E3

# Channel-related parameters
MaxTxPower, NSC = TXpowerCalc(BW, NSS)

# For reproducibility
np.random.seed(1)  

# Define output folder
h5file_path = os.path.join(os.getcwd(), 'deployments datasets' , sim, 'deployment_datasets.h5')

# Open the HDF5 file in read mode
with h5py.File(h5file_path, 'r') as f:
    # Load datasets into variables
    STA_matrix_save = f['STA_matrix_save'][:]
    channelMatrix_save = f['channelMatrix_save'][:]
    RSSI_dB_vector_to_export_save = f['RSSI_dB_vector_to_export_save'][:]

# Simulation parameters for parallel processing
ITERATIONS = 100

# # Result arrays for throughput
per_STA_DCF_throughput_bianchi = np.zeros((ITERATIONS, STA_NUMBER))
DL_throughput_CSR_bianchi = np.zeros((ITERATIONS, STA_NUMBER))

# Pre-allocate result arrays
per_STA_DCF_throughput_bianchi = np.zeros((ITERATIONS, STA_NUMBER))
DL_throughput_CSR_bianchi = np.zeros((ITERATIONS, STA_NUMBER))


# Run simulations with progress bar
MAX_WORKERS = 8  # Adjust the number of workers as needed
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    for i, (dcf_throughput, csr_throughput) in enumerate(
        tqdm(executor.map(simulate_iteration, range(ITERATIONS), chunksize=10), 
             total=ITERATIONS, desc="Simulating", unit=" iteration")
    ):
        per_STA_DCF_throughput_bianchi[i, :] = dcf_throughput
        DL_throughput_CSR_bianchi[i, :] = csr_throughput


# for i in range(ITERATIONS):
#     dcf_throughput, csr_throughput = simulate_iteration(i)
#     per_STA_DCF_throughput_bianchi[i, :] = dcf_throughput
#     DL_throughput_CSR_bianchi[i, :] = csr_throughput

# Aggregate Throughput
agg_thr_DCF_DL_vector = np.sum(per_STA_DCF_throughput_bianchi, axis=1)
agg_thr_cSR_bianchi = np.sum(DL_throughput_CSR_bianchi, axis=1)

# Flatten the results for plotting
allSTA_DCF = per_STA_DCF_throughput_bianchi.flatten()
allSTA_CSR = DL_throughput_CSR_bianchi.flatten()

# End Timer and print elapsed time
end_time = time.time()
print(f"Simulation took {end_time - start_time:.2f} seconds")

# Helper function for ECDF
def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

# Plotting ECDF for aggregate throughput
plt.figure()
x_DCF, y_DCF = ecdf(agg_thr_DCF_DL_vector)
x_CSR, y_CSR = ecdf(agg_thr_cSR_bianchi)
plt.plot(x_DCF, y_DCF, label="DCF")
plt.plot(x_CSR, y_CSR, label="CSR")
plt.xlabel('Aggregate Throughput (Mbps)')
plt.ylabel('Cumulative Distribution Function')
plt.legend()
plt.grid(True)
# plt.title('CDF of Throughput')

# Plotting ECDF for per-STA throughput
plt.figure()
x_STA_DCF, y_STA_DCF = ecdf(allSTA_DCF)
x_STA_CSR, y_STA_CSR = ecdf(allSTA_CSR)
plt.plot(x_STA_DCF, y_STA_DCF, label="DCF")
plt.plot(x_STA_CSR, y_STA_CSR, label="CSR")
plt.xlabel('Per-STA Throughput (Mbps)')
plt.ylabel('Cumulative Distribution Function')
plt.legend()
plt.grid(True)
# plt.title('CDF of Throughput')

plt.show()


# ################## Saving the results ##########################
# # Define the folder structure for saving
output_folder = os.path.join(os.getcwd(), 'Results', 'ThroughputCalculation', sim)
os.makedirs(output_folder, exist_ok=True)

# # # Save the results in .npy
# # np.save(os.path.join(output_folder, "per_STA_DCF_throughput_bianchi.npy"), per_STA_DCF_throughput_bianchi)
# # np.save(os.path.join(output_folder, "DL_throughput_CSR_bianchi.npy"), DL_throughput_CSR_bianchi)

# # Define file path
# h5file_path = os.path.join(output_folder, 'simulation_results.h5')

# # Save data to HDF5 file
# with h5py.File(h5file_path, 'w') as f:
#     f.create_dataset('per_STA_DCF_throughput_bianchi', data=per_STA_DCF_throughput_bianchi)
#     f.create_dataset('DL_throughput_CSR_bianchi', data=DL_throughput_CSR_bianchi)

# # Optionally, save as CSV for readability
# np.savetxt(os.path.join(output_folder, "per_STA_DCF_throughput_bianchi.csv"), per_STA_DCF_throughput_bianchi, delimiter=",")
# np.savetxt(os.path.join(output_folder, "DL_throughput_CSR_bianchi.csv"), DL_throughput_CSR_bianchi, delimiter=",")


