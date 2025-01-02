"""
######################################
Simulator for IEEE 802.11bn
"""

import time
import os
import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from AuxiliarFunctions import *
from MAPCsim import *


def simulate_iteration(sim, traffic_type, traffic_load, iteration):
    """
    Simulates one iteration and returns the STAs_arrivals_matrix.

    Parameters:
    sim (str): Simulation identifier.
    traffic_type (str): Type of traffic (e.g., 'Poisson', 'Bursty', 'VR').
    traffic_load (str): Load of the traffic (e.g., 'low', 'medium', 'high').
    iteration (int): Iteration number.
    STA_matrix_save (np.ndarray): Pre-saved STA matrix for all iterations.
    channelMatrix_save (np.ndarray): Pre-saved channel matrix for all iterations.
    RSSI_dB_vector_to_export_save (np.ndarray): Pre-saved RSSI vector for all iterations.

    Returns:
    np.ndarray: The STAs_arrivals_matrix for the given iteration.
    """

    EDCAaccessCategory = {'Poisson': 'BE', 'Bursty': 'BE', 'VR': 'VI'}.get(traffic_type, None)
    # Check if the traffic type is valid
    if EDCAaccessCategory is None:
        raise ValueError(f"Invalid traffic type: {traffic_type}. Valid types are 'Poisson', 'Bursty', 'VR'.")


    ### Deployment-dependent data
    AP_matrix, STA_matrix = AP_STA_coordinates(AP_number, STA_number, scenario_type, grid_value)
    # STA_matrix = STA_matrix_save[:, :, iteration]

    # Association
    association = AP_STA_Association(AP_number, STA_number, scenario_type)

    # Plot deployment
    # PlotDeployment(AP_matrix, STA_matrix, association, grid_value, walls)

    # Channel matrix  
    # channelMatrix = channelMatrix_save[:, :, iteration]

    # Compute the channelMatrix and RSSI_dB_vector_to_export if they aren't provided
    channelMatrix, _ = GetChannelMatrix(MaxTxPower, Cca, AP_matrix, STA_matrix, scenario_type, walls, checkSegmentIntersection, Getloss)
    
    # Compute the overheads
    preTX_overheadsDCF, preTX_overheadsCSR, DCFoverheads, CSRoverheads = OverheadsCalc(EDCAaccessCategory)

    CGs_STAs, TxPowerMatrix = CG_creationTPC(AP_number, STA_number, CSRoverheads, Pn_dBm, Nsc, Nss, association, channelMatrix, MaxTxPower, TXOP_duration) 
    
    # Simulation duration
    timestamp_to_stop = 5

    h5_file_path = os.path.join(os.getcwd(), 'traffic datasets', sim, traffic_type, traffic_load, f"STAs_arrivals_matrix{iteration}.h5")

    # Open and load the dataset
    with h5py.File(h5_file_path, 'r') as h5file:
      STAs_arrivals_matrix = np.array([h5file[key][:] for key in h5file.keys()])
    
    # Set the seed
    seed = 1

    np.random.seed(seed)
    simDCF = MAPCsim(AP_number, STA_number, association, MaxTxPower, channelMatrix, traffic_type, timestamp_to_stop, 
            simulation_system, validationFlag, TXOP_duration, Pn_dBm, Nss, Nsc, preTX_overheadsDCF, preTX_overheadsCSR, DCFoverheads, CSRoverheads)  # new "Traffic" object
    simDCF.STA_queue_timeline = STAs_arrivals_matrix  # Loading the traffic dataset and assigning it to the STAs
    simDCF.simulation_system = 'DCF'
    simDCF.accessCategory = EDCAaccessCategory
    simDCF.InitSettings()  # Initializing STAs
    simDCF.Start()


    np.random.seed(seed)
    simMNP = MAPCsim(AP_number, STA_number, association, MaxTxPower, channelMatrix, traffic_type, timestamp_to_stop, 
            simulation_system, validationFlag, TXOP_duration, Pn_dBm, Nss, Nsc, preTX_overheadsDCF, preTX_overheadsCSR, DCFoverheads, CSRoverheads)  # new "Traffic" object
    simMNP.STA_queue_timeline = STAs_arrivals_matrix  # Loading the traffic dataset and assigning it to the STAs
    simMNP.simulation_system = 'CSR'
    simMNP.scheduler = 'MNP'
    simMNP.CGs_STAs = CGs_STAs
    simMNP.TxPowerMatrix = TxPowerMatrix
    simMNP.accessCategory = EDCAaccessCategory
    simMNP.InitSettings()  # Initializing STAs
    simMNP.Start()

    np.random.seed(seed)
    simOP = MAPCsim(AP_number, STA_number, association, MaxTxPower, channelMatrix, traffic_type, timestamp_to_stop, 
            simulation_system, validationFlag, TXOP_duration, Pn_dBm, Nss, Nsc, preTX_overheadsDCF, preTX_overheadsCSR, DCFoverheads, CSRoverheads)  # new "Traffic" object
    simOP.STA_queue_timeline = STAs_arrivals_matrix  # Loading the traffic dataset and assigning it to the STAs
    simOP.simulation_system = 'CSR'
    simOP.scheduler = 'OP'
    simOP.CGs_STAs = CGs_STAs
    simOP.TxPowerMatrix = TxPowerMatrix
    simOP.accessCategory = EDCAaccessCategory
    simOP.InitSettings()  # Initializing STAs
    simOP.Start()

    np.random.seed(seed)
    simTAT = MAPCsim(AP_number, STA_number, association, MaxTxPower, channelMatrix, traffic_type, timestamp_to_stop, 
            simulation_system, validationFlag, TXOP_duration, Pn_dBm, Nss, Nsc, preTX_overheadsDCF, preTX_overheadsCSR, DCFoverheads, CSRoverheads)  # new "Traffic" object
    simTAT.STA_queue_timeline = STAs_arrivals_matrix  # Loading the traffic dataset and assigning it to the STAs
    simTAT.simulation_system = 'CSR'
    simTAT.scheduler = 'TAT'
    simTAT.CGs_STAs = CGs_STAs
    simTAT.TxPowerMatrix = TxPowerMatrix
    simTAT.accessCategory = EDCAaccessCategory
    simTAT.alpha_ = 0.5
    simTAT.beta_ = 0.5
    simTAT.InitSettings()  # Initializing STAs
    simTAT.Start()
    
    print(f'Iteration: {iteration}')
    print(f'DCF 50th {np.percentile(simDCF.delayvector,50)*1000}')
    print(f'DCF 99th {np.percentile(simDCF.delayvector,99)*1000}')
    print(f'MNP 50th {np.percentile(simMNP.delayvector,50)*1000}')
    print(f'MNP 99th {np.percentile(simMNP.delayvector,99)*1000}')
    print(f'OP 50th {np.percentile(simOP.delayvector,50)*1000}')
    print(f'OP 99th {np.percentile(simOP.delayvector,99)*1000}')
    print(f'TAT 50th {np.percentile(simTAT.delayvector,50)*1000}')
    print(f'TAT 99th {np.percentile(simTAT.delayvector,99)*1000}')
    print('-----------------------------------------')

    return simDCF.delayvector

def save_to_h5(output_dir, sim, traffic_type, traffic_load, iteration, DCFdelay):
    """
    Saves the the delay vectors into individual HDF5 files in a structured directory.
    """
    # Create the directory structure
    output_path = os.path.join(output_dir, sim, traffic_type, traffic_load, f"Deployment{iteration}")
    os.makedirs(output_path, exist_ok=True)

    # Save the current iteration to its own HDF5 file
    h5_file_path = os.path.join(output_path, f"DCFdelay.h5")
    # Save data to HDF5 file
    with h5py.File(h5_file_path, 'w') as f:
        f.create_dataset('DCFdelay', data=DCFdelay)
        # f.create_dataset('DCFdelay', data=DCFdelay, compression="gzip")   # with compression


# Start Timer
start_time = time.time()

###### Input parameters
simulation_system = 'DCF'    # Define the system simulation system (DCF, CSR)
validationFlag = 'no'

traffic_types = ['Poisson', 'Bursty', 'VR']
traffic_loads = {
    'Poisson': ['low', 'medium', 'high'],
    'Bursty': ['low', 'medium', 'high'],
    'VR': ['30-60', '30-90', '30-120']
}

# traffic_types = ['VR']
# traffic_loads = {
#     'VR': ['30-120']
# }


# Scenario-related
AP_number = 4
STA_number = 8
grid_value = 40
scenario_type = 'grid'
sim = '20metros-8STAs'
walls = np.array([[0, grid_value, grid_value/2, grid_value/2], 
                  [grid_value/2, grid_value/2, 0, grid_value]])

# System-related parameters
TXOP_duration = 5E-3
Pn_dBm = -95
Cca = -82
BW = 80
Nss = 2
L = 12E3

iterations = 100

### Channel-related parameters
MaxTxPower, Nsc = TXpowerCalc(BW, Nss)

# ### Load deployment data
# h5file_deployments_path = os.path.join(os.getcwd(), 'deployments datasets', sim, 'deployment_datasets.h5')
# with h5py.File(h5file_deployments_path, 'r') as f:
#     STA_matrix_save = f['STA_matrix_save'][:]
#     channelMatrix_save = f['channelMatrix_save'][:]
#     # RSSI_dB_vector_to_export_save = f['RSSI_dB_vector_to_export_save'][:]

### Output directory    
output_dir = os.path.join(os.getcwd(), 'Results/Simulation')

# Run simulations with progress bar
max_workers = 1  # Adjust the number of workers as needed
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    for traffic_type in traffic_types:
        for traffic_load in traffic_loads[traffic_type]:
            futures = [
                executor.submit(
                    # simulate_iteration, sim, traffic_type, traffic_load, i, 
                    # STA_matrix_save, channelMatrix_save
                    simulate_iteration, sim, traffic_type, traffic_load, i
                )
                for i in range(iterations)
            ]
            for i, future in enumerate(tqdm(futures, desc=f"{traffic_type} {traffic_load}", unit=" iteration")):
                try:
                    DCFdelay = future.result()
                    # save_to_h5(output_dir, sim, traffic_type, traffic_load, i, DCFdelay)
                except Exception as e:
                    print(f"Error in iteration {i} for {traffic_type} {traffic_load}: {e}")

# End Timer and print elapsed time
end_time = time.time()
print(f"Simulation took {end_time - start_time:.2f} seconds")