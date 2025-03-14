import numpy as np
import re
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Import tqdm for progress bar
from numpy.random import SeedSequence, default_rng
from Utils import *
from DeploymentGenerator import deployment_generator
import h5py
import os


# Helper function to simulate one iteration
def throughput_calculation(sim_config, seed, show_plot= None, iter_number=None):

    # Deployment-dependent data
    AP_matrix, STA_matrix, association, channelMatrix = deployment_generator(sim_config, seed, show_plot=show_plot)

    # if iter is not None:
    #     # Deployment-dependent data
    #     STA_matrix = STA_matrix_save[:,:,iter]
    #     channelMatrix = channelMatrix_save[:,:,iter]

    # Mode 1: Use pre-loaded data (if enabled)
    if sim_config['use_preloaded_deployments']:
        STA_matrix = STA_matrix_save[:, :, iter_number]
        sim_config['channelMatrix'] = channelMatrix_save[:, :, iter_number]
    # Mode 2: Generate fresh data
    else:
        AP_matrix, STA_matrix, sim_config['association'], sim_config['channelMatrix'] = deployment_generator(sim_config, seed)

    # Overheads
    preTX_overheadsEDCA, preTX_overheadsCSR, EDCAoverheads, CSRoverheads = OverheadsCalc(EDCAaccessCategory)

    map_matrix, TxPowerMatrixTemp, comb_ok, _ = CG_creationTPC(AP_NUMBER, 
                                                STA_NUMBER, 
                                                PN_DBM, 
                                                NSC, 
                                                NSS, 
                                                association, 
                                                channelMatrix, 
                                                MaxTxPower, 
                                                CG_filter='on', TPC_method='PSO')    # TPC Optimization method: None, 'PSO', 'IPOPT', 'DE'
    
    TxPowerMatrix = [row.tolist() for i, row in enumerate(TxPowerMatrixTemp) if comb_ok[i]==True]
    CGs_STAs = [row.tolist() for i, row in enumerate(map_matrix) if comb_ok[i]==True]

    # Validate that TxPowerMatrix and CGs_STAs have the same length
    if len(TxPowerMatrix) != len(CGs_STAs):
        raise ValueError('TxPowerMatrix and CGs_STAs have different lengths')
    
    per_STA_EDCA_throughput_bianchi = Throughput_EDCA_bianchi(AP_NUMBER, STA_NUMBER, association, channelMatrix, sim_config['MaxTxPower'],
                                                            PN_DBM, NSC, NSS, TXOP_DURATION, 
                                                            EDCAoverheads, EDCAaccessCategory)

    DL_throughput_CSR_bianchi = Throughput_CSR_bianchi(AP_NUMBER, STA_NUMBER, association, CGs_STAs, TxPowerMatrix, channelMatrix, PN_DBM, NSC, NSS, TXOP_DURATION,
                                                       CSRoverheads, EDCAaccessCategory)
                                                  
    return per_STA_EDCA_throughput_bianchi, DL_throughput_CSR_bianchi

def plot_cdf(per_STA_EDCA_throughput_bianchi, DL_throughput_CSR_bianchi):
    # Aggregate Throughput
    agg_thr_EDCA_DL_vector = np.sum(per_STA_EDCA_throughput_bianchi, axis=1)
    agg_thr_cSR_bianchi = np.sum(DL_throughput_CSR_bianchi, axis=1)

    # Flatten the results for plotting
    allSTA_EDCA = per_STA_EDCA_throughput_bianchi.flatten()
    allSTA_CSR = DL_throughput_CSR_bianchi.flatten()

    # Helper function for ECDF
    def ecdf(data):
        x = np.sort(data)
        y = np.arange(1, len(x)+1) / len(x)
        return x, y

    # Plotting ECDF for aggregate throughput
    plt.figure()
    x_EDCA, y_EDCA = ecdf(agg_thr_EDCA_DL_vector)
    x_CSR, y_CSR = ecdf(agg_thr_cSR_bianchi)
    plt.plot(x_EDCA, y_EDCA, label="EDCA")
    plt.plot(x_CSR, y_CSR, label="CSR")
    plt.xlabel('Aggregate Throughput (Mbps)')
    plt.ylabel('Cumulative Distribution Function')
    plt.legend()
    plt.grid(True)
    # plt.title('CDF of Throughput')

    # Plotting ECDF for per-STA throughput
    plt.figure()
    x_STA_EDCA, y_STA_EDCA = ecdf(allSTA_EDCA)
    x_STA_CSR, y_STA_CSR = ecdf(allSTA_CSR)
    plt.plot(x_STA_EDCA, y_STA_EDCA, label="EDCA")
    plt.plot(x_STA_CSR, y_STA_CSR, label="CSR")
    plt.xlabel('Per-STA Throughput (Mbps)')
    plt.ylabel('Cumulative Distribution Function')
    plt.legend()
    plt.grid(True)
    # plt.title('CDF of Throughput')

    plt.show()
    

def init_pool_processes(h5_path, use_preloaded):
    """Load HDF5 data only if required by the simulation mode"""
    global STA_matrix_save, channelMatrix_save
    if use_preloaded:
        with h5py.File(h5_path, 'r') as f:
            STA_matrix_save = f['STA_matrix_save'][:]
            channelMatrix_save = f['channelMatrix_save'][:]


# Define module-level globals
STA_matrix_save = None
channelMatrix_save = None


if __name__ == '__main__':
    # Start Timer
    start_time = time.time()

    # Simulation parameters
    EDCAaccessCategory = 'VI'

    sim = '30-16'  # Simulation name: 'APtoAPdistance-STA_NUMBER'
    numbers = re.findall(r'\d+', sim) # Extract numbers from the simulation name

    # Scenario-related
    AP_NUMBER = 4
    STA_NUMBER = int(numbers[1]) 
    GRID_VALUE = int(numbers[0]) * 2
    SCENARIO_TYPE = 'grid'

    walls = np.array([[0, GRID_VALUE, GRID_VALUE/2, GRID_VALUE/2], 
                    [GRID_VALUE/2, GRID_VALUE/2, 0, GRID_VALUE]])

    # System-related parameters
    TXOP_DURATION = 5E-3
    PN_DBM = -95
    CCA = -82
    BW = 80
    NSS = 2
    FRAME_LENGTH = 12E3

    # Channel-related parameters
    MaxTxPower, NSC = TXpowerCalc(BW, NSS)

    # For reproducibility
    np.random.seed(1)  

    # Simulation Configuration
    sim_config = {
        'use_preloaded_deployments': True,
        'AP_NUMBER': AP_NUMBER,
        'STA_NUMBER': STA_NUMBER,
        'SCENARIO_TYPE': SCENARIO_TYPE,
        'GRID_VALUE': GRID_VALUE,
        'walls': walls,
        'MaxTxPower': MaxTxPower,
        'TXOP_DURATION': TXOP_DURATION,
        'PN_DBM': PN_DBM,
        'CCA': CCA,
        'NSS': NSS,
        'NSC': NSC,
        'learning_timestamp_to_stop': 2, # seconds
        'training_flag': False,
        'timestamp_to_stop': 5, # seconds
        'FRAME_LENGTH': FRAME_LENGTH,
        'EVENT_NUMBER': int(1E5), # Number of events considered for traffic generation
        'seed': 1,
        'output_dir': os.path.join(os.getcwd(), 'traffic datasets', sim),
        'overheads' : ''
    }

    # Define output folder
    h5file_path = os.path.join(os.getcwd(), 'deployments datasets' , sim, 'deployment_datasets.h5')

    # Open the HDF5 file in read mode
    with h5py.File(h5file_path, 'r') as f:
        # Load datasets into variables
        STA_matrix_save = f['STA_matrix_save'][:]
        channelMatrix_save = f['channelMatrix_save'][:]

    # Simulation parameters for parallel processing
    ITERATIONS = 100

    seed_seq = SeedSequence(1)
    seeds = seed_seq.generate_state(ITERATIONS)

    # Deployment data path
    h5file_deployments_path = os.path.join(os.getcwd(), 'deployments datasets', sim, 'deployment_datasets.h5')

    # Pre-allocate result arrays
    per_STA_EDCA_throughput_bianchi = np.zeros((ITERATIONS, STA_NUMBER))
    DL_throughput_CSR_bianchi = np.zeros((ITERATIONS, STA_NUMBER))

    futures = []
    max_workers = min(os.cpu_count(), ITERATIONS)  # Optimize worker count

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_pool_processes,
        initargs=(h5file_deployments_path, sim_config['use_preloaded_deployments'])
        ) as executor:

        futures = [
            executor.submit(
                throughput_calculation, sim_config, seeds[iter_number], iter_number=iter_number
            )
            for iter_number in range(ITERATIONS)
        ]
        for iter_number, future in enumerate(tqdm(futures, desc="Processing", unit=" iterations")):
            try:
                EDCA_throughput, csr_throughput = future.result()
                per_STA_EDCA_throughput_bianchi[iter_number, :] = EDCA_throughput
                DL_throughput_CSR_bianchi[iter_number, :] = csr_throughput
            
            except Exception as e:
                print(f"Error in iterations {iter_number}")

    plot_cdf(per_STA_EDCA_throughput_bianchi, DL_throughput_CSR_bianchi)
    
    # # ################## Saving the results ##########################
    # # # Define the folder structure for saving
    # output_folder = os.path.join(os.getcwd(), 'Results', 'ThroughputCalculation', sim)
    # os.makedirs(output_folder, exist_ok=True)


    # # Define file path
    # h5file_outputpath = os.path.join(output_folder, 'simulation_results.h5')

    # # Save data to HDF5 file
    # with h5py.File(h5file_outputpath, 'w') as f:
    #     f.create_dataset('per_STA_EDCA_throughput_bianchi', data=per_STA_EDCA_throughput_bianchi)
    #     f.create_dataset('DL_throughput_CSR_bianchi', data=DL_throughput_CSR_bianchi)

    
    # End Timer and print elapsed time
    end_time = time.time()
    print(f"Simulation took {end_time - start_time:.2f} seconds")