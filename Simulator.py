
"""
######################################
Simulator for IEEE 802.11bn
"""

import time
import os
import h5py
import numpy as np
import re
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from numpy.random import SeedSequence

import Utils as utils
from MAPCsim import *
from TrafficGenerator import traffic_generator
from DeploymentGenerator import deployment_generator
import RLagent as RLagent

from CustomEnv import * # my Custom environment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback


def simulate_iterations(traffic_config, sim_config, learning_config, seed, iter_number=None):
    """
    Simulates one iterations and returns the delay vectors for EDCA, MNP, OP, and TAT.

    Parameters:
    sim (str): Simulation identifier.
    traffic_type (str): Type of traffic (e.g., 'Poisson', 'Bursty', 'CBR').
    traffic_load (str): Load of the traffic (e.g., 'low', 'medium', 'high').
    iter_number (int): Number of the current iteration.
    STA_matrix_save (np.ndarray): Pre-saved STA matrix for all iter_number.
    channelMatrix_save (np.ndarray): Pre-saved channel matrix for all iter_number.
    RSSI_dB_vector_to_export_save (np.ndarray): Pre-saved RSSI vector for all iter_number.

    Returns:
    np.ndarray, np.ndarray, np.ndarray, np.ndarray: delay vector for EDCA, MNP, OP, and TAT.
    """
         
    # Set the seed
    np.random.seed(seed)

    # Deployment
    AP_matrix, STA_matrix, sim_config['association'], sim_config['channelMatrix'] = deployment_generator(sim_config, seed)

    # Use pre-loaded data (if enabled)
    if sim_config['use_preloaded_deployments']:
        STA_matrix = STA_matrix_save[:, :, iter_number]
        sim_config['channelMatrix'] = channelMatrix_save[:, :, iter_number]

        
    # Compute the CGs and TxPowerMatrix
    map_matrix, TxPowerMatrixTemp, comb_ok, datarate = utils.CG_creationTPC(sim_config['AP_NUMBER'], 
                                                sim_config['STA_NUMBER'], 
                                                sim_config['PN_DBM'], 
                                                sim_config['NSC'], 
                                                sim_config['NSS'], 
                                                sim_config['association'], 
                                                sim_config['channelMatrix'], 
                                                sim_config['MaxTxPower'], 
                                                CG_filter='on', TPC_method='PSO')    # TPC Optimization method: None, 'PSO', 'IPOPT', 'DE'
    
    TxPowerMatrix = [row.tolist() for i, row in enumerate(TxPowerMatrixTemp) if comb_ok[i]==True]
    CGs_STAs = [row.tolist() for i, row in enumerate(map_matrix) if comb_ok[i]==True]

    # Validate that TxPowerMatrix and CGs_STAs have the same length
    if len(TxPowerMatrix) != len(CGs_STAs):
        raise ValueError('TxPowerMatrix and CGs_STAs have different lengths')

    ### Traffic dataset
    # Load the traffic dataset. Uncomment if using pre-saved dataset.
    if sim_config['use_preloaded_traffic']:
        h5_file_path = os.path.join(os.getcwd(), 'traffic datasets', sim, f"STAs_arrivals_matrix{iter_number}.h5")
        # Open and load the dataset
        with h5py.File(h5_file_path, 'r') as h5file:
            STAs_arrivals_matrix = [h5file[key][:] for key in h5file.keys()]
    else:
        # # # Generate the traffic dataset for the current value of ITERATIONS. Comment if using pre-saved dataset.
        STAs_arrivals_matrix = traffic_generator(
                traffic_config,
                sim_config
                ) 


    # model = 'l9dthv3v'
    # env = RLagent.evaluation(map_matrix, TxPowerMatrixTemp, comb_ok, datarate, STAs_arrivals_matrix, sim_config, learning_config, model)
    # env.simulator.TrafficAnalysis()

    np.random.seed(sim_config['seed'])
    simEDCA = MAPCsim(sim_config)  # new "MAPC simulator" object
    simEDCA.timestamp_to_stop = sim_config['timestamp_to_stop']
    simEDCA.STA_queue_timeline = STAs_arrivals_matrix  # Loading the traffic dataset and assigning it to the STAs
    simEDCA.simulation_system = 'EDCA'
    simEDCA.accessCategory = sim_config['EDCAaccessCategory']
    simEDCA.InitSettings()  # Initializing STAs
    simEDCA.Run()


    np.random.seed(sim_config['seed'])
    simMNP = MAPCsim(sim_config)  # new "MAPC simulator" object
    simMNP.timestamp_to_stop = sim_config['timestamp_to_stop']
    simMNP.STA_queue_timeline = STAs_arrivals_matrix  # Loading the traffic dataset and assigning it to the STAs
    simMNP.simulation_system = 'CSR'
    simMNP.scheduler = 'MNP'
    simMNP.CGs_STAs = CGs_STAs
    simMNP.TxPowerMatrix = TxPowerMatrix
    simMNP.accessCategory = sim_config['EDCAaccessCategory']
    simMNP.InitSettings()  # Initializing STAs
    simMNP.Run()

    np.random.seed(sim_config['seed'])
    simOP = MAPCsim(sim_config)  # new "MAPC simulator" object
    simOP.timestamp_to_stop = sim_config['timestamp_to_stop']
    simOP.STA_queue_timeline = STAs_arrivals_matrix  # Loading the traffic dataset and assigning it to the STAs
    simOP.simulation_system = 'CSR'
    simOP.scheduler = 'OP'
    simOP.CGs_STAs = CGs_STAs
    simOP.TxPowerMatrix = TxPowerMatrix
    simOP.accessCategory = sim_config['EDCAaccessCategory']
    simOP.InitSettings()  # Initializing STAs
    simOP.Run()

    np.random.seed(sim_config['seed'])
    simTAT = MAPCsim(sim_config)  # new "MAPC simulator" object
    simTAT.timestamp_to_stop = sim_config['timestamp_to_stop']
    simTAT.STA_queue_timeline = STAs_arrivals_matrix  # Loading the traffic dataset and assigning it to the STAs
    simTAT.simulation_system = 'CSR'
    simTAT.scheduler = 'TAT'
    simTAT.CGs_STAs = CGs_STAs
    simTAT.TxPowerMatrix = TxPowerMatrix
    simTAT.accessCategory = sim_config['EDCAaccessCategory']
    simTAT.alpha = 0.5
    simTAT.beta = 0.5
    simTAT.InitSettings()  # Initializing STAs
    simTAT.Run()

    print(f'Iteration: {iter_number}')
    # print(f'RL_PPO 50th {np.percentile(env.simulator.delayvector,50)*1000}')
    # print(f'RL_PPO 99th {np.percentile(env.simulator.delayvector,99)*1000}')
    print(f'EDCA 50th {np.percentile(simEDCA.delayvector,50)*1000}')
    print(f'EDCA 99th {np.percentile(simEDCA.delayvector,99)*1000}')
    print(f'MNP 50th {np.percentile(simMNP.delayvector,50)*1000}')
    print(f'MNP 99th {np.percentile(simMNP.delayvector,99)*1000}')
    print(f'OP 50th {np.percentile(simOP.delayvector,50)*1000}')
    print(f'OP 99th {np.percentile(simOP.delayvector,99)*1000}')
    print(f'TAT 50th {np.percentile(simTAT.delayvector,50)*1000}')
    print(f'TAT 99th {np.percentile(simTAT.delayvector,99)*1000}')
    print('-----------------------------------------')

    return
    # return simEDCA.delayvector, simMNP.delayvector, simOP.delayvector, simTAT.delayvector

def save_to_h5(output_dir, sim, traffic_type, traffic_load, ITERATIONS, EDCAdelay, MNPdelay, OPdelay, TATdelay):
    """
    Saves the the delay vectors into delay.h5 files in a structured directory.
    """
    # Create the directory structure
    output_path = os.path.join(output_dir, sim, traffic_type, traffic_load, f"Deployment{ITERATIONS}")
    os.makedirs(output_path, exist_ok=True)

    # Save the current ITERATIONS to its own HDF5 file
    h5_file_path = os.path.join(output_path, f"delay.h5")
    # Save data to HDF5 file
    with h5py.File(h5_file_path, 'w') as f:
        f.create_dataset('EDCAdelay', data=EDCAdelay)
        # f.create_dataset('EDCAdelay', data=EDCAdelay, compression="gzip")   # with compression
        f.create_dataset('MNPdelay', data=MNPdelay)
        f.create_dataset('OPdelay', data=OPdelay)
        f.create_dataset('TATdelay', data=TATdelay)

def init_pool_processes(h5_path, use_preloaded):
    """Load HDF5 data only if required by the simulation mode"""
    global STA_matrix_save, channelMatrix_save
    if use_preloaded:
        with h5py.File(h5_path, 'r') as f:
            STA_matrix_save = f['STA_matrix_save'][:]
            channelMatrix_save = f['channelMatrix_save'][:]

# Main function
if __name__ == "__main__":

    # Start Timer
    start_time = time.time()

    sim = '20-8'  # Simulation name: 'APtoAPdistance-STA_NUMBER'
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

    ### Channel-related parameters
    MaxTxPower, NSC = utils.TXpowerCalc(BW, NSS)

    # Number of iterations
    ITERATIONS = 100

    seed_seq = SeedSequence(1)
    seeds = seed_seq.generate_state(ITERATIONS)

    # Deployment data path
    h5file_deployments_path = os.path.join(os.getcwd(), 'deployments datasets', sim, 'deployment_datasets.h5')

    # Define the traffic profiles
    traffic_profiles = {
        'A' : {'traffic_model': 'Poisson', 'traffic_load' : 100, 'latency': 1E-4},
        'B' : {'traffic_model': 'Bursty', 'traffic_load' : 50, 'latency': 2E-4},
        'C' : {'traffic_model': 'CBR', 'traffic_load' : 25, 'fps': 60, 'latency': 5E-4}
    }

    # Assign a traffic profile to each STA
    traffic_profile_perSTA = np.random.choice(['A','B','C'], size=STA_NUMBER).tolist()

    # Traffic Configuration 
    traffic_config = {
        'traffic_profiles': traffic_profiles,
        'traffic_profile_perSTA': traffic_profile_perSTA,
        'EDCAaccessCategory' : [
            {'Poisson': 'BE',
            'Bursty': 'BE',
            'CBR': 'VI'
            }.get(traffic_profiles[traffic_profile_perSTA[i]]['traffic_model'], None) 
            for i in range(STA_NUMBER)]
    }  
    
    # Simulation Configuration
    sim_config = {
        'use_preloaded_deployments': True,
        'use_preloaded_traffic': True,
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
        'overheads' : {
            key: [
                utils.OverheadsCalc(traffic_config['EDCAaccessCategory'][i])[idx]
                for i in range(STA_NUMBER)
            ]
            for idx, key in enumerate([
                'preTX_overheadsEDCA',
                'preTX_overheadsCSR',
                'EDCAoverheads',
                'CSRoverheads'
            ])
        }
    }

    # Learning Configuration
    learning_config = {
        'log_dir': os.path.join(os.getcwd(),'trained_models'),
        'parallel_envs': 8,
        'num_episodes': 4E6,
        'simulator_attr' : 'simulator',
    }  

    # Run simulations with progress bar
    max_workers = 1  # Adjust the number of workers as needed

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_pool_processes,
        initargs=(h5file_deployments_path, sim_config['use_preloaded_deployments'])
        ) as executor:

        futures = [
            executor.submit(
                simulate_iterations, traffic_config, sim_config, learning_config, seeds[iter_number], iter_number
            )
            for iter_number in range(ITERATIONS)
        ]
        for iter_number, future in enumerate(tqdm(futures, desc="Processing", unit=" iterations")):
            try:
                # EDCAdelay, MNPdelay, OPdelay, TATdelay = future.result()
                future.result()

                ### Uncoment to save the delay vectors into HDF5 files for each iterations, traffic type, and traffic load in a structured directory
                # save_to_h5(output_dir, sim, traffic_type, traffic_load, i, EDCAdelay, MNPdelay, OPdelay, TATdelay)
            
            except Exception as e:
                print(f"Error in iterations {iter_number}")

    # End Timer and print elapsed time
    end_time = time.time()
    print(f"Simulation took {end_time - start_time:.2f} seconds")