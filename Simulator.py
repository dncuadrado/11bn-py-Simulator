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
from Utils import *
from MAPCsim import *
from TrafficGenerator import TrafficGenerator



# RL Model (e.g., PPO)
from CustomEnv import * # my Custom environment
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback





def simulate_iterations(sim, traffic_type, traffic_load, iter):
    """
    Simulates one iterations and returns the delay vectors for DCF, MNP, OP, and TAT.

    Parameters:
    sim (str): Simulation identifier.
    traffic_type (str): Type of traffic (e.g., 'Poisson', 'Bursty', 'VR').
    traffic_load (str): Load of the traffic (e.g., 'low', 'medium', 'high').
    iter (int): Number of the current iteration.
    STA_matrix_save (np.ndarray): Pre-saved STA matrix for all iter.
    channelMatrix_save (np.ndarray): Pre-saved channel matrix for all iter.
    RSSI_dB_vector_to_export_save (np.ndarray): Pre-saved RSSI vector for all iter.

    Returns:
    np.ndarray, np.ndarray, np.ndarray, np.ndarray: delay vector for DCF, MNP, OP, and TAT.
    """

    EDCAaccessCategory = {'Poisson': 'BE', 'Bursty': 'BE', 'VR': 'VI'}.get(traffic_type, None)
    # Check if the traffic type is valid
    if EDCAaccessCategory is None:
        raise ValueError(f"Invalid traffic type: {traffic_type}. Valid types are 'Poisson', 'Bursty', 'VR'.")


    ### Deployment-dependent data
    AP_matrix, STA_matrix = AP_STA_coordinates(AP_NUMBER, STA_NUMBER, SCENARIO_TYPE, GRID_VALUE)
    # STA_matrix = STA_matrix_save[:, :, iter]

    # Association
    association = AP_STA_Association(AP_NUMBER, STA_NUMBER, SCENARIO_TYPE)

    # Plot deployment
    # PlotDeployment(AP_matrix, STA_matrix, association, GRID_VALUE, walls)

    # Channel matrix. Uncomment if using pre-saved channel matrix.  
    channelMatrix = channelMatrix_save[:, :, iter]

    # RSSI vector. Uncomment if using pre-saved RSSI vector.
    RSSI_dB_vector_to_export = RSSI_dB_vector_to_export_save[:, :, iter]

    # Compute the channelMatrix and RSSI_dB_vector_to_export if they aren't provided as pre-saved datasets
    # channelMatrix, RSSI_dB_vector_to_export = GetChannelMatrix(MaxTxPower, CCA, AP_matrix, STA_matrix, SCENARIO_TYPE, walls, checkSegmentIntersection, Getloss)
    
    # Compute the overheads
    preTX_overheadsDCF, preTX_overheadsCSR, DCFoverheads, CSRoverheads = OverheadsCalc(EDCAaccessCategory)

    CGs_STAs, TxPowerMatrix = CG_creationTPC(AP_NUMBER, STA_NUMBER, PN_DBM, NSC, NSS, 
                                             association, channelMatrix, MaxTxPower, 
                                             CG_filter='on', TPC_method='PSO')    # TPC Optimization method: None, 'PSO', 'IPOPT', 'DE'
                                              
    
    per_STA_DCF_throughput_bianchi = Throughput_DCF_bianchi(AP_NUMBER, STA_NUMBER, association, RSSI_dB_vector_to_export, PN_DBM, NSC, NSS, TXOP_DURATION, 
                                                            DCFoverheads, EDCAaccessCategory)

    # Set the seed
    seed = 1

    # Simulation Configuration
    sim_config = {
        'AP_NUMBER': AP_NUMBER,
        'STA_NUMBER': STA_NUMBER,
        'association': association,
        'MaxTxPower': MaxTxPower,
        'channelMatrix': channelMatrix,
        'traffic_type': traffic_type,
        'traffic_load' : traffic_load,
        'EDCAaccessCategory': EDCAaccessCategory,
        'validation_flag': 'no',
        'TXOP_DURATION': TXOP_DURATION,
        'PN_DBM': PN_DBM,
        'NSS': NSS,
        'NSC': NSC,
        'preTX_overheadsDCF': preTX_overheadsDCF,
        'preTX_overheadsCSR': preTX_overheadsCSR,
        'DCFoverheads': DCFoverheads,
        'CSRoverheads': CSRoverheads,
        'learning_timestamp_to_stop': 1, # seconds
        'training_flag': False,
        'timestamp_to_stop': 5, # seconds
        'CGs_STAs': CGs_STAs,
        'TxPowerMatrix': TxPowerMatrix,
        'L': L,
        'per_STA_DCF_throughput_bianchi': per_STA_DCF_throughput_bianchi,
        'EVENT_NUMBER': 15000, # Number of events considered for traffic generation
        'seed': seed
    }

    # Learning Configuration
    learning_config = {
        'log_dir': '/home/david/Documents/Papers/journal_ML_CSR/python Code/trained_models',
        'parallel_envs': 1,
        'num_episodes': 2E6,
        'simulator_attr' : 'simulator',
    }
    
    # ### Traffic dataset
    # Load the traffic dataset. Uncomment if using pre-saved dataset.
    h5_file_path = os.path.join(os.getcwd(), 'traffic datasets', sim, traffic_type, traffic_load, f"STAs_arrivals_matrix{iter}.h5")
    # Open and load the dataset
    with h5py.File(h5_file_path, 'r') as h5file:
      STAs_arrivals_matrix = np.array([h5file[key][:] for key in h5file.keys()])

    # # # # Generate the traffic dataset for the current value of ITERATIONS. Comment if using pre-saved dataset.
    # STAs_arrivals_matrix = TrafficGenerator(
    #         sim_config['STA_NUMBER'], 
    #         sim_config['validation_flag'], 
    #         sim_config['traffic_type'], 
    #         sim_config['traffic_load'], 
    #         sim_config['L'], 
    #         sim_config['per_STA_DCF_throughput_bianchi'], 
    #         sim_config['EVENT_NUMBER']# Number of events considered for traffic generation
    #         ) 

        # Create a Gym-compatible environment
    def create_env():
        CGs_STAs1, TxPowerMatrix1 = CG_creationTPC(sim_config['AP_NUMBER'], 
                                                 sim_config['STA_NUMBER'], 
                                                 sim_config['PN_DBM'], 
                                                 sim_config['NSC'], 
                                                 sim_config['NSS'], 
                                                 sim_config['association'], 
                                                 sim_config['channelMatrix'], 
                                                 sim_config['MaxTxPower'], 
                                                 CG_filter='off', TPC_method='PSO')    # TPC Optimization method: None, 'PSO', 'IPOPT', 'DE'
        
        simulator = MAPCsim(sim_config)  # new "MAPC simulator" object
        simulator.simulation_system = 'CSR'                 # 'DCF' or 'CSR'
        simulator.CGs_STAs = CGs_STAs1
        simulator.TxPowerMatrix = TxPowerMatrix1
        simulator.accessCategory = sim_config['EDCAaccessCategory']
        return CustomEnv(sim_config, simulator)
    

    env = create_env()
    obs, _ = env.reset(seed=seed, STAs_arrivals_matrix=STAs_arrivals_matrix)

    terminated = False
    # Load the trained model
    loaded_model = A2C.load(os.path.join(learning_config['log_dir'], "best_model.zip"), env=env)
    while not terminated:
        action, _states = loaded_model.predict(obs, deterministic=False)
        obs, _, terminated, _, _ = env.step(action)
        # env.render()  # Optionally visualize the environment
    
    env.simulator.TrafficAnalysis()



    np.random.seed(seed)
    simDCF = MAPCsim(sim_config)  # new "MAPC simulator" object
    simDCF.timestamp_to_stop = sim_config['timestamp_to_stop']
    simDCF.STA_queue_timeline = STAs_arrivals_matrix  # Loading the traffic dataset and assigning it to the STAs
    simDCF.simulation_system = 'DCF'
    simDCF.accessCategory = EDCAaccessCategory
    simDCF.InitSettings()  # Initializing STAs
    simDCF.Run()


    np.random.seed(seed)
    simMNP = MAPCsim(sim_config)  # new "MAPC simulator" object
    simMNP.timestamp_to_stop = sim_config['timestamp_to_stop']
    simMNP.STA_queue_timeline = STAs_arrivals_matrix  # Loading the traffic dataset and assigning it to the STAs
    simMNP.simulation_system = 'CSR'
    simMNP.scheduler = 'MNP'
    simMNP.CGs_STAs = CGs_STAs
    simMNP.TxPowerMatrix = TxPowerMatrix
    simMNP.accessCategory = EDCAaccessCategory
    simMNP.InitSettings()  # Initializing STAs
    simMNP.Run()

    np.random.seed(seed)
    simOP = MAPCsim(sim_config)  # new "MAPC simulator" object
    simOP.timestamp_to_stop = sim_config['timestamp_to_stop']
    simOP.STA_queue_timeline = STAs_arrivals_matrix  # Loading the traffic dataset and assigning it to the STAs
    simOP.simulation_system = 'CSR'
    simOP.scheduler = 'OP'
    simOP.CGs_STAs = CGs_STAs
    simOP.TxPowerMatrix = TxPowerMatrix
    simOP.accessCategory = EDCAaccessCategory
    simOP.InitSettings()  # Initializing STAs
    simOP.Run()

    np.random.seed(seed)
    simTAT = MAPCsim(sim_config)  # new "MAPC simulator" object
    simTAT.timestamp_to_stop = sim_config['timestamp_to_stop']
    simTAT.STA_queue_timeline = STAs_arrivals_matrix  # Loading the traffic dataset and assigning it to the STAs
    simTAT.simulation_system = 'CSR'
    simTAT.scheduler = 'TAT'
    simTAT.CGs_STAs = CGs_STAs
    simTAT.TxPowerMatrix = TxPowerMatrix
    simTAT.accessCategory = EDCAaccessCategory
    simTAT.alpha = 0.5
    simTAT.beta = 0.5
    simTAT.InitSettings()  # Initializing STAs
    simTAT.Run()

    print(f'Iteration: {iter}')
    print(f'RL_A2C 50th {np.percentile(env.simulator.delayvector,50)*1000}')
    print(f'RL_A2C 99th {np.percentile(env.simulator.delayvector,99)*1000}')
    print(f'DCF 50th {np.percentile(simDCF.delayvector,50)*1000}')
    print(f'DCF 99th {np.percentile(simDCF.delayvector,99)*1000}')
    print(f'MNP 50th {np.percentile(simMNP.delayvector,50)*1000}')
    print(f'MNP 99th {np.percentile(simMNP.delayvector,99)*1000}')
    print(f'OP 50th {np.percentile(simOP.delayvector,50)*1000}')
    print(f'OP 99th {np.percentile(simOP.delayvector,99)*1000}')
    print(f'TAT 50th {np.percentile(simTAT.delayvector,50)*1000}')
    print(f'TAT 99th {np.percentile(simTAT.delayvector,99)*1000}')
    print('-----------------------------------------')

    return
    # return simDCF.delayvector, simMNP.delayvector, simOP.delayvector, simTAT.delayvector

def save_to_h5(output_dir, sim, traffic_type, traffic_load, ITERATIONS, DCFdelay, MNPdelay, OPdelay, TATdelay):
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
        f.create_dataset('DCFdelay', data=DCFdelay)
        # f.create_dataset('DCFdelay', data=DCFdelay, compression="gzip")   # with compression
        f.create_dataset('MNPdelay', data=MNPdelay)
        f.create_dataset('OPdelay', data=OPdelay)
        f.create_dataset('TATdelay', data=TATdelay)


# Main function
if __name__ == "__main__":

    # Start Timer
    start_time = time.time()

    ###### Input parameters
    validation_flag = 'no'

    # traffic_types = ['Poisson', 'Bursty', 'VR']
    # traffic_loads = {
    #     'Poisson': ['low', 'medium', 'high'],
    #     'Bursty': ['low', 'medium', 'high'],
    #     'VR': ['30-60', '30-90', '30-120']
    # }

    traffic_types = ['Bursty']
    traffic_loads = {
        'Bursty': ['high']
    }


    # Scenario-related
    AP_NUMBER = 4
    STA_NUMBER = 16
    GRID_VALUE = 60
    SCENARIO_TYPE = 'grid'

    sim = '30metros-16STAs'
    walls = np.array([[0, GRID_VALUE, GRID_VALUE/2, GRID_VALUE/2], 
                    [GRID_VALUE/2, GRID_VALUE/2, 0, GRID_VALUE]])

    # System-related parameters
    TXOP_DURATION = 5E-3
    PN_DBM = -95
    CCA = -82
    BW = 80
    NSS = 2
    L = 12E3

    ITERATIONS = 100

    ### Channel-related parameters
    MaxTxPower, NSC = TXpowerCalc(BW, NSS)


    # ### Load deployment data
    h5file_deployments_path = os.path.join(os.getcwd(), 'deployments datasets', sim, 'deployment_datasets.h5')
    with h5py.File(h5file_deployments_path, 'r') as f:
        STA_matrix_save = f['STA_matrix_save'][:]
        channelMatrix_save = f['channelMatrix_save'][:]
        RSSI_dB_vector_to_export_save = f['RSSI_dB_vector_to_export_save'][:]

    ### Output directory    
    output_dir = os.path.join(os.getcwd(), 'Results/Simulation')

    # Run simulations with progress bar
    max_workers = 1  # Adjust the number of workers as needed
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for traffic_type in traffic_types:
            for traffic_load in traffic_loads[traffic_type]:
                futures = [
                    executor.submit(
                        simulate_iterations, sim, traffic_type, traffic_load, i
                    )
                    for i in range(ITERATIONS)
                ]
                for i, future in enumerate(tqdm(futures, desc=f"{traffic_type} {traffic_load}", unit=" iterations")):
                    try:
                        # DCFdelay, MNPdelay, OPdelay, TATdelay = future.result()
                        future.result()

                        ### Uncoment to save the delay vectors into HDF5 files for each iterations, traffic type, and traffic load in a structured directory
                        # save_to_h5(output_dir, sim, traffic_type, traffic_load, i, DCFdelay, MNPdelay, OPdelay, TATdelay)
                    
                    except Exception as e:
                        print(f"Error in iterations {i} for {traffic_type} {traffic_load}: {e}")

    # End Timer and print elapsed time
    end_time = time.time()
    print(f"Simulation took {end_time - start_time:.2f} seconds")