
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
from DeploymentGenerator import deployment_generator



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
from stable_baselines3.common.env_util import make_vec_env

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback





def simulate_iterations(sim_config, learning_config, iter_number=None):
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

    sim_config['EDCAaccessCategory'] = {'Poisson': 'BE', 'Bursty': 'BE', 'CBR': 'VI'}.get(sim_config['traffic_type'], None)
    # Check if the traffic type is valid
    if sim_config['EDCAaccessCategory'] is None:
        raise ValueError(f"Invalid traffic type: {sim_config['traffic_type']}. Valid types are 'Poisson', 'Bursty', 'CBR'.")


    # Deployment-dependent data
    AP_matrix, STA_matrix, sim_config['association'], sim_config['channelMatrix'] = deployment_generator(sim_config)

    if iter_number is not None:
        STA_matrix = STA_matrix_save[:,:,iter_number]
        sim_config['channelMatrix'] = channelMatrix_save[:,:,iter_number]

    # Overheads
    sim_config['preTX_overheadsEDCA'], sim_config['preTX_overheadsCSR'], sim_config['EDCAoverheads'], sim_config['CSRoverheads'] = OverheadsCalc(sim_config['EDCAaccessCategory'])

    sim_config['per_STA_EDCA_throughput_bianchi'] = Throughput_EDCA_bianchi(sim_config['AP_NUMBER'], sim_config['STA_NUMBER'], sim_config['association'], sim_config['channelMatrix'], sim_config['MaxTxPower'],
                                                            sim_config['PN_DBM'], sim_config['NSC'], sim_config['NSS'], sim_config['TXOP_DURATION'], 
                                                            sim_config['EDCAoverheads'], sim_config['EDCAaccessCategory'])

    map_matrix, TxPowerMatrixTemp, comb_ok, datarate = CG_creationTPC(sim_config['AP_NUMBER'], 
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

    # ### Traffic dataset
    # # Load the traffic dataset. Uncomment if using pre-saved dataset.
    # h5_file_path = os.path.join(os.getcwd(), 'traffic datasets', sim, traffic_type, traffic_load, f"STAs_arrivals_matrix{iter_number}.h5")
    # # Open and load the dataset
    # with h5py.File(h5_file_path, 'r') as h5file:
    #   STAs_arrivals_matrix = np.array([h5file[key][:] for key in h5file.keys()])

    # # # Generate the traffic dataset for the current value of ITERATIONS. Comment if using pre-saved dataset.
    STAs_arrivals_matrix = TrafficGenerator(
            sim_config['STA_NUMBER'], 
            sim_config['validation_flag'], 
            sim_config['traffic_type'], 
            sim_config['traffic_load'], 
            sim_config['L'], 
            sim_config['per_STA_EDCA_throughput_bianchi'], 
            sim_config['EVENT_NUMBER']# Number of events considered for traffic generation
            ) 


    # Create a Gym-compatible environment
    def create_env():
        # Creating the simulator instance  

        simulator = MAPCsim(sim_config)  # new "MAPC simulator" object
        simulator.simulation_system = 'CSR'                 # 'EDCA' or 'CSR'
        simulator.CGs_STAs = map_matrix         # Entire groups matrix (all posible combinations)
        simulator.TxPowerMatrix = TxPowerMatrixTemp  # Entire Tx power matrix (all posible combinations)
        simulator.comb_ok = comb_ok # Combinations ok 
        simulator.datarate = datarate # Data rate for each combination (proportional tx rate)
        simulator.accessCategory = sim_config['EDCAaccessCategory']  # Access category of devices in the network
        simulator.timestamp_to_stop = sim_config['learning_timestamp_to_stop'] # training episode duration

        # Creating the custom environment
        env = CustomEnv(sim_config, simulator)  
        # check_env(env)  # Check the environment
        env.reset(seed=sim_config['seed'])
        # env = Monitor(env, learning_config['log_dir'])  # Wrap the environment
        return env
    
    # env = make_vec_env(create_env, n_envs=learning_config['parallel_envs'], vec_env_cls=DummyVecEnv)   # vec_env_cls = DummyVecEnv or SubprocVecEnv
    env = create_env()
    obs, _ = env.reset(seed=sim_config['seed'], STAs_arrivals_matrix=STAs_arrivals_matrix)

    terminated = False

    # Load the trained model
    loaded_model = MaskablePPO.load(os.path.join(learning_config['log_dir'], "models/qxm9k1l9/final_model.zip"), env=env)
    while not terminated:
        action_masks = env.action_masks()
        action, _states = loaded_model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, _, terminated, _, _ = env.step(action)
        # env.render()  # Optionally visualize the environment

    env.simulator.TrafficAnalysis()



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
    print(f'RL_PPO 50th {np.percentile(env.simulator.delayvector,50)*1000}')
    print(f'RL_PPO 99th {np.percentile(env.simulator.delayvector,99)*1000}')
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


# Main function
if __name__ == "__main__":

    # Start Timer
    start_time = time.time()

    ###### Input parameters
    validation_flag = 'no'

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
    
    traffic_profiles = {
        'A' : {'model': 'Poisson', 'bitrate' : 100, 'latency': 1E-4},
        'B' : {'model': 'Bursty', 'bitrate' : 50, 'latency': 2E-4},
        'C' : {'model': 'CBR', 'bitrate' : 25, 'latency': 5E-4}
    }

    
    # Simulation Configuration
    sim_config = {
        'AP_NUMBER': AP_NUMBER,
        'STA_NUMBER': STA_NUMBER,
        'SCENARIO_TYPE': SCENARIO_TYPE,
        'GRID_VALUE': GRID_VALUE,
        'walls': walls,
        'MaxTxPower': MaxTxPower,
        'validation_flag': 'no',
        'TXOP_DURATION': TXOP_DURATION,
        'PN_DBM': PN_DBM,
        'CCA': CCA,
        'NSS': NSS,
        'NSC': NSC,
        'learning_timestamp_to_stop': 2, # seconds
        'training_flag': False,
        'timestamp_to_stop': 5, # seconds
        'L': L,
        'EVENT_NUMBER': 30000, # Number of events considered for traffic generation
        'seed': 1
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
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for traffic_type in traffic_types:
            sim_config['traffic_type'] = traffic_type
            for traffic_load in traffic_loads[traffic_type]:
                sim_config['traffic_load'] = traffic_load
                futures = [
                    executor.submit(
                        simulate_iterations, sim_config, learning_config, i
                    )
                    for i in range(ITERATIONS)
                ]
                for i, future in enumerate(tqdm(futures, desc=f"{traffic_type} {traffic_load}", unit=" iterations")):
                    try:
                        # EDCAdelay, MNPdelay, OPdelay, TATdelay = future.result()
                        future.result()

                        ### Uncoment to save the delay vectors into HDF5 files for each iterations, traffic type, and traffic load in a structured directory
                        # save_to_h5(output_dir, sim, traffic_type, traffic_load, i, EDCAdelay, MNPdelay, OPdelay, TATdelay)
                    
                    except Exception as e:
                        print(f"Error in iterations {i} for {traffic_type} {traffic_load}: {e}")

    # End Timer and print elapsed time
    end_time = time.time()
    print(f"Simulation took {end_time - start_time:.2f} seconds")