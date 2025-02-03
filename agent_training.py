import os
import h5py
import numpy as np
from Utils import *
from MAPCsim import *
from CustomEnv import * # my Custom environment
from TrafficGenerator import TrafficGenerator
from DeploymentGenerator import deployment_generator

# RL Model (e.g., PPO)
import multiprocessing
from stable_baselines3 import PPO, TD3, DQN, A2C
from stable_baselines3.ppo import MlpPolicy, CnnPolicy, MultiInputPolicy 
from stable_baselines3.a2c import MlpPolicy, CnnPolicy, MultiInputPolicy
from stable_baselines3.dqn import MlpPolicy, CnnPolicy, MultiInputPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList

# For logging and monitoring
import wandb
from wandb.integration.sb3 import WandbCallback

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution

# From example notebook
import gymnasium as gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy



def agent_training(sim_config, learning_config, iter_number=None):
    """
    Simulates one iterations and returns the delay vectors for EDCA, MNP, OP, and TAT.

    Parameters:
    sim (str): Simulation identifier.
    traffic_type (str): Type of traffic (e.g., 'Poisson', 'Bursty', 'VR').
    traffic_load (str): Load of the traffic (e.g., 'low', 'medium', 'high').
    iter_number (int): Number of the current iteration.
    STA_matrix_save (np.ndarray): Pre-saved STA matrix for all iter_number.
    channelMatrix_save (np.ndarray): Pre-saved channel matrix for all iter_number.
    RSSI_dB_vector_to_export_save (np.ndarray): Pre-saved RSSI vector for all iter_number.

    Returns:

    """


    sim_config['EDCAaccessCategory'] = {'Poisson': 'BE', 'Bursty': 'BE', 'VR': 'VI'}.get(sim_config['traffic_type'], None)
    # Check if the traffic type is valid
    if sim_config['EDCAaccessCategory'] is None:
        raise ValueError(f"Invalid traffic type: {sim_config['traffic_type']}. Valid types are 'Poisson', 'Bursty', 'VR'.")

    # Deployment-dependent data
    AP_matrix, STA_matrix, sim_config['association'], sim_config['channelMatrix'] = deployment_generator(sim_config)

    if iter_number is not None:
        STA_matrix = STA_matrix_save[:,:,iter_number]
        channelMatrix = channelMatrix_save[:,:,iter_number]

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
                                              
    # Train the agent

    # Create log dir
    os.makedirs(learning_config['log_dir'], exist_ok=True)

    # Create a Gym-compatible environment
    def create_env():
        # Creating the simulator instance  

        simulator = MAPCsim(sim_config)  # new "MAPC simulator" object
        simulator.simulation_system = 'CSR'                 # 'EDCA' or 'CSR'
        simulator.CGs_STAs = map_matrix         # Entire groups matrix (all posible combinations)
        simulator.TxPowerMatrix = TxPowerMatrixTemp  # Entire Tx power matrix (all posible combinations)
        simulator.comb_ok = comb_ok # Combinations ok 
        simulator.datarate = datarate # Data rate for each combination
        simulator.accessCategory = sim_config['EDCAaccessCategory']  # Access category of devices in the network
        simulator.timestamp_to_stop = sim_config['learning_timestamp_to_stop'] # training episode duration

        # Creating the custom environment
        env = CustomEnv(sim_config, simulator)  
        # check_env(env)  # Check the environment
        env = Monitor(env, learning_config['log_dir'])  # Wrap the environment
        return env

    env = make_vec_env(create_env, n_envs=learning_config['parallel_envs'], vec_env_cls=SubprocVecEnv)   # vec_env_cls = DummyVecEnv or SubprocVecEnv

    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

    # Logging run
    logging_run = wandb.init(
        project="sb3",
        config=learning_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-Supload the videos of agents playing the game
        save_code=True,  # optional
    ) 


    model = MaskablePPO("MultiInputPolicy", env, verbose=0, tensorboard_log=os.path.join(learning_config['log_dir'], "tensorboard", logging_run.id), seed=sim_config['seed'])


    # maskeable_eval_callback = MaskableEvalCallback(env, best_model_save_path=os.path.join(learning_config['log_dir'], "models", logging_run.id),
    #                          log_path=learning_config['log_dir'], eval_freq=1000,
    #                          deterministic=False, render=False)
    
    # # Add WandbCallback
    # logging_callback = WandbCallback(
    #     gradient_save_freq=1000*learning_config['parallel_envs'],
    #     model_save_path=os.path.join(learning_config['log_dir'], "models", logging_run.id),
    #     model_save_freq=1000*learning_config['parallel_envs'],
    #     verbose=2,
    # )

    # callbacklist = CallbackList([logging_callback, maskeable_eval_callback])

    # Train the model
    model.learn(
        total_timesteps=learning_config['num_episodes'],
        callback=None
    )

    # # Save the model after training
    model.save(os.path.join(learning_config['log_dir'], "models", logging_run.id, "final_model.zip"))

    logging_run.finish()






if __name__ == '__main__':

    ###### Input parameters
    validation_flag = 'no'
    traffic_type = 'Bursty'
    traffic_load = 'high'



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

    ITERATIONS = 1

    ### Channel-related parameters
    MaxTxPower, NSC = TXpowerCalc(BW, NSS)


    # ### Load deployment data
    h5file_deployments_path = os.path.join(os.getcwd(),'deployments datasets', sim, 'deployment_datasets.h5')
    with h5py.File(h5file_deployments_path, 'r') as f:
        STA_matrix_save = f['STA_matrix_save'][:]
        channelMatrix_save = f['channelMatrix_save'][:]


    # Simulation Configuration
    sim_config = {
        'AP_NUMBER': AP_NUMBER,
        'STA_NUMBER': STA_NUMBER,
        'SCENARIO_TYPE': SCENARIO_TYPE,
        'GRID_VALUE': GRID_VALUE,
        'walls': walls,
        'MaxTxPower': MaxTxPower,
        'traffic_type': traffic_type,
        'traffic_load' : traffic_load,
        'validation_flag': 'no',
        'TXOP_DURATION': TXOP_DURATION,
        'PN_DBM': PN_DBM,
        'CCA': CCA,
        'NSS': NSS,
        'NSC': NSC,
        'learning_timestamp_to_stop': 2, # seconds
        'training_flag': True,
        'timestamp_to_stop': 5, # seconds
        'L': L,
        'EVENT_NUMBER': 30000, # Number of events considered for traffic generation
        'seed': 1
    }

    # Learning Configuration
    learning_config = {
        'log_dir': os.path.join(os.getcwd(),'trained_models'),
        'parallel_envs': 1,
        'num_episodes': 4E6,
        'simulator_attr' : 'simulator',
    }  
    
    # Simulate the iterations
    agent_training(sim_config, learning_config, iter_number=0)

