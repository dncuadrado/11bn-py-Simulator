import os
import h5py
import numpy as np
from Utils import *
from MAPCsim import *
from TrafficGenerator import TrafficGenerator
from SaveOnBestTrainingRewardCallback import SaveOnBestTrainingRewardCallback

# RL Model (e.g., PPO)
from CustomEnv import * # my Custom environment
from stable_baselines3 import PPO, TD3, DQN, A2C
from stable_baselines3.ppo import MlpPolicy, CnnPolicy, MultiInputPolicy 
from stable_baselines3.a2c import MlpPolicy, CnnPolicy, MultiInputPolicy
from stable_baselines3.dqn import MlpPolicy, CnnPolicy, MultiInputPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList


# From example notebook
import gymnasium as gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise


def agent_training(sim_config, learning_config):

    # Create log dir
    os.makedirs(learning_config['log_dir'], exist_ok=True)

    CGs_STAs1, TxPowerMatrix1 = CG_creationTPC(sim_config['AP_NUMBER'], 
                                            sim_config['STA_NUMBER'], 
                                            sim_config['PN_DBM'], 
                                            sim_config['NSC'], 
                                            sim_config['NSS'], 
                                            sim_config['association'], 
                                            sim_config['channelMatrix'], 
                                            sim_config['MaxTxPower'], 
                                            CG_filter='off', TPC_method='PSO')    # TPC Optimization method: None, 'PSO', 'IPOPT', 'DE'

    # Create a Gym-compatible environment
    def create_env():
        # Creating the simulator instance  

        simulator = MAPCsim(sim_config)  # new "MAPC simulator" object
        simulator.simulation_system = 'CSR'                 # 'DCF' or 'CSR'
        simulator.CGs_STAs = CGs_STAs1         # Coordinated Spatial Reuse groups
        simulator.TxPowerMatrix = TxPowerMatrix1  # Tx power matrix
        simulator.accessCategory = sim_config['EDCAaccessCategory']  # Access category of devices in the network
        simulator.timestamp_to_stop = sim_config['learning_timestamp_to_stop'] # training episode duration

        # Creating the custom environment
        env = CustomEnv(sim_config, simulator)  
        env.reset(seed=sim_config['seed'])
        env = Monitor(env, learning_config['log_dir'])  # Wrap the environment
        return env

    # env = Monitor(create_env(), learning_config['log_dir'])  # Wrap the environment
    # env = DummyVecEnv([create_env for i in range(learning_config['parallel_envs'])])
    env = SubprocVecEnv([create_env for i in range(learning_config['parallel_envs'])])
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

    # # Initialize PPO with fine-tuned parameters
    # model = PPO(
    #     "MlpPolicy",
    #     # "MultiInputPolicy",
    #     env,
    #     # normalize_advantage=True,
    #     learning_rate=3e-4,
    #     # learning_rate=1e-3,
    #     n_steps=2048,
    #     batch_size=64,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     ent_coef=0.01,
    #     policy_kwargs=policy_kwargs,
    #     verbose=0,
    # )

    # model = DQN('MlpPolicy', env, verbose=0, max_grad_norm=0.5, buffer_size=50000, learning_starts=1000,
    #         gamma=0.99, target_update_interval=100, exploration_fraction=0.1, exploration_final_eps=0.01,
    #         batch_size=64, train_freq=4, learning_rate=1e-4)

    model = PPO("MlpPolicy", env, verbose=0)
    # model = PPO("MlpPolicy", env, verbose=0)
    # model = PPO("MlpPolicy", env, verbose=0)

    # callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=learning_config['log_dir'])

    eval_callback = EvalCallback(env, best_model_save_path=learning_config['log_dir'],
                             log_path=learning_config['log_dir'], eval_freq=500,
                             deterministic=False, render=False)
    
    # Train the model
    model.learn(
        total_timesteps=learning_config['num_episodes'],
        callback=eval_callback
    )
    # Save the model after training
    model.save(os.path.join(learning_config['log_dir'], "./final_model_delay/"))

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

    EDCAaccessCategory = 'BE'
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
    
    # # # Simulation duration
    # timestamp_to_stop = 5  # seconds

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
        'training_flag': True,
        'timestamp_to_stop': 5, # seconds
        'CGs_STAs': CGs_STAs,
        'TxPowerMatrix': TxPowerMatrix,
        'L': L,
        'per_STA_DCF_throughput_bianchi': per_STA_DCF_throughput_bianchi,
        'EVENT_NUMBER': 30000, # Number of events considered for traffic generation
        'seed': seed
    }

    # Learning Configuration
    learning_config = {
        'log_dir': '/home/david/Documents/Papers/journal_ML_CSR/python Code/trained_models',
        'parallel_envs': 8,
        'num_episodes': 2E6,
        'simulator_attr' : 'simulator',
    }
    
    # Train the agent
    agent_training(sim_config, learning_config)



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
    h5file_deployments_path = os.path.join('/home/david/Documents/Papers/journal_ML_CSR/python Code/deployments datasets', sim, 'deployment_datasets.h5')
    with h5py.File(h5file_deployments_path, 'r') as f:
        STA_matrix_save = f['STA_matrix_save'][:]
        channelMatrix_save = f['channelMatrix_save'][:]
        RSSI_dB_vector_to_export_save = f['RSSI_dB_vector_to_export_save'][:]

    ### Output directory    
    output_dir = os.path.join(os.getcwd(), 'Results/Simulation')

    iter = 0

    # Simulate the iterations
    simulate_iterations(sim, traffic_type, traffic_load, iter)

