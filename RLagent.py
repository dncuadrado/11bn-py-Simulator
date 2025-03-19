import os
import h5py
import numpy as np
from numpy.random import SeedSequence
import time
import re
import Utils as utils
from MAPCsim import *
from CustomEnv import * # my Custom environment
from TrafficGenerator import traffic_generator
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

def create_env(traffic_config, sim_config, map_matrix, TxPowerMatrixTemp, comb_ok, datarate, seed, monitor_gym=False):
    # Creating the simulator instance  

    simulator = MAPCsim(sim_config)  # new "MAPC simulator" object
    simulator.simulation_system = 'CSR'                 # 'EDCA' or 'CSR'
    simulator.CGs_STAs = map_matrix         # Entire groups matrix (all posible combinations)
    simulator.TxPowerMatrix = TxPowerMatrixTemp  # Entire Tx power matrix (all posible combinations)
    simulator.comb_ok = comb_ok # Combinations ok 
    simulator.datarate = datarate # Data rate for each combination (proportional tx rate)
    simulator.accessCategory = traffic_config['EDCAaccessCategory']  # Access category of devices in the network
    simulator.timestamp_to_stop = sim_config['learning_timestamp_to_stop'] # training episode duration

    # Creating the custom environment
    env = CustomEnv(traffic_config, sim_config, simulator)  
    # check_env(env)  # Check the environment
    env.reset(seed=seed)  # Reset
    if monitor_gym:
        env = Monitor(env, learning_config['log_dir'])  # Wrap the environment
    return env

def training(traffic_config, sim_config, learning_config, iter_number=None):
    """
    Train the RL agent using the specified configuration
    """

    seed = sim_config['seed']

    # Set the seed
    np.random.seed(seed)

    # Deployment-dependent data
    _, _, sim_config['association'], sim_config['channelMatrix'] = deployment_generator(sim_config, seed)

    if sim_config['use_preloaded_deployments']:
        # ### Load deployment data
        h5file_deployments_path = os.path.join(os.getcwd(),'deployments datasets', sim, 'deployment_datasets.h5')
        with h5py.File(h5file_deployments_path, 'r') as f:
            STA_matrix_save = f['STA_matrix_save'][:]
            channelMatrix_save = f['channelMatrix_save'][:]

        STA_matrix = STA_matrix_save[:, :, iter_number]
        sim_config['channelMatrix'] = channelMatrix_save[:, :, iter_number]

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
                                              
    # Train the agent

    # Create log dir
    os.makedirs(learning_config['log_dir'], exist_ok=True)

    env = make_vec_env(
        lambda: create_env(traffic_config, sim_config, map_matrix, TxPowerMatrixTemp, comb_ok, datarate, seed=np.random.randint(1, int(1E8)), monitor_gym=True), 
        n_envs=learning_config['parallel_envs'], 
        vec_env_cls=SubprocVecEnv   # vec_env_cls = DummyVecEnv or SubprocVecEnv
    )

    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

    # Logging run
    logging_run = wandb.init(
        project="sb3",
        config=learning_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-Supload the videos of agents playing the game
        save_code=True,  # optional
    ) 


    model = MaskablePPO("MlpPolicy", env, verbose=0, tensorboard_log=os.path.join(learning_config['log_dir'], "tensorboard", logging_run.id), seed=sim_config['seed'])


    # maskeable_eval_callback = MaskableEvalCallback(env, best_model_save_path=os.path.join(learning_config['log_dir'], "models", logging_run.id),
    #                          log_path=learning_config['log_dir'], eval_freq=1000,
    #                          deterministic=False, render=False)
    
    # Add WandbCallback
    logging_callback = WandbCallback(
        gradient_save_freq=1000*learning_config['parallel_envs'],
        model_save_path=os.path.join(learning_config['log_dir'], "models", logging_run.id),
        model_save_freq=1000*learning_config['parallel_envs'],
        verbose=2,
    )

    # callbacklist = CallbackList([logging_callback, maskeable_eval_callback])

    # Train the model
    model.learn(
        total_timesteps=learning_config['num_episodes'],
        callback=logging_callback
    )

    # # Save the model after training
    model.save(os.path.join(learning_config['log_dir'], "models", logging_run.id, "final_model.zip"))

    logging_run.finish()


def evaluation(map_matrix, TxPowerMatrixTemp, comb_ok, datarate, STAs_arrivals_matrix, sim_config, learning_config, model):
        # Create a Gym-compatible environment

    
    # env = make_vec_env(create_env, n_envs=learning_config['parallel_envs'], vec_env_cls=DummyVecEnv)   # vec_env_cls = DummyVecEnv or SubprocVecEnv
    env = create_env(sim_config, map_matrix, TxPowerMatrixTemp, comb_ok, datarate, monitor_gym=False)
    obs, _ = env.reset(seed=sim_config['seed'], STAs_arrivals_matrix=STAs_arrivals_matrix)

    terminated = False

    # Load the trained model
    loaded_model = MaskablePPO.load(os.path.join(learning_config['log_dir'], "models", model,  "final_model.zip"), env=env)
    while not terminated:
        action_masks = env.action_masks()
        action, _states = loaded_model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, _, terminated, _, _ = env.step(action)

    return env


if __name__ == '__main__':

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

    # Traffic Configuration 
    traffic_config = {
        'traffic_profiles': {    # Define the traffic profiles
            'A' : {'traffic_model': 'Poisson', 'traffic_load' : 100, 'latency': 1E-4},
            'B' : {'traffic_model': 'Bursty', 'traffic_load' : 50, 'latency': 2E-4},
            'C' : {'traffic_model': 'CBR', 'traffic_load' : 25, 'fps': 60, 'latency': 5E-4}  # not used by now
    },
        'EDCAaccessCategory' : 'BE'
    }  

    # Simulation Configuration
    sim_config = {
        'use_preloaded_deployments': True,
        'use_preloaded_traffic': False,
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
        'training_flag': True,
        'timestamp_to_stop': 5, # seconds
        'FRAME_LENGTH': FRAME_LENGTH,
        'EVENT_NUMBER': int(3E4), # Number of events considered for traffic generation
        'seed': 1,
        'output_dir': os.path.join(os.getcwd(), 'Results', sim),
        'overheads' : utils.OverheadsCalc(traffic_config['EDCAaccessCategory'])   
    }

    # Learning Configuration
    learning_config = {
        'log_dir': os.path.join(os.getcwd(),'trained_models'),
        'parallel_envs': 8,
        'num_episodes': 4E6,
        'simulator_attr' : 'simulator',
    }  
    
    # Simulate the iterations
    training(traffic_config, sim_config, learning_config, iter_number=0)



