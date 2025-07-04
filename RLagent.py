import os
import h5py
import numpy as np
from numpy.random import SeedSequence
import time
import re
import math
from typing import Callable
import Utils as utils
from MAPCsim import *
from CustomEnv import * # my Custom environment
from TrafficGenerator import traffic_generator
from DeploymentGenerator import deployment_generator
from custom_feature_extractor import FiLMExtractor1, FiLMExtractor2, FiLMExtractor3, FiLMExtractor4, ProgressiveFiLMExtractor, SharedMLPExtractor, SharedMLPWithAttentionExtractor

# RL Model (e.g., PPO)
import torch as th
from torch import nn
import multiprocessing

# from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.ppo import MlpPolicy, CnnPolicy, MultiInputPolicy 
from stable_baselines3.a2c import MlpPolicy, CnnPolicy, MultiInputPolicy
from stable_baselines3.dqn import MlpPolicy, CnnPolicy, MultiInputPolicy


from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, CallbackList, StopTrainingOnNoModelImprovement

# For logging and monitoring
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback

from sb3_contrib import MaskablePPO, TRPO
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution

# From example notebook
import gymnasium as gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

def parse_args_from_slurm():
    """
    Parse arguments from command line
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='sb3-HPC-NEW')
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--n_steps', type=int, default=128)                 # default --- 2048    128               
    parser.add_argument('--batch_size', type=int, default=256)              # default --- 64        64
    parser.add_argument('--initial_lr', type=float, default=6.5E-4)       # default --- 3e-4              
    parser.add_argument('--learning_decay', type=str, default='cosine') # default --- 'cosine', 'linear', 'exp', 'square'
    parser.add_argument('--gamma', type=float, default=0.99)               # default --- 0.99
    parser.add_argument('--gae_lambda', type=float, default=0.95)          # default --- 0.95
    parser.add_argument('--clip_range', type=float, default=0.2)           # default --- 0.2
    parser.add_argument('--episode_threshold', type=int, default=0) 


    args = parser.parse_args()

    return vars(args)


def schedule_clip_range(
    clip_min_phase1: float = 0.1,
    clip_max: float        = 0.4,
    clip_min_phase2: float = 0.1,
    switch_point: float    = 0.5,  # when to switch from phase1 → phase2
) -> Callable[[float], float]:
    """
    Returns a schedule function that:
      - Increases clip_range from clip_min_phase1 → clip_max as 
        progress_remaining goes from 1.0 → switch_point
      - Decreases clip_range from clip_max → clip_min_phase2 as
        progress_remaining goes from switch_point → 0.0
    """
    def schedule(progress_remaining: float) -> float:
        # Phase 1: [0.0, switch_point]  →  clip_min_phase1 → clip_max
        if progress_remaining > switch_point:
            frac = (1.0 - progress_remaining) / (1.0 - switch_point)
            return clip_min_phase1 + (clip_max - clip_min_phase1) * frac
        # Phase 2: [switch_point, 1.0] → clip_max → clip_min_phase2
        else:
            frac = progress_remaining / switch_point
            return clip_min_phase2 + (clip_max - clip_min_phase2) * frac
    return schedule

def schedule_learning_rate(initial_lr, learning_decay=None):
    def schedule(progress_remaining):
        if learning_decay is None:
            return initial_lr
        elif learning_decay == 'linear':
            return initial_lr * progress_remaining
        elif learning_decay == 'cosine':
            return initial_lr * 0.5 * (1 + math.cos(math.pi * (1 - progress_remaining)))
        elif learning_decay == 'square':
            return initial_lr * (progress_remaining)**2
        elif learning_decay == 'exp':
            return initial_lr * np.exp(-1.38 * (1-progress_remaining))     # k=1.38 ---> log(1/2)*2  to achieve 50% decay at 50% of the training
        else:
            raise ValueError(f"Unknown decay type: {learning_decay}")
    return schedule

def create_env(traffic_config, sim_config, learning_config, channel_matrix, map_matrix, TxPowerMatrixTemp, comb_ok, seed, monitor_gym=False, training_flag=True):

    # Create copy of config to avoid mutation issues
    env_sim_config = sim_config.copy()
    env_sim_config["training_flag"] = training_flag  # Set flag per environment

    # Creating the simulator instance  

    simulator = MAPCsim(sim_config)  # new "MAPC simulator" object
    simulator.simulation_system = 'RL'                 # 'EDCA' or 'CSR'
    simulator.channel_matrix = channel_matrix
    simulator.CGs_STAs = map_matrix         # Entire groups matrix (all posible combinations)
    simulator.TxPowerMatrix = TxPowerMatrixTemp  # Entire Tx power matrix (all posible combinations)
    simulator.comb_ok = comb_ok # Combinations ok 
    simulator.accessCategory = traffic_config['EDCAaccessCategory']  # Access category of devices in the network
    simulator.timestamp_to_stop = sim_config['learning_timestamp_to_stop'] # training episode duration

    # Creating the custom environment
    env = CustomEnv(traffic_config, env_sim_config, learning_config, simulator)  
    # check_env(env)  # Check the environment
    if monitor_gym:
        env = Monitor(env, learning_config['log_dir'], info_keywords=('total_percentile99', 'worst_percentile99','mean_rew_shaping', 'mean_long_term_rew','mean_reward'))  # Wrap the environment
    return env

def training(traffic_config, sim_config, learning_config, iter_number=None):
    """
    Train the RL agent using the specified configuration
    """

    seed = sim_config['seed']

    # Set the seed
    np.random.seed(seed)

    # Deployment-dependent data
    AP_matrix, STA_matrix, sim_config['association'], channel_matrix = deployment_generator(sim_config, seed)

    if sim_config['use_preloaded_deployments']:
        # ### Load deployment data
        h5file_deployments_path = os.path.join(os.getcwd(),'deployments datasets', sim, 'deployment_datasets.h5')
        with h5py.File(h5file_deployments_path, 'r') as f:
            STA_matrix_save = f['STA_matrix_save'][:]
            channelMatrix_save = f['channelMatrix_save'][:]

        STA_matrix = STA_matrix_save[:, :, iter_number]
        channel_matrix = channelMatrix_save[:, :, iter_number]

    # utils.PlotDeployment(AP_matrix, STA_matrix, sim_config['association'], sim_config['GRID_VALUE'], sim_config['walls'])

    # Compute the CGs and TxPowerMatrix
    map_matrix, TxPowerMatrixTemp, comb_ok = utils.CG_creationTPC(sim_config['AP_NUMBER'], 
                                                sim_config['STA_NUMBER'], 
                                                sim_config['PN_DBM'], 
                                                sim_config['NSC'], 
                                                sim_config['NSS'], 
                                                sim_config['association'], 
                                                channel_matrix, 
                                                sim_config['MaxTxPower'], 
                                                is_filtering=sim_config['filtering'], TPC_method=None, # TPC Optimization method: None, 'PSO', 'IPOPT', 'DE'
                                                CG_size=4)  
    # map_matrix.append(None)
    # TxPowerMatrixTemp.append(None)    
    # comb_ok = np.append(comb_ok, True)

    # Train the agent

    # Create log dir
    os.makedirs(learning_config['log_dir'], exist_ok=True)

    # Create training environments with training_flag=True
    train_env = make_vec_env(
        lambda: create_env(
            traffic_config, sim_config, learning_config, 
            channel_matrix, map_matrix, TxPowerMatrixTemp, comb_ok,
            seed=np.random.randint(1, int(1E8)), 
            monitor_gym=True,
            training_flag=True  # Explicit training flag
        ),
        n_envs=learning_config['parallel_envs'],
        vec_env_cls=SubprocVecEnv
    ) 

    # train_env = VecFrameStack(train_env, n_stack=10)

    # # Create evaluation environments with training_flag=False
    # eval_env = make_vec_env(
    #     lambda: create_env(
    #         traffic_config, sim_config, learning_config,
    #         channel_matrix, map_matrix, TxPowerMatrixTemp, comb_ok,
    #         seed=np.random.randint(1, int(1E8)),
    #         training_flag=False  # Explicit evaluation flag
    #     ),
    #     n_envs=learning_config['parallel_envs'],  # Typically fewer envs for evaluation
    #     vec_env_cls=SubprocVecEnv
    # )

    # Start W&B run 
    logging_run = wandb.init(project=learning_config['project_name'], 
                    id=learning_config['run_id'], 
                    sync_tensorboard=True,
                    monitor_gym=False,
                    save_code=True,
                    )

    

    learning_rate_scheduled = schedule_learning_rate(
        initial_lr=learning_config['initial_lr'],
        learning_decay=learning_config['learning_decay']
    )

    clip_range_scheduled = schedule_clip_range(clip_min_phase1=0.1,
                                                clip_max=0.3,
                                                clip_min_phase2=0.2,
                                                switch_point=0.5
                                            )
    
    # model_name = 'n505nvdg'       
    # model = PPO.load(os.path.join(learning_config['log_dir'], "models", model_name, "model_5401600_steps.zip"), env=train_env)

    # policy_kwargs = dict(
    #     features_extractor_class=FiLMExtractor2,
    #     features_extractor_kwargs={"dyn_dim": sim_config['STA_NUMBER']*2, "stat_dim": sim_config['STA_NUMBER']},
    #     net_arch=[128, 128],   # now only one “body”
    # )

    # policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[64, 64])) 

    # policy_kwargs = dict(
    #     activation_fn=nn.LeakyReLU,  # Or nn.Tanh, nn.SiLU, etc.
    #     net_arch=dict(pi=[128, 128], vf=[128, 128]),
    #     share_features_extractor=True,  # Share the feature extractor between actor and critic
    # )

    # policy_kwargs = {
    #     "features_extractor_class": SharedMLPWithAttentionExtractor,
    #     "features_extractor_kwargs": {
    #         "sta_number": sim_config['STA_NUMBER'],
    #         "ap_number": sim_config['AP_NUMBER'],
    #         "hidden_dim": 64
    #     },
    # }

    model = MaskablePPO("MlpPolicy", 
                        train_env, 
                        verbose=0, 
                        tensorboard_log=os.path.join(learning_config['log_dir'], "tensorboard", logging_run.id), 
                        seed=sim_config['seed'],
                        n_steps=learning_config['n_steps'],
                        batch_size=learning_config['batch_size'],
                        learning_rate=learning_rate_scheduled,
                        # clip_range=clip_range_scheduled,
                        gamma=learning_config['gamma'],
                        gae_lambda=learning_config['gae_lambda'],
                        # policy_kwargs=policy_kwargs,
                        # ent_coef=0.02,
                        # clip_range=0.15,
                        )
    
    # model = PPO("MlpPolicy", 
    #                     train_env, 
    #                     verbose=0, 
    #                     tensorboard_log=os.path.join(learning_config['log_dir'], "tensorboard", logging_run.id),
    #                     device="cpu", 
    #                     seed=sim_config['seed'],
    #                     n_steps=learning_config['n_steps'],
    #                     batch_size=learning_config['batch_size'],
    #                     learning_rate=learning_rate_scheduled,
    #                     clip_range=clip_range_scheduled,
    #                     gamma=learning_config['gamma'],
    #                     gae_lambda=learning_config['gae_lambda'],
    #                     # policy_kwargs=policy_kwargs,
    #                     # ent_coef=0.02,
    #                     # clip_range=0.15,
    #                     )



    # model = TRPO("MlpPolicy", 
    #                     train_env, 
    #                     verbose=0, 
    #                     tensorboard_log=os.path.join(learning_config['log_dir'], "tensorboard", logging_run.id), 
    #                     seed=sim_config['seed'],
    #                     n_steps=learning_config['n_steps'],
    #                     batch_size=learning_config['batch_size'],
    #                     learning_rate=learning_rate_scheduled,
    #                     # clip_range=clip_range_scheduled,
    #                     gamma=learning_config['gamma'],
    #                     gae_lambda=learning_config['gae_lambda'],
    #                     policy_kwargs=policy_kwargs,
    #                     # ent_coef=0.02,
    #                     # clip_range=0.15,
    #                     )
    
    # model = A2C("MlpPolicy", 
    #                     train_env, 
    #                     verbose=0, 
    #                     tensorboard_log=os.path.join(learning_config['log_dir'], "tensorboard", logging_run.id), 
    #                     seed=sim_config['seed'],
    #                     learning_rate=learning_rate_scheduled,
    #                     policy_kwargs=policy_kwargs,
    #                     )

    # model = ACKTR("MlpPolicy", 
    #                 train_env, 
    #                 verbose=0, 
    #                 tensorboard_log=os.path.join(learning_config['log_dir'], "tensorboard", logging_run.id), 
    #                 seed=sim_config['seed'],
    #                 learning_rate=0.25,
    #                 # policy_kwargs=policy_kwargs,
    #                 )

    wandb.config.update({
        "n_steps": learning_config['n_steps'],
        "batch_size": learning_config['batch_size'],
        "initial_lr": learning_config['initial_lr'],
        "learning_decay": learning_config['learning_decay'],
        "gamma": learning_config['gamma'],
        "gae_lambda": learning_config['gae_lambda'],
        "clip_range": learning_config['clip_range'],
        "episode_threshold": learning_config['episode_threshold'],
    })
    
    ### Add WandbCallback
    logging_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=os.path.join(learning_config['log_dir'], "models", logging_run.id),
        model_save_freq=100,
        verbose=2,
    )

    if sim_config['save_model']:
        # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=25, min_evals=20, verbose=1)

        # eval_callback = MaskableEvalCallback(eval_env, best_model_save_path=os.path.join(learning_config['log_dir'], "models", logging_run.id),
        #                         log_path=learning_config['log_dir'], eval_freq=5120,
        #                         deterministic=False, render=False,
        #                         callback_after_eval=stop_train_callback,
        #                         n_eval_episodes= 10, # Number of parallel environments for evaluation
        #                      )
        
        # eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(learning_config['log_dir'], "models", logging_run.id),
        #                         log_path=learning_config['log_dir'], eval_freq=5120,
        #                         deterministic=False, render=False,
        #                         callback_after_eval=stop_train_callback,
        #                         n_eval_episodes= 10, # Number of parallel environments for evaluation
        #                      )
        
        # Save a checkpoint every 1000 steps
        checkpoint_callback = CheckpointCallback(
        save_freq=2560,   # 
        save_path=os.path.join(learning_config['log_dir'], "models", logging_run.id),
        name_prefix="model",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=2,
        )

        callbacklist = CallbackList([checkpoint_callback, logging_callback])
    else:
        callbacklist = logging_callback
    

    


    # Train the model
    model.learn(
        total_timesteps=learning_config['total_timesteps'],
        callback=callbacklist,
        reset_num_timesteps=False
    )

    # Save the model after training
    model.save(os.path.join(learning_config['log_dir'], "models", logging_run.id, "final_model.zip"))

    logging_run.finish()


def evaluation(traffic_config, sim_config, learning_config, channel_matrix, map_matrix, TxPowerMatrixTemp, comb_ok, STAs_arrivals_matrix, traffic_profile_perSTA, model):
    """
    Evaluate the trained model using the specified configuration
    """

    # env = make_vec_env(create_env, n_envs=learning_config['parallel_envs'], vec_env_cls=DummyVecEnv)   # vec_env_cls = DummyVecEnv or SubprocVecEnv
    env = create_env(traffic_config, sim_config, learning_config, channel_matrix, map_matrix, TxPowerMatrixTemp, comb_ok, sim_config['seed'], monitor_gym=False, training_flag=False)
    obs, _ = env.reset(seed=sim_config['seed'], STAs_arrivals_matrix=STAs_arrivals_matrix, traffic_profile_perSTA=traffic_profile_perSTA, is_deloyment_fixed=True)  # Reset the environment

    terminated = False

    # Load the trained model
    loaded_model = MaskablePPO.load(os.path.join(learning_config['log_dir'], "models", model['model_id'],  model['model_type']), env=env)
    # loaded_model = TRPO.load(os.path.join(learning_config['log_dir'], "models", model['model_id'],  model['model_type']), env=env)
    while not terminated:
        action_masks = env.action_masks()
        action, _states = loaded_model.predict(obs, action_masks=action_masks, deterministic=True)
        # action, _states = loaded_model.predict(obs,  deterministic=False)
        obs, _, terminated, _, _ = env.step(action)

    return env


if __name__ == '__main__':

    # multiprocessing.set_start_method('spawn') 

    # Start Timer
    start_time = time.time()

    args = parse_args_from_slurm()

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

    ### Channel-related parameters
    MaxTxPower, NSC = utils.TXpowerCalc(BW, NSS)

    # Traffic Configuration 
    traffic_config = {
        'load_min': 10,  # Minimum load in Mbps
        'load_max': 90,  # Maximum load in Mbps
        'EDCAaccessCategory': 'BE'
    }

    # Simulation Configuration
    sim_config = {
        'filtering': True,
        'save_model': True,
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
        'learning_timestamp_to_stop': 5, # seconds
        'training_flag': True,
        'timestamp_to_stop': 5, # seconds
        'FRAME_LENGTH': FRAME_LENGTH,
        'EVENT_NUMBER': int(1E5), # Number of events considered for traffic generation
        'seed': 1,
        'output_dir': os.path.join(os.getcwd(), 'Results', sim),
        'overheads' : utils.OverheadsCalc(traffic_config['EDCAaccessCategory'])   
    }

    # Learning Configuration — base + overrides from CLI args
    learning_config = {
        'log_dir': os.path.join(os.getcwd(), 'trained_models'),
        'parallel_envs': min(os.cpu_count(), 10),  # Number of parallel environments
        'total_timesteps': int(10E6),
        'simulator_attr': 'simulator',
        'project_name': args['project_name'],
        'run_id': args['run_id'],
        'n_steps': args['n_steps'],
        'batch_size': args['batch_size'],
        'initial_lr': args['initial_lr'],
        'learning_decay': args['learning_decay'],
        'gamma': args['gamma'],
        'gae_lambda': args['gae_lambda'],
        'clip_range': args['clip_range'],  
        'episode_threshold': args['episode_threshold'],
    }
    
    try:
        # Simulate the iterations
        training(traffic_config, sim_config, learning_config, iter_number=3)
    finally:
        if wandb.run is not None:
            wandb.finish()



