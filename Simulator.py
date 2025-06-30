
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
import argparse
import json

import Utils as utils
from MAPCsim import *
from TrafficGenerator import traffic_generator
from DeploymentGenerator import deployment_generator
import RLagent as RLagent
from histo_plot import plot_histogram

from CustomEnv import * # my Custom environment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import MaskablePPO 
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from concurrent.futures import ProcessPoolExecutor, as_completed

global STA_matrix_save, channelMatrix_save, summary_data

def parse_args_from_slurm():
    """
    Parse arguments from command line
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='sb3-HPC')
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--n_steps', type=int, default=2048)                 # default --- 2048                 
    parser.add_argument('--batch_size', type=int, default=64)              # default --- 64   
    parser.add_argument('--initial_lr', type=float, default=6.5e-4)       # default --- 3e-4              
    parser.add_argument('--learning_decay', type=str, default='cosine') # default --- 'cosine'
    parser.add_argument('--gamma', type=float, default=0.99)               # default --- 0.99
    parser.add_argument('--gae_lambda', type=float, default=0.95)          # default --- 0.95
    parser.add_argument('--clip_range', type=float, default=0.2)           # default --- 0.2
    parser.add_argument('--w_mean', type=float, default=0.13)
    parser.add_argument('--episode_threshold', type=int, default=0)
    parser.add_argument('--qos_threshold', type=float, default=0.01) # QoS threshold for the reward function
    parser.add_argument('--w_sparse', type=float, default=0.5)            # default --- 0.13

    args = parser.parse_args()

    return vars(args)

def simulate_iterations(traffic_config, sim_config, learning_config, seed, iter_number=None):
    """
    Simulates one iterations and returns the delay vectors for the different strategies.
    Parameters:
    traffic_config (dict): Configuration for traffic generation.
    sim_config (dict): Configuration for the simulation.
    learning_config (dict): Configuration for the learning agent.
    seed (int): Random seed for the simulation.
    iter_number (int): Iteration number for the deployment.

    Returns:
    deployment_key (str): Key for the deployment.
    deployment_summary (dict): Summary of the deployment.
    """ 

    iter_number = 3
    print('-----------------------------------------')
    print('-----------------------------------------')
    print(f"Deployment{iter_number}...")

    # # Set the seed
    np.random.seed(seed)

    deployment_key = f"Deployment{iter_number}"

    AP_matrix, STA_matrix, sim_config['association'], channel_matrix = deployment_generator(sim_config, seed)

    if sim_config['use_preloaded_deployments']:
        STA_matrix = STA_matrix_save[:, :, iter_number]
        channel_matrix= channelMatrix_save[:, :, iter_number]

    # utils.PlotDeployment(AP_matrix, STA_matrix, sim_config['association'], sim_config['GRID_VALUE'], sim_config['walls'])

    # Compute the CGs and TxPowerMatrix
    map_matrix, TxPowerMatrixTemp, comb_ok = CG_creationTPC(sim_config['AP_NUMBER'], 
                                                    sim_config['STA_NUMBER'], 
                                                    sim_config['PN_DBM'], 
                                                    sim_config['NSC'], 
                                                    sim_config['NSS'], 
                                                    sim_config['association'], 
                                                    channel_matrix, 
                                                    sim_config['MaxTxPower'], 
                                                    is_filtering=sim_config['filtering'], TPC_method=None, # TPC Optimization method: None, 'PSO', 'IPOPT', 'DE'
                                                    CG_size=4) 

    TxPowerMatrix = [row.tolist() for i, row in enumerate(TxPowerMatrixTemp) if comb_ok[i]]
    CGs_STAs = [row.tolist() for i, row in enumerate(map_matrix) if comb_ok[i]]

    # map_matrix.append(None)
    # TxPowerMatrixTemp.append(None)    
    # comb_ok = np.append(comb_ok, True)

    if len(TxPowerMatrix) != len(CGs_STAs):
        raise ValueError('Mismatch between TxPowerMatrix and CGs_STAs')

    traffic_ITERATIONS = 100
    seed_seq = SeedSequence(seeds[iter_number])
    traffic_seeds = seed_seq.generate_state(traffic_ITERATIONS)

    # Init the result
    deployment_summary = {}

    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(
                run_traffic_iteration,
                traffic_iter,
                traffic_seeds[traffic_iter],
                traffic_config,
                sim_config,
                learning_config,
                channel_matrix,
                map_matrix,
                TxPowerMatrixTemp,
                comb_ok,
                CGs_STAs,
                TxPowerMatrix,
                sim,
                iter_number
            )
            for traffic_iter in range(traffic_ITERATIONS)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Deployment {iter_number}"):
            try:
                traffic_iter, traffic_summary = future.result()
                deployment_summary[f"Traffic{traffic_iter}"] = traffic_summary
            except Exception as e:
                print(f"Error in traffic iteration: {e}")

    return deployment_key, deployment_summary

def run_traffic_iteration(
        traffic_iter, 
        traffic_seed, 
        traffic_config, 
        sim_config, 
        learning_config,
        channel_matrix, 
        map_matrix, 
        TxPowerMatrixTemp, 
        comb_ok, 
        CGs_STAs, 
        TxPowerMatrix, 
        sim, 
        iter_number
    ):
    np.random.seed(traffic_seed)
    if sim_config['use_preloaded_traffic']:
        h5_file_path = os.path.join(os.getcwd(), 'traffic datasets', sim, f"Deployment{iter_number}", f"STAs_arrivals_matrix{traffic_iter}.h5")
        with h5py.File(h5_file_path, 'r') as h5file:
            STAs_arrivals_matrix = [h5file[key][:] for key in h5file.keys()]
            traffic_config['traffic_profile_perSTA'] = h5file.attrs['traffic_profile_perSTA'].tolist()
    else:
        # traffic_profile_perSTA = np.random.choice(['C','D'], size=sim_config['STA_NUMBER']).tolist()
        traffic_profile_perSTA = [
                {
                    'traffic_load': np.random.uniform(traffic_config['load_min'], traffic_config['load_max']),  # Load in Mbps
                    'traffic_model': str(np.random.choice(['Poisson', 'Bursty']))  # Traffic model
                }
                for i in range(sim_config['STA_NUMBER'])
        ]

        STAs_arrivals_matrix = traffic_generator(traffic_config, sim_config, traffic_profile_perSTA)

    delay_dict = {}

    # # Baseline ML model
    # baseline = 'mvoz4x5g'
    # baseline_model = {'model_id': baseline, 'model_type': 'best_model.zip'}

    # np.random.seed(sim_config['seed'])
    # simML = RLagent.evaluation(
    #     traffic_config, sim_config, learning_config, 
    #     map_matrix, TxPowerMatrixTemp, comb_ok, datarate, 
    #     STAs_arrivals_matrix, baseline_model
    # )
    # simML.simulator.TrafficAnalysis()
    # baseline_delay = simML.simulator.delayvector

    # # # Include it in the results dict
    # delay_dict['baseline'] = baseline_delay

    # # Evaluate models
    models= ['6cbu38xr']
    # models= ['145tabtw']
    # # # # # # # # Add additional ML models

    ml_results = evaluate_models(models, sim_config, traffic_config, learning_config,
                                channel_matrix, map_matrix, TxPowerMatrixTemp, comb_ok, STAs_arrivals_matrix, traffic_profile_perSTA)
    
    delay_dict.update(ml_results)

    # EDCA
    np.random.seed(sim_config['seed'])
    simEDCA = MAPCsim(sim_config)
    simEDCA.timestamp_to_stop = sim_config['timestamp_to_stop']
    simEDCA.channel_matrix = channel_matrix
    simEDCA.STA_queue_timeline = STAs_arrivals_matrix
    simEDCA.simulation_system = 'EDCA'
    simEDCA.accessCategory = traffic_config['EDCAaccessCategory']
    simEDCA.InitSettings()
    simEDCA.Run()
    EDCAdelay = simEDCA.delayvector

    # MNP
    np.random.seed(sim_config['seed'])
    simMNP = MAPCsim(sim_config)
    simMNP.timestamp_to_stop = sim_config['timestamp_to_stop']
    simMNP.channel_matrix = channel_matrix
    simMNP.STA_queue_timeline = STAs_arrivals_matrix
    simMNP.simulation_system = 'CSR'
    simMNP.scheduler = 'MNP'
    simMNP.CGs_STAs = CGs_STAs
    simMNP.TxPowerMatrix = TxPowerMatrix
    simMNP.accessCategory = traffic_config['EDCAaccessCategory']
    simMNP.InitSettings()
    simMNP.Run()
    MNPdelay = simMNP.delayvector
    plot_histogram(simMNP.priority_selection_counter / simMNP.suc_TXOPs, name='MNP')

    # OP
    np.random.seed(sim_config['seed'])
    simOP = MAPCsim(sim_config)
    simOP.timestamp_to_stop = sim_config['timestamp_to_stop']
    simOP.channel_matrix = channel_matrix
    simOP.STA_queue_timeline = STAs_arrivals_matrix
    simOP.simulation_system = 'CSR'
    simOP.scheduler = 'OP'
    simOP.CGs_STAs = CGs_STAs
    simOP.TxPowerMatrix = TxPowerMatrix
    simOP.accessCategory = traffic_config['EDCAaccessCategory']
    simOP.InitSettings()
    simOP.Run()
    OPdelay = simOP.delayvector
    plot_histogram(simOP.priority_selection_counter / simOP.suc_TXOPs, name='OP')

    # TAT
    np.random.seed(sim_config['seed'])
    simTAT = MAPCsim(sim_config)
    simTAT.timestamp_to_stop = sim_config['timestamp_to_stop']
    simTAT.channel_matrix = channel_matrix
    simTAT.STA_queue_timeline = STAs_arrivals_matrix
    simTAT.simulation_system = 'CSR'
    simTAT.scheduler = 'TAT'
    simTAT.CGs_STAs = CGs_STAs
    simTAT.TxPowerMatrix = TxPowerMatrix
    simTAT.accessCategory = traffic_config['EDCAaccessCategory']
    simTAT.alpha = 0.5
    simTAT.beta = 0.5
    simTAT.InitSettings()
    simTAT.Run()
    TATdelay = simTAT.delayvector
    plot_histogram(simTAT.priority_selection_counter / simTAT.suc_TXOPs, name='TAT')

    ### Add the other strategies
    delay_dict.update({
        'EDCA': EDCAdelay,
        'MNP': MNPdelay,
        'OP': OPdelay,
        'TAT': TATdelay,
    })

    print(f'Iteration: {traffic_iter}')
    print('--- 99th percentile delay results ---')
    percentiles = {strategy: np.percentile(delay, 99) * 1000 for strategy, delay in delay_dict.items()}
    for strategy, p99 in sorted(percentiles.items(), key=lambda x: x[1]):
        print(f'{strategy:10s}: {p99:.2f} ms')

    # print(f'Traffic Profile per STA: {traffic_profile_perSTA}')
    print('-----------------------------------------')

    # save_to_h5(sim_config['output_dir'], sim, iter_number, traffic_iter, delay_dict)

    # Return traffic summary
    traffic_summary = {
        "traffic_profile": traffic_profile_perSTA,  # <--- compact
        "p99_delays_ms": {strategy: float(np.percentile(delay, 99) * 1000) for strategy, delay in delay_dict.items()}
    }

    return traffic_iter, traffic_summary 



def evaluate_models(models, sim_config, traffic_config, learning_config,
                    channel_matrix, map_matrix, TxPowerMatrixTemp, comb_ok, STAs_arrivals_matrix, traffic_profile_perSTA):
    delays = {}
    for model_id in models:
        np.random.seed(sim_config['seed'])  # Ensure fairness across runs
        model = {'model_id': model_id, 'model_type': 'best_model.zip'}
        
        match model_id:
            case '6cbu38xr':   # general model
                model['model_type'] = 'model_9830400_steps.zip'       # best
                name = 'ML-G'
            case 'jpjju421': # Expert deployment 0
                model['model_type'] = 'model_9984000_steps.zip'       # best
            case '145tabtw': # Expert deployment 3
                model['model_type'] = 'model_8601600_steps.zip'       # model_6553600_steps     model_8601600_steps [ok]        model_8652800_steps
                name = 'ML-E'
            case _:
                model['model_type'] = 'best_model.zip'  # Default case for other models
                name = model_id

        simML = RLagent.evaluation(
            traffic_config, sim_config, learning_config,
            channel_matrix, map_matrix, TxPowerMatrixTemp, comb_ok,
            STAs_arrivals_matrix, traffic_profile_perSTA, model
        )
        simML.simulator.TrafficAnalysis()

        
        plot_histogram(simML.simulator.priority_selection_counter / simML.simulator.suc_TXOPs, name=name)
        
        delay_vector = simML.simulator.delayvector
        delays[model_id] = delay_vector
    
    return delays

def save_to_h5(output_dir, sim, iter_number, traffic_iter, delay_dict):
    """
    Saves delay metrics into an HDF5 file named 'delay.h5' in a structured directory.

    Parameters:
    - output_dir: str, base directory to save the output.
    - sim: unused (kept for compatibility).
    - iter_number: int or str, current deployment iteration.
    - traffic_iter: int or str, current traffic iteration.
    - delay_dict: dict, mapping from delay metric names to their corresponding arrays.
                  Example: {'EDCAdelay': ..., 'ModelA': ..., 'ModelB': ..., ...}
    """
    # Create the directory structure
    output_path = os.path.join(output_dir, f"Deployment{iter_number}", f"Traffic{traffic_iter}")
    os.makedirs(output_path, exist_ok=True)

    # Define the HDF5 file path
    h5_file_path = os.path.join(output_path, "delay.h5")
    
    # Write all delay metrics to the HDF5 file
    with h5py.File(h5_file_path, 'a') as f:
        for name, data in delay_dict.items():
            if name in f:
                del f[name]  # Optional: overwrite specific dataset
            f.create_dataset(name, data=data)

def init_pool_processes(h5_path, use_preloaded):
    """Load HDF5 data only if required by the simulation mode"""
    global STA_matrix_save, channelMatrix_save
    if use_preloaded:
        with h5py.File(h5_path, 'r') as f:
            STA_matrix_save = f['STA_matrix_save'][:]
            channelMatrix_save = f['channelMatrix_save'][:]

# Main function
if __name__ == "__main__":
    start_time = time.time()

    args = parse_args_from_slurm()

    sim = '30-16'
    numbers = re.findall(r'\d+', sim)

    AP_NUMBER = 4
    STA_NUMBER = int(numbers[1])
    GRID_VALUE = int(numbers[0]) * 2
    SCENARIO_TYPE = 'grid'

    walls = np.array([
        [0, GRID_VALUE, GRID_VALUE/2, GRID_VALUE/2],
        [GRID_VALUE/2, GRID_VALUE/2, 0, GRID_VALUE]
    ])

    TXOP_DURATION = 5E-3
    PN_DBM = -95
    CCA = -82
    BW = 80
    NSS = 2
    FRAME_LENGTH = 12E3

    MaxTxPower, NSC = utils.TXpowerCalc(BW, NSS)

    ITERATIONS = 1
    seed_seq = SeedSequence(1)
    seeds = seed_seq.generate_state(max(ITERATIONS,100))

    h5file_deployments_path = os.path.join(os.getcwd(), 'deployments datasets', sim, 'deployment_datasets.h5')

    traffic_config = {
        'load_min': 30,  # Minimum load in Mbps
        'load_max': 90,  # Maximum load in Mbps
        'EDCAaccessCategory': 'BE'
    }

    sim_config = {
        'filtering': True,
        'save_model': False,
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
        'learning_timestamp_to_stop': 5,
        'training_flag': False,
        'timestamp_to_stop': 5,
        'FRAME_LENGTH': FRAME_LENGTH,
        'EVENT_NUMBER': int(1E5),
        'seed': 1,
        'output_dir': os.path.join(os.getcwd(), 'Results/Simulation', '30-16 (expert3_30-90)'),
        'overheads': utils.OverheadsCalc('BE')
    }

    learning_config = {
        'log_dir': os.path.join(os.getcwd(), 'trained_models'),
        'parallel_envs': 10,
        'total_timesteps': int(5E6),
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
        'w_mean': args['w_mean'],
        'episode_threshold': args['episode_threshold'],
        'qos_threshold': args['qos_threshold'],
        'w_sparse': args['w_sparse'],
    }

    if sim_config['use_preloaded_deployments']:
        with h5py.File(h5file_deployments_path, 'r') as f:
            STA_matrix_save = f['STA_matrix_save'][:]
            channelMatrix_save = f['channelMatrix_save'][:]

    summary_data = {}

    for iter_number in range(ITERATIONS):
        deployment_key, deployment_summary = simulate_iterations(
            traffic_config, sim_config, learning_config, seeds[iter_number], iter_number
        )
        summary_data[deployment_key] = deployment_summary

    summary_path = os.path.join(sim_config['output_dir'], "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
        
    print(f"Summary saved to {summary_path}")

    end_time = time.time()
    print(f"Simulation took {end_time - start_time:.2f} seconds")