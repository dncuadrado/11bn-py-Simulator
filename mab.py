import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import joblib
import copy
import time
import re
from linucb_agent import LinUCBAgent
from mapc_sim import MAPCsim  # your existing simulator
from deployment_generator import deployment_generator
from utils import get_association, cg_creation_tpc, get_channel_coefficient
from traffic_generator import traffic_generator
from constants import SYSTEM, MAC
import utils as utils
import argparse

import wandb

def parse_args_from_slurm():
    """
    Parse arguments from command line
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='linucb_optimization')  # WandB project name
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--n_steps', type=int, default=128)                 # default --- 2048                   
    parser.add_argument('--batch_size', type=int, default=256)              # default --- 64        
    parser.add_argument('--n_epochs', type=int, default=10)        # default --- 10
    parser.add_argument('--initial_lr', type=float, default=6.5E-4)       # default --- 3e-4              
    parser.add_argument('--learning_decay', type=str, default='cosine') # default --- 'cosine', 'linear', 'exp', 'square'
    parser.add_argument('--gamma', type=float, default=0.99)               # default --- 0.99
    parser.add_argument('--gae_lambda', type=float, default=0.95)          # default --- 0.95
    parser.add_argument('--clip_range', type=float, default=0.2)           # default --- 0.2
    parser.add_argument('--episode_threshold', type=int, default=0) 
    parser.add_argument('--w_throughput', type=float, default=4E-7)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--mab_alpha', type=float, default=0.9)
    parser.add_argument('--mab_coeff', type=float, default=0.4) 
    parser.add_argument('--mab_penalty_weight', type=float, default=0.5)  

    args = parser.parse_args()

    return vars(args)


def plot_rewards(records):
    df = pd.DataFrame(records)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- First subplot: delay trends with twin y-axis ---
    color1 = 'tab:blue'
    color2 = 'tab:orange'

    ax1 = axes[0]
    ax2 = ax1.twinx()  # create a second y-axis sharing the same x-axis

    l1 = ax1.plot(df['step'], df['new_point'], color=color1, marker='o', markersize=4, alpha=0.7, label='New Delay')
    l2 = ax2.plot(df['step'], df['mean'], color=color2, linestyle='--', label='Reward Mean')

    # Labeling and titles
    ax1.set_title("Delay and Reward Mean Over Time")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Worst 99th Percentile Delay", color=color1)
    ax2.set_ylabel("Reward Mean", color=color2)

    # Color sync for tick labels
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combine legends
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    # Grid
    ax1.grid(True, alpha=0.3)

    # --- Second subplot: reward ---
    axes[1].plot(df['step'], df['reward'], label='Reward', color='purple')
    axes[1].set_title("Reward Over Time")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Reward")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def traffic_generation(traffic_config, sim_config, iteration):
    ''' Generate traffic arrivals matrix '''
    np.random.seed(iteration)
    if sim_config['use_preloaded_traffic']:
        h5_file_path = os.path.join(base_dir, 'traffic_datasets', campaign, sim, f"deployment{iteration}", f"stas_arrivals_matrix{0}.h5")
        with h5py.File(h5_file_path, 'r') as h5file:
            stas_arrivals_matrix = [h5file[key][:] for key in h5file.keys()]
            traffic_profile_per_sta = h5file.attrs['traffic_profile_per_sta']
    else:
        # traffic_profile_per_sta = np.random.choice(['C','D'], size=sim_config['sta_number']).tolist()
        traffic_profile_per_sta = [
                {
                    'traffic_load': np.random.uniform(traffic_config['load_min'], traffic_config['load_max']),  # Load in Mbps
                    'traffic_model': str(np.random.choice(['poisson', 'bursty']))  # Traffic model
                }
                for i in range(sim_config['sta_number'])
        ]

        stas_arrivals_matrix = traffic_generator(traffic_config, sim_config, traffic_profile_per_sta)
    return stas_arrivals_matrix, traffic_profile_per_sta

def create_env(traffic_config, sim_config, learning_config, mobility_config, sta_mobility, channel_matrix, map_matrix, tx_power_matrix_temp, comb_ok, stas_arrivals_matrix, agent=None):
    
    # Initialize simulator
    simulator = MAPCsim(sim_config, mobility_config=mobility_config)
    simulator.sta_mobility = sta_mobility
    simulator.simulation_system = 'rl'
    simulator.channel_matrix = channel_matrix.copy()
    simulator.cgs_stas = copy.deepcopy(map_matrix)
    simulator.tx_power_matrix = copy.deepcopy(tx_power_matrix_temp)
    simulator.comb_ok = copy.deepcopy(comb_ok)
    simulator.access_category = traffic_config['edca_access_category']
    simulator.timestamp_to_stop = sim_config['learning_timestamp_to_stop']
    simulator.sta_queue_timeline = copy.deepcopy(stas_arrivals_matrix)
    simulator.init_settings()
    simulator.sim_forward()

    # Initialize agent
    n_arms = len(simulator.comb_ok)
    context_dim = sim_config['sta_number']  # context dimension equal to number of STAs (worst-case delay per STA)
    if agent is None:
        alpha = learning_config['mab_alpha']
        mab_coeff = learning_config['mab_coeff']
        mab_penalty_weight = learning_config['mab_penalty_weight']
        agent = LinUCBAgent(n_arms=n_arms, context_dim=context_dim, alpha=alpha, mab_coeff=mab_coeff, mab_penalty_weight=mab_penalty_weight)    # baseline: alpha=0.9
    else:
        # sanity check for loaded agent compatibility with environment
        if agent.n_arms != n_arms or agent.context_dim != context_dim:
            raise ValueError(
                f"Loaded agent expects n_arms={agent.n_arms}, context_dim={agent.context_dim}, "
                f"but environment requires n_arms={n_arms}, context_dim={context_dim}"
            )   

    return simulator, agent

def training(traffic_config, sim_config, learning_config, mobility_config=None, sta_mobility=None, iteration=None):
    
    print(f"\n=== deployment{iteration} ===")

    # Generate deployment
    sim_config['ap_matrix'], sta_matrix, sim_config['association'], channel_matrix = deployment_generator(sim_config, seed=iteration)

    if sim_config['use_preloaded_deployments']:
        sta_matrix = sta_matrix_save[:, :, iteration]
        channel_matrix = channel_matrix_save[:, :, iteration]

    # Plot deployment
    # utils.plot_deployment(ap_matrix, sta_matrix, sim_config['association'], grid_value, walls)

    if mobility_config:
        sta_mobility = utils.generate_sta_mobility(
            sim_config['ap_matrix'],
            sta_matrix,
            sim_config['walls'],
            sim_config['grid_value'],
            sim_config['association'],
            sim_config['timestamp_to_stop'],
            mobility_config['ch_realization_duration'],
            mobility_config['speed'],
            min_dist_to_ap=1.0,
            max_attempts=30,
            rng=np.random.default_rng(iteration),
            )
        # utils.plot_mobility_trajectories(sta_mobility, sim_config['ap_matrix'], sim_config['association'], sim_config['walls'], sim_config['grid_value'])


    # Generate TX power and combinations
    map_matrix, tx_power_matrix_temp, comb_ok = cg_creation_tpc(
        sim_config['association'],
        channel_matrix,
        sim_config['max_tx_power_dbm'],
        sim_config['nsc'],
        is_filtering=sim_config['filtering'],
        tpc_method=sim_config['tpc_method'], # TPC Optimization method: None, 'PSO'
        cg_size=sim_config['cg_size']
    )

    # Generate traffic
    stas_arrivals_matrix, _ = traffic_generation(traffic_config, sim_config, iteration)

    # Create environment
    simulator, agent = create_env(traffic_config, sim_config, learning_config, mobility_config, sta_mobility, channel_matrix, map_matrix, tx_power_matrix_temp, comb_ok, stas_arrivals_matrix)

    # Run training
    simulator, agent, records = run_linucb_agent(sim_config, simulator, agent, deployment_id=iteration)
    
    warm_up_time = 10.0
    simulator.traffic_analysis(warm_up_time=warm_up_time)
    delay = simulator.delay_vector
    percentile = np.percentile(delay, 99) * 1000
    
    print(f'{percentile:.2f} ms ----> 99th Percentile Delay after a warm-up of {warm_up_time}s')
    
    # plot_rewards(records)

    # Save model per deployment
    mab_campaign = learning_config['run_id']
    save_path = os.path.join(learning_config['log_dir'], 'mab', campaign, sim, mab_campaign, 'linucb' + str(iteration) + '.pkl')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(agent, save_path)
    # print(f"✅ Model saved to {save_path}")
    return percentile, records

def evaluation(traffic_config, sim_config, learning_config, mobility_config, sta_mobility, channel_matrix, map_matrix, tx_power_matrix_temp, comb_ok, stas_arrivals_matrix, trained_model_path):

    # Load model
    agent = joblib.load(trained_model_path)
    # print(f"✅ Model linucb_agent{iteration} loaded successfully!")
    
    # Create environment
    simulator, _ = create_env(traffic_config, sim_config, learning_config, mobility_config, sta_mobility, channel_matrix, map_matrix, tx_power_matrix_temp, comb_ok, stas_arrivals_matrix, agent=agent)
    
    # Run evaluation
    simulator, _, records = run_linucb_agent(sim_config, simulator, agent)
    # plot_rewards(records)

    return simulator

def run_linucb_agent(sim_config, simulator, agent, deployment_id=None):

    # ============================================================
    #   CONTEXT EXTRACTION & CANDIDATE MASK
    # ============================================================
    def get_context(simulator):
        """
        Builds a context vector summarizing per-sta worst case delay.
        """

        context = np.array([
            (simulator.sim_timeline - simulator.first_pos_timestamp[sta]) / sim_config['timestamp_to_stop']
            if simulator.first_pos_timestamp[sta] <= simulator.sim_timeline
            else 0.0
            for sta in range(sim_config['sta_number'])
        ])

        return context.astype(np.float32)


    def get_candidate_actions(simulator):
        """
        Returns a numpy array of candidate action indices:
        - respects simulator.comb_ok (True = feasible combination)
        - excludes arms whose all STAs currently have zero packets (no work to do)
        """
        comb_ok = np.array(simulator.comb_ok, dtype=bool)  # shape (n_arms,)
        valid_indices = np.flatnonzero(comb_ok)  # indices where comb_ok is True

        # Filter out groups where all participating STAs have no queued packets
        candidates = []
        for a in valid_indices:
            uni = simulator.cgs_stas[a]
            if uni is None:
                # if this combination represents a 'wait' action
                continue

            # check if any STA in group has packets ready at current sim_timeline
            has_packets = False
            for sta in uni:
                if simulator.first_pos_timestamp[sta] <= simulator.sim_timeline:
                    if len(simulator.get_queue(sta)) > 0:
                        has_packets = True
                        break

            if has_packets:
                candidates.append(a)

        # If no candidate remains (all groups empty), fall back to comb_ok indices (so we don't crash)
        if len(candidates) == 0:
            candidates = valid_indices.tolist()

        return np.array(candidates, dtype=int)

    # ============================================================
    #   TRAINING LOOP
    # ============================================================

    # Run bandit
    step = 0
    rewards = []
    records = []  # store metrics for analysis
    delay_dict = {}
    last_txop_timestamp = np.min(simulator.first_pos_timestamp)

    while simulator.sim_timeline < sim_config['learning_timestamp_to_stop']:
        context = get_context(simulator)

        # Build candidate set (masked)
        candidates = get_candidate_actions(simulator)  # array of valid arm indices

        # Select arm only among candidates
        arm = agent.select_arm(context, candidates)

        # translating the arm into a valid subset of ap-sta pairs
        uni = simulator.cgs_stas[arm]
        if uni is None:
            agent_decision = None
            raise ValueError('Selected arm corresponds to any group')     # validation
        else:
            sta_rx = [sta for sta in uni if simulator.first_pos_timestamp[sta] <= simulator.sim_timeline]
            aps = get_association(simulator.association, sta_rx)
            agent_decision = [sta_rx, aps]

        # Run simulator step
        simulator.run_step(agent_decision)
        simulator.sim_forward()

        # # # ================== Reward calculation ==================
        # # # ========================================================
        # # reward shaping targetting the oldest packet
        reward_shaping = np.min(simulator.first_pos_timestamp) - last_txop_timestamp    
        last_txop_timestamp = np.min(simulator.first_pos_timestamp)         # update last txop timestamp                  

        # packet-based reward
        n_users = sum(simulator.per_txop_sta_tx_packets > 0)                    # number of users served in the last txop
        stas = np.where(simulator.per_txop_sta_tx_packets>0)[0]                 # indices of STAs that received packets in the last txop
        packet_reward = sum(simulator.per_txop_sta_tx_packets[stas])/1E4        # normalized packet signal

        # reward = agent.mab_coeff + n_users*packet_reward if reward_shaping > 0 else 0.0  
        reward = agent.mab_coeff*n_users*packet_reward if reward_shaping > 0 else 0.0        



        # # ================================================================================================================
        # ## The following reward function has been tested for penalty weights {0.1, 0.3, 0.5, 0.7, 0.9}, 
        # ## and also the reward function without the 0.1 value  but the version above is still better

        # # Delay statistics
        # reward_shaping = np.min(simulator.first_pos_timestamp) - last_txop_timestamp
        # last_txop_timestamp = np.min(simulator.first_pos_timestamp)   

        # # Packet statistics
        # n_users = np.sum(simulator.per_txop_sta_tx_packets > 0)
        # total_packets = np.sum(simulator.per_txop_sta_tx_packets)
        # packet_reward = total_packets / 1E4 if total_packets > 0 else 0.0   # keep your scaling

        # # Check if any packets in queue
        # packets_in_queue = simulator.first_pos_timestamp <= simulator.sim_timeline
        # if not np.any(packets_in_queue):   # validation: if no packets are in queue, we shouldn't be here 
        #     raise ValueError("No packets available. We shouldn't be here - check candidate generation logic")
        
        # norm_queuing_delays = np.array([
        #     (simulator.sim_timeline - simulator.first_pos_timestamp[sta]) / sim_config['timestamp_to_stop']
        #     if simulator.first_pos_timestamp[sta] <= simulator.sim_timeline
        #     else 0.0
        #     for sta in range(sim_config['sta_number'])
        # ])

        # max_norm_queuing_delays = np.max(norm_queuing_delays)

        # if reward_shaping > 0:
        #     # Success: oldest packet cleared
        #     reward = agent.mab_coeff  + n_users * packet_reward
        # else:
        #     # Failure: oldest packets remain – give a penalty based on the current worst delay
        #     reward = -agent.mab_penalty_weight * max_norm_queuing_delays


        rewards.append(reward)

        # if deployment_id is not None:
        #     agent.update(arm, reward, context)

        agent.update(arm, reward, context)

        # worst-case delay (just for statistics)
        current_worst_delay = simulator.sim_timeline - np.min(simulator.first_pos_timestamp)   

        avg_reward = np.mean(rewards[-100:])
        records.append({
            'step': step,
            'mean': avg_reward,
            'reward': reward
        })

        if (step + 1) % 100 == 0:
            if deployment_id is not None: 
                # --- W&B logging ---
                wandb.log({
                    "sim_timeline": float(simulator.sim_timeline),
                    f"deployment_{deployment_id}/worst_delay": current_worst_delay,
                    f"deployment_{deployment_id}/avg_reward": avg_reward,
                })
        
            # # ## For debugging:
            # # print(f"Step {step + 1}, recent avg reward (last 100 steps): {np.mean(rewards[-100:]):.4f}, candidates={len(candidates)}")
            
            # print(f"---------\n\
            # #       Agent decision: {agent_decision} \n\
            # #       Current worst delay: {simulator.sim_timeline - np.min(simulator.first_pos_timestamp)} \n\
            # #       Timestamp: {simulator.sim_timeline} \n\
            # #       -----------------------------------------------------------------------------------------------------\n\
            # #       -----------------------------------------------------------------------------------------------------")

        step += 1

    return simulator, agent, records




# ============================================================
#   MAIN
# ============================================================
if __name__ == '__main__':

    start_time = time.time()
    args = parse_args_from_slurm()

    # Initialize W&B sweep
    run = wandb.init(
        project=args['project_name'],
        id=args['run_id'],
        resume="allow",
        config=args
    )
    # Per deployment logging setup
    wandb.define_metric("sim_timeline")
    wandb.define_metric("deployment_*", step_metric="sim_timeline")
    # For global metrics (all deployments)
    wandb.define_metric("deployment")
    wandb.define_metric("custom_*", step_metric="deployment")


    sim = '30-16'
    campaign = 'general[10,90]' # 'general[10,90]', 'expert[10,90]', 'low[10,30]', 'medium[30,50]', 'high[50,70]'

    # Scenario
    ap_number = 4
    sta_number = int(re.findall(r'\d+', sim)[1])
    grid_value = int(re.findall(r'\d+', sim)[0]) * 2
    scenario_type = 'grid'

    walls = np.array([
        [0, grid_value, grid_value/2, grid_value/2],
        [grid_value/2, grid_value/2, 0, grid_value]
    ])

    # Channel-related
    max_tx_power_dbm, nsc = utils.tx_power_calc()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h5file_deployments_path = os.path.join(base_dir, 'deployments_datasets', sim, 'deployment_datasets.h5')

    match sta_number:
        case 8:
            min_load, max_load = 43.42, 156.58
        case 12:
            min_load, max_load = 20.47, 112.87
        case 16:
            min_load, max_load = 10.0, 90.0
        case 20:
            min_load, max_load = 4.21, 75.79

    # Traffic configuration
    traffic_config = {
        'load_min': min_load,
        'load_max': max_load,
        'edca_access_category': 'BE'
    }

    # Simulation configuration
    sim_config = {
        'filtering': True,
        'save_model': False,
        'use_preloaded_deployments': True,
        'use_preloaded_traffic': True,
        'ap_number': ap_number,
        'sta_number': sta_number,
        'scenario_type': scenario_type,
        'grid_value': grid_value,
        'walls': walls,
        'max_tx_power_dbm': max_tx_power_dbm,
        'tpc_method': None,  # TPC Optimization method: None, 'PSO'
        'cg_size': ap_number,
        'txop_duration': SYSTEM.TXOP_DURATION,
        'pn_dbm': SYSTEM.PN_DBM,
        'cca': SYSTEM.CCA,
        'nss': SYSTEM.NSS,
        'nsc': nsc,
        'learning_timestamp_to_stop': 20,    ############
        'training_flag': True,
        'timestamp_to_stop': 20,        ############
        'frame_length': MAC.FRAME_LENGTH,
        'event_number': int(1E6),   ############
        'seed': 42,
        'output_dir': os.path.join(base_dir, 'results', campaign, sim),
        'overheads': utils.overheads_calc(traffic_config['edca_access_category']),
    }

    # Learning Configuration — base + overrides from CLI args
    learning_config = {
        'log_dir': os.path.join(base_dir, 'trained_models'),
        'parallel_envs': min(os.cpu_count(), 10),  # Number of parallel environments
        'total_timesteps': int(10E6),
        'simulator_attr': 'simulator',
        'project_name': args['project_name'],
        'run_id': args['run_id'],
        'n_steps': args['n_steps'],
        'batch_size': args['batch_size'],
        'n_epochs': args['n_epochs'],
        'initial_lr': args['initial_lr'],
        'learning_decay': args['learning_decay'],
        'gamma': args['gamma'],
        'gae_lambda': args['gae_lambda'],
        'clip_range': args['clip_range'],  
        'episode_threshold': args['episode_threshold'],
        'w_throughput': args['w_throughput'],
        'window_size': args['window_size'],
        'mab_alpha': args['mab_alpha'],
        'mab_coeff': args['mab_coeff'],
        'mab_penalty_weight': args['mab_penalty_weight']  
    }

    mobility_config = None
    # mobility_config = {
    #     'ch_realization_duration': 0.1,  # seconds
    #     'ch_realizations_per_update': 5,
    #     'speed' : 1  # meters per second
    # }

    if sim_config['use_preloaded_deployments']:
        with h5py.File(h5file_deployments_path, 'r') as f:
            sta_matrix_save = f['sta_matrix_save'][:]
            channel_matrix_save = f['channel_matrix_save'][:]

    iterations = 100
    final_delays = []
    final_rewards = []

    for iteration in range(iterations):

        percentile, records = training(
            traffic_config,
            sim_config,
            learning_config,
            mobility_config=None,
            iteration=iteration
        )

        final_delays.append(percentile)
        final_rewards.append(np.mean([r['reward'] for r in records]))

        

        wandb.log({
            "deployment": iteration,
            "custom_p99_delay": percentile,
            "custom_running_mean_p99_delay": np.mean(final_delays),
        })

    # Exclude outliers (due to saturation) for final metrics calculation (if any)
    match sim:
        case '30-16':
            to_exclude = {6, 19, 26, 29, 33, 50, 57, 59, 65, 66}
        case _:
            to_exclude = set()

    final_delays_excluded = [
        d for i, d in enumerate(final_delays)
        if i not in to_exclude
    ]

    final_rewards_excluded = [
        r for i, r in enumerate(final_rewards)
        if i not in to_exclude
    ]

    # FINAL METRICS 
    wandb.log({
        "custom/final_mean_p99_delay": np.mean(final_delays_excluded),
        "custom/final_std_p99_delay": np.std(final_delays_excluded),
        "custom/final_avg_reward": np.mean(final_rewards_excluded),
        "custom/p99_delay_distribution": wandb.Histogram(final_delays_excluded)
    })

    wandb.finish()
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

            

