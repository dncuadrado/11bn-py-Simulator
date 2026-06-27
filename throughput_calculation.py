import numpy as np
import re
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Import tqdm for progress bar
from numpy.random import SeedSequence, default_rng
import utils as utils
from constants import SYSTEM, MAC, CHANNEL
from deployment_generator import deployment_generator
import h5py
import os


# Helper function to simulate one iteration
def throughput_calculation(sim_config, seed, show_plot= None, iter_number=None):

    # Deployment-dependent data
    ap_matrix, sta_matrix, association, channel_matrix = deployment_generator(sim_config, seed, show_plot=show_plot)

    # Mode 1: Use pre-loaded data (if enabled)
    if sim_config['use_preloaded_deployments']:
        sta_matrix = sta_matrix_save[:, :, iter_number]
        channel_matrix = channel_matrix_save[:, :, iter_number]
    # Mode 2: Generate fresh data
    else:
        ap_matrix, sta_matrix, association, channel_matrix = deployment_generator(sim_config, seed)


    map_matrix, tx_power_matrix_temp, comb_ok = utils.cg_creation_tpc(
        association, 
        channel_matrix, 
        max_tx_power_dbm, 
        nsc,
        is_filtering='on', 
        tpc_method=sim_config['tpc_method'],    # TPC Optimization method: None, 'PSO'
        cg_size=sim_config['cg_size']
    )    
    # print(f"Iteration {iter_number}: Valid combinations found: {np.sum(comb_ok)}")
    tx_power_matrix = [row.tolist() for i, row in enumerate(tx_power_matrix_temp) if comb_ok[i]==True]
    cgs_stas = [row.tolist() for i, row in enumerate(map_matrix) if comb_ok[i]==True]

    # Validate that tx_power_matrix and cgs_stas have the same length
    if len(tx_power_matrix) != len(cgs_stas):
        raise ValueError('tx_power_matrix and cgs_stas have different lengths')

    per_sta_edca_throughput_bianchi = utils.throughput_edca_bianchi(ap_number, sta_number, association, channel_matrix, sim_config['max_tx_power_dbm'],
                                                            sim_config['nsc'], sim_config['overheads']['edca_overheads'], 'BE')


    dl_throughput_csr_bianchi = utils.throughput_csr_bianchi(ap_number, sta_number, association, cgs_stas, tx_power_matrix, channel_matrix, 
                                                            sim_config['nsc'], sim_config['overheads']['csr_overheads'], 'BE', nss=None)
                                                  
    return per_sta_edca_throughput_bianchi, dl_throughput_csr_bianchi, np.sum(comb_ok)

def plot_cdf(per_sta_edca_throughput_bianchi, dl_throughput_csr_bianchi):
    # Aggregate Throughput
    agg_thr_edca_dl_vector = np.sum(per_sta_edca_throughput_bianchi, axis=1)
    agg_thr_csr_bianchi = np.sum(dl_throughput_csr_bianchi, axis=1)

    # Flatten the results for plotting
    all_sta_edca = per_sta_edca_throughput_bianchi.flatten()
    all_sta_csr = dl_throughput_csr_bianchi.flatten()

    # Helper function for ECDF
    def ecdf(data):
        x = np.sort(data)
        y = np.arange(1, len(x)+1) / len(x)
        return x, y

    # Plotting ECDF for aggregate throughput
    plt.figure()
    x_edca, y_edca = ecdf(agg_thr_edca_dl_vector)
    x_csr, y_csr = ecdf(agg_thr_csr_bianchi)
    plt.plot(x_edca, y_edca, label="EDCA")
    plt.plot(x_csr, y_csr, label="CSR")
    plt.xlabel('Aggregate Throughput (Mbps)')
    plt.ylabel('Cumulative Distribution Function')
    plt.legend()
    plt.grid(True)
    # plt.title('CDF of Throughput')

    # Plotting ECDF for per-STA throughput
    plt.figure()
    x_sta_edca, y_sta_edca = ecdf(all_sta_edca)
    x_sta_csr, y_sta_csr = ecdf(all_sta_csr)
    plt.plot(x_sta_edca, y_sta_edca, label="EDCA")
    plt.plot(x_sta_csr, y_sta_csr, label="CSR")
    plt.xlabel('Per-STA Throughput (Mbps)')
    plt.ylabel('Cumulative Distribution Function')
    plt.legend()
    plt.grid(True)
    # plt.title('CDF of Throughput')

    plt.show()
    
def init_pool_processes(h5_path, use_preloaded):
    """Load HDF5 data only if required by the simulation mode"""
    global sta_matrix_save, channel_matrix_save
    if use_preloaded:
        with h5py.File(h5_path, 'r') as f:
            sta_matrix_save = f['sta_matrix_save'][:]
            channel_matrix_save = f['channel_matrix_save'][:]


# Define module-level globals
sta_matrix_save = None
channel_matrix_save = None


if __name__ == '__main__':
    # Start Timer
    start_time = time.time()

    # Simulation parameters
    edca_access_category = 'BE'

    sim = '30-16'  # Simulation name: 'APtoAPdistance-STA_NUMBER'
    numbers = re.findall(r'\d+', sim) # Extract numbers from the simulation name

    # Scenario-related
    ap_number = 4
    sta_number = int(numbers[1]) 
    grid_value = int(numbers[0]) * 2
    scenario_type = 'grid'

    walls = np.array([
        [0, grid_value, grid_value/2, grid_value/2],
        [grid_value/2, grid_value/2, 0, grid_value]
    ])

    ### Channel-related parameters
    max_tx_power_dbm, nsc = utils.tx_power_calc() # default bw=80 MHz, nss=2 spatial streams

    # Deployment data path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up one level
    h5file_deployments_path = os.path.join(base_dir, 'deployments_datasets', sim, 'deployment_datasets.h5')

    # Simulation Configuration
    sim_config = {
        'use_preloaded_deployments': False,
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
        'training_flag': False,
        'timestamp_to_stop': 5, # seconds
        'frame_length': MAC.FRAME_LENGTH,
        'event_number': int(1E5), # Number of events considered for traffic generation
        'seed': 1,
        'output_dir': os.path.join(base_dir, 'Results', sim),
        'overheads' : utils.overheads_calc('BE'), 
    }

    # Simulation parameters for parallel processing
    iterations = 100

    # For reproducibility
    seed_seq = SeedSequence(1)
    seeds = seed_seq.generate_state(iterations)

    # Pre-allocate result arrays
    per_sta_edca_throughput_bianchi = np.zeros((iterations, sta_number))
    dl_throughput_csr_bianchi = np.zeros((iterations, sta_number))
    comb_numbers = np.zeros(iterations)

    max_workers = min(os.cpu_count(), iterations)  # Optimize worker count
    # max_workers = 1  # Optimize worker count

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_pool_processes,
        initargs=(h5file_deployments_path, sim_config['use_preloaded_deployments'])
        ) as executor:

        futures = [
            executor.submit(
                throughput_calculation, sim_config, seeds[iter_number], iter_number=iter_number
            )
            for iter_number in range(iterations)
        ]
        for iter_number, future in enumerate(tqdm(futures, desc="Processing", unit=" iterations")):
            try:
                edca_throughput, csr_throughput, comb_number = future.result()
                per_sta_edca_throughput_bianchi[iter_number, :] = edca_throughput
                dl_throughput_csr_bianchi[iter_number, :] = csr_throughput
                comb_numbers[iter_number] = comb_number


            except Exception as e:
                print(f"Error in iterations {iter_number}")

    percent_reduction = 100-(100 * sum(comb_numbers) / (iterations * 624))
    print(f"Percentage of reduction: {percent_reduction:.2f}")
    plot_cdf(per_sta_edca_throughput_bianchi, dl_throughput_csr_bianchi)

    # ################## Saving the results ##########################
    # # Define the folder structure for saving

    output_folder = os.path.join(base_dir, 'results', 'throughput_calculation', sim, 'pruning_off')
    os.makedirs(output_folder, exist_ok=True)

    # Define file path
    h5file_outputpath = os.path.join(output_folder, 'simulation_results.h5')

    # # Save data to HDF5 file
    with h5py.File(h5file_outputpath, 'w') as f:
        f.create_dataset('per_sta_edca_throughput_bianchi', data=per_sta_edca_throughput_bianchi)
        f.create_dataset('dl_throughput_csr_bianchi', data=dl_throughput_csr_bianchi)
        f.create_dataset('comb_numbers', data=comb_numbers)

    
    # End Timer and print elapsed time
    end_time = time.time()
    print(f"Simulation took {end_time - start_time:.2f} seconds")