import time
import os
import h5py
import numpy as np
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from numpy.random import SeedSequence, default_rng
import json

import Utils as utils
from DeploymentGenerator import deployment_generator

global STA_matrix_save, channelMatrix_save

def traffic_generator(traffic_config, sim_config, traffic_profile_perSTA):
    """
    Generates a list of arrival times for each STA based on the specified traffic model.
    """
    # Initialize the STAs_arrivals_matrix
    STAs_arrivals_matrix = []

    for sta in range(sim_config['STA_NUMBER']):
        
        # Loading STA profile
        traffic_profile = traffic_profile_perSTA[sta]

        # Loading the traffic model
        traffic_model = traffic_profile['traffic_model']

        # Loading the traffic load
        traffic_load = traffic_profile['traffic_load']
        
        traffic_generation_rate = traffic_load  * 1E6 / sim_config['FRAME_LENGTH']  # in packets/sec
        match traffic_model:
            case 'Poisson': # Poisson traffic model
                arrivals = poisson_fixed_events(sim_config['EVENT_NUMBER'], traffic_generation_rate)
            case 'Bursty': # Bursty traffic model
                arrivals = generate_burst_traffic(sim_config['EVENT_NUMBER'], traffic_generation_rate)
            case 'CBR': # CBR traffic model
                fps = traffic_config['traffic_profiles'][traffic_profile_perSTA[sta]]['fps']
                arrivals = generate_CBR_traffic(sim_config['FRAME_LENGTH'], traffic_load, fps)
            case _:
                raise ValueError("Invalid traffic type specified.")
    
        STAs_arrivals_matrix.append(arrivals)

    return STAs_arrivals_matrix

def poisson_fixed_events(EVENT_NUMBER, traffic_generation_rate):
    """
    Generate arrival times using a Poisson process for each STA.
    """

    # Generate exponential inter-arrival times
    w = np.random.exponential(scale=1/traffic_generation_rate, size=EVENT_NUMBER)
    arrivals = np.cumsum(w)

    # import matplotlib.pyplot as plt
    # # Plot the exponential inter-arrival times histogram
    # plt.figure()
    # plt.hist(w, bins=50, density=True, alpha=0.75)
    # x = np.linspace(0, np.max(w), 100)
    # plt.plot(x, traffic_generation_rate * np.exp(-traffic_generation_rate * x), 'r-', lw=2)
    # plt.title('Histogram of Inter-Arrival Times with Exponential PDF')
    # plt.xlabel('Inter-Arrival Time')
    # plt.ylabel('Probability Density')
    # plt.show()

    return arrivals

def generate_burst_traffic(EVENT_NUMBER, traffic_generation_rate):
    """
    Generate bursty traffic arrivals for each STA.
    """

    # Average ON and OFF times
    average_on_time = 1E-3
    average_off_time = 10E-3

    # Expected proportion of time spent in the ON state
    on_off_ratio = average_on_time / (average_on_time + average_off_time)

    # Adjusted generation rate during ON periods
    adjusted_generation_rate = traffic_generation_rate / on_off_ratio


    arrivals = np.zeros(EVENT_NUMBER)  # Preallocate space for arrival times
    current_time = 0  # Start at time 0
    total_packets_generated = 0  # Track the total number of packets generated

    while total_packets_generated < EVENT_NUMBER:
        # ON period: Generate packets based on adjusted_generation_rate
        on_period_duration = np.random.exponential(average_on_time)
        packets_in_burst = int(on_period_duration * adjusted_generation_rate)

        for _ in range(packets_in_burst):
            if total_packets_generated >= EVENT_NUMBER:
                break
            inter_arrival_time = np.random.exponential(1 / adjusted_generation_rate)
            current_time += inter_arrival_time
            arrivals[total_packets_generated] = current_time
            total_packets_generated += 1

        # OFF period: No packets generated
        off_period_duration = np.random.exponential(average_off_time)
        current_time += off_period_duration

        ##### Verifying the code:
        # # Check if the arrival times are in increasing order
        # assert np.all(np.diff(arrival_times) > 0), f"Non-monotonic times found in STA {i}"

        # # Check the actual rate of packets generated
        # total_time = arrival_times[-1] - arrival_times[0]
        # actual_rate = len(arrival_times) / total_time
        # print(f"STA {i}: Actual rate = {actual_rate:.2f} packets/sec and expected rate = {traffic_generation_rate:.2f} packets/sec")

        # Add the generated arrival times to the result matrix


    return arrivals

def generate_CBR_traffic(FRAME_LENGTH, traffic_load, fps):
    """
    Generate CBR traffic arrivals for each STA.
    """
    bitrate = traffic_load
    frame_interval = 1 / fps
    frames_per_burst = int(np.ceil((bitrate * 1E6 * frame_interval) / FRAME_LENGTH))
    frame_spacing = 5E-6
    stop_timestamp = 20

    current_time = np.random.uniform(0, frame_interval)
    arrivals = []
    while current_time < stop_timestamp:
        burst_times = current_time + np.arange(frames_per_burst) * frame_spacing
        arrivals.extend(burst_times)
        current_time += frame_interval

    return np.array(arrivals)

def save_to_h5(sim_config, traffic_config, sim, STAs_arrivals_matrix, deployment_iteration, traffic_iteration):
    """
    Saves the STAs_arrivals_matrix into individual HDF5 files in a structured directory.
    """
    # Create the directory structure
    output_path = os.path.join(sim_config['output_dir'], f"Deployment{deployment_iteration}")
    os.makedirs(output_path, exist_ok=True)

    # Save the current iteration to its own HDF5 file
    h5_file_path = os.path.join(output_path, f"STAs_arrivals_matrix{traffic_iteration}.h5")
    with h5py.File(h5_file_path, 'w') as h5file:
        for i, arrivals in enumerate(STAs_arrivals_matrix):
            h5file.create_dataset(f"STA_{i}", data=arrivals) # no compression
            # h5file.create_dataset(f"STA_{i}", data=arrivals, compression="gzip")   # with compression

        # Store each key-value pair as an attribute
        for key, value in traffic_config.items():
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                h5file.attrs[key] = np.array(value, dtype=h5py.string_dtype(encoding='utf-8'))
            elif isinstance(value, dict):
                h5file.attrs[key] = str(value) 
            else:
                h5file.attrs[key] = value

def simulate_iterations(traffic_config, sim_config, learning_config, seed, iter_number=None):
    """
    Simulates one iterations and returns the delay vectors for the different strategies.
    Parameters:
    traffic_config (dict): Configuration for traffic generation.
    sim_config (dict): Configuration for the simulation.
    learning_config (dict): Configuration for the learning agent.
    seed (int): Random seed for the simulation.
    iter_number (int): Iteration number for the deployment.

    """ 
    print('-----------------------------------------')
    print('-----------------------------------------')
    print(f"Deployment{iter_number}...")

    # # Set the seed
    np.random.seed(seed)

    AP_matrix, STA_matrix, sim_config['association'], sim_config['channelMatrix'] = deployment_generator(sim_config, seed)

    if sim_config['use_preloaded_deployments']:
        STA_matrix = STA_matrix_save[:, :, iter_number]
        sim_config['channelMatrix'] = channelMatrix_save[:, :, iter_number]

    traffic_ITERATIONS = 10
    seed_seq = SeedSequence(seeds[iter_number])
    traffic_seeds = seed_seq.generate_state(traffic_ITERATIONS)

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                run_traffic_iteration,
                traffic_iter,
                traffic_seeds[traffic_iter],
                traffic_config,
                sim_config,
                learning_config,
                sim,
                iter_number
            )
            for traffic_iter in range(traffic_ITERATIONS)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Deployment {iter_number}"):
            try:
                future.result()
            except Exception as e:
                print(f"Error in traffic iteration: {e}")

    return

def run_traffic_iteration(
        traffic_iter, 
        traffic_seed, 
        traffic_config, 
        sim_config, 
        learning_config, 
        sim, 
        iter_number
    ):
    np.random.seed(traffic_seed)
    # traffic_config['traffic_profile_perSTA'] = np.random.choice(['A','B'], size=sim_config['STA_NUMBER']).tolist()

    traffic_profile_perSTA = [
            {
                'traffic_load': np.random.uniform(10, 90),  # Load in Mbps
                'traffic_model': str(np.random.choice(['Poisson', 'Bursty']))  # Traffic model
            }
            for i in range(sim_config['STA_NUMBER'])
    ]

    STAs_arrivals_matrix = traffic_generator(traffic_config, sim_config, traffic_profile_perSTA)

    ### Uncoment to save the delay vectors into HDF5 files for each iterations, traffic type, and traffic load in a structured directory
    save_to_h5(sim_config, traffic_config, sim, STAs_arrivals_matrix, iter_number, traffic_iter)


# main function
if __name__ == "__main__":
    # Start Timer
    start_time = time.time()

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

    # Number of iterations
    ITERATIONS = 100
    seed_seq = SeedSequence(1)
    seeds = seed_seq.generate_state(ITERATIONS)

    # Deployment data path
    h5file_deployments_path = os.path.join(os.getcwd(), 'deployments datasets', sim, 'deployment_datasets.h5')

    # Traffic Configuration 
    traffic_config = {
        'traffic_profiles': {
            'A': {'traffic_model': 'Poisson', 'traffic_load': 10, 'latency': 1E-4},
            'B': {'traffic_model': 'Bursty', 'traffic_load': 90, 'latency': 2E-4},
        },
        'EDCAaccessCategory': 'BE'
    }   
    
    # Simulation Configuration
    sim_config = {
        'use_preloaded_deployments': True,
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
        'training_flag': False,
        'timestamp_to_stop': 5, # seconds
        'FRAME_LENGTH': FRAME_LENGTH,
        'EVENT_NUMBER': int(1E5), # Number of events considered for traffic generation
        'seed': 1,
        'output_dir': os.path.join(os.getcwd(), 'traffic datasets', sim),
        'overheads' : utils.OverheadsCalc(traffic_config['EDCAaccessCategory'])
    }

    learning_config = {
        'log_dir': os.path.join(os.getcwd(), 'trained_models'),
        'parallel_envs': 10,
        'num_episodes': int(5E6),
    }

    if sim_config['use_preloaded_deployments']:
        with h5py.File(h5file_deployments_path, 'r') as f:
            STA_matrix_save = f['STA_matrix_save'][:]
            channelMatrix_save = f['channelMatrix_save'][:]

    for iter_number in range(ITERATIONS):
        simulate_iterations(
            traffic_config, sim_config, learning_config, seeds[iter_number], iter_number
        )

    # End Timer and print elapsed time
    end_time = time.time()
    print(f"Simulation took {end_time - start_time:.2f} seconds")