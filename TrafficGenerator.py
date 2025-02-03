import time
import os
import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from Utils import *

def TrafficGenerator(STA_NUMBER, validation_flag, traffic_type, traffic_load, L, 
                     per_STA_EDCA_throughput_bianchi, EVENT_NUMBER):
    """
    Generates a list of arrival times for each STA based on the specified traffic model.
    """

    # Traffic load configuration
    traffic_loads = {'low': 0.3, 'medium': 0.6, 'high': 0.9}

    if traffic_load in traffic_loads:
        C = traffic_loads[traffic_load]
        traffic_generation_rate = C * np.min(per_STA_EDCA_throughput_bianchi) * 1E6 / L  # in packets/s
    elif traffic_load in ['30-60', '30-90', '30-120']:
        pass  # No need to set C or traffic_generation_rate for CBR traffic
    else:
        raise ValueError("Invalid traffic load specified.")

    

    # Traffic type selection
    if traffic_type == 'Poisson':
        STAs_arrivals_matrix = poisson_fixed_events(STA_NUMBER, validation_flag, EVENT_NUMBER, traffic_generation_rate)
    elif traffic_type == 'Bursty':
        STAs_arrivals_matrix = generate_burst_traffic(STA_NUMBER, EVENT_NUMBER, traffic_generation_rate)
    elif traffic_type == 'CBR':
        STAs_arrivals_matrix = generate_CBR_traffic(STA_NUMBER, traffic_load, L)
    else:
        raise ValueError("Invalid traffic type specified.")

    return STAs_arrivals_matrix

def poisson_fixed_events(STA_NUMBER, validation_flag, EVENT_NUMBER, traffic_generation_rate):
    """
    Generate arrival times using a Poisson process for each STA.
    """
    STAs_arrivals_matrix = []
    for _ in range(STA_NUMBER):
        # Generate exponential inter-arrival times
        w = np.random.exponential(scale=1/traffic_generation_rate, size=EVENT_NUMBER)
        t = np.cumsum(w)

        # Validation flag handling
        if validation_flag == 'yes':
            arrivals = t[:-1]  # Ensure a packet at t=0
        else:
            arrivals = t

        STAs_arrivals_matrix.append(arrivals)

        # # Plot the exponential inter-arrival times histogram
        # plt.figure()
        # plt.hist(w, bins=50, density=True, alpha=0.75)
        # x = np.linspace(0, np.max(w), 100)
        # plt.plot(x, traffic_generation_rate * np.exp(-traffic_generation_rate * x), 'r-', lw=2)
        # plt.title('Histogram of Inter-Arrival Times with Exponential PDF')
        # plt.xlabel('Inter-Arrival Time')
        # plt.ylabel('Probability Density')
        # plt.show()

    return STAs_arrivals_matrix

def generate_burst_traffic(STA_NUMBER, EVENT_NUMBER, traffic_generation_rate):
    """
    Generate bursty traffic arrivals for each STA.
    """
    STAs_arrivals_matrix = []

    # Average ON and OFF times
    average_on_time = 1E-3
    average_off_time = 10E-3

    # Expected proportion of time spent in the ON state
    on_off_ratio = average_on_time / (average_on_time + average_off_time)

    # Adjusted generation rate during ON periods
    adjusted_generation_rate = traffic_generation_rate / on_off_ratio

    for i in range(STA_NUMBER):
        arrival_times = np.zeros(EVENT_NUMBER)  # Preallocate space for arrival times
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
                arrival_times[total_packets_generated] = current_time
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
        STAs_arrivals_matrix.append(arrival_times)

    return STAs_arrivals_matrix

def generate_CBR_traffic(STA_NUMBER, traffic_load, L):
    """
    Generate CBR traffic arrivals for each STA.
    """
    bitrate, fps = map(float, traffic_load.split('-'))
    frame_interval = 1 / fps
    frames_per_burst = int(np.ceil((bitrate * 1E6 * frame_interval) / L))
    frame_spacing = 5E-6
    stop_timestamp = 20

    STAs_arrivals_matrix = []
    for _ in range(STA_NUMBER):
        current_time = np.random.uniform(0, frame_interval)
        interarrival_times = []
        while current_time < stop_timestamp:
            burst_times = current_time + np.arange(frames_per_burst) * frame_spacing
            interarrival_times.extend(burst_times)
            current_time += frame_interval

        STAs_arrivals_matrix.append(np.array(interarrival_times))

    return STAs_arrivals_matrix

def save_to_h5(output_dir, sim, traffic_type, traffic_load, iteration, STAs_arrivals_matrix):
    """
    Saves the STAs_arrivals_matrix into individual HDF5 files in a structured directory.
    """
    # Create the directory structure
    output_path = os.path.join(output_dir, sim, traffic_type, traffic_load)
    os.makedirs(output_path, exist_ok=True)

    # Save the current iteration to its own HDF5 file
    h5_file_path = os.path.join(output_path, f"STAs_arrivals_matrix{iteration}.h5")
    with h5py.File(h5_file_path, 'w') as h5file:
        for i, arrivals in enumerate(STAs_arrivals_matrix):
            h5file.create_dataset(f"STA_{i}", data=arrivals) # no comrpession
            # h5file.create_dataset(f"STA_{i}", data=arrivals, compression="gzip")   # with compression

def simulate_iteration(sim, traffic_type, traffic_load, iteration, STA_matrix_save, channelMatrix_save, RSSI_dB_vector_to_export_save):
    """
    Simulates one iteration and returns the STAs_arrivals_matrix.
    """
    # Deployment-dependent data
    STA_matrix = STA_matrix_save[:, :, iteration]
    channelMatrix = channelMatrix_save[:, :, iteration]
    RSSI_dB_vector_to_export = RSSI_dB_vector_to_export_save[:, :, iteration]

    # Generate per-STA EDCA throughput using your existing logic
    # Simulate necessary steps for the deployment
    association = AP_STA_Association(AP_NUMBER, STA_NUMBER, SCENARIO_TYPE)
    _, _, EDCAoverheads, _ = OverheadsCalc(EDCAaccessCategory)

    per_STA_EDCA_throughput_bianchi = Throughput_EDCA_bianchi(
        AP_NUMBER, STA_NUMBER, association, RSSI_dB_vector_to_export, PN_DBM, Nsc, NSS, 
        TXOP_DURATION, EDCAoverheads, EDCAaccessCategory
    )

    # Generate traffic for the current iteration
    STAs_arrivals_matrix = TrafficGenerator(
            STA_NUMBER, validation_flag, traffic_type, traffic_load, L, per_STA_EDCA_throughput_bianchi, 
            EVENT_NUMBER = 150000
        )
    return STAs_arrivals_matrix

# main function
if __name__ == "__main__":
    # Start Timer
    start_time = time.time()

    ###### Input parameters
    validation_flag = 'no'
    EDCAaccessCategory = 'VI'
    traffic_types = ['Poisson', 'Bursty', 'CBR']
    traffic_loads = {
        'Poisson': ['low', 'medium', 'high'],
        'Bursty': ['low', 'medium', 'high'],
        'CBR': ['30-60', '30-90', '30-120']
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
    MaxTxPower, Nsc = TXpowerCalc(BW, NSS)

    # Seed for reproducibility
    rndGeneration = {
        '20metros-8STAs': 1, 
        '20metros-16STAs': 2,
        '30metros-16STAs': 3,
    }
    np.random.seed(rndGeneration[sim])

    # Load deployment data
    h5file_deployments_path = os.path.join(os.getcwd(), 'deployments datasets', sim, 'deployment_datasets.h5')
    with h5py.File(h5file_deployments_path, 'r') as f:
        STA_matrix_save = f['STA_matrix_save'][:]
        channelMatrix_save = f['channelMatrix_save'][:]
        RSSI_dB_vector_to_export_save = f['RSSI_dB_vector_to_export_save'][:]

    # Output directory    
    output_dir = os.path.join(os.getcwd(), 'traffic datasets')

    # Run simulations with progress bar
    MAX_WORKERS = 8  # Adjust the number of workers as needed
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for traffic_type in traffic_types:
            for traffic_load in traffic_loads[traffic_type]:
                futures = [
                    executor.submit(
                        simulate_iteration, sim, traffic_type, traffic_load, i, 
                        STA_matrix_save, channelMatrix_save, RSSI_dB_vector_to_export_save
                    )
                    for i in range(ITERATIONS)
                ]
                for i, future in enumerate(tqdm(futures, desc=f"{traffic_type} {traffic_load}", unit=" iteration")):
                    try:
                        STAs_arrivals_matrix = future.result()
                        save_to_h5(output_dir, sim, traffic_type, traffic_load, i, STAs_arrivals_matrix)
                    except Exception as e:
                        print(f"Error in iteration {i} for {traffic_type} {traffic_load}: {e}")

    # End Timer and print elapsed time
    end_time = time.time()
    print(f"Simulation took {end_time - start_time:.2f} seconds")