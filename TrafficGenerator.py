import time
import os
import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from AuxiliarFunctions import *

def TrafficGenerator(STA_number, validation_flag, traffic_type, traffic_load, L, 
                     per_STA_DCF_throughput_bianchi):
    """
    Generates a list of arrival times for each STA based on the specified traffic model.
    """
    # Number of packets transmitted during the simulation
    event_number = 150000  

    # Traffic load configuration
    traffic_loads = {'low': 0.3, 'medium': 0.6, 'high': 0.9}

    if traffic_load in traffic_loads:
        C = traffic_loads[traffic_load]
        traffic_generation_rate = C * np.min(per_STA_DCF_throughput_bianchi) * 1E6 / L  # in packets/s
    elif traffic_load in ['30-60', '30-90', '30-120']:
        pass  # No need to set C or traffic_generation_rate for VR traffic
    else:
        raise ValueError("Invalid traffic load specified.")

    

    # Traffic type selection
    if traffic_type == 'Poisson':
        STAs_arrivals_matrix = poisson_fixed_events(STA_number, validation_flag, event_number, traffic_generation_rate)
    elif traffic_type == 'Bursty':
        STAs_arrivals_matrix = generate_burst_traffic(STA_number, event_number, traffic_generation_rate)
    elif traffic_type == 'VR':
        STAs_arrivals_matrix = generate_vr_traffic(STA_number, traffic_load, L)
    else:
        raise ValueError("Invalid traffic type specified.")

    return STAs_arrivals_matrix

def poisson_fixed_events(STA_number, validation_flag, event_number, traffic_generation_rate):
    """
    Generate arrival times using a Poisson process for each STA.
    """
    STAs_arrivals_matrix = []
    for _ in range(STA_number):
        # Generate exponential inter-arrival times
        w = np.random.exponential(scale=1/traffic_generation_rate, size=event_number)
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

def generate_burst_traffic(STA_number, event_number, traffic_generation_rate):
    """
    Generate bursty traffic arrivals for each STA.
    """
    STAs_arrivals_matrix = []
    average_on_time = 1E-3
    average_off_time = 10E-3
    on_off_ratio = average_on_time / (average_on_time + average_off_time)
    adjusted_rate = traffic_generation_rate / on_off_ratio

    for _ in range(STA_number):
        arrival_times = []
        current_time = 0
        while len(arrival_times) < event_number:
            # ON period
            on_duration = np.random.exponential(average_on_time)
            packets_in_burst = int(on_duration * adjusted_rate)
            inter_arrival_times = np.random.exponential(scale=1/adjusted_rate, size=packets_in_burst)
            arrival_times.extend(current_time + np.cumsum(inter_arrival_times))

            # OFF period
            off_duration = np.random.exponential(average_off_time)
            current_time += on_duration + off_duration

        STAs_arrivals_matrix.append(np.array(arrival_times[:event_number]))

    return STAs_arrivals_matrix

def generate_vr_traffic(STA_number, traffic_load, L):
    """
    Generate VR traffic arrivals for each STA.
    """
    bitrate, fps = map(float, traffic_load.split('-'))
    frame_interval = 1 / fps
    frames_per_burst = int(np.ceil((bitrate * 1E6 * frame_interval) / L))
    frame_spacing = 5E-6
    stop_timestamp = 20

    STAs_arrivals_matrix = []
    for _ in range(STA_number):
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

    # Generate per-STA DCF throughput using your existing logic
    # Simulate necessary steps for the deployment
    association = AP_STA_Association(AP_number, STA_number, scenario_type)
    _, _, DCFoverheads, _ = OverheadsCalc(EDCAaccessCategory)

    per_STA_DCF_throughput_bianchi = Throughput_DCF_bianchi(
        AP_number, STA_number, association, RSSI_dB_vector_to_export, Pn_dBm, Nsc, Nss, 
        TXOP_duration, DCFoverheads, EDCAaccessCategory
    )

    # Generate traffic for the current iteration
    STAs_arrivals_matrix = TrafficGenerator(STA_number, validation_flag, traffic_type, traffic_load, L, 
                                            per_STA_DCF_throughput_bianchi)
    return STAs_arrivals_matrix




# Start Timer
start_time = time.time()

###### Input parameters
validation_flag = 'no'
EDCAaccessCategory = 'VI'
traffic_types = ['Poisson', 'Bursty', 'VR']
traffic_loads = {
    'Poisson': ['low', 'medium', 'high'],
    'Bursty': ['low', 'medium', 'high'],
    'VR': ['30-60', '30-90', '30-120']
}


# Scenario-related
AP_number = 4
STA_number = 16
grid_value = 60
scenario_type = 'grid'
sim = '30metros-16STAs'
walls = np.array([[0, grid_value, grid_value/2, grid_value/2], 
                  [grid_value/2, grid_value/2, 0, grid_value]])

# System-related parameters
TXOP_duration = 5E-3
Pn_dBm = -95
Cca = -82
BW = 80
Nss = 2
L = 12E3

iterations = 100


### Channel-related parameters
MaxTxPower, Nsc = TXpowerCalc(BW, Nss)

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
max_workers = 8  # Adjust the number of workers as needed
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    for traffic_type in traffic_types:
        for traffic_load in traffic_loads[traffic_type]:
            futures = [
                executor.submit(
                    simulate_iteration, sim, traffic_type, traffic_load, i, 
                    STA_matrix_save, channelMatrix_save, RSSI_dB_vector_to_export_save
                )
                for i in range(iterations)
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