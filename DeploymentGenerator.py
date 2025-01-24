import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Import tqdm for progress bar
from Utils import *
import h5py
import os

################## Deployment Generator ###########################


# Helper function to simulate one iteration
def deployment_generator(sim_config, show_plot=False):
    """ 
    Simulate one iteration and return the deployment matrices.
    """
    # Generate the AP and STA coordinates
    AP_matrix, STA_matrix = AP_STA_coordinates(sim_config['AP_NUMBER'], 
                                                sim_config['STA_NUMBER'], 
                                                sim_config['SCENARIO_TYPE'], 
                                                sim_config['GRID_VALUE'])
    
    # Compute the channelMatrix
    channelMatrix = GetChannelMatrix(sim_config['MaxTxPower'], 
                                        sim_config['CCA'], 
                                        AP_matrix, 
                                        STA_matrix, 
                                        sim_config['SCENARIO_TYPE'], 
                                        sim_config['walls'])

    # Association list
    association = AP_STA_Association(sim_config['AP_NUMBER'], 
                                        sim_config['STA_NUMBER'], 
                                        sim_config['SCENARIO_TYPE'])
    

    
    # Call the function to plot
    if show_plot:
        PlotDeployment(AP_matrix, STA_matrix, association, sim_config['GRID_VALUE'], sim_config['walls'])

    
    return AP_matrix, STA_matrix, association, channelMatrix

if __name__ == '__main__':
    # Start Timer
    start_time = time.time()

    ###### Input parameters

    # Simulation parameters
    traffic_type = 'VR'
    traffic_load = '30-60'
    EDCAaccessCategory = 'VI'

    # Scenario-related
    AP_NUMBER = 4
    STA_NUMBER = 16
    GRID_VALUE = 40
    SCENARIO_TYPE = 'grid'
    sim = '20metros-16STAs'
    walls = np.array([[0, GRID_VALUE, GRID_VALUE/2, GRID_VALUE/2], 
                    [GRID_VALUE/2, GRID_VALUE/2, 0, GRID_VALUE]])

    # System-related parameters
    TXOP_DURATION = 5E-3
    PN_DBM = -95
    CCA = -82
    BW = 80
    NSS = 2
    L = 12E3

    # Channel-related parameters
    MaxTxPower, NSC = TXpowerCalc(BW, NSS)

    # Seed for reproducibility
    rndGeneration = {
        '20metros-8STAs': 1, 
        '20metros-16STAs': 2,
        '30metros-16STAs': 3,
    }
    np.random.seed(rndGeneration[sim])

    # Simulation parameters for parallel processing
    ITERATIONS = 100

    # Pre-allocate variables
    STA_matrix_save = np.empty((STA_NUMBER, 2, ITERATIONS))        
    channelMatrix_save = np.empty((STA_NUMBER, AP_NUMBER, ITERATIONS))

    # Pre-allocate result arrays
    per_STA_DCF_throughput_bianchi = np.zeros((ITERATIONS, STA_NUMBER))
    DL_throughput_CSR_bianchi = np.zeros((ITERATIONS, STA_NUMBER))


    EDCAaccessCategory = 'VI' if traffic_type == 'VR' else 'BE'

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
        'EDCAaccessCategory': EDCAaccessCategory,
        'validation_flag': 'no',
        'TXOP_DURATION': TXOP_DURATION,
        'PN_DBM': PN_DBM,
        'CCA': CCA,
        'NSS': NSS,
        'NSC': NSC,
        'seed': 1
    }

    futures = []
    max_workers = min(os.cpu_count(), ITERATIONS)  # Optimize worker count
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        for i in range(ITERATIONS):
            futures.append(executor.submit(deployment_generator, sim_config))
        
        # Process results as they complete
        for i, future in enumerate(tqdm(futures, desc="Processing", unit=" iterations")):
            try:
                _, STA_matrix, _, channelMatrix = future.result()
                STA_matrix_save[:, :, i] = STA_matrix
                channelMatrix_save[:, :, i] = channelMatrix
            except Exception as e:
                print(f"Error in iteration {i}")


    # # Define output folder
    # output_folder = os.path.join(os.getcwd(), 'deployments datasets' , sim)
    # os.makedirs(output_folder, exist_ok=True)

    # # Define file path
    # h5file_path = os.path.join(output_folder, 'deployment_datasets.h5')

    # # Save data to HDF5 file
    # with h5py.File(h5file_path, 'w') as f:
    #     f.create_dataset('STA_matrix_save', data=STA_matrix_save)
    #     f.create_dataset('channelMatrix_save', data=channelMatrix_save)

    # End Timer and print elapsed time
    end_time = time.time()
    print(f"Simulation took {end_time - start_time:.2f} seconds")