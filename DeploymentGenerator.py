import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Import tqdm for progress bar
from numpy.random import SeedSequence, default_rng

import Utils as utils
import h5py
import os

################## Deployment Generator ###########################


# Helper function to simulate one iteration
def deployment_generator(sim_config, seed, show_plot=False):
    """ 
    Simulate one iteration and return the deployment matrices.
    """
    # Set the seed
    np.random.seed(seed)

    # Generate the AP and STA coordinates
    AP_matrix, STA_matrix = utils.AP_STA_coordinates(sim_config['AP_NUMBER'], 
                                                sim_config['STA_NUMBER'], 
                                                sim_config['SCENARIO_TYPE'], 
                                                sim_config['GRID_VALUE'])
    
    # Compute the channelMatrix
    channelMatrix = utils.GetChannelMatrix(sim_config['MaxTxPower'], 
                                        sim_config['CCA'], 
                                        AP_matrix, 
                                        STA_matrix, 
                                        sim_config['SCENARIO_TYPE'], 
                                        sim_config['walls'])

    # Association list
    association = utils.AP_STA_Association(sim_config['AP_NUMBER'], 
                                        sim_config['STA_NUMBER'], 
                                        sim_config['SCENARIO_TYPE'])
    

    
    # Call the function to plot
    if show_plot:
        utils.PlotDeployment(AP_matrix, STA_matrix, association, sim_config['GRID_VALUE'], sim_config['walls'])

    
    return AP_matrix, STA_matrix, association, channelMatrix

if __name__ == '__main__':
    # Start Timer
    start_time = time.time()

    sim = '30-20'  # Simulation name: 'APtoAPdistance-STA_NUMBER'
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
    L = 12E3

    # Channel-related parameters
    MaxTxPower, NSC = utils.TXpowerCalc(BW, NSS)

    # Number of iterations
    ITERATIONS = 100

    seed_seq = SeedSequence(1)
    seeds = seed_seq.generate_state(ITERATIONS)

    # Pre-allocate variables
    STA_matrix_save = np.empty((STA_NUMBER, 2, ITERATIONS))        
    channelMatrix_save = np.empty((STA_NUMBER, AP_NUMBER, ITERATIONS))

    # Simulation Configuration
    sim_config = {
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
        'seed': 1
    }

    max_workers = min(os.cpu_count(), ITERATIONS)  # Optimize worker count
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(
                deployment_generator, sim_config, seeds[iter_number]
            )
            for iter_number in range(ITERATIONS)
        ]        
        # Process results as they complete
        for iter_number, future in enumerate(tqdm(futures, desc="Processing", unit=" iterations")):
            try:
                _, STA_matrix, _, channelMatrix = future.result()
                STA_matrix_save[:, :, iter_number] = STA_matrix
                channelMatrix_save[:, :, iter_number] = channelMatrix
            except Exception as e:
                print(f"Error in iteration {iter_number}")



    # Define output folder
    output_folder = os.path.join(os.getcwd(), 'deployments datasets' , sim)
    os.makedirs(output_folder, exist_ok=True)

    # Define file path
    h5file_path = os.path.join(output_folder, 'deployment_datasets.h5')

    # Save data to HDF5 file
    with h5py.File(h5file_path, 'w') as f:
        f.create_dataset('STA_matrix_save', data=STA_matrix_save)
        f.create_dataset('channelMatrix_save', data=channelMatrix_save)

    # End Timer and print elapsed time
    end_time = time.time()
    print(f"Simulation took {end_time - start_time:.2f} seconds")