import numpy as np
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Import tqdm for progress bar
from Utils import *
from DeploymentGenerator import deployment_generator
import h5py
import os


# Helper function to simulate one iteration
def throughput_calculation(sim_config, show_plot= None, iter=None):

    # Deployment-dependent data
    AP_matrix, STA_matrix, association, channelMatrix = deployment_generator(sim_config, show_plot=show_plot)

    if iter is not None:
        # Deployment-dependent data
        STA_matrix = STA_matrix_save[:,:,iter]
        channelMatrix = channelMatrix_save[:,:,iter]

    # Overheads
    preTX_overheadsDCF, preTX_overheadsCSR, DCFoverheads, CSRoverheads = OverheadsCalc(EDCAaccessCategory)

    map_matrix, TxPowerMatrixTemp, comb_ok, _ = CG_creationTPC(AP_NUMBER, 
                                                STA_NUMBER, 
                                                PN_DBM, 
                                                NSC, 
                                                NSS, 
                                                association, 
                                                channelMatrix, 
                                                MaxTxPower, 
                                                CG_filter='on', TPC_method='PSO')    # TPC Optimization method: None, 'PSO', 'IPOPT', 'DE'
    
    TxPowerMatrix = [row.tolist() for i, row in enumerate(TxPowerMatrixTemp) if comb_ok[i]==True]
    CGs_STAs = [row.tolist() for i, row in enumerate(map_matrix) if comb_ok[i]==True]

    # Validate that TxPowerMatrix and CGs_STAs have the same length
    if len(TxPowerMatrix) != len(CGs_STAs):
        raise ValueError('TxPowerMatrix and CGs_STAs have different lengths')
    
    per_STA_DCF_throughput_bianchi = Throughput_DCF_bianchi(AP_NUMBER, STA_NUMBER, association, channelMatrix, sim_config['MaxTxPower'],
                                                            PN_DBM, NSC, NSS, TXOP_DURATION, 
                                                            DCFoverheads, EDCAaccessCategory)

    DL_throughput_CSR_bianchi = Throughput_CSR_bianchi(AP_NUMBER, STA_NUMBER, association, CGs_STAs, TxPowerMatrix, channelMatrix, PN_DBM, NSC, NSS, TXOP_DURATION,
                                                       CSRoverheads, EDCAaccessCategory)
                                                  
    return per_STA_DCF_throughput_bianchi, DL_throughput_CSR_bianchi


if __name__ == '__main__':
    # Start Timer
    start_time = time.time()

    # Simulation parameters
    EDCAaccessCategory = 'VI'

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

    # Channel-related parameters
    MaxTxPower, NSC = TXpowerCalc(BW, NSS)

    # For reproducibility
    np.random.seed(1)  

    # Simulation Configuration
    sim_config = {
        'AP_NUMBER': AP_NUMBER,
        'STA_NUMBER': STA_NUMBER,
        'SCENARIO_TYPE': SCENARIO_TYPE,
        'GRID_VALUE': GRID_VALUE,
        'walls': walls,
        'MaxTxPower': MaxTxPower,
        'EDCAaccessCategory': EDCAaccessCategory,
        'validation_flag': 'no',
        'TXOP_DURATION': TXOP_DURATION,
        'PN_DBM': PN_DBM,
        'CCA': CCA,
        'NSS': NSS,
        'NSC': NSC,
        'seed': 1
    }

    # Define output folder
    h5file_path = os.path.join(os.getcwd(), 'deployments datasets' , sim, 'deployment_datasets.h5')

    # Open the HDF5 file in read mode
    with h5py.File(h5file_path, 'r') as f:
        # Load datasets into variables
        STA_matrix_save = f['STA_matrix_save'][:]
        channelMatrix_save = f['channelMatrix_save'][:]

    # Simulation parameters for parallel processing
    ITERATIONS = 100

    # Pre-allocate result arrays
    per_STA_DCF_throughput_bianchi = np.zeros((ITERATIONS, STA_NUMBER))
    DL_throughput_CSR_bianchi = np.zeros((ITERATIONS, STA_NUMBER))

    futures = []
    max_workers = min(os.cpu_count(), ITERATIONS)  # Optimize worker count
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        for i in range(ITERATIONS):
            futures.append(executor.submit(throughput_calculation, sim_config))
        
        # Process results as they complete
        for i, future in enumerate(tqdm(futures, desc="Processing", unit=" iterations")):
            try:
                dcf_throughput, csr_throughput = future.result()
                per_STA_DCF_throughput_bianchi[i, :] = dcf_throughput
                DL_throughput_CSR_bianchi[i, :] = csr_throughput
            except Exception as e:
                print(f"Error in iteration {i}")

    # Aggregate Throughput
    agg_thr_DCF_DL_vector = np.sum(per_STA_DCF_throughput_bianchi, axis=1)
    agg_thr_cSR_bianchi = np.sum(DL_throughput_CSR_bianchi, axis=1)

    # Flatten the results for plotting
    allSTA_DCF = per_STA_DCF_throughput_bianchi.flatten()
    allSTA_CSR = DL_throughput_CSR_bianchi.flatten()

    # End Timer and print elapsed time
    end_time = time.time()
    print(f"Simulation took {end_time - start_time:.2f} seconds")

    # Helper function for ECDF
    def ecdf(data):
        x = np.sort(data)
        y = np.arange(1, len(x)+1) / len(x)
        return x, y

    # Plotting ECDF for aggregate throughput
    plt.figure()
    x_DCF, y_DCF = ecdf(agg_thr_DCF_DL_vector)
    x_CSR, y_CSR = ecdf(agg_thr_cSR_bianchi)
    plt.plot(x_DCF, y_DCF, label="DCF")
    plt.plot(x_CSR, y_CSR, label="CSR")
    plt.xlabel('Aggregate Throughput (Mbps)')
    plt.ylabel('Cumulative Distribution Function')
    plt.legend()
    plt.grid(True)
    # plt.title('CDF of Throughput')

    # Plotting ECDF for per-STA throughput
    plt.figure()
    x_STA_DCF, y_STA_DCF = ecdf(allSTA_DCF)
    x_STA_CSR, y_STA_CSR = ecdf(allSTA_CSR)
    plt.plot(x_STA_DCF, y_STA_DCF, label="DCF")
    plt.plot(x_STA_CSR, y_STA_CSR, label="CSR")
    plt.xlabel('Per-STA Throughput (Mbps)')
    plt.ylabel('Cumulative Distribution Function')
    plt.legend()
    plt.grid(True)
    # plt.title('CDF of Throughput')

    plt.show()


    # ################## Saving the results ##########################
    # # Define the folder structure for saving
    output_folder = os.path.join(os.getcwd(), 'Results', 'ThroughputCalculation', sim)
    os.makedirs(output_folder, exist_ok=True)

    # # # Save the results in .npy
    # # np.save(os.path.join(output_folder, "per_STA_DCF_throughput_bianchi.npy"), per_STA_DCF_throughput_bianchi)
    # # np.save(os.path.join(output_folder, "DL_throughput_CSR_bianchi.npy"), DL_throughput_CSR_bianchi)

    # # Define file path
    # h5file_path = os.path.join(output_folder, 'simulation_results.h5')

    # # Save data to HDF5 file
    # with h5py.File(h5file_path, 'w') as f:
    #     f.create_dataset('per_STA_DCF_throughput_bianchi', data=per_STA_DCF_throughput_bianchi)
    #     f.create_dataset('DL_throughput_CSR_bianchi', data=DL_throughput_CSR_bianchi)

    # # Optionally, save as CSV for readability
    # np.savetxt(os.path.join(output_folder, "per_STA_DCF_throughput_bianchi.csv"), per_STA_DCF_throughput_bianchi, delimiter=",")
    # np.savetxt(os.path.join(output_folder, "DL_throughput_CSR_bianchi.csv"), DL_throughput_CSR_bianchi, delimiter=",")


