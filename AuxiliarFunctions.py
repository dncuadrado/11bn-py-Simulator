import math  # Import math module here
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# optimization problem
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from pyswarm import pso  # Particle Swarm Optimization library

# For CGcreationTPC
from itertools import product



####################################################################################################################
# Function to calculate the TX power and the number of subcarriers 
def TXpowerCalc(BW, Nss):
    """
    Compute the number of subcarriers, Nsc, as well as the total power used depending on the bandwidth and the number of spatial streams
    """

    bw_to_nsc = {20: 234, 40: 468, 80: 980, 160: 1960}
    Nsc = bw_to_nsc.get(BW, None)
    if Nsc is None:
        raise ValueError(f"Unsupported bandwidth: {BW}")
    
    PSdensity = 5               # power spectral density in dBm/MHz (it cannot exceed 10 dBm/MHz by regulation, and the EIRP is even limited to 23 dBm in Europe)
    EIRP = PSdensity + 10 * math.log10(BW)                      # EIRP is constant by regulation in the 6Gz band
    EIRP = min(EIRP, 23)                                        # By ETSI constraint EIRP cannot exceed 23 dBm in Europe
    tx_power_ss = EIRP - 10 * math.log10(Nss)                   # transmission power per spatial stream
    
    return tx_power_ss, Nsc
####################################################################################################################

# Function to calculate the overheads
def OverheadsCalc(EDCAaccessCategory):
    """
    Computes the needed overheads for the transmission process.
    Returns: preTX_overheadsDCF, preTX_overheadsCSR, DCFoverheads, CSRoverheads
    """
    # Computes the needed overheads
    time_preamble_data = 100e-6
    
    # Time durations for different overheads
    TRTS = 56E-6
    TCTS = 48E-6
    TSIFS = 16e-6  # Shortest Interframe spacing (SIFS time)
    
    # DIFS = 34e-6  # DCF Interframe spacing (DIFS time)
    Te = 9e-6  # Duration of a single backoff slot
    TBACK = 100E-6

    TMAPC_ICF = 74.4E-6
    TMAPC_ICR = 88E-6
    TMAPC_TF = 74.4E-6

    AIFSN = 2 if EDCAaccessCategory in ['VI', 'VO'] else 3
    AIFS = AIFSN * Te + TSIFS

    # DCF Overheads
    preTX_overheadsDCF = TRTS + TSIFS + TCTS + TSIFS + time_preamble_data
    DCFoverheads = TRTS + TSIFS + TCTS + TSIFS + time_preamble_data + TSIFS + TBACK + AIFS + Te

    # CSR Overheads
    preTX_overheadsCSR = TMAPC_ICF + TSIFS + TMAPC_ICR + TSIFS + TMAPC_TF + TSIFS + time_preamble_data
    CSRoverheads = TMAPC_ICF + TSIFS + TMAPC_ICR + TSIFS + TMAPC_TF + TSIFS + time_preamble_data + TSIFS + TBACK + AIFS + Te

    return preTX_overheadsDCF, preTX_overheadsCSR, DCFoverheads, CSRoverheads
####################################################################################################################

# Function to calculate the AP-STA coordinates
def AP_STA_coordinates(AP_number, STA_number, scenario_type, grid_value):
    """
    Computes the matrices with the coordinates of the devices (AP_matrix and STA_matrix).
    Returns: AP_matrix, STA_matrix
    """
    STAs_per_AP = STA_number // AP_number

    # Max Distance between AP and STA (used only in 'grid')
    AP_STA_max_distance = min(10, grid_value / 4)  # in meters

    # Initialize matrices
    AP_matrix = np.zeros((AP_number, 2))
    STA_matrix = np.zeros((STA_number, 2))

    if scenario_type == 'grid':
        # APs are manually placed in a grid
        AP_matrix = np.array([[grid_value / 4, grid_value / 4],
                              [grid_value / 4, 3 * grid_value / 4],
                              [3 * grid_value / 4, grid_value / 4],
                              [3 * grid_value / 4, 3 * grid_value / 4]])

        # STA placement around each AP
        t = 2 * np.pi * np.random.rand(STA_number)
        g = 1 + (AP_STA_max_distance - 1) * np.random.rand(STA_number)
        AP_indices = np.repeat(np.arange(AP_number), STAs_per_AP)
        STA_matrix[:, 0] = AP_matrix[AP_indices, 0] + g * np.cos(t)
        STA_matrix[:, 1] = AP_matrix[AP_indices, 1] + g * np.sin(t)

    elif scenario_type == 'random':
        # APs randomly placed all over the area
        AP_matrix = grid_value * np.random.rand(AP_number, 2)

        # STAs randomly placed all over the area
        STA_matrix = grid_value * np.random.rand(STA_number, 2)

    # Validations
    if AP_matrix.shape[0] != AP_number:
        raise ValueError('AP_number does not match with AP_matrix dimension')
    if STA_matrix.shape[0] != STA_number:
        raise ValueError('STA_number does not match with STA_matrix dimension')

    return AP_matrix, STA_matrix
####################################################################################################################

# Function to calculate the AP-STA association
def AP_STA_Association(AP_number, STA_number, scenario_type):
    """
    Association process. STAs are associated independently of the distance to their corresponding AP.
    Returns a list with the list of STAs by AP.
    """
    if scenario_type == 'grid':
        STAs_per_AP = STA_number // AP_number
        # Create a list of STAs and reshape it to match the AP-STA association
        association = np.arange(STA_number).reshape(AP_number, STAs_per_AP).tolist()

    return association
####################################################################################################################

# Function to plot the devices' locations in the deployment
def PlotDeployment(AP_matrix, STA_matrix, association, grid_value, walls):
    """
    Plots the deployment with APs, STAs, and walls.
    """
    
    # # Get a list of all available font names
    # import matplotlib.font_manager as fm
    # available_fonts = sorted([f.name for f in fm.fontManager.ttflist])
    # # Print the list of available fonts
    # for font in available_fonts:
    #     print(font)


    plt.rcParams['font.family'] = 'Noto Sans Mono'
    # plt.rcParams['font.family'] = 'Tlwg Typewriter'

    # Create figure
    plt.figure(figsize=(8, 8))

    # Define AP colors
    if AP_matrix.shape[0] < 10:
        AP_colours = np.array([[0.0763082893739572, 0.499882500825560, 0.931206019689022],
                               [0.779918792240115, 0.679229996120941, 0.0248992275503480],
                               [0.438409231440894, 0.803739036104376, 0.600548917464123],
                               [0.723465177830941, 0.380941133148538, 0.950129500413646],
                               [0.977989511996603, 0.0659363469059051, 0.230302879020965],
                               [0.538495870410434, 0.288145599307994, 0.548489919236030],
                               [0.501120463659938, 0.909593527719614, 0.909128374886731],
                               [0.0720511333597615, 0.213385353579916, 0.133169445759250],
                               [0.268438980101871, 0.452123961817683, 0.523412580673766]])
    else:
        AP_colours = np.random.rand(AP_matrix.shape[0], 3)

    # Plot APs
    for k in range(AP_matrix.shape[0]):
        plt.plot(AP_matrix[k, 0], AP_matrix[k, 1], 'v', markersize=6, color=AP_colours[k, :])
        plt.text(AP_matrix[k, 0], AP_matrix[k, 1], f'AP${{ {k} }}$', fontsize=14, ha='center', va='bottom')

    # Plot STAs
    for j in range(STA_matrix.shape[0]):
        # Find the position where the current STA is associated
        idxCol = None
        for i, assoc in enumerate(association):
            if j in assoc:
                idxCol = i
                break
        
        if idxCol is not None:  # If a valid association was found
            plt.plot(STA_matrix[j, 0], STA_matrix[j, 1], 's', markersize=6, color=AP_colours[idxCol, :])
            plt.text(STA_matrix[j, 0], STA_matrix[j, 1], f'STA${{ {j} }}$', fontsize=14, ha='center', va='bottom')
        else:
            print(f"STA {j} is not associated with any AP.")

    # Plot walls
    for i in range(walls.shape[0]):
        plt.plot([walls[i, 0], walls[i, 1]], [walls[i, 2], walls[i, 3]], color='k', linewidth=2)

    # Add grid and labels
    plt.grid(True)
    plt.xlabel('X-axis [meters]', fontsize=16)
    plt.ylabel('Y-axis [meters]', fontsize=16)
    plt.xticks(np.arange(0, grid_value + 1, 5))
    plt.yticks(np.arange(0, grid_value + 1, 5))
    plt.xlim([0, grid_value])
    plt.ylim([0, grid_value])

    plt.show()
####################################################################################################################

# Function to calculate the pathloss
def Getloss(a_position, b_position, m_NumberOfWalls, std_dev):
    """
    Calculates the path loss between two devices using the Enterprise model 
    defined for 802.11ax.
    
    Parameters:
    - a_position (array-like): Coordinates (x, y) of device A.
    - b_position (array-like): Coordinates (x, y) of device B.
    - m_NumberOfWalls (int): Number of walls between the devices.
    - std_dev (float): Standard deviation for shadowing in dB.

    Returns:
    - loss (float): Path loss in dB.
    """
    loss = 0.0            
    
    m_dBP = 10.0           # Breaking point distance from which an additional loss factor is added 
    m_frequency = 6        # Frequency in GHz; 6-GHz band

    distance = np.linalg.norm(np.array(a_position) - np.array(b_position))

    # Additional loss (when distance is greater than the breaking point)
    addLoss = 0.0
    if distance >= m_dBP:
        addLoss = 35 * np.log10(distance / m_dBP)

    shadowing = std_dev * np.random.randn()  # Shadowing in dB

    loss = 40.05 + 20 * np.log10(m_frequency / 2.4) + 20 * np.log10(min(distance, m_dBP)) + addLoss + 7 * m_NumberOfWalls + shadowing

    return loss
####################################################################################################################

# Function to check if two segments intersect (number of walls between devices)
def checkSegmentIntersection(x1a1, x2a1, y1a1, y2a1, a2):
    """
    Returns True if segment a1 crosses segment a2. This function is used to determine
    whether the transmission defined by the two points in a1 intersects with the wall
    defined by segment a2.

    Parameters:
    x1a1 (float): x-coordinate of the first point of segment a1.
    x2a1 (float): x-coordinate of the second point of segment a1.
    y1a1 (float): y-coordinate of the first point of segment a1.
    y2a1 (float): y-coordinate of the second point of segment a1.
    a2 (array-like): Coordinates [x1, x2, y1, y2] defining the second segment a2.

    Returns:
    bool: True if segments intersect, False otherwise.
    """

    x1a2, x2a2, y1a2, y2a2 = a2
    # Helper function to calculate orientation
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise

    # Helper function to check if a point is on a segment
    def on_segment(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    # Define the endpoints
    p1, q1 = (x1a1, y1a1), (x2a1, y2a1)
    p2, q2 = (x1a2, y1a2), (x2a2, y2a2)

    # Calculate orientations
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special cases
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False
####################################################################################################################

# Function to calculate channelMatrix and RSSI_dB_vector_to_export
def GetChannelMatrix(MaxTxPower, Cca, AP_matrix, STA_matrix, scenario_type, walls, checkSegmentIntersection, Getloss):
    """
    Calculates the channel matrix and RSSI [dB] values for each AP-STA and AP-AP pairs. Validate RSSI between APs is above the CCA

    returns: channelMatrix, RSSI_dB_vector_to_export
    """

    # Matrix to store all AP-STA channel coefficients
    channelMatrix = np.zeros((STA_matrix.shape[0], AP_matrix.shape[0]))
    
    # Stores the RSSI value considering the channel effect and the maximum TX power allowed
    RSSI_dB_vector_to_export = np.zeros((STA_matrix.shape[0], AP_matrix.shape[0]))

    NumberOfWallsAP_STA_Matrix = np.zeros((STA_matrix.shape[0], AP_matrix.shape[0]))  # Matrix for walls between APs and STAs
    NumberOfWallsAP_AP_Matrix = np.zeros((AP_matrix.shape[0], AP_matrix.shape[0]))  # Matrix for walls between APs

    AP_to_AP_RSSI_matrix = np.zeros((AP_matrix.shape[0], AP_matrix.shape[0]))  # AP to AP RSSI matrix

    # Loop to calculate channel matrix and RSSI values for each STA-AP pair
    for k in range(AP_matrix.shape[0]):
        for kk in range(STA_matrix.shape[0]):
            for kkk in range(walls.shape[0]):  # Checking the number of walls between AP_k and STA_kk
                isIntersecting = checkSegmentIntersection(AP_matrix[k, 0], STA_matrix[kk, 0], AP_matrix[k, 1], STA_matrix[kk, 1], walls[kkk, :])
                if isIntersecting:
                    NumberOfWallsAP_STA_Matrix[kk, k] += 1

            std_dev = 5  # Standard deviation for shadowing
            channelCoefficient_dB = Getloss(AP_matrix[k, :], STA_matrix[kk, :], NumberOfWallsAP_STA_Matrix[kk, k], std_dev)
            channelMatrix[kk, k] = 1 / 10 ** (channelCoefficient_dB / 10)
            RSSI_dB_vector_to_export[kk, k] = MaxTxPower - channelCoefficient_dB

        # AP to AP interactions
        AP_other_vector = list(set(range(AP_matrix.shape[0])) - set([k]))  # All APs except AP k
        for i in AP_other_vector:
            for ii in range(walls.shape[0]):  # Checking the number of walls between AP_k and AP_i
                isIntersecting = checkSegmentIntersection(AP_matrix[k, 0], AP_matrix[i, 0], AP_matrix[k, 1], AP_matrix[i, 1], walls[ii, :])
                if isIntersecting:
                    NumberOfWallsAP_AP_Matrix[k, i] += 1

            # Calculate RSSI for AP to AP connections
            if scenario_type == 'random':
                std_dev = 0  # No shadowing between APs
                channelCoefficient_dB = Getloss(AP_matrix[k, :], AP_matrix[i, :], NumberOfWallsAP_AP_Matrix[k, i], std_dev)
                AP_to_AP_RSSI_matrix[k, i] = MaxTxPower - channelCoefficient_dB
            elif scenario_type == 'grid':
                std_dev = 0  # No shadowing between APs
                channelCoefficient_dB = Getloss(AP_matrix[k, :], AP_matrix[i, :], 1, std_dev)  # Only 1 wall between APs
                AP_to_AP_RSSI_matrix[k, i] = MaxTxPower - channelCoefficient_dB

    # Validation check for RSSI between APs
    if np.min(AP_to_AP_RSSI_matrix) < Cca:
        raise ValueError('Scenario constraint: RSSI between APs is under the Cca threshold. All APs should be in the coverage area of the others')

    return channelMatrix, RSSI_dB_vector_to_export
####################################################################################################################

# Function to calculate the MCS, N_bps, and Rc for a given SINR value
def MCS_cal_PER_001(SINR_db):
    """
    Calculates the 802.11be MCS and related parameters (N_bps, Rc) for a given SINR value.
    
    Args:
    SINR_db (float): SINR value in dB
    
    Returns:
    MCS (int): Modulation and Coding Scheme index (NaN if invalid)
    N_bps (int): Number of coded bits per subcarrier per stream (NaN if invalid)
    Rc (float): Rate of coding (NaN if invalid)
    """
    if SINR_db < 14.2862:
        MCS = np.nan
        N_bps = np.nan
        Rc = np.nan
    elif 14.2862 <= SINR_db < 19.5154:
        MCS = int(0)
        N_bps = 1
        Rc = 1 / 2
    elif 19.5154 <= SINR_db < 25.5501:
        MCS = int(1)
        N_bps = 2
        Rc = 1 / 2
    elif 25.5501 <= SINR_db < 27.9312:
        MCS = int(2)
        N_bps = 2
        Rc = 3 / 4
    elif 27.9312 <= SINR_db < 33.7179:
        MCS = int(3)
        N_bps = 4
        Rc = 1 / 2
    elif 33.7179 <= SINR_db < 36.6008:
        MCS = int(4)
        N_bps = 4
        Rc = 3 / 4
    elif 36.6008 <= SINR_db < 38.8428:
        MCS = int(5)
        N_bps = 6
        Rc = 2 / 3
    elif 38.8428 <= SINR_db < 41.9447:
        MCS = int(6)
        N_bps = 6
        Rc = 3 / 4
    elif 41.9447 <= SINR_db < 43.9603:
        MCS = int(7)
        N_bps = 6
        Rc = 5 / 6
    elif 43.9603 <= SINR_db < 46.5902:
        MCS = int(8)
        N_bps = 8
        Rc = 3 / 4
    elif 46.5902 <= SINR_db < 49.1915:
        MCS = int(9)
        N_bps = 8
        Rc = 5 / 6
    elif 49.1915 <= SINR_db < 52.3450:
        MCS = int(10)
        N_bps = 10
        Rc = 3 / 4
    elif 52.3450 <= SINR_db < 53.8530:
        MCS = int(11)
        N_bps = 10
        Rc = 5 / 6
    elif 53.8530 <= SINR_db < 57.3929:
        MCS = int(12)
        N_bps = 12
        Rc = 3 / 4
    elif SINR_db >= 57.3929:
        MCS = int(13)
        N_bps = 12
        Rc = 5 / 6
    else:
        MCS = np.nan
        N_bps = np.nan
        Rc = np.nan

    return MCS, N_bps, Rc
####################################################################################################################

# Function to calculate the optimal power allocation
def power_allocation(N, noise_power, H, P_max, Nsc, Nss, method):
    """
    Optimizes power allocation using the specified method to maximize proportional fairness.
    Returns: P_opt
    """

    # Function to compute the product of rates for the given power allocation
    def compute_rates(P, H, noise_power, N, Nsc, Nss):
        """
        Computes the product of rates for the given power allocation.
        
        Args:
        P: Power allocation vector (N,)
        H: Channel gain matrix (NxN)
        noise_power: Noise power (scalar)
        N: Number of links
        Nsc: Number of subcarriers
        Nss: Number of spatial streams
        
        Returns:
        product_rate: Product of rates for all links
        """
        
        T_DFT = 12.8e-6  # OFDM symbol duration
        T_GI = 0.8e-6    # Guard Interval
        
        rates = np.zeros(N)
        
        # Calculate SINR in linear scale
        # sinr = (P * np.diag(H)) / (noise_power + np.sum(H * P[:, None], axis=0) - np.diag(H) * P)
        sinr = (P * np.diag(H)) / (noise_power + np.sum(H * P, axis=1) - np.diag(H) * P)
        sinr_dB = 10 * np.log10(sinr)
        
        for i in range(N):
            MCS, N_bps, Rc = MCS_cal_PER_001(sinr_dB[i])
            if np.isnan(MCS):
                rates[i] = 0
            else:
                rates[i] = (Nsc * N_bps * Rc * Nss) / (T_DFT + T_GI)
                # rates[i] = np.log2(1 + sinr[i])
        
        # Return the product of all rates

        if np.any(rates == 0):
            product_rate = 0  # Penalize invalid rates
        else:
            # product_rate = np.prod(rates)
            product_rate = np.sum(np.log10(rates))
        
        return product_rate
        # return np.min(rates)
    
    # Function to calculate the optimal power allocation using the Particle Swarm Optimization (PSO) algorithm
    def power_allocation_particleswarm(N, noise_power, H, P_max, Nsc, Nss):
        """
        Optimizes power allocation using Particle Swarm Optimization (PSO)
        to maximize proportional fairness.
        
        Args:
        N: Number of links (transmitters)
        noise_power: Noise power (scalar)
        H: Channel gain matrix (NxN)
        P_max: Maximum power constraint (scalar)
        Nsc: Number of subcarriers
        Nss: Number of spatial streams
        
        Returns:
        P_opt: Optimized power allocation vector (N,)
        """
        

        # Bounds: 0 <= P <= P_max for each power allocation
        lb = np.ones(N)  # Lower bound for power allocation
        ub = P_max * np.ones(N)  # Upper bound for power allocation

        # Use PSO with optimized swarm size and iterations
        swarm_size = 10  # Reduced swarm size (can be tuned)
        max_iter = 20   # Reduced number of iterations (can be tuned)

        def objective(P):
            return -compute_rates(P, H, noise_power, N, Nsc, Nss)
        
        # Silence stdout and stderr
        with open(os.devnull, 'w') as fnull:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = fnull
                sys.stderr = fnull
                # P_opt, _ = pso(objective, lb, ub, swarmsize=swarm_size, maxiter=max_iter, debug=False, 
                #        minfunc=1e-8, omega=0.9, phip=0.1, phig=0.1)
                
                P_opt, _ = pso(objective, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
                    swarmsize=swarm_size, omega=0.9, phip=0.1, phig=0.1, maxiter=max_iter, 
                    minstep=1e-8, minfunc=1e-8, debug=False)
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        # P_opt, _ = pso(objective, lb, ub, swarmsize=swarm_size, maxiter=max_iter, debug=False, 
        #                minfunc=1e-6, omega=0.5, phip=0.5, phig=0.5)
        

        return P_opt
    ####################################################################################################################

    def power_allocation_ipyopt(N, noise_power, H, P_max, Nsc, Nss):
        """
        Optimizes power allocation using IPOPT to maximize proportional fairness.

        IMPORTANT! It does not work at all when datarates are computed using: 
        rates[i] = (Nsc * N_bps * Rc * Nss) / (T_DFT + T_GI)

        instead use: 
        rates[i] = np.log2(1 + sinr[i])

        Args:
        N: Number of links (transmitters)
        noise_power: Noise power (scalar)
        H: Channel gain matrix (NxN)
        P_max: Maximum power constraint (scalar)
        Nsc: Number of subcarriers
        Nss: Number of spatial streams

        Returns:
        P_opt: Optimized power allocation vector (N,)
        """

        # Precompute constants
        T_DFT = 12.8e-6  # OFDM symbol duration
        T_GI = 0.8e-6    # Guard Interval

        # # Initial guess (equal power allocation within bounds)
        P0 = np.clip(np.full(N, P_max / N), 0, P_max)

        # Randomize the initial guess within the bounds
        # P0 = np.clip(np.random.uniform(0, P_max, N), 0, P_max)

        # Define bounds for power allocation
        bounds = [(0, P_max) for _ in range(N)]

        # Objective function to minimize (negative of log-product of rates)
        def objective(P):
            return -compute_rates(P, H, noise_power, N, Nsc, Nss)

        # Run optimization via scipy.optimize.minimize
        result = minimize(
            fun=objective,
            x0=P0,
            bounds=bounds,
            method='SLSQP',  # Sequential Least Squares Programming
            options={'maxiter': 100, 'disp': False}
        )

        if result.success:
            return result.x
        else:
            return P0
    ####################################################################################################################

    def power_allocation_differential_evolution(N, noise_power, H, P_max, Nsc, Nss):
        """
        Optimizes power allocation using Differential Evolution (DE)
        to maximize proportional fairness.
        """

        # Negative of compute_rates for maximization
        def objective(P):
            return -compute_rates(P, H, noise_power, N, Nsc, Nss)

        # Define bounds for power allocation
        bounds = [(0, P_max) for _ in range(N)]

        # Run Differential Evolution
        # result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=20, popsize=200, tol=1e-3)
        result = differential_evolution(objective, bounds, args=(), strategy='best1bin',
                            maxiter=1000, popsize=15, tol=0.01,
                            mutation=(0.5, 1), recombination=0.7, seed=None,
                            callback=None, disp=False, polish=True,
                            init='latinhypercube', atol=0, updating='immediate',
                            workers=1, constraints=(), x0=None,
                            integrality=None, vectorized=False)
        return result.x  # Optimized power allocation
    
    if method == None:
        P_opt = P = np.full(N, P_max)
    elif method == 'PSO':   # Particle Swarm Optimization
        P_opt = power_allocation_particleswarm(N, noise_power, H, P_max, Nsc, Nss)
    elif method == 'IPOPT': # Several methods inside. CUrrently selected: SQP
        P_opt = power_allocation_ipyopt(N, noise_power, H, P_max, Nsc, Nss)
    elif method == 'DE':    # DIfferential Evolution
        P_opt = power_allocation_differential_evolution(N, noise_power, H, P_max, Nsc, Nss)
    else:
        raise ValueError(f"Unsupported optimization method: {method}")

    return P_opt 
####################################################################################################################

# Function to compute the number of aggregated packets that can be transmitted in a given time
def tx_packets(Nsc, N_bps, Rc, Nss, data_tx_time):
    """
    Calculates the number of aggregated packets (A-MPDU length) that can be transmitted 
    within the given data transmission time.

    Args:
    Nsc (int): Number of subcarriers
    N_bps (int): Number of coded bits per subcarrier per stream
    Rc (float): Rate of coding
    Nss (int): Number of spatial streams
    data_tx_time (float): Data transmission time in seconds

    Returns:
    int: Number of packets that can be transmitted
    """

    T_DFT = 12.8e-6  # OFDM symbol duration (seconds)
    T_GI = 0.8e-6    # Guard interval duration (seconds)

    Lsf = 16         # Length of service field (bits)
    Lmh = 240        # MAC header (bits)
    Ld = 12000       # Frame size (bits)
    Ltail = 18       # Tail bits (bits)
    Lmd = 32         # MPDU Delimiter (bits)

    # Calculate the number of packets that can be transmitted
    bits_available = np.floor(data_tx_time / (T_DFT + T_GI)) * (Nsc * N_bps * Rc * Nss)              # Total bits that can be transmitted
    agg_packets = int(np.floor((bits_available - Lsf - Ltail) / (Lmd + Lmh + Ld)))  # Number of aggregated packets
    
    if agg_packets > 1024:
        raise ValueError('Number of aggregated packets exceeds the maximum limit')
    
    return int(agg_packets)
####################################################################################################################

# Function to compute the elapsed time for transmitting the given number of aggregated packets
def elapsed_time_tx(Nsc, N_bps, Rc, Nss, tx_Packets):
    """
    Calculates the elapsed time for transmitting the given number of aggregated packets.

    Args:
    Nsc (int): Number of subcarriers
    N_bps (int): Number of coded bits per subcarrier per stream
    Rc (float): Rate of coding
    Nss (int): Number of spatial streams
    tx_Packets (int): Number of aggregated packets

    Returns:
    float: time_tx is the elapsed time for transmitting data packets
    """

    T_DFT = 12.8e-6  # OFDM symbol duration (seconds)
    T_GI = 0.8e-6    # Guard interval duration (seconds)

    Lsf = 16         # Length of service field (bits)
    Lmh = 240        # MAC header (bits)
    Ld = 12000       # Frame size (bits)
    Ltail = 18       # Tail bits (bits)

    if tx_Packets == 0:
        Lmd = 0
    else:
        Lmd = 32         # MPDU Delimiter (bits)

    # Calculate the elapsed time for transmitting the packets
    time_tx = np.ceil((Lsf + tx_Packets*(Lmd + Lmh + Ld) + Ltail)/(Nsc*N_bps*Rc*Nss))*(T_DFT + T_GI)

    return time_tx
####################################################################################################################

# Function to compute the C-SR groups and the corresponding power allocation
def CG_creationTPC(AP_number, STA_number, CSRoverheads, Pn_dBm, Nsc, Nss, 
                   association, channelMatrix, MaxTxPower, TXOP_duration):
    # Initialization
    CG_size = AP_number  # Maximum number of APs per group
    noise_power = 10**(Pn_dBm / 10)
    MaxTxPower = 10**(MaxTxPower / 10)

    
    def generate_combinations(association, CG_size):
        AP_number = len(association)
        STA_number = sum(len(stas) for stas in association)

        # Step 1: Generate all possible combinations of STAs for each AP, including np.nan for NaN replacement
        u_with_placeholders = [np.array([np.nan] + stas, dtype=float) for stas in association]

        # Step 2: Generate all possible combinations of AP-STAs using itertools.product
        all_combinations = np.array(list(product(*u_with_placeholders)), dtype=float)

        # Step 3: Filter out rows where all elements are the placeholder (np.nan)
        valid_combinations = all_combinations[~np.all(np.isnan(all_combinations), axis=1)]

        # Step 4: Apply filtering based on CG_size
        # Only keep rows where the number of non-placeholder elements is <= CG_size
        idx_row = np.sum(~np.isnan(valid_combinations), axis=1)
        valid_combinations = valid_combinations[idx_row <= CG_size]

        # Step 5: Remove duplicate rows
        valid_combinations = np.unique(valid_combinations, axis=0)

        # Step 6: Sort rows by the number of placeholder elements (descending) and smallest non-placeholder integers
        valid_combinations = valid_combinations[
            np.lexsort((
                np.min(np.where(~np.isnan(valid_combinations), valid_combinations, np.inf), axis=1),
                -np.sum(np.isnan(valid_combinations), axis=1)
            ))
        ]

        return valid_combinations

    # Function to generate all possible combinations of AP-STAs for a given association and CG_size
    map_matrix = generate_combinations(association, CG_size)

    # Create TxPowerMatrixTemp with the shape of map_matrix
    TxPowerMatrixTemp = np.full(map_matrix.shape, np.nan)
    # Fill the first [0:STA_number] rows with MaxTxPower in the columns where map_matrix[0:STA_number] is non-NaN
    TxPowerMatrixTemp[0:STA_number] = np.where(~np.isnan(map_matrix[0:STA_number]), MaxTxPower, np.nan)

    # Other matrices initialization
    datarate = np.full_like(TxPowerMatrixTemp, np.nan)
    comb_ok = np.zeros(map_matrix.shape[0], dtype=bool)
    Discardlist = np.ones(map_matrix.shape[0], dtype=bool)

    # Main loop for verifying groups
    for i in range(map_matrix.shape[0]):
        if Discardlist[i] == False:
            continue

        APs = np.where(~np.isnan(map_matrix[i,:]))[0].astype(int)
        STAs = map_matrix[i, APs].astype(int)
        H = channelMatrix[np.ix_(STAs, APs)]

        if len(STAs) == 1:  # Use maximum power for a single STA
            P = np.full(len(STAs), MaxTxPower)  # Equivalent to MaxTxPower * ones(length(STAs), 1)
        else:  # Compute the subset of power that maximizes the proportional fair transmission
            # TPC (Power allocation)
            method = None  # Optimization method: None, 'PSO', 'IPOPT', 'DE'
            # Solving the Opt problem with the selected method
            P = power_allocation(len(STAs), noise_power, H, MaxTxPower, Nsc, Nss, method)

            # Store the power vector in TxPowerMatrixTemp
            TxPowerMatrixTemp[i, APs] = P  

        # Compute the SINR and datarate for each STA
        SINR = (P * np.diag(H)) / (noise_power + np.sum(H * P, axis=1) - np.diag(H) * P)
        SINR_db = 10 * np.log10(SINR)

        for k, _ in enumerate(STAs):
            MCS, N_bps, Rc = MCS_cal_PER_001(SINR_db[k])
            if np.isnan(MCS):
                datarate[i, APs[k]] = 0 #SINR under the threshold
            else:
                datarate[i, APs[k]] = Nsc * N_bps * Rc * Nss / (12.8e-6 + 0.8e-6) # Datarate in bps
            
            # if the datarate using CSR is lower than the datarate without CSR, discard the combination
            if len(STAs) * datarate[i, APs[k]] >= datarate[STAs[k], APs[k]]:
                continue
            else:
                # Discarding the rest of combs where this STAs appear
                Discardlist[np.sum(np.isin(map_matrix,STAs), axis=1) >= len(STAs)] = False
                break
        else:
            comb_ok[i] = True

    TxPowerMatrix = TxPowerMatrixTemp[comb_ok]
    CGs_STAs = map_matrix[comb_ok]

    return CGs_STAs, TxPowerMatrix
####################################################################################################################

# Compute of the probaility of a transmission slot, expected backoff and conditional collision probability
def SimpleDCF_modelWithBEB(N, EDCAaccessCategory):
    """
    Computes the transmission probability (tau), expected backoff (EB), and collision probability (p) 
    for a Distributed Coordination Function (DCF) with Binary Exponential Backoff (BEB).
    
    Args:
    N (int): Number of contending nodes
    EDCAaccessCategory (str): EDCA access category ('BE', 'VI', or 'VO')
    
    Returns:
    tau_ (float): Transmission probability
    EB_ (float): Expected backoff
    p_ (float): COnditional collision probability
    """
    
    MaxIter = 100
    
    # Set CWmin and m based on EDCA access category
    if EDCAaccessCategory == 'BE':
        CWmin = 15
        m = 6  # CWmax = 1023
    elif EDCAaccessCategory == 'VI':
        CWmin = 7
        m = 1  # CWmax = 15
    elif EDCAaccessCategory == 'VO':
        CWmin = 3
        m = 1
    else:
        raise ValueError("Invalid EDCA access category. Choose 'BE' or 'VI'.")
    
    # Initialize values
    tau = np.zeros(MaxIter)
    p = np.zeros(MaxIter)
    EB = np.zeros(MaxIter)
    
    tau[0] = 2 / (CWmin + 2)
    p[0] = 0
    EB[0] = 0
    
    # Fixed-point iterative approach to obtain tau and p
    for i in range(MaxIter - 1):
        # Collision Probability
        p[i + 1] = 1 - (1 - tau[i]) ** (N - 1)
        
        # Expected Backoff Duration (Binary Exponential Backoff)
        A = (1 - (2 * p[i + 1]) ** m) / (1 - 2 * p[i + 1])
        B = (2 * p[i + 1]) ** m / (1 - p[i + 1])
        EB[i + 1] = ((CWmin + 1) / 2) * (1 - p[i + 1]) * (A + B) - 1 / 2
        
        # Transmission Probability
        tau[i + 1] = 1 / (EB[i + 1] + 1)
        
        # Average to improve convergence
        if i > 4:
            tau[i + 1] = (1 / 4) * (tau[i + 1] + tau[i] + tau[i - 1] + tau[i - 2])
    
    tau_ = tau[MaxIter - 1]
    EB_ = EB[MaxIter - 1]
    p_ = p[MaxIter - 1]
    
    return tau_, EB_, p_
####################################################################################################################

# Function to compute the throughput using Bianchi's model
def Throughput_DCF_bianchi(AP_number, STA_number, association, RSSI_dB_vector_to_export, 
                            Pn_dBm, Nsc, Nss, TXOP_duration, DCFoverheads, EDCAaccessCategory):
    """
    Computes the per-station throughput using Bianchi's model.
    
    Args:
    AP_number (int): Number of access points
    STA_number (int): Number of stations
    association (list): List of associations for each AP
    RSSI_dB_vector_to_export (numpy array): RSSI values in dB
    Pn_dBm (float): Noise power in dBm
    Nsc (int): Number of subcarriers
    Nss (int): Number of spatial streams
    TXOP_duration (float): TXOP duration
    DCFoverheads (float): DCF overheads
    EDCAaccessCategory (str): EDCA access category ('VI' or 'BE')
    
    Returns:
    per_STA_DCF_throughput_bianchi (numpy array): Throughput for each station
    """
    
    # Constants
    TSIFS = 16e-6       # Shortest Interframe spacing (SIFS time)
    TRTS = 56E-6        # RTS duration
    TCTS = 48E-6        # CTS duration
    
    L = 12e3            # Single frame length
    Te = 9e-6           # Duration of a single backoff slot
    
    AIFSN = 2 if EDCAaccessCategory in ['VI', 'VO'] else 3
    
    AIFS = AIFSN * Te + TSIFS  # AIFSN*slotTime + SIFS
    Tcoll = TRTS + TSIFS + TCTS + AIFS + Te  # Collision duration
    
    # Bianchi's parameters
    tau, _, _ = SimpleDCF_modelWithBEB(AP_number, EDCAaccessCategory)
    pe = (1 - tau) ** AP_number
    ps = AP_number * tau * (1 - tau) ** (AP_number - 1)
    pc = 1 - pe - ps
    
    # Initialize throughput calculations
    rx_packets = np.zeros(STA_number)
    per_STA_DCF_throughput_bianchi = np.zeros(STA_number)
    
    # Initialize MCS parameters
    MCS = np.zeros(STA_number)
    N_bps = np.zeros(STA_number)
    Rc = np.zeros(STA_number)
    
    # Loop through each STA
    for kk in range(STA_number):
        # Find the rows (APs) where STA kk is associated
        ap_indices = [ap_idx for ap_idx, ap_stas in enumerate(association) if kk in ap_stas]
    
        # Find the AP indices where STA kk is associated
        # num_associated_stas = sum(len(association[ap_idx]) for ap_idx, ap_stas in enumerate(association) if kk + 1 in ap_stas)
    
        # Probability of STA_kk being selected
        p_STA = 1 / (AP_number * len(association[ap_indices[0]]))
        
        # Calculate SINR and MCS
        SINR_db = RSSI_dB_vector_to_export[kk, ap_indices] - Pn_dBm  # SINR calculation
        MCS[kk], N_bps[kk], Rc[kk] = MCS_cal_PER_001(SINR_db)  # MCS calculation
        
        if np.isnan(MCS[kk]):
            raise ValueError("Invalid MCS")
        
        # Calculate received packets
        rx_packets[kk] = tx_packets(Nsc, N_bps[kk], Rc[kk], Nss, TXOP_duration - DCFoverheads)
        if rx_packets[kk] > 1024:
            raise ValueError("Impossible to transmit more than 1024 MSDUs")
        
        # Throughput calculation following Bianchi's model
        per_STA_DCF_throughput_bianchi[kk] = p_STA * ps * rx_packets[kk] * L / (1e6 * (pe * Te + ps * TXOP_duration + pc * Tcoll))
    
    return per_STA_DCF_throughput_bianchi
####################################################################################################################

# Function to compute the CSR throughput extending Bianchi's model 
def Throughput_CSR_bianchi(AP_number, STA_number, CGs_STAs, TxPowerMatrix, channelMatrix, Pn_dBm, Nsc, Nss, TXOP_duration, CSRoverheads, EDCAaccessCategory):
    noise_power = 10**(Pn_dBm / 10)
    
    # MAPC overheads
    TSIFS = 16e-6             # Shortest Interframe spacing (SIFS time)
    TMAPC_ICF = 74.4E-6
    TMAPC_ICR = 88E-6
    Te = 9e-6

    # Initialize the rx_packets and per_STA_rx_packets arrays
    rx_packets = np.zeros(CGs_STAs.shape)
    per_STA_rx_packets = {sta: [] for sta in range(STA_number)}

    for i in range(CGs_STAs.shape[0]):

        APs = np.where(~np.isnan(CGs_STAs[i,:]))[0].astype(int)
        STAs = CGs_STAs[i, APs].astype(int)
        H = channelMatrix[np.ix_(STAs, APs)]
    
        MCS = np.full(len(STAs), np.nan)
        N_bps = np.full(len(STAs), np.nan)
        Rc = np.full(len(STAs), np.nan)
        
        P = TxPowerMatrix[i, APs]
        SINR_db = 10 * np.log10((P * np.diag(H)) / (noise_power + np.sum(H * P, axis=1) - np.diag(H) * P))
        
        for k in range(len(STAs)):
            # Assuming MCS_cal_PER_001 is a function that calculates MCS, N_bps, and Rc based on SINR_db
            MCS[k], N_bps[k], Rc[k] = MCS_cal_PER_001(SINR_db[k])

            if np.isnan(MCS[k]):
                rx_packets[i, APs[k]] = 0
            else:
                # Assuming tx_packets is a function that calculates the number of packets transmitted
                rx_packets[i, APs[k]] = tx_packets(Nsc, N_bps[k], Rc[k], Nss, TXOP_duration - CSRoverheads)
            
            if rx_packets[i, APs[k]] > 1024:
                raise ValueError('Impossible to transmit more than 1024 MSDUs')
    
            per_STA_rx_packets[STAs[k]].append(rx_packets[i, APs[k]])

    # Bianchi section
    L = 12e3  # Frame length (bytes)
    
    # DL calculation
    tau_DL, _, prob_col_bianchi = SimpleDCF_modelWithBEB(AP_number, EDCAaccessCategory)
    
    pe_DL = (1 - tau_DL) ** AP_number
    ps_DL = AP_number * tau_DL * (1 - tau_DL) ** (AP_number - 1)
    pc_DL = 1 - pe_DL - ps_DL

    # Access category AIFSN and AIFS calculation
    AIFSN = 2 if EDCAaccessCategory in ['VI', 'VO'] else 3
    AIFS = AIFSN * Te + TSIFS
    Tcoll = TMAPC_ICF + TSIFS + TMAPC_ICR + AIFS + Te;       # Collision duration

    p_comb = 1 / CGs_STAs.shape[0]  # Round-robin transmission probability
    
    DL_throughput_CSR_bianchi = np.zeros(STA_number)
    for kk in range(STA_number):
        DL_throughput_CSR_bianchi[kk] = p_comb * ps_DL * L * np.sum(per_STA_rx_packets[kk]) / (1e6 * (pe_DL * Te + ps_DL * TXOP_duration + pc_DL * Tcoll))
        if DL_throughput_CSR_bianchi[kk] <= 0:
            raise ValueError('Throughput <= 0 is not allowed')
    
    return DL_throughput_CSR_bianchi
####################################################################################################################




