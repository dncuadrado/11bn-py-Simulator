
from Utils import generate_combinations, get_association, MCS_cal_PER_001    
import numpy as np       
import os
import sys

# optimization problem
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from pyswarm import pso  # Particle Swarm Optimization library



class GroupCreator():
    def __init__(self, sim_config, channel_matrix, CG_size, TPC_method, is_filtering=True):
        
        self._AP_NUMBER = sim_config['AP_NUMBER']
        self._STA_NUMBER = sim_config['STA_NUMBER']
        self._noise_power = 10**(sim_config['PN_DBM'] / 10) # noise power in mW
        self._MaxTxPower = 10**(sim_config['MaxTxPower'] / 10)  # max tx power in mW
        self._NSC = sim_config['NSC']
        self._NSS = sim_config['NSS']
        self.association = sim_config['association']
        self.channel_matrix = channel_matrix

        self.TPC_method = TPC_method
        self.CG_size = CG_size
        self.is_filtering = is_filtering


        # Output variables
        self.map_matrix = []
        self.CGs_STAs = []
        self.TxPowerMatrix = []
        self.TxPowerMatrixFull = []
        self.comb_ok = []


    def group_management(self):

        noise_power = 10**(self._noise_power / 10)
        MaxTxPower = 10**(self._MaxTxPower / 10) 

        # Function to generate all possible combinations of AP-STAs for a given association
        map_matrix = generate_combinations(self._AP_NUMBER, self._STA_NUMBER, self.association, self.CG_size)

        # Create TxPowerMatrixTemp with the same shape as map_matrix
        TxPowerMatrixTemp = [np.full_like(row, MaxTxPower, dtype=float) for _, row in enumerate(map_matrix)]

        if not self.is_filtering:
            self.map_matrix = map_matrix
            self.TxPowerMatrix = self.TxPowerMatrixFull = TxPowerMatrixTemp
            self.comb_ok = np.ones(len(map_matrix), dtype=bool)
            return 
    
        # Other matrices initialization
        datarateTemp = [np.zeros(len(row), dtype=float) for _, row in enumerate(map_matrix)]
        comb_ok = np.zeros(len(map_matrix), dtype=bool)
        discard_list = np.ones(len(map_matrix), dtype=bool)

        # Main loop for verifying groups
        for i in range(len(map_matrix)):
            if discard_list[i] == False:
                continue
            
            STAs = map_matrix[i]
            APs = get_association(self.association, STAs)
            
            H = self.channel_matrix[np.ix_(STAs, APs)]

            if len(STAs) == 1:  # Use maximum power for a single STA
                P = MaxTxPower  
            else:  # Compute the subset of power that maximizes the proportional fair transmission
                # TPC (Power allocation)
                
                # Solving the Opt problem with the selected method
                P = self._power_allocation(len(STAs), noise_power, H, MaxTxPower, self._NSC, self._NSS, self.TPC_method)

                # Store the power vector in TxPowerMatrixTemp
                TxPowerMatrixTemp[i] = P  

            # Compute the SINR and datarate for each STA
            SINR = (P * np.diag(H)) / (noise_power + np.sum(H * P, axis=1) - np.diag(H) * P)
            SINR_db = 10 * np.log10(SINR)

            for k, _ in enumerate(STAs):
                MCS, N_bps, Rc = MCS_cal_PER_001(SINR_db[k])
                if MCS == -1:
                    datarateTemp[i][k] = 0 #SINR under the threshold
                else:
                    # datarateTemp[i][k] = self._NSC * N_bps * Rc * self._NSS / (12.8e-6 + 0.8e-6) # Datarate in bps
                    datarateTemp[i][k] = N_bps * Rc / (12 * 5/6)  # Number of bits to code a symbol
                
                # To filter out combinations using a specific criterium 
                # if the datarate using CSR is lower than the datarate without CSR, discard the combination
                if len(STAs) * datarateTemp[i][k] >= datarateTemp[STAs[k]]:
                    continue
                else:
                    # Discarding the rest of combs where this STAs appear
                    true_indices = np.where(discard_list)[0]
                    mask = np.array([set(STAs).issubset(set(map_matrix[idx])) for idx in true_indices])
                    discard_list[true_indices[mask]] = False
                    break


            if discard_list[i] == True:
                comb_ok[i] = True
                # datarate[i] = len(datarateTemp[i])*np.prod(datarateTemp[i])
        
        TxPowerMatrix = [row.tolist() for i, row in enumerate(TxPowerMatrixTemp) if comb_ok[i]]
        CGs_STAs = [row.tolist() for i, row in enumerate(map_matrix) if comb_ok[i]]

        if len(TxPowerMatrix) != len(CGs_STAs):
            raise ValueError('Mismatch between TxPowerMatrix and CGs_STAs')
        
        
        self.CGs_STAs = CGs_STAs
        self.TxPowerMatrix = TxPowerMatrix
        self.TxPowerMatrixFull = TxPowerMatrixTemp

        self.map_matrix = map_matrix
        self.comb_ok = comb_ok

    # Function to calculate the optimal power allocation
    def _power_allocation(self, N, noise_power, H, P_max, NSC, NSS, TPC_method):
        """
        Optimizes power allocation using the specified method to maximize proportional fairness.
        Returns: P_opt
        """

        if TPC_method == None:
            P_opt = P = np.full(N, P_max)
        elif TPC_method == 'PSO':   # Particle Swarm Optimization
            P_opt = self._power_allocation_particleswarm(N, noise_power, H, P_max, NSC, NSS)
        else:
            raise ValueError(f"Unsupported optimization method: {TPC_method}")

        return P_opt 

    # Function to compute the product of rates for the given power allocation
    def _compute_rates(self, P, H, noise_power, N, NSC, NSS):
        """
        Computes the product of rates for the given power allocation.
        
        Args:
        P: Power allocation vector (N,)
        H: Channel gain matrix (NxN)
        noise_power: Noise power (scalar)
        N: Number of links
        NSC: Number of subcarriers
        NSS: Number of spatial streams
        
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
            if MCS == -1:
                rates[i] = 0
            else:
                rates[i] = (NSC * N_bps * Rc * NSS) / (T_DFT + T_GI)
                # rates[i] = np.log2(1 + sinr[i])
        
        # Return the product of all rates

        if np.any(rates == 0):
            product_rate = 0  # Penalize invalid rates
        else:
            product_rate = np.sum(np.log10(np.maximum(rates, 1e-9)))
        
        return product_rate
        # return np.min(rates)

    # Function to calculate the optimal power allocation using the Particle Swarm Optimization (PSO) algorithm
    def _power_allocation_particleswarm(self, N, noise_power, H, P_max, NSC, NSS):
        """
        Optimizes power allocation using Particle Swarm Optimization (PSO)
        to maximize proportional fairness.
        
        Args:
        N: Number of links (transmitters)
        noise_power: Noise power (scalar)
        H: Channel gain matrix (NxN)
        P_max: Maximum power constraint (scalar)
        NSC: Number of subcarriers
        NSS: Number of spatial streams
        
        Returns:
        P_opt: Optimized power allocation vector (N,)
        """
        

        # Bounds: 0 <= P <= P_max for each power allocation
        lb = np.ones(N)  # Lower bound for power allocation
        ub = P_max * np.ones(N)  # Upper bound for power allocation

        # Use PSO with optimized swarm size and iterations
        SWARM_SIZE = 20  # Reduced swarm size (can be tuned)
        MAX_ITER = 20   # Reduced number of iterations (can be tuned)

        def objective(P):
            return -self._compute_rates(P, H, noise_power, N, NSC, NSS)
        
        # Silence stdout and stderr
        with open(os.devnull, 'w') as fnull:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = fnull
                sys.stderr = fnull

                P_opt, _ = pso(objective, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
                    swarmsize=SWARM_SIZE, omega=0.5, phip=0.5, phig=0.5, maxiter=MAX_ITER, 
                    minstep=1e-8, minfunc=1e-8, debug=False)  # minstep=1e-8, minfunc=1e-8
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        # P_opt, _ = pso(objective, lb, ub, swarmsize=SWARM_SIZE, maxiter=MAX_ITER, debug=False, 
        #                minfunc=1e-6, omega=0.5, phip=0.5, phig=0.5)
        

        return P_opt
    ####################################################################################################################
