import numpy as np
from Utils import MCS_cal_PER_001, tx_packets, elapsed_time_tx, get_association, CG_creationTPC
import copy

class MAPCsim:
    """
    Traffic class to handle the traffic generated for the STAs
    """

    def __init__(self, sim_config):

        self.sim_config = sim_config

        # System-related
        self._TXOP_DURATION = sim_config['TXOP_DURATION']                              # Duration of a TXOP
        self._NOISE_POWER = 10 ** (sim_config['PN_DBM'] / 10)                          # Noise power in mW
        self._MaxTxPower = 10 ** (sim_config['MaxTxPower'] / 10)                       # Maximum transmission power in mW
        self._NSS = sim_config['NSS']                                                  # Number of spatial streams
        self._NSC = sim_config['NSC']                                                  # Number of subcarriers

        # Scenario-related
        self.AP_NUMBER = sim_config['AP_NUMBER']                                                # Number of APs                                                   
        self.STA_NUMBER = sim_config['STA_NUMBER']                                              # Number of STAs
        self._association = sim_config['association']                                   # Association matrix
        self.channel_matrix = np.zeros((self.STA_NUMBER, self.AP_NUMBER))               # Channel matrix
        self.channel_matrix_fading = np.zeros((self.STA_NUMBER, self.AP_NUMBER))                   # Channel matrix
        
        self.txops_between_sounding = int(10)
        self.txop_counter_for_sounding = int(0)
        self.sounding_overheads = 0
        

        
        # Simulation-related
        self.sim_timeline = 0                                                # Simulation timeline
        self.timestamp_to_stop : float                        # Timestamp to stop the simulation

        # Traffic-related
        self.accessCategory : str                                         # Access category                  
        self.STA_queue_timeline = []                                     # Stores the arrival time of the packets of all STAs
        self.delivery_timestamp_record : list                               # Stores the delivery time of the packets of all STAs
        self._STA_queue_state : list                                        # Stores the state of the packets of all STAs. True = not transmitted, False = transmitted
        
        self._firstPosTimestamp : np.ndarray                                # Stores the timestamp of the first non-transmitted packet of each STA
        self._firstPosPosition : np.ndarray                                 # Stores the position of the first non-transmitted packet of each STA




        # Simulation-related
        self.simulation_system : str                                      # Simulation system -> EDCA or CSR

        # Backoff-related
        self.TXOPwinner : int                                              # Winner of the TXOP
        self._APs_packet_indicator : np.ndarray                             # Vector indicating whether each AP has packets to transmit
        self._backoffValues : np.ndarray                                    # Backoff values for each AP
        self._backoffStage : np.ndarray                                     # Backoff stage for each AP
        self._CWmin : int                                                 # Minimum contention window
        self._maxBackoffStage : int                                       # Maximum backoff stage
        self._AIFS : float                                                # Arbitration Inter-Frame Space
        

        # TXOP-related
        self._TXOPwinNumber : np.ndarray                                  # Number of TXOP wins for each AP
        self._TXOPcollision : np.ndarray                                  # Number of TXOP collisions for each AP
        self.preTX_overheadsEDCA = sim_config['overheads']['preTX_overheadsEDCA']                      # Amount of time per TXOP before the data transmission begins using EDCA 
        self.preTX_overheadsCSR = sim_config['overheads']['preTX_overheadsCSR']                      # Amount of time per TXOP before the data transmission begins using CSR
        self.info_overheadsCSR = sim_config['overheads']['info_overheadsCSR']                            # Amount of time per TXOP for the information exchange in CSR
        self.EDCAoverheads = sim_config['overheads']['EDCAoverheads']                                  # Total amount of EDCA overheads 
        self.CSRoverheads = sim_config['overheads']['CSRoverheads']                                  # Total amount of CSR overheads

        # CSR-related
        self.map_matrix : np.ndarray                                      # Matrix with the mapping of the STAs to the APs
        self.CGs_STAs : list                                            # C-SR compatible groups of STAs
        self.TxPowerMatrix : list                                 # Transmission power matrix
        self.comb_ok : np.ndarray                                 # Combination vector to indicate whether the combination is selected or not
        self.scheduler : str                            # scheduling: - Number of packets: 'MNP' 
                                                        #             - Oldest packet: 'OP'
                                                        #             - Random selection: 'Random'
                                                        #             - TAT selection: 'TAT'
                                                        #             - Hybrid selection: 'Hybrid'
        self.alpha = 0.5                                                  # Alpha value for TAT. Default value is 0.5
        self.beta = 0.5                                                   # Beta value for TAT. Default value is 0.5
        
        # Results
        self.nominal_data_rate : float                                   # Nominal data rate of the TXOP
        self.last_txop_queue_sizes = np.zeros(self.STA_NUMBER, dtype=int)  # Queue sizes at the last TXOP
        self.per_TXOP_STA_tx_packets : np.ndarray                         # Number of packets transmitted per STA per TXOP
        self.STAselectionCounter : np.ndarray                             # Counter for the number of times each STA is selected
        self.throughput_sim : np.ndarray                                  # Throughput of each STA
        self.delay_per_STA : list                                         # Delay of each STA
        self.delayvector : list                                           # Delay matrix (all STAs)         
        self.APcollision_prob : np.ndarray                                # Collision probability of each AP

        self.suc_TXOPs = int(0)  # Counter for successful transmissions
        self.priority_selection_counter = np.zeros((self.STA_NUMBER,), dtype=int)  # Priority selection for each STA

    def UpdateAP(self):
        """
        Updates the vector indicating whether each AP has packets to transmit.
        """
        # Vectorized operation to check if any associated STA has packets to transmit
        self._APs_packet_indicator = np.array([
            np.any(self._firstPosTimestamp[self._association[k]] <= self.sim_timeline)
            for k in range(self.AP_NUMBER)
        ], dtype=bool)

    def UpdateSTA(self, STA_rx, rx_vector_pos, temp_elapsed_time):
        """
        Updates STA properties when packets are received.
        """
        if rx_vector_pos:
            # Update the queue state of the STA
            self._STA_queue_state[STA_rx][rx_vector_pos] = False
            
            # Update the delivery timestamp record
            self.delivery_timestamp_record[STA_rx][rx_vector_pos] = self.sim_timeline + temp_elapsed_time
            
            # Update the first position timestamp and position
            self._firstPosPosition[STA_rx] = np.argmax(self._STA_queue_state[STA_rx])
            self._firstPosTimestamp[STA_rx] = self.STA_queue_timeline[STA_rx][self._firstPosPosition[STA_rx]]

        # Update the number of times each STA is selected    
        self.STAselectionCounter[STA_rx] += 1

    def Backoff(self):
        """
        Executes the backoff process.
        """
        if  self.simulation_system == 'EDCA':
            Tc = 56E-6 + 16E-6 + 48E-6 + self._AIFS + 9E-6;      # collision duration -----> Tc = RTS + SIFS + CTS + AIFS + Te
        elif (self.simulation_system == 'CSR') or (self.simulation_system == 'RL'):
            Tc = 74.4E-6 + 16e-6 + 88E-6 + self._AIFS + 9e-6;      # collision duration -----> Tc = MAPC_ICF + SIFS + MAPC_ICR + AIFS + Te
        
        # Extract the minimum backoff value among the APs with packets
        APs_with_packets = np.where(self._APs_packet_indicator)[0]
        # print(f'Backoff values = {self._backoffValues}')
        # print(f'APs with packets = {APs_with_packets}')

        slotnum = np.min(self._backoffValues[APs_with_packets])

        self._backoffValues[APs_with_packets] -= slotnum
        idx = np.where(self._backoffValues[APs_with_packets] == 0)[0]

        collision_counter = 0
        while True:
            if len(idx) == 1:
                self.TXOPwinner = APs_with_packets[idx[0]]
                # print(f'TXOP winner - AP{self.TXOPwinner}')
                self._TXOPwinNumber[self.TXOPwinner] += 1
                self._backoffValues[self.TXOPwinner] = np.random.randint(0, self._CWmin)
                self._backoffStage[self.TXOPwinner] = 0
                # print(f'Slot number = {slotnum}')
                # print(f'Number of collisions = {collision_counter}')
                # print(f'Backoff values = {self._backoffValues}')
                # print(f'Backoff stage = {self._backoffStage}')
                # print('----------------------------------')

                return slotnum * 9e-6 + collision_counter * Tc, self.TXOPwinner
            else:
                # print(f'Collisioned APs = {idx}')
                for i in idx:
                    ap = APs_with_packets[i]
                    self._TXOPwinNumber[ap] += 1
                    self._TXOPcollision[ap] += 1
                    if self._backoffStage[ap] < self._maxBackoffStage:
                        self._backoffStage[ap] += 1
                    self._backoffValues[ap] = np.random.randint(0, self._CWmin * (2 ** self._backoffStage[ap]))
                    # print(f'Selected backoff value: {self._backoffValues[APs_with_packets[i]]}')
                    
                
                # print(f'Backoff values = {self._backoffValues}')
                slotnum += np.min(self._backoffValues[APs_with_packets])
                collision_counter += 1
                self._backoffValues[APs_with_packets] -= np.min(self._backoffValues[APs_with_packets])
                idx = np.where(self._backoffValues[APs_with_packets] == 0)[0]
                # print(f'Backoff values = {self._backoffValues}')
                # print('')

    def get_queue(self, sta):
        """ Vector with the available packets to be transmitted to a given STA at sim_timeline. 
        Returns:
        tx_vector_pos : list   -----   List of the available packets to be transmitted to sta at sim_timeline
        
        """

        # Find the first packet available to be transmitted 
        firstPos = self._firstPosPosition[sta]

        # Find the last packet available to be transmitted
        lastPos = np.where(self.STA_queue_timeline[sta] <= self.sim_timeline)[0][-1]
        
        # Find the vector of packets that can be transmitted
        packet_range = range(firstPos, lastPos + 1)
        tx_vector_pos = [p for p in packet_range if self._STA_queue_state[sta][p] == True]

        return tx_vector_pos

    def SchedulingV1(self, agent_decision=None):
        """
        Scheduling logic.
        
        Returns:
        STA_rx : list   -----   List of STAs to be served
        APs : list      -----   List of the corresponding APs to served the STAs in STA_rx
        """
        # If an external agent decision is provided
        if agent_decision is not None:
            STA_rx, APs = agent_decision
        else:
            if self.simulation_system == 'EDCA':
                # STA selection based on the oldest packet
                STAidx = np.argmin(self._firstPosTimestamp[self._association[self.TXOPwinner]])
                STA_rx = np.atleast_1d(self._association[self.TXOPwinner][STAidx])

                APs = np.atleast_1d(self.TXOPwinner)
            elif self.simulation_system == 'CSR':

                # Initialize variables
                CGs = copy.deepcopy(self.CGs_STAs)
                per_STA_ScorePackets = np.zeros(self.STA_NUMBER, dtype=int)
                placeholder = -1 # Placeholder for inactive STAs

                # Identify inactive STAs and update CGs
                inactive_STAs = np.where(self._firstPosTimestamp > self.sim_timeline)[0]
                inactive_set = set(inactive_STAs)
                for cg in CGs:
                    cg[:] = [placeholder if sta in inactive_set else sta for sta in cg]

                # Calculate packets for all STAs
                per_STA_ScorePackets = np.array([len(self.get_queue(j)) if j not in inactive_set else 0 for j in range(self.STA_NUMBER)])

                # Scoring and unique STA extraction
                uni = []                           # Unique STAs
                ScorePackets = []                  # Score based on the number of packets
                ScoreTimeOldest = []               # Score based on the oldest packet (timestamp)
                ScoreTAT = []                      # Score based on TAT

                # Iterate through CGs that contain valid STAs
                for cg in CGs:
                    u = [sta for sta in cg if sta != placeholder]
                    if not u:
                        continue

                    # u = np.array(active_u)
                    uni.append(u)
                    ScorePackets.append(np.sum(per_STA_ScorePackets[u]))
                    ScoreTimeOldest.append(np.min(self._firstPosTimestamp[u]))

                    if self.scheduler in ['TAT', 'Hybrid']:
                        ei_min = np.min(self._firstPosTimestamp[u])
                        ei_max = np.max(self._firstPosTimestamp[u])
                        delta_nt = self.sim_timeline - ei_min
                        Delta_nt = self.sim_timeline - ei_max
                        ScoreTAT.append(
                            delta_nt if len(u) == 1 else delta_nt + self.beta * (Delta_nt - self.alpha * delta_nt)
                        )

                if self.scheduler == 'MNP':  # Maximum number of packets
                    maxScore = max(ScorePackets)
                    idx_score = np.argmax(ScorePackets)
                    equalScoreIdx = [i for i, score in enumerate(ScorePackets) if score == maxScore]

                    if len(equalScoreIdx) != 1:
                        idx_score = equalScoreIdx[np.argmin([ScoreTimeOldest[i] for i in equalScoreIdx])]
                elif self.scheduler == 'OP': # Oldest packet
                    minOldestScore = min(ScoreTimeOldest)
                    idx_score = np.argmin(ScoreTimeOldest)
                    equalScoreIdx = [i for i, score in enumerate(ScoreTimeOldest) if score == minOldestScore]

                    if len(equalScoreIdx) != 1:
                        idx_score = equalScoreIdx[np.argmax([ScorePackets[i] for i in equalScoreIdx])]
                elif self.scheduler == 'Random':
                    idx_score = np.random.randint(len(uni))
                elif self.scheduler == 'TAT': # Traffic-Alignment Tracker
                    maxScore = max(ScoreTAT)
                    idx_score = np.argmax(ScoreTAT)
                    equalScoreIdx = [i for i, score in enumerate(ScoreTAT) if score == maxScore]
                    if len(equalScoreIdx) != 1:
                        idx_score = equalScoreIdx[np.argmax([ScorePackets[i] for i in equalScoreIdx])]

                STA_rx = uni[idx_score]

                # compute the corresponding APs which the STAs in STA_rx are associated with
                APs = get_association(self._association, STA_rx)

        return STA_rx, APs

    def TXtimeCalc(self, STA_rx, APs, data_tx_time):
        """
        Transmission time calculation.
        """
        # Initialize variables
        # agg_packets = np.zeros(len(STA_rx), dtype=int)
        # tx_Packets = np.zeros(len(STA_rx), dtype=int)
        temp_elapsed_time = np.zeros(len(STA_rx))
        agg_packets_group = []  # List to store the number of aggregated packets for each STA

        # Channel matrix for the corresponding APs and STA_rx
        H = self.channel_matrix_fading[np.ix_(STA_rx, APs)]

        # If STA_rx == 1, no TPC
        if len(STA_rx) == 1:
            P = self._MaxTxPower
        else:
            try:
                idx = next(i for i, cg in enumerate(self.CGs_STAs) if np.array_equal(cg, STA_rx))
            except StopIteration:
                raise ValueError('Not found') # STA_rx not found
            
            P = self.TxPowerMatrix[idx]

        # Computing the SINR in dB 
        SINR_db = 10 * np.log10((P * np.diag(H)) / (self._NOISE_POWER + np.sum(H * P, axis=1) - np.diag(H) * P))
        
        for k, sta in enumerate(STA_rx):
            # Compute MCS and related parameters
            MCS, N_bps, Rc = MCS_cal_PER_001(SINR_db[k])
            
            # Fetch the queue of available packets for the current STA
            tx_vector_pos = self.get_queue(sta)
            
            if MCS != -1:
                Pe = 1e-2  # Packet error probability for valid MCS
            else:
                Pe = 1  # All packets are lost
            
            agg_packets = tx_packets(self._NSC, N_bps, Rc, self._NSS, data_tx_time)

            # Determine the number of packets to transmit
            tx_Packets = min(len(tx_vector_pos), agg_packets)

            # Updating the tx_vector_pos considering the minimum between the available packets and the packets that can be aggregated
            tx_vector_pos = tx_vector_pos[:tx_Packets]

            if Pe < 1:  # If transmission is possible
                # Packets successfully transmitted
                received_packets = np.random.binomial(tx_Packets, 1 - Pe)
                lost_packets = np.random.choice(tx_vector_pos, tx_Packets - received_packets, replace=False)
                rx_vector_pos = [p for p in tx_vector_pos if p not in lost_packets]
            else:
                rx_vector_pos = []  # No packets received

            # Compute transmission time
            temp_elapsed_time[k] = elapsed_time_tx(self._NSC, N_bps, Rc, self._NSS, tx_Packets)

            # Update the number of transmitted packets in the current TXOP
            self.per_TXOP_STA_tx_packets[sta] = tx_Packets if Pe < 1 else -100
            agg_packets_group.append(agg_packets)  # Store the number of aggregated packets

            # Update STA state
            self.UpdateSTA(sta, rx_vector_pos, temp_elapsed_time[k])

        self.nominal_data_rate = sum(agg_packets_group) / data_tx_time

        return max(temp_elapsed_time)
    
    def apply_fading(self, fading_type="rician", K_dB=10):
        """
        Apply fast fading to a base channel matrix.
        
        Parameters:
        - channel_matrix: matrix with pathloss-based channel gains
        - fading_type: 'rayleigh' or 'rician'
        - K_dB: Rician K-factor in dB (ignored if rayleigh)

        Returns:
        - faded_channel_matrix: same shape with fast fading applied
        """
        shape = self.channel_matrix.shape

        if fading_type == "rayleigh":
            fading = (np.random.normal(0, 1, shape) + 1j * np.random.normal(0, 1, shape)) / np.sqrt(2)
        elif fading_type == "rician":
            K = 10**(K_dB / 10)
            s = np.sqrt(K / (K + 1))   # LOS component
            sigma = np.sqrt(1 / (2 * (K + 1)))  # NLOS component

            fading = (
                s + np.random.normal(0, sigma, shape) + 1j * np.random.normal(0, sigma, shape)
            )
        else:
            raise ValueError("Unsupported fading type.")

        self.channel_matrix_fading = self.channel_matrix * np.abs(fading) ** 2  # Power gain

        return 

    def TrafficAnalysis(self):
        """Analysis of the results."""

        self.delay_per_STA = [[] for _ in range(self.STA_NUMBER)]  # Delay of each STA
        self.delayvector = []                                      # Delay matrix (all STAs)

        for j in range(self.STA_NUMBER):  # Per STA Analysis

            queue_timeline = self.STA_queue_timeline[j]
            delivery_record = self.delivery_timestamp_record[j]
            state_record = self._STA_queue_state[j]

            # Separate transmitted and non-transmitted packets
            transmitted_mask = (state_record == False)
            not_transmitted_mask = (state_record == True)

            transmitted_delays = delivery_record[transmitted_mask] - queue_timeline[transmitted_mask]

            if np.any(transmitted_delays <= 0):
                raise ValueError("Delay cannot be negative or zero for transmitted packets")

            # For untransmitted packets, assume delivery at the end of simulation
            pending_arrivals = queue_timeline[not_transmitted_mask]
            pending_delays = self.timestamp_to_stop - pending_arrivals

            if np.any(pending_delays <= 0):
                raise ValueError("Pending packet delay is negative or zero; check timestamp_to_stop.")

            # Concatenate both
            all_delays = np.concatenate((transmitted_delays, pending_delays))

            self.delay_per_STA[j] = all_delays
            self.delayvector = np.concatenate((self.delayvector, all_delays))

        # Per AP analysis (unchanged)
        for jj in range(self.AP_NUMBER):
            if self._TXOPwinNumber[jj] == 0:
                self.APcollision_prob[jj] = 0
            else:
                self.APcollision_prob[jj] = self._TXOPcollision[jj] / self._TXOPwinNumber[jj]

    def InitSettings(self):
        """Initialization. Restarts the parameters to start a new simulation"""
     
        if self.accessCategory == 'VO':
            self._CWmin = 4
            self._maxBackoffStage = 1
            AIFSN = 2
        elif self.accessCategory == 'VI':
            self._CWmin = 8
            self._maxBackoffStage = 1
            AIFSN = 2
        elif self.accessCategory == 'BE':
            self._CWmin = 16
            self._maxBackoffStage = 6
            AIFSN = 3
        else:
            raise ValueError("Invalid access category")
    
        # Simulation related
        self.sim_timeline = 0

        # Traffic-related
        self._firstPosTimestamp = np.zeros((self.STA_NUMBER,), dtype=float)     # Stores the timestamp
        self._firstPosPosition = np.zeros((self.STA_NUMBER,), dtype=int)      # Stores the position of the first non-transmitted packet of each STA
        self.delivery_timestamp_record = []                               # Stores the delivery time of the packets of all STAs
        self._STA_queue_state = []                                       # Stores the state of the packets of all STAs. True = not transmitted, False = transmitted

        # Initialize queues and traffic-related vars
        for i, timeline in enumerate(self.STA_queue_timeline):
            # Convert and filter
            timeline = np.array(timeline, dtype=float)
            timeline = timeline[timeline <= self.timestamp_to_stop]  # ✅ Remove late arrivals

            actual_length = len(timeline)

            # Initialize state arrays with correct length
            self.delivery_timestamp_record.append(np.full(actual_length, np.nan, dtype=float))
            self._STA_queue_state.append(np.ones(actual_length, dtype=bool))

            # Other assignments
            self._firstPosTimestamp[i] = timeline[0] if actual_length > 0 else np.nan
            self.STA_queue_timeline[i] = timeline

        # Scenario-related
        self.channel_matrix_fading = self.channel_matrix.copy()  # Copy the channel matrix for fading
        # self.txop_counter_for_sounding = int(0)
        # self.apply_fading()
        # match self.simulation_system: 
        #     case 'CSR': 
        #         map_matrix, TxPowerMatrixTemp, self.comb_ok = CG_creationTPC(self.AP_NUMBER, self.STA_NUMBER, self.sim_config['PN_DBM'], self._NSC, self._NSS, self._association, self.channel_matrix_fading, self.sim_config['MaxTxPower'], self.sim_config['filtering'], TPC_method=None, CG_size=self.AP_NUMBER)
        #         self.TxPowerMatrix = [row.tolist() for i, row in enumerate(TxPowerMatrixTemp) if self.comb_ok[i]]
        #         self.CGs_STAs = [row.tolist() for i, row in enumerate(map_matrix) if self.comb_ok[i]]
            
        #     case 'RL':
        #         map_matrix, TxPowerMatrixTemp, self.comb_ok = CG_creationTPC(self.AP_NUMBER, self.STA_NUMBER, self.sim_config['PN_DBM'], self._NSC, self._NSS, self._association, self.channel_matrix_fading, self.sim_config['MaxTxPower'], self.sim_config['filtering'], TPC_method=None, CG_size=self.AP_NUMBER)
        #         self.TxPowerMatrix = TxPowerMatrixTemp
        #         self.CGs_STAs = map_matrix
        
        
  
        # Backoff-related
        self._APs_packet_indicator = np.zeros((self.AP_NUMBER,), dtype=bool)  # Vector indicating whether each AP has packets to transmit
        self._backoffValues = np.random.randint(0, self._CWmin, size=self.AP_NUMBER)
        self._backoffStage = np.zeros((self.AP_NUMBER,), dtype=int)           # Backoff stage for each AP
        self._AIFS = AIFSN * 9e-6 + 16e-6
        

        # TXOP-related
        self._TXOPwinNumber = np.zeros((self.AP_NUMBER,), dtype=int)          # Number of TXOP wins for each AP
        self._TXOPcollision = np.zeros((self.AP_NUMBER,), dtype=int)          # Number of TXOP collisions for each AP
        

        # Results
        self.last_txop_queue_sizes = np.zeros(self.STA_NUMBER, dtype=int)  # Queue sizes at the last TXOP
        self.per_TXOP_STA_tx_packets = np.zeros((self.STA_NUMBER,), dtype=int)          # Number of packets transmitted per STA per TXOP
        self.STAselectionCounter = np.zeros((self.STA_NUMBER,), dtype=int)    # Counter for the number of times each STA is selected
        self.throughput_sim = np.zeros((self.STA_NUMBER,), dtype=float)                    # Throughput of each STA       
        self.APcollision_prob = np.zeros((self.AP_NUMBER,), dtype=float)      # Collision probability of each AP

        self.suc_TXOPs = int(0)  # Counter for successful transmissions
        self.priority_selection_counter = np.zeros((self.STA_NUMBER,), dtype=int)  # Priority selection for each STA

    def sim_forward(self):
        """ Forward the simulation just right before the scheduling process. """

        if min(self._firstPosTimestamp) > self.sim_timeline:
            self.sim_timeline = min(self._firstPosTimestamp)
        
        # Update APs_packet_indicator vector to indicate whether each AP has packets to transmit
        self.UpdateAP()

        # Backoff process
        backofftime, self.TXOPwinner = self.Backoff()

        # Update simulation timeline with backoff time and backoff collision time
        self.sim_timeline += backofftime

        if self.simulation_system in ['CSR', 'RL']:
            self.sim_timeline += self.info_overheadsCSR           # ICF + SIFS + ICR + SIFS


    def run_step(self, agent_decision=None):
        """Run a single step of the simulation, optionally with external intervention."""
        
        # Reset the number of packets transmitted per STA per TXOP and the total queue size
        self.per_TXOP_STA_tx_packets = np.zeros((self.STA_NUMBER,), dtype=int)
        self.nominal_data_rate = 0.0
        self.last_txop_queue_sizes = np.array([len(self.get_queue(sta)) if self._firstPosTimestamp[sta] <= self.sim_timeline else 0 for sta in range(self.sim_config['STA_NUMBER'])])

        delay = np.array([self.sim_timeline - self._firstPosTimestamp[sta] if self._firstPosTimestamp[sta] <= self.sim_timeline else 0.0 for sta in range(self.STA_NUMBER)])
        ordered = np.argsort(delay)
        

        # Verify whether agent_decision is None (non-agent decision) or is not empty (agent decision with valid decision) 
        if (agent_decision not in [None, [], [[], []]]) or self.simulation_system in ['EDCA', 'CSR']:
   
            # Scheduling
            STA_rx, APs = self.SchedulingV1(agent_decision)

            self.suc_TXOPs += 1  # Increment the successful TXOP counter
            pos_priority = [np.where(ordered==sta)[0] for sta in STA_rx]
            self.priority_selection_counter[pos_priority] += 1

            # Pre-TX overheads and data transmission time calculation
            if self.simulation_system == "EDCA":
                self.sim_timeline += self.preTX_overheadsEDCA
                data_tx_time = self._TXOP_DURATION - self.EDCAoverheads
            elif self.simulation_system in ['CSR', 'RL']:
                self.sim_timeline += self.preTX_overheadsCSR
                data_tx_time = self._TXOP_DURATION - self.CSRoverheads

            
            # Compute the transmission time and update the simulation timeline
            elapsed_time = self.TXtimeCalc(STA_rx, APs, data_tx_time) # Transmission time calculation

            if elapsed_time > data_tx_time:
                raise ValueError("Transmission time cannot be greater than the allowed transmission time")
            
            self.sim_timeline += elapsed_time + 16e-6 + 100e-6 + self._AIFS + 9e-6 # elapsed time + SIFS + ACK + AIFS + Te

            # self.txop_counter_for_sounding += 1
            # if (self.txop_counter_for_sounding == self.txops_between_sounding):
            #     self.txop_counter_for_sounding = int(0)
            #     self.apply_fading()
            #     match self.simulation_system: 
            #         case 'CSR': 
            #             map_matrix, TxPowerMatrixTemp, self.comb_ok = CG_creationTPC(self.AP_NUMBER, self.STA_NUMBER, self.sim_config['PN_DBM'], self._NSC, self._NSS, self._association, self.channel_matrix_fading, self.sim_config['MaxTxPower'], self.sim_config['filtering'], TPC_method=None, CG_size=self.AP_NUMBER)
            #             self.TxPowerMatrix = [row.tolist() for i, row in enumerate(TxPowerMatrixTemp) if self.comb_ok[i]]
            #             self.CGs_STAs = [row.tolist() for i, row in enumerate(map_matrix) if self.comb_ok[i]]
                    
            #             self.sim_timeline += self.sounding_overheads
            #         case 'RL':
            #             map_matrix, TxPowerMatrixTemp, self.comb_ok = CG_creationTPC(self.AP_NUMBER, self.STA_NUMBER, self.sim_config['PN_DBM'], self._NSC, self._NSS, self._association, self.channel_matrix_fading, self.sim_config['MaxTxPower'], self.sim_config['filtering'], TPC_method=None, CG_size=self.AP_NUMBER)
            #             self.TxPowerMatrix = TxPowerMatrixTemp
            #             self.CGs_STAs = map_matrix
                    
            #             self.sim_timeline += self.sounding_overheads


        return 
        
    def Run(self):
        """Simulation process."""

        while self.sim_timeline < self.timestamp_to_stop:

            self.sim_forward() # Forward the simulation just right before the scheduling process.
            self.run_step() # Run a single step of the simulation, optionally with external agent decision.

        self.TrafficAnalysis()
