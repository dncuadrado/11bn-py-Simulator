import numpy as np
from Utils import MCS_cal_PER_001, tx_packets, elapsed_time_tx, get_association
import copy

class MAPCsim:
    """
    Traffic class to handle the traffic generated for the STAs
    """

    def __init__(self, sim_config):

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
        self._channelMatrix = sim_config['channelMatrix']                               # Channel matrix
        
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
        self.validation_flag = sim_config['validation_flag']                              # Validation flag -> 'yes' or 'no'

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
        self.preTX_overheadsEDCA = sim_config['preTX_overheadsEDCA']                      # Amount of time per TXOP before the data transmission begins using EDCA 
        self.preTX_overheadsCSR = sim_config['preTX_overheadsCSR']                      # Amount of time per TXOP before the data transmission begins using CSR
        self.EDCAoverheads = sim_config['EDCAoverheads']                                  # Total amount of EDCA overheads 
        self.CSRoverheads = sim_config['CSRoverheads']                                  # Total amount of CSR overheads

        # CSR-related
        self.CGs_STAs : list                                            # C-SR compatible groups of STAs
        self.TxPowerMatrix : list                                 # Transmission power matrix
        self.comb_ok : np.ndarray                                 # Combination vector to indicate whether the combination is selected or not
        self.datarate : np.ndarray                                 # Vector that contains the proportional data rate per group
        self.scheduler : str                            # scheduling: - Number of packets: 'MNP' 
                                                        #             - Oldest packet: 'OP'
                                                        #             - Random selection: 'Random'
                                                        #             - TAT selection: 'TAT'
                                                        #             - Hybrid selection: 'Hybrid'
        self.alpha = 0.5                                                  # Alpha value for TAT. Default value is 0.5
        self.beta = 0.5                                                   # Beta value for TAT. Default value is 0.5
        
        # Results
        self.per_TXOP_STA_tx_packets : np.ndarray                         # Number of packets transmitted per STA per TXOP
        self.STAselectionCounter : np.ndarray                             # Counter for the number of times each STA is selected
        self.throughput_sim : np.ndarray                                  # Throughput of each STA
        self.delay_per_STA : list                                         # Delay of each STA
        self.delayvector : list                                           # Delay matrix (all STAs)         
        self.APcollision_prob : np.ndarray                                # Collision probability of each AP

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
            
            # Update the number of tx packet in the current TXOP
            self.per_TXOP_STA_tx_packets[STA_rx] = len(rx_vector_pos)

        # Update the number of times each STA is selected    
        self.STAselectionCounter[STA_rx] += 1

    def Backoff(self):
        """
        Executes the backoff process.
        """
        if  self.simulation_system == 'EDCA':
            Tc = 56E-6 + 16E-6 + 48E-6 + self._AIFS + 9E-6;      # collision duration -----> Tc = RTS + SIFS + CTS + AIFS + Te
        elif self.simulation_system == 'CSR':
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
                if self.validation_flag == 'yes':
                    # TODO: Check!!!!!!!
                    # Round Robin scheduling for validation
                    if sum(self._firstPosPosition) == self.STA_NUMBER:
                        for k in range(self.AP_NUMBER):
                            self.rrobin_EDCA_group_selector[k][:] = [1] + [0] * (len(self._association[k]) - 1)
                    
                    STA_rx = self._association[self.TXOPwinner][np.where(self.rrobin_EDCA_group_selector[self.TXOPwinner] == 1)[0][0]]
                    self.rrobin_EDCA_group_selector[self.TXOPwinner] = np.roll(self.rrobin_EDCA_group_selector[self.TXOPwinner], 1)
                else:
                    # STA selection based on the oldest packet
                    STAidx = np.argmin(self._firstPosTimestamp[self._association[self.TXOPwinner]])
                    STA_rx = np.atleast_1d(self._association[self.TXOPwinner][STAidx])

                APs = np.atleast_1d(self.TXOPwinner)
            else:

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

                if self.validation_flag == 'yes':
                    if self.rrobin_CSR_group_selector == 0:
                        self.rrobin_CSR_group_selector = np.zeros(len(self.CGs_STAs))
                        self.rrobin_CSR_group_selector[0] = 1

                    idx_score = np.where(self.rrobin_CSR_group_selector == 1)[0][0]
                    self.rrobin_CSR_group_selector = np.roll(self.rrobin_CSR_group_selector, 1)
                else:
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
        agg_packets = np.zeros(len(STA_rx), dtype=int)
        tx_Packets = np.zeros(len(STA_rx), dtype=int)
        temp_elapsed_time = np.zeros(len(STA_rx))

        # Channel matrix for the corresponding APs and STA_rx
        H = self._channelMatrix[np.ix_(STA_rx, APs)]

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
            # MCS-related
            MCS, N_bps, Rc = MCS_cal_PER_001(SINR_db[k])

            if np.isnan(MCS):
                MCS, N_bps, Rc, Pe = 0, 1, 0.5, 1
            else:
                Pe = 1e-2

            # Number of packets that can be aggregated
            agg_packets[k] = tx_packets(self._NSC, N_bps, Rc, self._NSS, data_tx_time)

            # Vector with the available packets to be transmitted to a given STA at sim_timeline
            tx_vector_pos = self.get_queue(sta)

            # number of packets that can be transmitted, i.e., min between aggregation and the number of available packets
            tx_Packets[k] = min(len(tx_vector_pos), agg_packets[k])

            # packets finally transmitted
            tx_vector_pos = tx_vector_pos[:tx_Packets[k]]

            if self.validation_flag == 'yes':
                temp_elapsed_time[k] = data_tx_time
            else:
                # received packets
                received_packets = np.random.binomial(tx_Packets[k], 1 - Pe)
                # lost packets (if any)
                lost_packets = np.random.choice(tx_vector_pos, int(tx_Packets[k] - received_packets), replace=False)
                # properly received packets, considering the ones lost (if any)
                rx_vector_pos = [p for p in tx_vector_pos if p not in lost_packets]
                # elapsed time due to the transmission
                temp_elapsed_time[k] = elapsed_time_tx(self._NSC, N_bps, Rc, self._NSS, tx_Packets[k])
            # Update STA
            self.UpdateSTA(sta, rx_vector_pos, temp_elapsed_time[k])

        return max(temp_elapsed_time)
    
    def TrafficAnalysis(self):
        """Analysis of the results."""
        for j in range(self.STA_NUMBER):  # Per STA Analysis
            # Find the last transmitted packet index
            last_tx_packet = np.where(self._STA_queue_state[j] == False)[0][-1] if np.any(self._STA_queue_state[j] == False) else None

            if last_tx_packet is None:
                continue  # Skip if no transmitted packet is found

            valid_indices = np.concatenate([~np.isnan(self.delivery_timestamp_record[j][:last_tx_packet]), 
                                                    [True] + [False] * (len(self.delivery_timestamp_record[j]) - last_tx_packet - 1)])

            # Compute delays and check for any negative or zero delays
            delay = self.delivery_timestamp_record[j][valid_indices] - self.STA_queue_timeline[j][valid_indices]
            if np.any(delay <= 0):
                raise ValueError("Delay cannot be negative or equal to zero")

            # Append delays to the global delay vector
            self.delay_per_STA.append(delay)
            self.delayvector = np.concatenate((self.delayvector, delay))

        for jj in range(self.AP_NUMBER):  # Per AP analysis
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
            actual_length = len(timeline)

            # Ensure delivery_timestamp_record and _STA_queue_state are lists of arrays
            self.delivery_timestamp_record.append(np.full(actual_length, np.nan, dtype=float))
            self._STA_queue_state.append(np.ones(actual_length, dtype=bool))

            # Other assignments
            self._firstPosTimestamp[i] = timeline[0]
            self.STA_queue_timeline[i] = np.array(timeline, dtype=float)  # Convert to numpy array if necessary


        
        # Backoff-related
        self._APs_packet_indicator = np.zeros((self.AP_NUMBER,), dtype=bool)  # Vector indicating whether each AP has packets to transmit
        self._backoffValues = np.random.randint(0, self._CWmin, size=self.AP_NUMBER)
        self._backoffStage = np.zeros((self.AP_NUMBER,), dtype=int)           # Backoff stage for each AP
        self._AIFS = AIFSN * 9e-6 + 16e-6
        

        # TXOP-related
        self._TXOPwinNumber = np.zeros((self.AP_NUMBER,), dtype=int)          # Number of TXOP wins for each AP
        self._TXOPcollision = np.zeros((self.AP_NUMBER,), dtype=int)          # Number of TXOP collisions for each AP

        # Results
        self.per_TXOP_STA_tx_packets = np.zeros((self.STA_NUMBER,), dtype=int)          # Number of packets transmitted per STA per TXOP
        self.STAselectionCounter = np.zeros((self.STA_NUMBER,), dtype=int)    # Counter for the number of times each STA is selected
        self.throughput_sim = np.zeros((self.STA_NUMBER,), dtype=float)                    # Throughput of each STA
        self.delay_per_STA = []                                           # Delay of each STA
        self.delayvector = []                                             # Delay matrix (all STAs)         
        self.APcollision_prob = np.zeros((self.AP_NUMBER,), dtype=float)      # Collision probability of each AP

    def sim_forward(self):
        """ Forward the simulation just right before the scheduling process. """

        if min(self._firstPosTimestamp) > self.sim_timeline:
            self.sim_timeline = min(self._firstPosTimestamp)
        
        # Update APs_packet_indicator vector to indicate whether each AP has packets to transmit
        self.UpdateAP()

        # Reset the number of packets transmitted per STA per TXOP
        self.per_TXOP_STA_tx_packets = np.zeros((self.STA_NUMBER,), dtype=int)

        # Backoff process
        backofftime, self.TXOPwinner = self.Backoff()

        # Update simulation timeline with backoff time and backoff collision time
        self.sim_timeline += backofftime

    def run_step(self, agent_decision=None):
        """Run a single step of the simulation, optionally with external intervention."""
        
        
        # Verify whether agent_decision is None (non-agent decision) or is not empty (agent decision with valid decision) 
        if agent_decision is None or (len(agent_decision[0]) > 0):
            # Scheduling
            STA_rx, APs = self.SchedulingV1(agent_decision)
            
            # Pre-TX overheads and data transmission time calculation
            if self.simulation_system == "EDCA":
                self.sim_timeline += self.preTX_overheadsEDCA
                data_tx_time = self._TXOP_DURATION - self.EDCAoverheads
            else:
                self.sim_timeline += self.preTX_overheadsCSR
                data_tx_time = self._TXOP_DURATION - self.CSRoverheads

            elapsed_time = self.TXtimeCalc(STA_rx, APs, data_tx_time) # Transmission time calculation
            self.sim_timeline += elapsed_time + 16e-6 + 100e-6 + self._AIFS + 9e-6 # elapsed time + SIFS + ACK + AIFS + Te


        return 
        
    def Run(self):
        """Simulation process."""

        while self.sim_timeline < self.timestamp_to_stop:

            self.sim_forward() # Forward the simulation just right before the scheduling process.
            self.run_step() # Run a single step of the simulation, optionally with external agent decision.

        self.TrafficAnalysis()
