import simpy
import numpy as np
import re
from numpy.random import SeedSequence
import time
import os
from collections import deque
import h5py
import bisect
import Utils as utils
from TrafficGenerator import traffic_generator
from DeploymentGenerator import deployment_generator
import RLagent as RLagent

# Physical layer parameters
SLOT_TIME = 9e-6
DIFS = 34e-6
SIFS = 16e-6
RTS_DURATION = 54e-6
CTS_DURATION = 32e-6
CTS_TIMEOUT = 163e-6
BACK_DURATION = 100e-6
MAX_AGGREGATED_PACKETS = 1024

class SimPyWiFiNetwork:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.packet_counter = np.zeros(config['STA_NUMBER'], dtype=int)
        self.nodes = []
        self.aps = []
        self.stas = []
        self.last_activity = {'timestamp': 0, 'action': 'SIM_START', 'node_id': None}
        self.channel_state = 'IDLE'
        self.nav = 0  # Absolute timestamp until the channel is busy
        self.stats = {
            'tx_success': 0,
            'collisions': 0,
            'delays': [],
            'rts_cts_success': 0,
            'cts_timeouts': 0
        }

        # Create association list with all the APs and STAs
        self.association = config['association']
        self.channelMatrix = config['channelMatrix']
        
        self._create_nodes()
        self._setup_associations()
        
    def _create_nodes(self):
        for ap_id in range(self.config['AP_NUMBER']):
            ap = AccessPoint(
                env=self.env,
                node_id=ap_id,
                network=self,
                config=self.config,
                is_ap=True
            )
            self.aps.append(ap)
            self.nodes.append(ap)
        
        for sta_id in range(self.config['STA_NUMBER']):
            sta = Station(
                env=self.env,
                node_id=sta_id,
                network=self,
                config=self.config,
                is_ap=False
            )
            self.stas.append(sta)
            if self.config['uplink']:
                self.nodes.append(sta)
    
    def _setup_associations(self):
        num_aps = self.config['AP_NUMBER']
        num_stas = self.config['STA_NUMBER']
        stas_per_ap = num_stas // num_aps

        for i, sta in enumerate(self.stas):
            ap_idx = i // stas_per_ap
            ap = self.aps[ap_idx]
            sta.associated_ap = ap
            ap.associated_stas.append(sta)
            
            # Initialize per-STA queue in AP
            ap.queues[sta] = deque()

    def channel_is_idle(self, node_id):
        if self.channel_state == 'IDLE':
            return True
        if self.last_activity['node_id'] == node_id:
            return self.env.now >= self.last_activity['timestamp'] + CTS_TIMEOUT
        return False
    
    # def scheduling(self, node_id):
    #         return np.argmax(self.nodes[node_id].packet_availability())

    def scheduling(self, node_id):
        """Selects STA with oldest packet using efficient timestamp tracking"""
        ap = self.nodes[node_id]

        valid_stas = [sta for sta in ap.associated_stas if len(ap.queues[sta]) > 0]  
    
        if not valid_stas:
            return None, -1  # Prevent scheduling empty queues
        
        # Fast path for single-STA case
        if len(ap.associated_stas) == 1:
            return ap.associated_stas[0].node_id
        
        # Precompute STA indices and their first packet timestamps
        oldest_time = float('inf')
        # print(f'Timestamp: {self.env.now}')
        selected_sta = -1
        
        for sta in ap.associated_stas:
            queue = ap.queues.get(sta, [])
            if queue:
                packet_time = queue[0]['arrival']  # O(1) access to deque head
                if packet_time < oldest_time:
                    oldest_time = packet_time
                    selected_sta = sta
                    selected_sta_id = sta.node_id
    
        return selected_sta, selected_sta_id
    
    def notify_activity(self, action, node_id, duration=0):
        """Update channel state and NAV with activity duration"""
        self.last_activity.update({
            'timestamp': self.env.now,
            'duration': duration,
            'action': action,
            'node_id': node_id
        })
        self.channel_state = 'BUSY'
        
        # Update NAV if duration is provided
        if duration > 0:
            self.nav = max(self.nav, self.env.now + duration + SIFS)

    def run(self):
        for node in self.nodes:
            self.env.process(node.traffic_generator())
        self.env.run(until=self.config['timestamp_to_stop'])
        
        print(f"\nSimulation Results ({self.config['timestamp_to_stop']}s)")
        print(f"Successful transmissions: {self.stats['tx_success']}")
        print(f"Collisions detected: {self.stats['collisions']}")
        print(f"RTS successes: {self.stats['rts_cts_success']}")
        if self.stats['tx_success'] > 0:
            percentile99th_delay = np.percentile(self.stats['delays'],99) * 1e3
            print(f"99th pecentile delay: {percentile99th_delay:.2f} ms")

class NetworkNode:
    def __init__(self, env, node_id, network, config, is_ap):
        self.env = env
        self.node_id = node_id
        self.network = network
        self.config = config
        self.is_ap = is_ap
        self.cw_min = config['cw_min']['ap'] if is_ap else config['cw_min']['sta']
        self.cw_max = config['cw_max']
        self.backoff_stage = 0
        self.backoff_counter = 0
        self.aifs = config['aifs']['ap'] if is_ap else config['aifs']['sta']

        self.max_tx_power = float(10 ** (config['MaxTxPower']/10))  # linear scale
        self.noise_power = float(10 ** (config['PN_DBM']/10)) # linear scale
         
        self.associated_stas = [] if is_ap else None
        self.associated_ap = None if is_ap else None
        self.contention_process_active = False  # Flag to prevent multiple processes
        self.current_transmission = False
        self.packet_bits = config['FRAME_LENGTH'] * 8

        # Queue structure per node
        if self.is_ap:
            # Per-STA queues
            self.queues = {}  # {station: deque}
        else:
            # Single queue for stations
            self.queue = deque()


    def traffic_generator(self):
        if self.is_ap:
            # AP's traffic generator for each associated STA
            for sta in self.associated_stas:
                self.env.process(self._generate_dl_traffic(sta))
        else:
            # Uplink
            while True:
                yield self.env.timeout(np.random.exponential(1/self.config['ul_rate']))
                self.queue.append({
                    'arrival': self.env.now,
                    'destination': self.associated_ap
                })
                # Start contention process only if not already running
                if not self.contention_process_active and not self.current_transmission:
                    self.contention_process_active = True
                    self.env.process(self.contention_process())

    def _generate_dl_traffic(self, sta):
        """Generate downlink traffic for a specific STA"""
        while True:
            arrival = np.random.exponential(1/self.config['dl_rate'])
            yield self.env.timeout(arrival)
            self.queues[sta].append({
                'arrival': self.env.now,
                'destination': sta
            })
            # Update the packet counter for the STA
            self.network.packet_counter[sta.node_id] += 1

            if not self.contention_process_active and not self.current_transmission:
                self.contention_process_active = True
                self.env.process(self.contention_process())

    def contention_process(self):
        """Handles channel contention and transmission for one TXOP"""
        try:
            while True:
                
                # Exit condition when no packets remain
                if self.is_ap and not any(len(q) > 0 for q in self.queues.values()):
                    break  # NEW: Exit loop for empty APs

               
                if not np.any(self.packet_availability()!= 0): 
                    yield self.env.timeout(SLOT_TIME)
                    continue

                # Backoff countdown
                while self.backoff_counter > 0:
                    if self.network.channel_state == 'IDLE':
                        yield self.env.timeout(SLOT_TIME)
                        self.backoff_counter -= 1
                    else:
                        # Wait for NAV to expire if needed
                        if self.env.now < self.network.nav:
                            remaining_nav = self.network.nav - self.env.now
                            yield self.env.timeout(remaining_nav)
                        continue

                # Check for collisions
                contenders = [n for n in self.network.nodes 
                            if n.backoff_counter == 0 and np.any(n.packet_availability()!= 0)]
                
                if len(contenders) > 1:
                    self.network.stats['collisions'] += 1
                    self._handle_collision()
                    yield self.env.timeout(CTS_TIMEOUT)
                else:
                    # Lock backoff at zero during TXOP
                    self.backoff_counter = 0
                    yield self.env.process(self._attempt_rts_cts())

                    rx_nodes, rx_nodes_id = self.network.scheduling(self.node_id)
                    
                    yield self.env.process(self._transmit_data(rx_nodes))
        finally:
            self.contention_process_active = False  # Allow new process to start
            self._reset_backoff()  # NEW: Ensure clean state

    def _attempt_rts_cts(self):
        duration = RTS_DURATION + SIFS + CTS_DURATION + SIFS
        self.network.notify_activity(action='RTS_CTS', node_id=self.node_id, duration=duration)
        yield self.env.timeout(duration)
        self.network.stats['rts_cts_success'] += 1
        return 

    def _transmit_data(self, rx_node):

        queue = self.queues[rx_node]
        available_packets = len(queue)
    
        if not available_packets:
            return

        # Channel matrix for the corresponding tx and rx nodes
        H = self.network.channelMatrix[rx_node.node_id, self.node_id]
        P = self.max_tx_power

        # # Computing the SINR in dB - safely handle various H dimensions
        if np.isscalar(H):
            # For scalar channel coefficient
            sinr_db = 10 * np.log10((P * H) / self.noise_power)
        else:
            # For matrix channel coefficients
            signal_power = P * np.diag(H) if H.ndim == 2 else P * H
            interference = np.sum(H * P, axis=1) - np.diag(H) * P if H.ndim == 2 else 0
            sinr_db = 10 * np.log10(signal_power / (self.noise_power + interference))

        TODO 
        # MCS related parameters 
        MCS, N_bps, Rc = utils.MCS_cal_PER_001(sinr_db)
        
        # Handle invalid MCS case
        if np.isnan(MCS):
            MCS, N_bps, Rc = 0, 1, 0.5  # Fallback to lowest MCS
            Pe = 1.0  # 100% packet error
        else:
            Pe = 0.01  # 1% PER example,

        data_tx_time = self.config['TXOP_DURATION'] - utils.OverheadsCalc('BE')['EDCAoverheads']
        agg_packets = utils.tx_packets(self.config['NSC'], N_bps, Rc, self.config['NSS'], data_tx_time)

        tx_packet_number = min(available_packets, agg_packets)

        # Calculate transmission parameters
        tx_time = utils.elapsed_time_tx(
            self.config['NSC'],
            N_bps,
            Rc,
            self.config['NSS'],
            tx_packet_number
        )

        # # Process packets
        success_mask = np.random.rand(tx_packet_number) 
        rx_packets = [queue.popleft() for packet_id in range(tx_packet_number) if success_mask[packet_id] > Pe]

        # Update the packet counter for the STA
        self.network.packet_counter[rx_node.node_id] -= len(rx_packets)

        # Notify network of activity
        self.network.notify_activity(action='TX_DATA', node_id=self.node_id, duration=tx_time)
        self.current_transmission = True
        self.network.channel_state = 'BUSY'

        # Simulate transmission
        yield self.env.timeout(tx_time)

        # Update statistics
        now = self.env.now
        self.network.stats['tx_success'] += len(rx_packets)
        self.network.stats['delays'].extend(now - p['arrival'] for p in rx_packets)

        # Notify network of activity
        self.network.notify_activity(action='BACK', node_id=self.node_id, duration=BACK_DURATION)   # node_id is the receiver in this case
        self.current_transmission = True
        
        # Simulate transmission
        yield self.env.timeout(BACK_DURATION)

        # Indicate the channel is idle
        self.current_transmission = False
        self.network.channel_state = 'IDLE'

        # Wait an AIFS slot before starting the next contention process
        yield self.env.timeout(self.aifs)
        # Reset backoff counter
        self._reset_backoff()
        # # Start the new contention process
        # yield self.env.process(self.contention_process())
    
    def packet_availability(self):
        return self.network.packet_counter[self.network.association[self.node_id]]
    
    def _handle_collision(self):
        self.backoff_stage = min(self.backoff_stage + 1, self.config['max_backoff_stage'])
        self._reset_backoff()

    def _reset_backoff(self):
        # Only reset if packets exist
        if (self.is_ap and any(len(q) > 0 for q in self.queues.values())) or \
        (not self.is_ap and len(self.queue) > 0):
            self.cw = min(self.cw_min * (2 ** self.backoff_stage), self.cw_max)
            self.backoff_counter = np.random.randint(0, self.cw)
        else:
            self.backoff_counter = -1  # Flag for no packets

class AccessPoint(NetworkNode):
    pass

class Station(NetworkNode):
    pass

if __name__ == "__main__":
    # # Start Timer
    # start_time = time.time()

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

    # Define the traffic profiles
    traffic_profiles = {
        'A' : {'traffic_model': 'Poisson', 'traffic_load' : 100, 'latency': 1E-4},
        'B' : {'traffic_model': 'Bursty', 'traffic_load' : 50, 'latency': 2E-4},
        'C' : {'traffic_model': 'CBR', 'traffic_load' : 25, 'fps': 60, 'latency': 5E-4}
    }

    # Assign a traffic profile to each STA
    traffic_profile_perSTA = np.random.choice(['A','B', 'C'], size=STA_NUMBER).tolist()

    # Traffic Configuration 
    traffic_config = {
        'traffic_profiles': {    # Define the traffic profiles
            'A' : {'traffic_model': 'Poisson', 'traffic_load' : 100, 'latency': 1E-4},
            'B' : {'traffic_model': 'Bursty', 'traffic_load' : 50, 'latency': 2E-4},
            'C' : {'traffic_model': 'CBR', 'traffic_load' : 25, 'fps': 60, 'latency': 5E-4}  # not used by now
    },
        'EDCAaccessCategory' : 'BE'
    } 

    # Simulation Configuration
    sim_config = {
        'use_preloaded_deployments': True,
        'use_preloaded_traffic': False,
        'AP_NUMBER': AP_NUMBER,
        'STA_NUMBER': STA_NUMBER,
        'SCENARIO_TYPE': SCENARIO_TYPE,
        'GRID_VALUE': GRID_VALUE,
        'walls': walls,
        'MaxTxPower': MaxTxPower,
        'cw_min': {'ap': 16, 'sta': 16},
        'cw_max': 1023,
        'max_backoff_stage': 6,
        'aifs': {'ap': 43e-6, 'sta': 43e-6},
        'TXOP_DURATION': TXOP_DURATION,
        'PN_DBM': PN_DBM,
        'CCA': CCA,
        'NSS': NSS,
        'NSC': NSC,
        'learning_timestamp_to_stop': 2, # seconds
        'training_flag': False,
        'timestamp_to_stop': 5, # seconds
        'FRAME_LENGTH': FRAME_LENGTH, 
        'dl_rate': 1000,
        'ul_rate': 500,
        'uplink': False,
        'EVENT_NUMBER': int(1E5), # Number of events considered for traffic generation
        'seed': 1,
        'output_dir': os.path.join(os.getcwd(), 'traffic datasets', sim),
        'overheads' : utils.OverheadsCalc(traffic_config['EDCAaccessCategory'])
    }

    # Deployment
    _, _, sim_config['association'], sim_config['channelMatrix'] = deployment_generator(sim_config, sim_config['seed'])

    # Deployment data path
    h5file_deployments_path = os.path.join(os.getcwd(), 'deployments datasets', sim, 'deployment_datasets.h5')

    # Use pre-loaded data (if enabled)
    if sim_config['use_preloaded_deployments']:
        with h5py.File(h5file_deployments_path, 'r') as f:
            STA_matrix_save = f['STA_matrix_save'][:]
            channelMatrix_save = f['channelMatrix_save'][:]
        STA_matrix = STA_matrix_save[:, :, 0]
        sim_config['channelMatrix'] = channelMatrix_save[:, :, 0]

    # Start Timer
    start_time = time.time()
    np.random.seed(sim_config['seed'])
    env = simpy.Environment()
    network = SimPyWiFiNetwork(env, sim_config)
    network.run()
    print(f"Simulation took {time.time() - start_time:.2f} seconds")