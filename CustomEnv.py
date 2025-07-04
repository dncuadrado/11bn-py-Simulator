from typing import List, Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from TrafficGenerator import traffic_generator
from Utils import get_association, Throughput_EDCA_bianchi, CG_creationTPC, get_channel_coefficient, get_channel_coefficient_bss
from DeploymentGenerator import deployment_generator
from collections import deque

from torch.distributions import Distribution 
Distribution.set_default_validate_args(False)


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self, traffic_config, sim_config, learning_config, simulator):
        super().__init__()

        # Simulation configuration
        self.sim_config = sim_config
        self.traffic_config = traffic_config

        self.training_flag = sim_config["training_flag"]  # Store flag

        # Set the duration of each episode depending on whether it is a training or validation episode
        if self.sim_config['training_flag'] == True:
            self.learning_timestamp_to_stop = self.sim_config['learning_timestamp_to_stop']
        else:
            self.learning_timestamp_to_stop = self.sim_config['timestamp_to_stop']

        # Loading the simulator into the environment
        self.simulator = simulator

        # Define the action space
        self.action_space = spaces.Discrete(len(self.simulator.comb_ok))  # Number of valid actions

        # Initialize the state
        self.delays = np.zeros(self.sim_config['STA_NUMBER'], dtype=float)                      # delays
        self.queue_sizes = np.zeros(self.sim_config['STA_NUMBER'], dtype=float)                 # queue sizes
        # self.channel_coef = np.zeros(self.sim_config['STA_NUMBER'], dtype=float)
        self.channel_coef = np.zeros(self.sim_config['STA_NUMBER']*self.sim_config['AP_NUMBER'], dtype=float)


        # obs space
        # self.observation_space = spaces.Box(low=0, high=1, shape=(sim_config['STA_NUMBER']*3,), dtype=float)
        self.observation_space = spaces.Box(low=0, high=1, shape=(sim_config['STA_NUMBER']*2 + len(self.channel_coef),), dtype=float)

        # self.observation_space = spaces.Dict({
        #     "dynamic": spaces.Box(0, 1, shape=(sim_config['STA_NUMBER']*2,), dtype=np.float32),
        #     "static":  spaces.Box(0, 1, shape=(len(self.channel_coef),), dtype=np.float32),
        # })

        self.step_number = int(0)

        self.agent_decision = [[], []]  # Initialize agent decision
 
        # Masking the actions
        self.mask = np.array(self.simulator.comb_ok, dtype=bool)
        self.valid_indices = np.flatnonzero(self.mask)

        self.rew_shaping: list = []
        self.long_term_rew: list = []
        self.reward: list = []

        self.training_episode_counter = int(0)  # Counter for the number of episodes
        self.episode_counter_threshold =  learning_config['episode_threshold']  # Number of episodes to wait before generating a new deployment

        self.w_sparse = learning_config['w_sparse']  # Weight for the sparse reward
        self.w_short_term = 1 - self.w_sparse  # Weight for the short term reward
        self.w_packet_rew = 1 - self.w_sparse

    def reset(self, seed=None, STAs_arrivals_matrix=None, traffic_profile_perSTA=None, is_deloyment_fixed=False):
        """
        Resets the environment to the initial state. Get the observation in the initial state
        """

        # Seed the environment if seed is provided
        super().reset(seed=seed)  #

        ########### deployment generation 
        if (self.training_episode_counter > self.episode_counter_threshold) or (self.training_flag == False) and (not is_deloyment_fixed):
            # Generate a new deployment
            self._deployment_generator(seed=seed)

            if self.training_flag:
                self.training_episode_counter = int(0)  # Reset the episode counter

        if self.training_flag:
            self.training_episode_counter += 1  # Increment the episode counter


        ########### traffic generation
        if STAs_arrivals_matrix is None:
            # Set the seed
            np.random.seed(seed)
            if traffic_profile_perSTA is None:
                # traffic_profile_perSTA = np.random.choice(['A','B'], size=self.sim_config['STA_NUMBER']).tolist()
                traffic_profile_perSTA = [
                        {
                            'traffic_load': np.random.uniform(self.traffic_config['load_min'], self.traffic_config['load_max']),  # Load in Mbps
                            'traffic_model': str(np.random.choice(['Poisson', 'Bursty']))  # Traffic model
                        }
                        for i in range(self.sim_config['STA_NUMBER'])
                ]

            STAs_arrivals_matrix = traffic_generator(
                self.traffic_config,
                self.sim_config,
                traffic_profile_perSTA,
                )  

        # Validate traffic
        if any(STAs_arrivals_matrix[i][-1] < self.learning_timestamp_to_stop for i in range(self.sim_config['STA_NUMBER'])):
            raise ValueError(f"Traffic should last more than {self.learning_timestamp_to_stop} seconds")                  

        # Loading the traffic dataset into the buffers in the simulator
        self.simulator.STA_queue_timeline = STAs_arrivals_matrix.copy()    

        # Reset the simulator (initialize the settings)
        self.simulator.InitSettings()

        # Advance until the first event
        self.simulator.sim_forward()

        # Initialize the state
        self.delays = np.zeros(self.sim_config['STA_NUMBER'], dtype=float)                      # delays
        self.queue_sizes = np.zeros(self.sim_config['STA_NUMBER'], dtype=float)                 # queue sizes
        self.agent_decision = [[], []]  # Initialize agent decision  

        # Masking the bad actions (actions that are not possible independently of the queue ocupancy)
        self.mask = np.array(self.simulator.comb_ok, dtype=bool)
        self.valid_indices = np.flatnonzero(self.mask)

        self.reward_shaping = []
        self.long_term_reward = []
        self.reward = []

        # Get the observation
        obs = self.get_state()

        self.last_txop_timestamp = np.min(self.simulator._firstPosTimestamp)
        self.step_number = int(0)

        # Optionally we can pass additional info
        info = {}

        return obs, info
    
    def step(self, action):
        """
        Executes one time step in the simulator and returns:
        - next_state: The new state of the environment.
        - reward: The reward received for the action.
        - done: Whether the episode has ended.
        """

        # Get the action
        self.get_action(action)

        # increase the step number
        self.step_number += 1
        
        # Execute the action
        self.simulator.run_step(self.agent_decision)

        # if self.agent_decision is None:
        #     self.simulator.sim_timeline += 48E-6 + 34E-6 + 9E-6  # the agent decides to wait a little more (no tx) ---> fake_frame + DIFS + Te
            
        # Forward the simulation
        self.simulator.sim_forward()

        # Get the observation
        obs = self.get_state()

        # Get the reward
        reward = self.get_reward()

        # Check termination conditions
        terminated = truncated = bool(self.simulator.sim_timeline >= self.learning_timestamp_to_stop)

        info = {}
        if terminated or truncated:
            self.simulator.TrafficAnalysis()
            prctile99 = [np.percentile(self.simulator.delay_per_STA[sta],99) for sta in range(self.sim_config['STA_NUMBER'])]

            info['total_percentile99'] = -np.percentile(self.simulator.delayvector,99)
            info['worst_percentile99'] = -max(prctile99)
            info['mean_rew_shaping'] = np.mean(self.reward_shaping)
            info['mean_long_term_rew'] = np.mean(self.long_term_reward)
            info['mean_reward'] = np.mean(self.reward)



        return obs, reward, terminated, truncated, info
    
    def get_action(self, action):
        """
        Get the action.
        Returns:
            agent_decision (list): The action. [STA_rx, APs] where STA_rx is the list with the index of the STAs that will be served and 
                                                                       APs is the list of APs that will transmit.
        """
        uni = self.simulator.CGs_STAs[action]

        if uni is None:    # EMPTY ACTION, USED FOR WAITING MORE TIME TO ALLOW THE QUEUES TO FILL
            # ValueError("Empty action received, this should not happen. The agent should always choose a valid action.")
            self.agent_decision = None
        else:
            STA_rx = [sta for sta in uni if self.simulator._firstPosTimestamp[sta] <= self.simulator.sim_timeline]
            APs = get_association(self.simulator._association, STA_rx)
                
            # Agent decision to be passed to the simulator
            self.agent_decision = [STA_rx, APs]

        return self.agent_decision

    def get_state(self):
        """
        Get the observation.
        Returns:
            observation : The observation.
        """
        # # Queue sizes Normalized 
        self.queue_sizes = np.array([min(len(self.simulator.get_queue(sta))/1E4,1) if self.simulator._firstPosTimestamp[sta] <= self.simulator.sim_timeline else 0 for sta in range(self.sim_config['STA_NUMBER'])])
 
        # Delays
        self.delays = np.array([(self.simulator.sim_timeline - self.simulator._firstPosTimestamp[sta]) / float(self.learning_timestamp_to_stop) if self.simulator._firstPosTimestamp[sta] <= self.simulator.sim_timeline else 0.0 for sta in range(self.sim_config['STA_NUMBER'])])
        
        # Channel coefficients
        # self.channel_coef = [r / 0.005 for r in get_channel_coefficient_bss(self.simulator.channel_matrix_fading, range(self.sim_config['STA_NUMBER']), get_association(self.sim_config['association'], range(self.sim_config['STA_NUMBER'])))] # RSSI normalized to 0-1          0.005 is the minimum channel coeficient, i.e., no walls, distance= 1m
        self.channel_coef = [r / 0.005 for r in get_channel_coefficient(self.simulator.channel_matrix_fading)] # RSSI normalized to 0-1          0.005 is the minimum channel coeficient, i.e., no walls, distance= 1m

        # Observation
        obs = np.concatenate((self.delays, self.queue_sizes, self.channel_coef))

        # obs_dyn = np.concatenate([self.delays, self.queue_sizes])
        # obs_stat = np.array(self.channel_coef, dtype=np.float32)

        # return {"dynamic": obs_dyn, "static": obs_stat}
        return obs
    
    def get_reward(self):
        """ Compute the reward """

        ################################################################
        

        # # # Compute the short term reward (for reward shaping)
        reward_shaping = np.min(self.simulator._firstPosTimestamp) - self.last_txop_timestamp   
        self.last_txop_timestamp = np.min(self.simulator._firstPosTimestamp)
        self.reward_shaping.append(reward_shaping)


        current_worst_delay = self.simulator.sim_timeline - np.min(self.simulator._firstPosTimestamp)
        long_term_reward = min(0.001/(current_worst_delay + 1E-6),1.0)
        self.long_term_reward.append(long_term_reward)
        ####################################################################################

        if any(self.simulator.per_TXOP_STA_tx_packets<0):
            packet_reward = 0.0
            throughput_reward = 0.0
            delay_reward = 0.0
        else:
            packet_reward = 1E-2 * sum(self.simulator.per_TXOP_STA_tx_packets)/sum(self.simulator.last_txop_queue_sizes) 
            throughput_reward = 1E-8 * self.simulator.nominal_data_rate
            delay_reward = reward_shaping + long_term_reward
            
        
        
        reward = delay_reward
        


        ####################################################################################
        ####################################################################################

        if self.agent_decision is not None:
            # if (not self.agent_decision[0]):
            #     reward = -1.0  # Negative reward for no action or invalid actions

            if any(self.simulator.per_TXOP_STA_tx_packets<0) or (not self.agent_decision[0]):
                # If any STA has negative packets, it means that the action was invalid
                reward = 0.0


        ################################################################

        # print(f"Step: {self.step_number} \n\
        # #       STAs - APs: {self.agent_decision} \n\
        # #       Queue sizes: {self.queue_sizes} \n\
        # #       Delays: {[(self.simulator.sim_timeline - self.simulator._firstPosTimestamp[sta]) if self.simulator._firstPosTimestamp[sta] <= self.simulator.sim_timeline else 0 for sta in range(self.sim_config['STA_NUMBER'])]} \n\
        # #       Transmitted packets: {self.simulator.per_TXOP_STA_tx_packets} \n\
        # #       Packet reward: {packet_reward} \n\
        # #       Throughput reward: {throughput_reward}  \n\
        # #       Delay reward: {delay_reward} \n\
        # #       Current worst delay: {self.simulator.sim_timeline - np.min(self.simulator._firstPosTimestamp)} \n\
        # #       Timestamp: {self.simulator.sim_timeline} \n\
        # #       Total reward: {reward} \n\
        # #       -----------------------------------------------------------------------------------------------------")

        self.reward.append(reward)
        return reward 

    def _choose_next_state(self) -> None:
        self.state = self.action_space.sample()

    def action_masks(self) -> List[Any]:
        """
        Updates the action masks according to the environment's state.
        """
        # Mask the actions that are not possible considering also the queue occupancy
        mask = self.mask.copy()   # mask is a boolean array indicating valid actions, at this point it already masks bad actions
        no_queues = [i for i in self.valid_indices[:-1] if np.all(self.queue_sizes[self.simulator.CGs_STAs[i]] == 0)]   # indices of actions with no packets in all the participants in the group
        mask[no_queues] = 0

        # Ensure at least one valid action remains
        if not np.any(mask):
            raise ValueError("No valid actions available in the current state.")
    
        return mask.tolist()
    
    def _deployment_generator(self, seed=None):
        """
        Generates a new deployment.
        """
        # Generate a new deployment
        _, _, self.sim_config['association'], channel_matrix = deployment_generator(self.sim_config, seed)
        self.simulator.channel_matrix = channel_matrix  # Channel matrix

        _, TxPowerMatrixTemp, comb_ok = CG_creationTPC(self.sim_config['AP_NUMBER'], 
                                                self.sim_config['STA_NUMBER'], 
                                                self.sim_config['PN_DBM'], 
                                                self.sim_config['NSC'], 
                                                self.sim_config['NSS'], 
                                                self.sim_config['association'], 
                                                channel_matrix, 
                                                self.sim_config['MaxTxPower'], 
                                                is_filtering=self.sim_config['filtering'], TPC_method=None, # TPC Optimization method: None, 'PSO', 'IPOPT', 'DE'
                                                CG_size=4)  

        self.simulator.TxPowerMatrixTemp = TxPowerMatrixTemp 
        # self.simulator.TxPowerMatrixTemp.append(None) 

        self.simulator.comb_ok = comb_ok
        # self.simulator.comb_ok = np.append(self.simulator.comb_ok, True)



    
