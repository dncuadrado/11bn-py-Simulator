from typing import List, Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from TrafficGenerator import TrafficGenerator
from Utils import get_association, Throughput_DCF_bianchi, CG_creationTPC
from DeploymentGenerator import deployment_generator

from torch.distributions import Distribution 
Distribution.set_default_validate_args(False)

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self, sim_config, simulator):
        super().__init__()

        # Simulation configuration
        self.sim_config = sim_config

        # Set the duration of each episode depending on whether it is a training or validation episode
        if self.sim_config['training_flag'] == True:
            self.learning_timestamp_to_stop = self.sim_config['learning_timestamp_to_stop']
        else:
            self.learning_timestamp_to_stop = self.sim_config['timestamp_to_stop']

        # Loading the simulator into the environment
        self.simulator = simulator


        # Define the action space
        self.action_space = spaces.Discrete(len(self.simulator.CGs_STAs))  # Number of valid actions

        # # # # Environment with multi-dimensional observation space
        self.observation_space = spaces.Dict({
            # "queue_sizes": spaces.Box(low=0, high=1000, shape=(sim_config['STA_NUMBER'],), dtype=int),
            "delays": spaces.Box(low=0, high=1000, shape=(sim_config['STA_NUMBER'],), dtype=float),
            "datarates": spaces.Box(low=0, high=1000, shape=(len(self.simulator.datarate),), dtype=float),
        })

        # # Environment with flatten observation space
        # packet-based obs
        # self.observation_space = spaces.Box(low=0, high=1023, shape=(self.sim_config['STA_NUMBER'],), dtype=int)

        # delay-based obs
        # self.observation_space = spaces.Box(low=0, high=1000, shape=(sim_config['STA_NUMBER'],), dtype=float)

        # Flag to control whether to forward the simulation or not. Used when the agent takes an action with no STAs to serve in the previous step           
        self.forward_flag = bool(True)


        self.episode_counter = int(0)

    def step(self, action):
        """
        Executes one time step in the simulator and returns:
        - next_state: The new state of the environment.
        - reward: The reward received for the action.
        - done: Whether the episode has ended.
        """

        # Get the action
        agent_decision = self.get_action(action)
        
        # Execute the action
        self.simulator.run_step(agent_decision)

        # Get the reward
        reward = self.get_reward()

        if self.forward_flag == True: # foward the simulation if True. Set to False initially after reset
            self.simulator.sim_forward()
        else:
            raise ValueError("Empty actions are not allowed")

        # Get the observation
        obs = self.get_state()

        # Check termination conditions
        terminated = truncated = bool(self.simulator.sim_timeline >= self.learning_timestamp_to_stop)

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, STAs_arrivals_matrix=None):
        """
        Resets the environment to the initial state. Get the observation in the initial state
        """

        # Seed the environment if seed is provided
        super().reset(seed=seed)  #

        if self.episode_counter == 5:
            AP_matrix, STA_matrix, self.sim_config['association'], self.sim_config['channelMatrix'] = deployment_generator(self.sim_config)

            self.sim_config['per_STA_DCF_throughput_bianchi'] = Throughput_DCF_bianchi(self.sim_config['AP_NUMBER'], self.sim_config['STA_NUMBER'], self.sim_config['association'], self.sim_config['channelMatrix'], self.sim_config['MaxTxPower'],
                                                            self.sim_config['PN_DBM'], self.sim_config['NSC'], self.sim_config['NSS'], self.sim_config['TXOP_DURATION'], 
                                                            self.sim_config['DCFoverheads'], self.sim_config['EDCAaccessCategory'])

            map_matrix, TxPowerMatrixTemp, comb_ok, datarate = CG_creationTPC(self.sim_config['AP_NUMBER'], 
                                                        self.sim_config['STA_NUMBER'], 
                                                        self.sim_config['PN_DBM'], 
                                                        self.sim_config['NSC'], 
                                                        self.sim_config['NSS'], 
                                                        self.sim_config['association'], 
                                                        self.sim_config['channelMatrix'], 
                                                        self.sim_config['MaxTxPower'], 
                                                        CG_filter='on', TPC_method='PSO')    # TPC Optimization method: None, 'PSO', 'IPOPT', 'DE'

            self.simulator.CGs_STAs = map_matrix         # Entire groups matrix (all posible combinations)
            self.simulator.TxPowerMatrix = TxPowerMatrixTemp  # Entire Tx power matrix (all posible combinations)
            self.simulator.comb_ok = comb_ok # Combinations ok 
            self.simulator.datarate = datarate # Data rate for each combination
            self.episode_counter = 0

        if STAs_arrivals_matrix is None:
            STAs_arrivals_matrix = TrafficGenerator(
                        self.sim_config['STA_NUMBER'], # Number of STAs
                        self.sim_config['validation_flag'], # Validation flag
                        self.sim_config['traffic_type'], # Traffic type
                        self.sim_config['traffic_load'], # Traffic load
                        self.sim_config['L'], # Packet length
                        self.sim_config['per_STA_DCF_throughput_bianchi'], # DCF throughput per STA                                    
                        self.sim_config['EVENT_NUMBER']# Number of events considered for traffic generation
                        )  

        # Validate traffic
        if any(STAs_arrivals_matrix[i][-1] < self.learning_timestamp_to_stop for i in range(self.sim_config['STA_NUMBER'])):
            raise ValueError(f"Traffic should last more than {self.learning_timestamp_to_stop} seconds")                  

        # Loading the traffic dataset into the buffers in the simulator
        self.simulator.STA_queue_timeline = STAs_arrivals_matrix    

        # Reset the simulator (initialize the settings)
        self.simulator.InitSettings()

        # Advance until the first event
        self.simulator.sim_forward()

        # Get the observation
        obs = self.get_state()

        # Flag to control whether to forward the simulation or not. Used when the agent takes an action with no STAs to serve in the previous step           
        self.forward_flag = bool(True)

        # Increase the episode counter
        self.episode_counter += 1

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return obs, info
    
    def get_action(self, action):
        """
        Get the action.
        Returns:
            agent_decision (list): The action. [STA_rx, APs] where STA_rx is the list with the index of the STAs that will be served and 
                                                                       APs is the list of APs that will transmit.
        """
        uni = self.simulator.CGs_STAs[action]

        STA_rx = [sta for sta in uni if self.simulator._firstPosTimestamp[sta] <= self.simulator.sim_timeline]
        APs = get_association(self.simulator._association, STA_rx)

        if STA_rx:
            self.forward_flag = True
        else:
            self.forward_flag = False
            raise ValueError("Empty actions are not allowed")
            

        # Agent decision to be passed to the simulator
        agent_decision = [STA_rx, APs]

        return agent_decision

    def get_state(self):
        """
        Get the observation.
        Returns:
            observation (dict): The observation.
        """
        
        # queue_sizes = np.array([len(self.simulator.get_queue(sta)) if self.simulator._firstPosTimestamp[sta] <= self.simulator.sim_timeline else 0 for sta in range(self.sim_config['STA_NUMBER'])])
        delays = np.array([self.simulator.sim_timeline - self.simulator._firstPosTimestamp[sta] if self.simulator._firstPosTimestamp[sta] <= self.simulator.sim_timeline else 0 for sta in range(self.sim_config['STA_NUMBER'])])

        # # # # # # Environment with multi-dimensional observation space
        obs = {
            # "queue_sizes": queue_sizes,
            "delays": delays,
            "datarates": self.simulator.datarate,
        }

        # Environment with only one observation space
        # obs = queue_sizes
        # obs = delays


        return obs
    
    def get_reward(self):
        """ Compute the reward
        Returns:
            reward (float): The reward value if delay-based
            reward (int): The reward value if packet-based
        """
        
        # # Delay-based reward. Considering the worst delay of the hol packets among all STAs
        if self.forward_flag:  
            reward = -(self.simulator.sim_timeline - np.min(self.simulator._firstPosTimestamp))
            # if reward*1000 < -50:
            #     print(f"Reward: {reward*1000} at timeline {self.simulator.sim_timeline}")
        else:
            reward = -1
            raise ValueError("Empty actions are not allowed")

        # # Packet-based reward
        # if self.forward_flag:
        #     reward = np.sum(self.simulator.per_TXOP_STA_tx_packets)
        # else:
        #     reward = -100
        #     raise ValueError("Empty actions are not allowed")
        

        return reward
    
    def _choose_next_state(self) -> None:
        self.state = self.action_space.sample()

    def action_masks(self) -> List[Any]:
        """
        Updates the action masks according to the environment's state.
        """
        # Get queue sizes and valid indices
        queue_sizes = np.array([len(self.simulator.get_queue(sta)) if self.simulator._firstPosTimestamp[sta] <= self.simulator.sim_timeline else 0 for sta in range(self.sim_config['STA_NUMBER'])])
        mask = np.array(self.simulator.comb_ok, dtype=bool)
        
        # Ensure comb_ok is valid and returns at least one valid action
        if mask.size == 0 or not np.any(mask):
            raise ValueError("self.simulator.comb_ok is invalid or all actions are masked.")
    
        valid_indices = np.flatnonzero(mask)
        no_queues = [i for i in valid_indices if np.all(queue_sizes[self.simulator.CGs_STAs[i]] == 0)]
        mask[no_queues] = 0

        # Ensure at least one valid action remains
        if not np.any(mask):
            raise ValueError("No valid actions available in the current state.")
    
        return mask.tolist()



    
