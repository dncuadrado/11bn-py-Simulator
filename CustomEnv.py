import gymnasium as gym
import numpy as np
from gymnasium import spaces
import copy
from TrafficGenerator import TrafficGenerator   
from Utils import get_association

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

        # # # Environment with multi-dimensional observation space
        # self.observation_space = spaces.Dict({
        #     "queue_sizes": spaces.Box(low=0, high=1000, shape=(sim_config['STA_NUMBER'],), dtype=int),
        #     "delays": spaces.Box(low=0, high=1000, shape=(sim_config['STA_NUMBER'],), dtype=float)
        # })

        # # Environment with flatten observation space
        # self.observation_space = spaces.Box(low=0, high=1023, shape=(self.sim_config['STA_NUMBER'],), dtype=int)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(sim_config['STA_NUMBER'],), dtype=float)

        # Flag to control whether to forward the simulation or not. Used when the agent takes an action with no STAs to serve in the previous step           
        self.forward_flag = bool(False)

    def step(self, action):
        """
        Executes one time step in the simulator and returns:
        - next_state: The new state of the environment.
        - reward: The reward received for the action.
        - done: Whether the episode has ended.
        """
        
        if self.forward_flag == True: # foward the simulation if True. Set to False initially after reset
            self.simulator.sim_forward()


        # Get the action
        agent_decision = self.get_action(action)
        
        # Execute the action
        self.simulator.run_step(agent_decision)

        # Get the observation
        obs = self.get_state()

        # Check termination conditions
        terminated = truncated = bool(self.simulator.sim_timeline >= self.learning_timestamp_to_stop)

        # Get the reward
        reward = self.get_reward()
  

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, STAs_arrivals_matrix=None):
        """
        Resets the environment to the initial state. Get the observation in the initial state
        """

        # Seed the environment if seed is provided
        super().reset(seed=seed)  #

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
        self.forward_flag = bool(False)

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
        APs = [next((idx for idx, assoc in enumerate(self.simulator._association) if sta in assoc), -1) for sta in STA_rx]

        if STA_rx:
            self.forward_flag = True
        else:
            self.forward_flag = False

        # if self.sim_config['training_flag'] == False:
        #     raise ValueError("The environment is in validation mode. Void actions are not allowed")

        # Agent decision to be passed to the simulator
        agent_decision = [STA_rx, APs]

        return agent_decision

    def get_state(self):
        """
        Get the observation.
        Returns:
            observation (dict): The observation.
        """
        
        queue_sizes = np.array([len(self.simulator.get_queue(sta)) if self.simulator._firstPosTimestamp[sta] <= self.simulator.sim_timeline else 0 for sta in range(self.sim_config['STA_NUMBER'])])
        delays = np.array([self.simulator.sim_timeline - self.simulator._firstPosTimestamp[sta] if self.simulator._firstPosTimestamp[sta] <= self.simulator.sim_timeline else 0 for sta in range(self.sim_config['STA_NUMBER'])])

        # # # # Environment with multi-dimensional observation space
        # obs = {
        #     "queue_sizes": queue_sizes,
        #     "delays": delays
        # }

        # Environment with only one observation space
        # obs = queue_sizes
        obs = delays


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

        # # Packet-based reward
        # if self.forward_flag:
        #     reward = np.sum(self.simulator.per_TXOP_STA_tx_packets)
        # else:
        #     reward = -100
        

        return reward



    
