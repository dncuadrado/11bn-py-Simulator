import gymnasium as gym
import numpy as np
from gymnasium import spaces

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self, sim_config, simulator):
        super().__init__()

        self.learning_timestamp_to_stop = sim_config['learning_timestamp_to_stop']
        self.simulator = simulator

        self.STA_NUMBER = sim_config['STA_NUMBER']

        if 'CGs_STAs' not in sim_config or 'STA_NUMBER' not in sim_config:
            raise KeyError("sim_config must contain 'CGs_STAs' and 'STA_NUMBER' keys")

        # Define the action space
        self.action_space = spaces.Discrete(len(sim_config['CGs_STAs']))  # Number of valid actions

        # # Environment with multi-dimensional observation space
        # self.observation_space = spaces.Dict({
        #     "queue_sizes": spaces.Box(low=0, high=1000, shape=(sim_config['STA_NUMBER'],), dtype=int),
        #     "delays": spaces.Box(low=0, high=1000, shape=(sim_config['STA_NUMBER'],), dtype=float)
        # })

        # Environment with flatten observation space
        self.observation_space = spaces.Box(low=0, high=1023, shape=(sim_config['STA_NUMBER'],), dtype=int)

        # Flag to control whether to forward the simulation or not. Used when the agent takes an action with no STAs to serve in the previous step           
        self.forward_flag = bool(True)

    def step(self, action):
        """
        Executes one time step in the simulator and returns:
        - next_state: The new state of the environment.
        - reward: The reward received for the action.
        - done: Whether the episode has ended.
        """
        
        if self.forward_flag == True: # False: the last action had no STAs to serve. Nothing happened, no need to forward the simulation now
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
    
    def reset(self, seed=None):
        """
        Resets the environment to the initial state. Get the observation in the initial state
        """

        # Seed the environment if seed is provided
        super().reset(seed=seed)  #

        self.simulator.InitSettings()
        self.simulator.sim_forward()

        obs = self.get_state()
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

        # Agent decision to be passed to the simulator
        agent_decision = [STA_rx, APs]

        return agent_decision

    def get_state(self):
        """
        Get the observation.
        Returns:
            observation (dict): The observation.
        """
        
        queue_sizes = np.array([len(self.simulator.get_queue(sta)) if self.simulator._firstPosTimestamp[sta] <= self.simulator.sim_timeline else 0 for sta in range(self.STA_NUMBER)])
        # delays = np.array([self.simulator.sim_timeline - self.simulator._firstPosTimestamp[sta] if self.simulator._firstPosTimestamp[sta] <= self.simulator.sim_timeline else 0 for sta in range(self.STA_NUMBER)])

        # # # Environment with multi-dimensional observation space
        # obs = {
        #     "queue_sizes": queue_sizes,
        #     "delays": delays
        # }

        # Environment with only one observation space
        obs = queue_sizes


        return obs
    
    def get_reward(self):
        """ Compute the reward
        Returns:
            reward (float): The reward value if delay-based
            reward (int): The reward value if packet-based
        """
        
        # # Delay-based reward. Considering the worst delay of the hol packets among all STAs
        # if self.forward_flag:  
        #     reward = -(self.simulator.sim_timeline - np.min(self.simulator._firstPosTimestamp))
        #     # if reward*1000 < -50:
        #     #     print(f"Reward: {reward*1000} at timeline {self.simulator.sim_timeline}")
        # else:
        #     reward = -0.1

        # Packet-based reward
        if self.forward_flag:
            reward = np.sum(self.simulator.per_TXOP_STA_tx_packets)
        else:
            reward = 0
        

        return reward



    
