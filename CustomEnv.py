import gymnasium as gym
import numpy as np
from gymnasium import spaces

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self, sim_config, simulator):
        super().__init__()
        # super(CustomEnv, self).__init__()

        self.simulator = simulator

        self.STA_NUMBER = sim_config['STA_NUMBER']

        if 'CGs_STAs' not in sim_config or 'STA_NUMBER' not in sim_config:
            raise KeyError("sim_config must contain 'CGs_STAs' and 'STA_NUMBER' keys")

        # Define the action space
        self.action_space = spaces.Discrete(sim_config['CGs_STAs'].shape[0])  # Number of valid actions

        # Environment with multi-dimensional observation space
        self.observation_space = spaces.Dict({
            "queue_sizes": spaces.Box(low=0, high=1000, shape=(sim_config['STA_NUMBER'],), dtype=int),
            "delays": spaces.Box(low=0, high=1000, shape=(sim_config['STA_NUMBER'],), dtype=float)
        })


    def step(self, action):
        """
        Executes the action in the simulator and returns:
        - next_state: The new state of the environment.
        - reward: The reward received for the action.
        - done: Whether the episode has ended.
        """
        # Execute one time step within the environment
        self.simulator.sim_forward()
        agent_decision = self.get_action(action)

        self.simulator.run_step(agent_decision)

        obs = self.get_state()

        # self.apply_action(action)  # Map action index to simulator's logic


        # Check termination conditions
        terminated = bool(self.simulator.sim_timeline >= self.simulator.timestamp_to_stop)
        truncated = bool(self.simulator.sim_timeline >= self.simulator.timestamp_to_stop)

        # Get the reward
        reward = self.get_reward() if terminated else -np.inf

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None):
        """
        Resets the environment to the initial state. Get the observation in the initial state
        """

        # Seed the environment if seed is provided
        super().reset(seed=seed)  # This will seed self.np_random

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
        uni = self.simulator.CGs_STAs[action][~np.isnan(self.simulator.CGs_STAs[action])].astype(int)

        STA_rx = [sta for sta in uni if self.simulator._firstPosTimestamp[sta] <= self.simulator.sim_timeline]
        APs = [next((idx for idx, assoc in enumerate(self.simulator._association) if sta in assoc), -1) for sta in STA_rx]

        agent_decision = [STA_rx, APs]

        return agent_decision

    def get_state(self):
        """
        Get the observation.
        Returns:
            observation (dict): The observation.
        """
        queue_sizes = np.array([len(self.simulator.get_queue(sta)) if self.simulator._firstPosTimestamp[sta] <= self.simulator.sim_timeline else 0 for sta in range(self.STA_NUMBER)])
        delays = np.array([self.simulator.sim_timeline - self.simulator._firstPosTimestamp[sta] if self.simulator._firstPosTimestamp[sta] <= self.simulator.sim_timeline else 0 for sta in range(self.STA_NUMBER)])

        obs = {
            "queue_sizes": queue_sizes,
            "delays": delays
        }
        return obs
    
    def get_reward(self):
        """ Compute the reward
        Returns:
            reward (float): The reward value
        """

        self.simulator.TrafficAnalysis()
        reward = np.min(-np.percentile(self.simulator.delayvector, 99))

        return reward



    
