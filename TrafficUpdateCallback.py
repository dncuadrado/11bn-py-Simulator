from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from TrafficGenerator import TrafficGenerator


# Custom Callback to Update Traffic
class TrafficUpdateCallback(BaseCallback):
    def __init__(self, sim_config, learning_config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Learning parameters
        self.parallel_envs = learning_config['parallel_envs']
        self.simulator_attr = learning_config['simulator_attr']
        self.total_timesteps_per_episode = learning_config['total_timesteps_per_episode']

        # Simulation parameters
        self.STA_NUMBER = sim_config['STA_NUMBER']
        self.validation_flag = sim_config['validation_flag']
        self.traffic_type = sim_config['traffic_type']
        self.traffic_load = sim_config['traffic_load']
        self.learning_timestamp_to_stop = sim_config['learning_timestamp_to_stop']
        self.L = sim_config['L']
        self.per_STA_DCF_throughput_bianchi = sim_config['per_STA_DCF_throughput_bianchi']

    def _on_step(self) -> bool:
        # Update traffic periodically
        if self.num_timesteps % self.total_timesteps_per_episode == 0:
            # Generate traffic and validate it in one step
            traffic = []
            for _ in range(self.parallel_envs):
                STAs_arrivals_matrix = TrafficGenerator(
                    self.STA_NUMBER, self.validation_flag, self.traffic_type,
                    self.traffic_load, self.L, self.per_STA_DCF_throughput_bianchi,
                    EVENT_NUMBER=15000
                )
                # Validate traffic inline
                if any(STAs_arrivals_matrix[i][-1] < self.learning_timestamp_to_stop for i in range(self.STA_NUMBER)):
                    raise ValueError(f"Traffic should last more than {self.learning_timestamp_to_stop} seconds")
                traffic.append(STAs_arrivals_matrix)

            # Update all simulators in a single loop
            simulators = self.training_env.get_attr(self.simulator_attr)
            for env_instance, STAs_arrivals_matrix in zip(simulators, traffic):
                env_instance.STA_queue_timeline = STAs_arrivals_matrix

        return True