# 11bn-py-Simulator

A Python-based simulator for IEEE 802.11bn candidate features with integrated Deep Reinforcement Learning (DRL) framework for intelligent scheduling optimization.

## Authors

- David Nunez, david.nunez@upf.edu

## Introduction

The 11bn-py-Simulator is a comprehensive simulation platform designed to model and evaluate IEEE 802.11bn wireless networks. It provides a flexible environment for developing and testing scheduling algorithms, with built-in support for Deep Reinforcement Learning (DRL) agents. The simulator enables researchers and developers to train Proximal Policy Optimization (PPO) agents that can learn to make optimal scheduling decisions in complex network scenarios. 

>Note that, the current PPO implementation uses MaskablePPO from https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

This project bridges the gap between network simulation and machine learning, allowing for the development of intelligent, data-driven scheduling solutions for next-generation wireless networks.

## Features

- **IEEE 802.11bn Simulation**: Comprehensive modeling of 802.11bn network features and behaviors
- **Traffic Generation**: Realistic traffic pattern generation for various network conditions
- **Deployment Scenarios**: Multiple deployment configurations with scalable network topologies
- **DRL Integration**: Built-in support for training reinforcement learning agents
- **PPO Agent**: Pre-configured Proximal Policy Optimization (MaskablePPO) agent for scheduling optimization
- **Custom Environment**: Gymnasium-compatible custom environment for RL agent training
- **Dataset Support**: Pre-generated traffic and deployment datasets for consistent evaluation
- **Training Monitoring**: Wandb integration for real-time training visualization
- **Model Evaluation**: Comprehensive evaluation metrics and analysis tools

## Overview

### Architecture

The simulator is organized into the following core components:

- **`simulator_main.py`**: Main entry point for running simulations
- **`custom_env.py`**: Custom Gymnasium (https://gymnasium.farama.org/api/env/) environment for DRL agent interaction
- **`rl_agent.py`**: RL agent implementation and training logic
- **`traffic_generator.py`**: Generates realistic network traffic patterns
- **`deployment_generator.py`**: Creates network deployment scenarios
- **`mapc_sim.py`**: MAPC simulator integration (main simulator class)
- **`constants.py`**: Project-wide constants and configuration parameters
- **`utils.py`**: Utility functions for logging, data handling, and processing

### Project Structure

```
├── code/                          # Source code modules
│   ├── constants.py              # Configuration constants
│   ├── custom_env.py             # Custom Gym environment
│   ├── deployment_generator.py   # Deployment generation
│   ├── mapc_sim.py               # MAPC simulator
│   ├── rl_agent.py               # DRL agent implementation
│   ├── simulator_main.py          # Main simulator
│   ├── traffic_generator.py       # Traffic pattern generation
│   └── utils.py                  # Utility functions
├── deployments_datasets/          # Pre-generated deployment configurations
├── traffic_datasets/              # Pre-generated traffic patterns
├── trained_models/                # Trained models and evaluation results
│   ├── models/                   # Saved model checkpoints
│   ├── evaluations.npz           # Evaluation metrics
│   ├── monitor.csv               # Training monitor logs
│   └── tensorboard/              # TensorBoard logs
├── results/                      # To store the results
└── README.md                      # This file
```

## Usage

### Prerequisites

- Python 3.7+
- Required dependencies (see requirements)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd 11bn-py-Simulator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Simulator

#### Basic Simulation

To run a basic simulation:

```bash
python code/simulator_main.py
```

For reproducibility, both the simulator and the learning agent support configurable seeds. For simulation, deployments (i.e., device locations) and traffic patterns can either be generated dynamically during execution or pre-generated and stored as datasets for reuse in subsequent simulations by configuring the following parameters:
- `sim_config['use_preloaded_deployments']: True`
- `sim_config['use_preloaded_traffic']: True`

#### Training a DRL Agent

To train a PPO agent for scheduling optimization:

```bash
python code/rl_agent.py
```

The training process will:
- Initialize the custom environment with deployment and traffic datasets
- Train the PPO agent using the specified hyperparameters
- Log training metrics to TensorBoard
- Save model checkpoints to `trained_models/models/`


The training support parallel environments through `SubprocVecEnv`.

##### Wandb Integration

The training process can be monitored via wandb API from https://wandb.ai. For that purpose, in rl_agent.py set:
- `learning_config['wandb_log']: True`

The following two files have been customized in order to show /custom statistics in Wandb:
- In `stable_baselines3.common.monitor:101` add:

            ep_info = {
                "r": round(ep_rew, 6), 
                "l": ep_len, 
                "t": round(time.time() - self.t_start, 6),
                "total_percentile99": round(info['total_percentile99'],6),
                "worst_percentile99": round(info['worst_percentile99'],6),
                "mean_rew_shaping": round(info['mean_rew_shaping'],6),
                "mean_long_term_rew": round(info['mean_long_term_rew'],6),
                "mean_reward": round(info['mean_reward'],6),
                } 
- In `stable_baselines3.common.on_policy_algorithm:293` add:
            
            self.logger.record("custom/ep_total_percentile99", safe_mean([ep_info["total_percentile99"] for ep_info in self.ep_info_buffer]))
            self.logger.record("custom/ep_worst_percentile99", safe_mean([ep_info["worst_percentile99"] for ep_info in self.ep_info_buffer]))
            self.logger.record("custom/step_mean_rew_shaping", safe_mean([ep_info["mean_rew_shaping"] for ep_info in self.ep_info_buffer]))
            self.logger.record("custom/step_mean_long_term_rew", safe_mean([ep_info["mean_long_term_rew"] for ep_info in self.ep_info_buffer]))
            self.logger.record("custom/step_mean_reward", safe_mean([ep_info["mean_reward"] for ep_info in self.ep_info_buffer])) 



##### Saving models
For saving the model, two methods are available:
1. `learning_config['save_best_model']: True` -> Save the best model through EvalCallback
2. `learning_config['checkpoint_log']: True` -> Save the model periodically

#### Evaluation

The model is evaluate from rl_agent.evaluation which is called from simulator_main.py
Don't forget to add the proper name of the model to `model` list.   

### Configuration

The constants groups, SYSTEM, MAC, CHANNEL are stored in `code/constants.py`.

Apart from that, some settings can be set in:
- traffic_config
- sim_config
- learning_config

## Contribute

Contributions are welcome! But first, send an email to david.nunez@upf.edu to discuss in advance the scope/goal of the contribution

After that:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows the project's style guidelines
- Changes include appropriate documentation
- Tests are added for new functionality

## Acknowledgements

This work is supported by the following projects:
MLDR (Chist-ERA ANR-23-CHR4-0005) PCI2023-145958-2 MCIU/AEI/10.13039/
TRUE Wi-Fi PID2024-155470NB-I00 (MICIU/AEI/10,13039/501100011033/FEDER, UE)
Wi-XR Wi-XR PID2021-123995NB-I00 (MCIU/AEI/FEDER,UE)
AGAUR ICREA Academia 00077
Maria de Maeztu Units of Excellence Programme (CEX2021-001195-M)

---
