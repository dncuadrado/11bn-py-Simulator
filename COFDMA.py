import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import product

np.random.seed(0)
# Parameters
num_aps = 4  # Number of APs
bandwidth_options = [0, 40, 80, 120, 160, 200, 240, 280, 320]  # Bandwidth options
total_bandwidth = 160  # Total available bandwidth

episodes = 5000  # Number of training episodes
ep_duration = 2  # Duration of each episode (number of steps)
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
epsilon_min = 0.1  # Minimum exploration rate

# Generate all valid actions (bandwidth allocation combinations)
all_combinations = list(product(bandwidth_options, repeat=num_aps))
valid_actions = [combo for combo in all_combinations if sum(combo) <= total_bandwidth]

state_to_index = {tuple(state): idx for idx, state in enumerate(all_combinations)}


# Map actions to indices for easier handling
action_to_index = {action: idx for idx, action in enumerate(valid_actions)}
index_to_action = {idx: action for action, idx in action_to_index.items()}

# Q-table
num_states = len(all_combinations)  # Allow states to represent any valid sum
num_actions = len(valid_actions)
Q_table = np.random.uniform(low=0, high=0.1, size=(num_states, num_actions))  # Initialize with small random values

# # Helper function: Convert state (bandwidth allocation) to an integer index
# def state_to_index(state):
#     index = 0
#     for i, bw in enumerate(state):
#         index += bw * ((total_bandwidth + 1) ** i)
#     return index

# Fixed throughput values for consistency (generated once)
def generate_fixed_throughput():
    throughput_values = {}
    for action in valid_actions:
        # Generate base throughput for 160 MHz
        base_throughput = [random.uniform(1, 10) if bw > 0 else 0 for bw in action]
        # Scale throughput proportionally to the allocated bandwidth
        scaled_throughput = [th * (bw / total_bandwidth) for th, bw in zip(base_throughput, action)]
        throughput_values[action] = scaled_throughput
    return throughput_values

throughput_values = generate_fixed_throughput()

# Helper function: Retrieve throughput for a given action
def calculate_throughput(action):
    return throughput_values[action]

# Track rewards for plotting
rewards_per_episode = []

# Training loop
for episode in range(episodes):
    # Initialize state (all APs unallocated)
    state = [0] * num_aps  # Represent state as actual bandwidth allocations
    total_reward = 0

    for step in range(ep_duration):
        # Choose action (epsilon-greedy)
        state_idx = state_to_index[tuple(state)]
        if random.uniform(0, 1) < epsilon:
            action_idx = random.randint(0, num_actions - 1)  # Explore
        else:
            action_idx = np.argmax(Q_table[state_idx])  # Exploit

        # Get the action and apply it
        action = index_to_action[action_idx]

        # Update state (add bandwidth allocations, ensuring valid values)
        new_state = [(s + a) for s, a in zip(state, action)]

        # Intermediate reward: Sum of scaled individual throughputs
        throughput = calculate_throughput(action)
        intermediate_reward = sum(throughput)

        # Update Q-table for intermediate reward
        new_state_idx = state_to_index[tuple(new_state)]
        Q_table[state_idx, action_idx] += alpha * (
            intermediate_reward + gamma * np.max(Q_table[new_state_idx]) - Q_table[state_idx, action_idx]
        )

        # Update state and accumulate reward
        state = new_state
        total_reward += intermediate_reward

    # Final reward adjustment at the end of the episode
    if all(s > 0 for s in state):  # Check if all APs are allocated
        final_throughput = calculate_throughput(action)
        final_reward = np.prod(final_throughput) * 10  # Scale up the final reward
        # total_reward = final_reward
    else:
        final_reward = -5  # Reduced penalty for missing allocations
        # total_reward = 0

    # Add final reward to the Q-table
    state_idx = state_to_index[tuple(state)]
    Q_table[state_idx, action_idx] += alpha * (
        final_reward + gamma * np.max(Q_table[state_idx]) - Q_table[state_idx, action_idx]
    )
    
    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Store total reward for this episode
    rewards_per_episode.append(final_reward)

    # Optional: Print progress
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}")

# Plot rewards over episodes
plt.figure(figsize=(10, 6))
plt.plot(rewards_per_episode, label='Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-learning Convergence with Proportional Throughput and Valid States')
plt.legend()
plt.grid()
plt.show()

# Display final Q-table (optional)
print("\nTraining complete. Q-table:")
print(Q_table)
