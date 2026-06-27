import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


# # Start Timer
# start_time = time.time()
for deplo in range(30):
    # --- Simulation settings ---
    np.random.seed(deplo)              # For reproducibility
    reward_function = 8            # Choose reward function type
    num_steps = 5000                 # Number of iterations
    window_size = 10                 # Rolling window for mean/median
    initial_delay = 1E-3              # Starting delay valu2

    



    # --- Initialize ---
    # historical_delay = np.random.choice(range(5, 30), 1, replace=False).tolist()
    # --- Initialize ---
    historical_delay = [initial_delay]
    records = []  # store metrics for analysis


    # --- Simulation loop ---
    for i in range(num_steps):
        # Compute statistics over recent history
        window = historical_delay[-window_size:]
        data_mean = np.mean(window)
        data_median = np.median(window)


        # Generate a new data point with random noise
        new_slot = np.random.randint(-6, 6) * 1E-3 + np.random.normal(0, 20)* 1E-3  # random fluctuation guranteing some increase bias
        new_data_point = max(historical_delay[-1] + new_slot, 1E-3)  # avoid non-positive delays
        historical_delay.append(new_data_point)

        if reward_function == 1:
            reward = 1E-3/(historical_delay[-1] + 1E-9)    # baseline: inverse of delay
        elif reward_function == 2:
            reward = data_mean - new_data_point            # rolling mean reward: positive if new delay < recent mean
        elif reward_function == 3:
            reward = np.log(data_mean / new_data_point)    # Log-coefficient reward
        elif reward_function == 4:
            lambda_ = 100
            reward = np.exp(-lambda_ * historical_delay[-1])  # Exponential decay reward
        elif reward_function == 5:
            k1, k2 = 1.0, 2.0                                       # penalize increases more strongly
            delta = data_mean - historical_delay[-1]
            reward = k1 * delta if delta > 0 else k2 * delta
        elif reward_function == 6:
            alpha = 0.7
            reward = - (alpha * window[-1] + (1 - alpha) * np.std(window))
        elif reward_function == 7:
            D_target, beta, gamma = 10, 0.1, 0.05
            current, prev = historical_delay[-1], historical_delay[-2]
            reward = -np.tanh(beta * (current - D_target)) + gamma * (prev - current)
        elif reward_function == 8:
            D_cutoff, k = 0.1, 5
            # D_cutoff, k = 0.2, 10

            # reward = 1 / (1 + np.exp(k * (historical_delay[-1] - D_cutoff)))
            # delay_reward = 2 / (1 + np.exp(k * (historical_delay[-1] - D_cutoff))) - 1 # scaled to [-1, 1]
            delay_reward = 1 / (1 + np.exp(k * (historical_delay[-1])))  # scaled to [-1, 0.23]
            # delay_reward = 2 / (1 + np.exp(k * (historical_delay[-1] - D_cutoff))) - 1 # scaled to [-1, 1]
            
            reward = delay_reward

        # Store results
        records.append({
            'step': i,
            'mean': data_mean,
            'median': data_median,
            'new_point': new_data_point,
            'reward': reward
        })

        # Print summary every 10 steps
        # if i % 10 == 0 or i == num_steps - 1:
        #     print(f"Step {i:3d} | Mean: {data_mean:6.2f} | New: {new_data_point:6.2f} | Reward: {reward:+.3f}")

    pos_cum_reward = sum(r['reward'] for r in records if r['reward'] > 0)
    neg_cum_reward = sum(r['reward'] for r in records if r['reward'] < 0)

    # # End Timer
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"\nElapsed Time: {elapsed_time:.2f} seconds")

    print(f"\nCumulative Positive Reward: {pos_cum_reward:.3f}")
    print(f"Cumulative Negative Reward: {neg_cum_reward:.3f}")
    # --- Convert to DataFrame for analysis ---
    df = pd.DataFrame(records)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- First subplot: delay trends ---
    axes[0].plot(df['step'], df['new_point'], label='New Delay', marker='o', markersize=4, alpha=0.7)
    axes[0].plot(df['step'], df['mean'], label='Rolling Mean', linestyle='--')
    axes[0].set_title("Delay Evolution Over Time")
    axes[0].set_ylabel("Delay")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Second subplot: reward ---
    axes[1].plot(df['step'], df['reward'], label='Reward', color='purple')
    axes[1].set_title("Reward Over Time")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Reward")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
