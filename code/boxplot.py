import os
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def plot_delay_distribution(sim, strategies=None, threshold_ms=100, plot_type='box'):
    """
    Loads simulation data, filters deployments, and plots delay distributions
    
    Args:
        sim (str): Simulation directory name
        strategies (list): Strategies to include (None for auto-discovery)
        threshold_ms (int): P99 delay threshold for filtering
        plot_type (str): 'box' for boxplot or 'violin' for violin plot
    """
    base_path = os.path.join('Results', 'Simulation', sim)
    summary_file = os.path.join(base_path, 'summary.json')
    
    if not os.path.isfile(summary_file):
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    
    # Auto-discover strategies if not provided
    if strategies is None:
        dep0 = next(iter(summary_data.values()))
        traf0 = next(iter(dep0.values()))
        strategies = list(traf0['p99_delays_ms'].keys())
    
    # Identify valid deployment-traffic pairs
    valid_pairs = []
    for dep, traffic_data in summary_data.items():
        for traf, metrics in traffic_data.items():
            non_edca_delays = [
                delay for strat, delay in metrics['p99_delays_ms'].items() 
                if strat != 'EDCA' and strat in strategies
            ]
            if non_edca_delays and min(non_edca_delays) <= threshold_ms:
                valid_pairs.append((dep, traf))
    
    print(f"Selected {len(valid_pairs)} deployment-traffic pairs")
    
    # Aggregate delay data
    delay_data = {s: [] for s in strategies}
    for dep, traf in valid_pairs:
        filepath = os.path.join(base_path, dep, traf, 'delay.h5')
        if os.path.exists(filepath):
            with h5py.File(filepath, 'r') as h5f:
                for strat in strategies:
                    if strat in h5f:
                        # data = np.array(h5f[strat]).ravel() * 1000  # Convert to ms
                        data = np.percentile(np.array(h5f[strat]).ravel(),99) * 1000
                        # data = np.percentile(np.array(h5f[strat]).ravel(),50) * 1000
                        delay_data[strat].append(data)
    
    # Flatten nested lists
    # for strat in strategies:
    #     if delay_data[strat]:
    #         delay_data[strat] = np.concatenate(delay_data[strat])
    
    # Generate visualization
    if plot_type == 'box':
        plot_boxplot(delay_data, strategies, threshold_ms)
    else:
        raise ValueError("Invalid plot_type. Choose 'box'")

def plot_boxplot(delay_data, strategies, threshold_ms):
    """Generates boxplot visualization with 99th percentile markers"""
    # Setup color palette
    colors = [
        '#36B3D9', '#FF9E4A', '#8C6BBE', '#66C2A5', 
        '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854'
    ]
    colors = colors[:len(strategies)]
    
    # Configure plot
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    
    # Create boxplot
    positions = np.arange(len(strategies))
    boxes = ax.boxplot(
        [delay_data[s] for s in strategies],
        positions=positions,
        patch_artist=True,
        widths=0.3,
        showfliers=True,  # Cleaner visualization
        zorder=1  # Place behind markers
    )
    
    # Color boxes with transparency
    for i, box in enumerate(boxes['boxes']):
        box.set_facecolor(colors[i])
        box.set_alpha(0.6)
    
    # # Add 99th percentile markers
    # for i, strat in enumerate(strategies):
    #     p99 = np.percentile(delay_data[strat], 99)
    #     plt.scatter(i, p99, s=120, marker='D', 
    #                edgecolor='#E41A1C', facecolor='white', 
    #                linewidth=2, zorder=2, label='99th Percentile' if i == 0 else "")
    
    # Format plot
    ax.set_xticks(positions)
    # Apply your label modification
    modified_strategies = ['ML-G' if s == '6cbu38xr' else s for s in strategies]
    ax.set_xticklabels(modified_strategies)
    ax.set_ylabel('99th percentile distribution [ms]', fontsize=12, fontweight='bold')

    # ax.set_title(f'Delay Distribution (Threshold: {threshold_ms}ms)', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    # Create unified legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper right', framealpha=1)
    
    plt.tight_layout()
    plt.show()


# Box plot
plot_delay_distribution('30-16 (general_30-50)', 
                       strategies=['MNP', 'OP', 'TAT', '6cbu38xr'], 
                       threshold_ms=100,
                       plot_type='box')
