import wandb
import os
import subprocess
import numpy as np


project_name = "cg_size=2"


# sweep_config = {
#     'method': 'grid',
#     'metric': {
#         'name': 'custom/ep_worst_percentile99',
#         'goal': 'minimize'
#     },
#     'parameters': {
#         'window_size': {
#             'values': np.random.randint(8, 21, size=10).tolist()
#         },
#     }
# }


# sweep_config = {
#     'method': 'random',
#     'metric': {
#         'name': 'custom/ep_worst_percentile99',
#         'goal': 'minimize'
#     },
#     'parameters': {
#         'w_throughput': {
#             'distribution': 'log_uniform_values',  # Log-uniform distribution
#             'min': 1.9e-8,
#             'max': 2e-8
#         },
#     }
# }

sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'custom/ep_worst_percentile99',
        'goal': 'minimize'
    },
    'parameters': {
        'w_shaping_coef': {
            'distribution': 'log_uniform_values',  # Log-uniform distribution
            'min': 2.5,
            'max': 5.0
        },
    }
}




# # Sweep configuration using only valid n_steps/batch_size pairs
# sweep_config = {
#     'method': 'grid',  # or 'random' for simpler setups
#     'metric': {
#         'name': 'custom/ep_worst_percentile99',
#         'goal': 'minimize'
#     },
#     'parameters': {
#         'n_steps': {
#             'values': [128, 256, 512, 1024, 2048]
#             },
#         'batch_size': {
#             'values': [32, 64, 128, 256, 512, 1024, 2048],
#             },
#     }
# }



# Create or reuse sweep
sweep_id = wandb.sweep(sweep_config, project=project_name)

# sweep_id = 'o7v7h0oh'

# --- SLURM job submission ---
def submit_slurm_job(run_id, config):
    os.makedirs(f"slurm_jobs/{sweep_id}", exist_ok=True)
    os.makedirs(f"slurm_logs/{sweep_id}", exist_ok=True)

    stdout_log = f"slurm_logs/{sweep_id}/output_{run_id}.log"
    stderr_log = f"slurm_logs/{sweep_id}/error_{run_id}.log"

    job_script = f"""#!/bin/bash
#SBATCH --job-name=sweep_{run_id}
#SBATCH --output={stdout_log}
#SBATCH --error={stderr_log}
#SBATCH -p high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=18G

module load Miniconda3/4.9.2
eval "$(conda shell.bash hook)"
conda activate myenv

cd /home/dnunez/Papers/conference_ML_CSR/pythonCode/802.11bn-py-sim-ML

# Run training script
python3.10 -u /home/dnunez/Papers/conference_ML_CSR/pythonCode/802.11bn-py-sim-ML/code/rl_agent.py \\
    --project_name {project_name} \\
    --run_id {run_id} \\
    --w_shaping_coef {config['w_shaping_coef']} \\
"""

    job_path = f"slurm_jobs/{sweep_id}/{run_id}.sh"
    with open(job_path, "w") as f:
        f.write(job_script)

    subprocess.run(["sbatch", job_path])


# --- W&B agent logic ---
def train_agent():
    run = wandb.init()
    config = run.config
    run_id = run.id

    # # Only run if valid n_steps % batch_size == 0
    # if config.n_steps % config.batch_size != 0:
    #     print(f"Invalid pair: n_steps={config.n_steps}, batch_size={config.batch_size} — Skipping")
    #     run.finish(exit_code=0)
    #     return

    submit_slurm_job(run_id, config)


# Launch agent to sample and dispatch jobs
wandb.agent(sweep_id, function=train_agent, count=14, project=project_name)