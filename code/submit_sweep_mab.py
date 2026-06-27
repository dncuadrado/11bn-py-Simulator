import wandb
import os
import subprocess

project_name = "linucb_optimization"

# =============================================================
# SWEEP CONFIG (IMPROVED)
# =============================================================
sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'custom_running_mean_p99_delay',   
        'goal': 'minimize'
    },
    'parameters': {
        'mab_alpha': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 3
        },
        'mab_coeff': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 20.0
        },
        'mab_penalty_weight': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 1.0
        }
    }
}

# sweep_id = wandb.sweep(sweep_config, project=project_name)


# sweep_id = 'g0dgolwd'
# sweep_id = '90chibqi'
sweep_id = 'z175uc41'   # mab_coeff up to 20


# =============================================================
# SLURM JOB SUBMISSION
# =============================================================
def submit_slurm_job(run_id, config):

    os.makedirs(f"slurm_jobs/{sweep_id}", exist_ok=True)
    os.makedirs(f"slurm_logs/{sweep_id}", exist_ok=True)

    stdout_log = f"slurm_logs/{sweep_id}/output_{run_id}.log"
    stderr_log = f"slurm_logs/{sweep_id}/error_{run_id}.log"

    job_script = f"""#!/bin/bash
#SBATCH --job-name=linucb_{run_id}
#SBATCH --output={stdout_log}
#SBATCH --error={stderr_log}
#SBATCH -p high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load Miniconda3/4.9.2
eval "$(conda shell.bash hook)"
conda activate myenv

cd /home/dnunez/Papers/conference_ML_CSR/pythonCode/802.11bn-py-sim-ML

python3.10 -u /home/dnunez/Papers/conference_ML_CSR/pythonCode/802.11bn-py-sim-ML/code/mab.py\\
    --project_name {project_name} \\
    --run_id {run_id} \\
    --mab_alpha {config['mab_alpha']} \\
    --mab_coeff {config['mab_coeff']} \\
    --mab_penalty_weight {config['mab_penalty_weight']}
"""

    job_path = f"slurm_jobs/{sweep_id}/{run_id}.sh"
    with open(job_path, "w") as f:
        f.write(job_script)

    subprocess.run(["sbatch", job_path], check=True)


# =============================================================
# W&B AGENT
# =============================================================
def train_agent():
    run = wandb.init()
    config = run.config
    run_id = run.id

    print(f"Submitting job {run_id} with config: {dict(config)}")

    submit_slurm_job(run_id, config)

    run.finish()


# =============================================================
# LAUNCH SWEEP
# =============================================================
wandb.agent(
    sweep_id,
    function=train_agent,
    count=30,   # number of jobs
    project=project_name
)