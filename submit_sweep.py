import wandb
import os
import subprocess

# Define your sweep configuration inline or load it from a YAML if preferred
sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'mean_reward',
        'goal': 'maximize'
    },
    'parameters': {
        'n_steps': {
            'min': 32,
            'max': 2048, 
        },
    }
}

# Define sweep
sweep_id = wandb.sweep(sweep_config, project="sb3-sweep")

def submit_slurm_job(run_id, config):

    # Make sure folders exist
    os.makedirs("slurm_jobs", exist_ok=True)
    os.makedirs("slurm_logs", exist_ok=True)

    # Build SLURM script
    job_script = f"""#!/bin/bash
    #SBATCH --job-name={run_id}
    #SBATCH -p high                    # short, medium, high, high-cpu
    #SBATCH --nodes=1                      # Request 1 node
    #SBATCH --ntasks=1                     # Single task
    #SBATCH --cpus-per-task=4              # CPUs per task
    #SBATCH --mem=20G                      # Memory allocation

    # Load Miniconda module
    module load Miniconda3/4.9.2

    # Conda base
    eval "$(conda shell.bash hook)"

    # Activate the conda environment
    conda activate myenv

    # Change to the required working directory
    cd /home/dnunez/Papers/journal_ML_CSR/pythonCode

    python3.10 -u /home/dnunez/Papers/journal_ML_CSR/pythonCode/11bn-py-Simulator/RLagent.py --run_id {run_id} --n_steps {config['n_steps']}
    """

    job_path = f"slurm_jobs/{run_id}.sh"
    with open(job_path, "w") as f:
        f.write(job_script)

    subprocess.run(["sbatch", job_path])

def train_agent():
    with wandb.init() as run:
        config = run.config
        run_id = run.id
        submit_slurm_job(run_id, config)
        print(f"Launching SLURM job: run_{run_id}")
        

# Launch the sweep agent
wandb.agent(sweep_id, function=train_agent, count=5)