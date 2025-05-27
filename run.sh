#!/bin/bash
#SBATCH --job-name=train_fovea_peripheral  # Job name
#SBATCH --output=logs/%x_%j.out           # Output log file (%x = job name, %j = job ID)
#SBATCH --error=logs/%x_%j.err            # Error log file
#SBATCH --time=24:00:00                   # Maximum runtime (hh:mm:ss)
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH --gres=gpu:1                      # Number of GPUs (adjust if needed)
#SBATCH --mem=16G                         # Memory per node (adjust as needed)



# Run the Python script
python train_fovea_lightened.py --config config/dqn_separated_config.yml