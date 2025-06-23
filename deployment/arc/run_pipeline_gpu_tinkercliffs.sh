#!/bin/bash

#SBATCH -J haven
#SBATCH --account=seqevol
#SBATCH --partition=a100_normal_q

#SBATCH --mem=450G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -t 144:00:00 # wall-time required (# 144hrs = 6 days)


# Load modules
module reset
module load
# Load conda


module load Miniconda3
#Load CUDA
module load CUDA/12.6.0

# Load conda environment
source activate haven
echo "Conda information:"
conda info

# Setup project and result directories
PROJECT_DIR=$1
LOGS_DIR=$PROJECT_DIR/output/logs
echo "Project directory: $PROJECT_DIR"

# Execute python script
SCRIPT_LOCATION=$PROJECT_DIR/src/run.py
CONFIG_FILE=$2
LOG_FILE=$LOGS_DIR/$(date +%Y_%b_%d_%H_%M_%s).log
echo "Config File: $CONFIG_FILE"
echo "Log File: $LOG_FILE"

conda activate haven
echo "Python version in $CONDA_PREFIX"
$CONDA_PREFIX/bin/python --version

echo "GPU check"
$CONDA_PREFIX/bin/python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}. Available GPU devices: {torch.cuda.device_count()}')"
echo "Pipeline START"
date
$CONDA_PREFIX/bin/python $SCRIPT_LOCATION -c $CONFIG_FILE > $LOG_FILE 2>&1
echo "Pipeline END"
date