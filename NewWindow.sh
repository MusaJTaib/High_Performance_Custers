#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=Window_SW2
#SBATCH --array=1-800
#SBATCH --output=Windows_Size-%A_%a.out

echo Starting code to make seperate size windows #This script runs the main python code as a Job Array 

runtime
id

echo start initialization the Environment

export PATH=/home/musa.taib/software/miniconda3/envs/Tensor/bin:$PATH
source activate Tensor

which python
conda env list

echo finished initializaing

echo $SLURM_ARRAY_TASK_ID 

echo starting python code

python FinalCode.py $SLURM_ARRAY_TASK_ID #I am actually using the SLURM ARRAY TASK ID as the window size in my python code so that Array 1 will have a window size of 1 and so on

echo Main Python Code Slurm file ended
