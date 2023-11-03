#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=CSV_Output
#SBATCH --output=CSV_Output-%j.out

echo Starting code to make CSV File

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

python SaveToCSV.py 

echo ending slurm script to add noise

