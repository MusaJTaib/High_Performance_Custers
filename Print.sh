#!/bin/bash

# This script just sends me an email once all the codes have finished running


#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=CSV_Output
#SBATCH --output=CSV_Output-%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=musa.taib@ucalgary.ca

echo Code Completed 
