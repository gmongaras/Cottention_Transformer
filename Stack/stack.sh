#!/bin/bash

#SBATCH -J StackDown
#SBATCH -N 1
#SBATCH -t 120:00:00
#SBATCH -p highmem
#SBATCH --mem=1000G
#SBATCH --ntasks=1
#SBATCH -o job_output_%j.txt
#SBATCH -e job_errors_%j.txt

# Download Dataset
python stack.py
