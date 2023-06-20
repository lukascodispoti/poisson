#!/bin/bash

#SBATCH --ntasks=64
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=poisson
#SBATCH --output=poissonlog

mpirun -n 64 bin/main <file> <dataset>