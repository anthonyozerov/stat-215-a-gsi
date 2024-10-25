#!/bin/bash

# EXAMPLE USAGE:
# sbatch job.sh configs/default.yaml

#SBATCH --job-name=lab4-autoencoder
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4

python run_autoencoder.py $1