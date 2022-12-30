#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=gpu04,gpu06
#SBATCH --mem=10000
#SBATCH --mail-type=end
#SBATCH --mail-user=bahonya98@gmail.com

module load anaconda/3-5.0.1

source activate my_env

cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888