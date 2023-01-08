#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=10000
#SBATCH --exclude=gpu04,gpu06
#SBATCH --mail-type=end
#SBATCH --mail-user=bakytzhan.konratov@gmail.com

module load anaconda/3-5.0.1

source activate my_env

srun python -u $1