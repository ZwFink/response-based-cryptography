#!/bin/bash

#SBATCH --job-name=rbc_4x_v100  # the name of your job
#SBATCH --output=/scratch/jaw566/avgcase_4x_v100.out #
#SBATCH --error=/scratch/jaw566/avgcase_4x_v100.err #
#SBATCH --time=00:02:00        # 2 min, shorter time, quicker start
#SBATCH --mem=1000         #1 GiB memory requested
#SBATCH --gres=gpu:tesla:4 #resource requirement the :4 is 4 GPUs
#SBATCH --qos=gpu
#SBATCH --constraint=v100 #this is the volta node
#SBATCH --exclusive 
#SBATCH --reservation=jaw566_183

module load gcc/6.2.0
module load cuda/10.2

srun --gres=gpu:4 ./avgcase_v100.sh
