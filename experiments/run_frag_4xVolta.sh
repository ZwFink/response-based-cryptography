#!/bin/bash

#SBATCH --job-name=gpu_info_v100  # the name of your job
#SBATCH --output=/scratch/jaw566/frag_info_v100.out #
#SBATCH --error=/scratch/jaw566/frag_info_v100.err #
#SBATCH --time=00:30:00        # 2 min, shorter time, quicker start
#SBATCH --mem=1000         #1 GiB memory requested
#SBATCH --gres=gpu:tesla:4 #resource requirement the :4 is 4 GPUs
#SBATCH --qos=gpu
#SBATCH --constraint=v100 #this is the volta node
#SBATCH --exclusive 
#SBATCH --reservation=jaw566_183

module load cuda/10.2
module load gcc/6.2.0

srun --gres=gpu:4 ./frag_experiment_4xVolta.sh
