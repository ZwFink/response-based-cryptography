#!/bin/bash

#SBATCH --job-name=test_frag  # the name of your job
#SBATCH --output=/scratch/jaw566/test_frag.out #
#SBATCH --error=/scratch/jaw566/test_frag.err #
#SBATCH --time=00:60:00        # 2 min, shorter time, quicker start
#SBATCH --mem=1000         #1 GiB memory requested
#SBATCH --gres=gpu:tesla:4 #resource requirement the :4 is 4 GPUs
#SBATCH --qos=gpu
#SBATCH --constraint=v100 #this is the volta node
#SBATCH --exclusive 
#SBATCH --reservation=jaw566_183

module load cuda/10.2
module load gcc/6.2.0

outfile="spread_analysis_fragmentation_4xVolta.txt"

#echo "------------" >> "$outfile"
#echo "2 fragments" >> "$outfile"
#echo "------------" >> "$outfile"
#echo "" >> "$outfile"
#for((d=2; d<7; d++)); do
#    for((t=0; t<100; t++)); do
#        echo Hamming distance: "$d" >> "$outfile"
#        echo "" >> "$outfile"
#        ./sbench "$d" 0 4 2 >> "$outfile"
#    done
#done
#echo "------------" >> "$outfile"
#echo "4 fragments" >> "$outfile"
#echo "------------" >> "$outfile"
#echo "" >> "$outfile"
#for((d=2; d<9; d++)); do
#    for((t=0; t<100; t++)); do
#        echo Hamming distance: "$d" >> "$outfile"
#        echo "" >> "$outfile"
#        ./sbench "$d" 0 4 4 >> "$outfile"
#    done
#done
echo "------------" >> "$outfile"
echo "8 fragments" >> "$outfile"
echo "------------" >> "$outfile"
echo "" >> "$outfile"
for((d=2; d<16; d++)); do
    for((t=0; t<50; t++)); do
        echo "" >> "$outfile"
        echo Hamming distance: "$d" >> "$outfile"
        echo "" >> "$outfile"
        ./sbench "$d" 1 4 8 >> "$outfile"
    done
done


# srun --gres=gpu:4 ./sbench 7 1 4 4
# srun --gres=gpu:4 ./analyze_spread_fragmentation.sh
