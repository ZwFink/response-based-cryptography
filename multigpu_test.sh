#!/bin/bash
trials=5
outfile="time_trialing_multigpu.txt"

echo "" >> "$outfile"
echo "Single GPU Time Trials" >> "$outfile"

# 1 gpu
for ((i=0; i<trials; i++)); do
    ./sbench 5 0 1 >> "$outfile"
done

echo "" >> "$outfile"
echo "Multi GPU Time Trials" >> "$outfile"

# 2 gpus
for ((i=0; i<trials; i++)); do
    ./sbench 5 0 2 >> "$outfile"
done
