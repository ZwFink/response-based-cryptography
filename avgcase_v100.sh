#!/bin/bash
trials=5
outfile="avgcase_v100.txt"

echo "" >> "$outfile"
echo "4 v100 GPUs Early Exit Time Trialing" >> "$outfile"

for ((i=0; i<trials; i++)); do
    ./sbench 5 0 1 >> "$outfile"
done
echo "" >> "$outfile"

for ((i=0; i<trials; i++)); do
    ./sbench 5 0 2 >> "$outfile"
done
echo "" >> "$outfile"

for ((i=0; i<trials; i++)); do
    ./sbench 5 0 4 >> "$outfile"
done
echo "" >> "$outfile"

