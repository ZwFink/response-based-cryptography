#!/bin/bash
trials=5
outfile="fragment_4x_v100.txt"

echo "FRAGMENTATION TIME TRIALING" >> "$outfile"
echo "" >> "$outfile"

echo "g=2" >> "$outfile"
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 2 0 4 2 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 3 0 4 2 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 4 0 4 2 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 5 0 4 2 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 6 0 4 2 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 7 0 4 2 >> "$outfile"
done

echo "" >> "$outfile"
echo "g=4" >> "$outfile"
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 2 0 4 4 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 3 0 4 4 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 4 0 4 4 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 5 0 4 4 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 6 0 4 4 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 7 0 4 4 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 8 0 4 4 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 9 0 4 4 >> "$outfile"
done

echo "" >> "$outfile"
echo "g=8" >> "$outfile"
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 2 0 4 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 3 0 4 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 4 0 4 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 5 0 4 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 6 0 4 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 7 0 4 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 8 0 4 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 9 0 4 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 10 0 4 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 11 0 4 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 12 0 4 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 13 0 4 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 14 0 4 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 15 0 4 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 16 0 4 8 >> "$outfile"
done


