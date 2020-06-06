#!/bin/bash
trials=5
outfile="fragment_3x_v100.txt"

echo "FRAGMENTATION TIME TRIALING" >> "$outfile"
echo "" >> "$outfile"

echo "g=2" >> "$outfile"
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 2 0 3 2 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 3 0 3 2 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 4 0 3 2 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 5 0 3 2 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 6 0 3 2 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 7 0 3 2 >> "$outfile"
done

echo "" >> "$outfile"
echo "g=4" >> "$outfile"
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 2 0 3 4 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 3 0 3 4 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 4 0 3 4 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 5 0 3 4 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 6 0 3 4 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 7 0 3 4 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 8 0 3 4 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 9 0 3 4 >> "$outfile"
done

echo "" >> "$outfile"
echo "g=8" >> "$outfile"
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 2 0 3 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 3 0 3 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 4 0 3 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 5 0 3 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 6 0 3 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 7 0 3 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 8 0 3 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 9 0 3 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 10 0 3 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 11 0 3 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 12 0 3 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 13 0 3 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 14 0 3 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 15 0 3 8 >> "$outfile"
done
echo "" >> "$outfile"
for ((i=0; i<trials; i++)); do
    ./sbench 16 0 3 8 >> "$outfile"
done


