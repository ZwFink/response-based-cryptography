#!/bin/bash

trials=5
outfile1="fragmentation_g-1.txt"
outfile2="fragmentation_g-2.txt"
outfile4="fragmentation_g-4.txt"
outfile8="fragmentation_g-8.txt"

echo "------------" >> "$outfile1"
echo "1 fragment, 4 Voltas" >> "$outfile1"
echo "------------" >> "$outfile1"
echo "" >> "$outfile1"
for((d=1; d<6; d++)); do
    echo "" >> "$outfile1"
    echo Hamming distance: "$d" >> "$outfile1"
    for((t=0; t<trials; t++)); do
        ./sbench "$d" 0 2 1 >> "$outfile1"
    done
done
echo "------------" >> "$outfile2"
echo "2 fragments, 4 Voltas" >> "$outfile2"
echo "------------" >> "$outfile2"
echo "" >> "$outfile2"
for((d=1; d<8; d++)); do
    echo "" >> "$outfile2"
    echo Hamming distance: "$d" >> "$outfile2"
    for((t=0; t<trials; t++)); do
        ./sbench "$d" 0 2 2 >> "$outfile2"
    done
done
echo "------------" >> "$outfile4"
echo "4 fragments, 4 Voltas" >> "$outfile4"
echo "------------" >> "$outfile4"
echo "" >> "$outfil4"
for((d=1; d<10; d++)); do
    echo "" >> "$outfile4"
    echo Hamming distance: "$d" >> "$outfile4"
    for((t=0; t<trials; t++)); do
        ./sbench "$d" 0 2 4 >> "$outfile4"
    done
done
echo "------------" >> "$outfile8"
echo "8 fragments, 4 Voltas" >> "$outfile8"
echo "------------" >> "$outfile8"
echo "" >> "$outfile8"
for((d=1; d<17; d++)); do
    echo "" >> "$outfile8"
    echo Hamming distance: "$d" >> "$outfile8"
    for((t=0; t<trials; t++)); do
        ./sbench "$d" 0 2 8 >> "$outfile8"
    done
done

