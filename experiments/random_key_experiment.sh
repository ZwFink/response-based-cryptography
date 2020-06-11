#!/bin/bash

trials=50
outfile="random_keygen_fragmentation_2xTitan.txt"

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
    for((t=0; t<trials; t++)); do
        echo "" >> "$outfile"
        echo Hamming distance: "$d" >> "$outfile"
        echo "" >> "$outfile"
        ./sbench "$d" 0 2 8 >> "$outfile"
    done
done


