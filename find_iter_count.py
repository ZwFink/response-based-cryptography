#!/usr/bin/env python3
import subprocess
import os
cwd= os.getcwd()
outfile="iter_frac_experiment.txt"
trials=5
frac=0.00075
frac_list=[0.00005,0.00225,0.0225,0.225]
i=0
j=0
while( frac < 1.0 ):
    for i in range(0,trials):
        subprocess.call([str(cwd)+"/./sbench 5 0 2 "+str(frac)+" >> "+outfile],shell=True)
    frac += frac_list[j]
    i+=1
    if( i == 4 ):
        j+=1
        i=0

