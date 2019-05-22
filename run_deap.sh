#!/bin/sh
#source activate nibabel
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -v -n 50
systemctl suspend

