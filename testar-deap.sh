#!/bin/sh
#python deap_alzheimer.py -s ./seeds.txt -a ../../attributes_amostra/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../temp -n 3 -v -m KNN
python deap_alzheimer.py -s ./seeds.txt -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../temp -n 3 -m KNN -v -p
