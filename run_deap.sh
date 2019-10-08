#!/bin/sh
#source activate nibabel
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m KNN > ../results/saida_MF_knn.txt&
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m LDA > ../results/saida_MF_lda.txt&
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m CART > ../results/saida_MF_cart.txt&
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m NB > ../results/saida_MF_nb.txt&
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m SVM > ../results/saida_MF_svm.txt&
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m RF > ../results/saida_MF_rl.txt&
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m LR > ../results/saida_MF_lr.txt&

#systemctl suspend


