#!/bin/sh
#source activate nibabel
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m KNN > ../results/saida_F_knn.txt&
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m LDA > ../results/saida_F_lda.txt&
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m CART > ../results/saida_F_cart.txt&
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m NB > ../results/saida_F_nb.txt&
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m SVM > ../results/saida_F_svm.txt&
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m RF > ../results/saida_F_rl.txt&
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m LR > ../results/saida_F_lr.txt&

#systemctl suspend


