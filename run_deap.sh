#!/bin/sh
#source activate nibabel
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m KNN > saida_knn.txt&
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m LDA > saida_lda.txt&
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m CART > saida_cart.txt&
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m NB > saida_nb.txt&
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m SVM > saida_svm.txt&
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m RF > saida_rl.txt&
python deap_alzheimer.py -a ../../attributes2/ -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results -n 50 -m LR > saida_lr.txt&

#systemctl suspend


