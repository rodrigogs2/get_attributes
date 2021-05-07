#!/bin/sh
#source activate nibabel
python deap_alzheimer.py -a ../bet_attribs -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results_M -n 50 -m KNN > ../results/saida_M_knn.txt&
python deap_alzheimer.py -a ../bet_attribs -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results_M -n 50 -m LDA > ../results/saida_M_lda.txt&
python deap_alzheimer.py -a ../bet_attribs -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results_M -n 50 -m CART > ../results/saida_M_cart.txt&
python deap_alzheimer.py -a ../bet_attribs -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results_M -n 50 -m NB > ../results/saida_M_nb.txt&
python deap_alzheimer.py -a ../bet_attribs -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results_M -n 50 -m SVM > ../results/saida_M_svm.txt&
python deap_alzheimer.py -a ../bet_attribs -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results_M -n 50 -m RF > ../results/saida_M_rl.txt&
python deap_alzheimer.py -a ../bet_attribs -c ./ADNI1_Complete_All_Yr_3T.csv -o ../results_M -n 50 -m LR > ../results/saida_M_lr.txt&

#systemctl suspend


