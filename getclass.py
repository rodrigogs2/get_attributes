#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 16:43:47 2019

@author: rodrigo
"""
import os, re, csv

def build_classes_dictionary(csv_file):
    alzheimer_dic = {'CN': 0, 'MCI': 1, 'AD': 2}
    dic = {}

    if os.path.exists(csv_file):
        try:
            with open(csv_file, 'r') as file:
                print('CSV File received: ', csv_file)
                reader = csv.reader(file)
                headers = next(reader) 
                for row in reader:
                    image_id = 'I' + row[3]
                    image_class = alzheimer_dic[row[5]]
                    dic[image_id] = image_class
        except os.error:
            print("*** ERROR: The csv file %s can not be readed (os.error in build_classes_dictionary)" % csv_file)    

    else:
        message = str("file %s does not exist!" % csv_file)
        raise ValueError(message)
    return dic

def get_class(attributes_file, all_classes_dictionary):
    image_id = re.findall(r'I[0-9]+',attributes_file)
    return all_classes_dictionary[image_id[0]]
        
def main(argv):
    print("Iniciando main...")
    
    attribs_file = '/home/rodrigo/Documents/_phd/attributes/ADNI_002_S_0413_MR_MPR____N3__Scaled_2_Br_20081001114937668_S14782_I118675.txt'
    csv_file = '/home/rodrigo/Documents/_phd/csv_files/ADNI1_Complete_All_Yr_3T.csv'
    
    d = build_classes_dictionary(csv_file)
    
    print(d['I107779'])
    
    image_class = get_class(attribs_file,d)
    print('Image class of I107779 file = ',image_class)
    
if __name__ == "__main__":    
    main(sys.argv)
