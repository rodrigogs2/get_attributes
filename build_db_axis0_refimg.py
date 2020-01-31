#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 18:00:54 2020

@author: rodrigo
"""
import os, csv, sys
import numpy as np

input_csv = "./com_OFFSET_ADNI1_Complete_All_Yr_3T.csv"
step_size = 2
max_row = 70

def get_reference_img_index(attributes_filename, csv_file, axis_num=0):
    


def load_IDs_and_ref_img_from_axis(csv_file, axis_num=0):
    image_id_column_index = 3
    slice_column_index = 19 + axis_num
    
    image_id_list = []
    slicenum_list = []
    #attributes_list = []
    print('Reading csv file: ' + csv_file)
    
    if os.path.exists(csv_file):
        print('File exists!')
        try:
            with open(csv_file, 'r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    index_ref_slice = row[slice_column_index]
                    
                    if index_ref_slice:
                        try:
                            int_index = int(index_ref_slice)
                            slicenum_list.append(int_index)
                        
                            image_id = row[image_id_column_index]
                            image_id_list.append(image_id)
                            
                        except ValueError:
                            print('* Value of variable int_index ({0}) can not yield a integer!'.format(int_index))
                        
                        
                    
                    
                    
        except os.error:
            print("*** ERROR: Attributes file %s can not be readed (os.error)" % csv_file)
        except UnicodeDecodeError:
            print('Error processing file ({0})'.format(csv_file))
    
#    attribs_as_floats_lists = []
#    #print("*** Processing attributes file %s)" % attributes_file)
#    for attribs_as_string in attributes_list:
#        #print("Attribs as strings: ", attribs_as_string)
#        a = []
#        for str_attrib in attribs_as_string:
#            #print('String attribute: ', str_attrib)
#            if str_attrib != '':
#                try:
#                    value = float(str_attrib)
#                except ValueError:
#                    print('*** ERROR: Fail to convert an string attribute ("{0}") to float in load_attribs_and_metadata().\n Attributes File: {1}'.format(str_attrib, attributes_file))
#                    sys.exit(-1)
#                
#                a.append(value)
#                    
#        attribs_as_floats_lists.append(a)
#    
#    # Counting slices from each body axis
#    plane0 = body_plane_list.count('0') - 1 #256 slices are indexed between 0 and 255
#    plane1 = body_plane_list.count('1') - 1
#    plane2 = body_plane_list.count('2') - 1
#    
#    #plane0 = len(body_plane_list) - body_plane_list[::-1].index('0')
#    #plane1 = len(body_plane_list) - body_plane_list[::-1].index('1') - plane0
#    #plane2 = len(body_plane_list) - body_plane_list[::-1].index('2') - plane0 - plane1
#    slice_amount_per_plane = [plane0,plane1,plane2]
#    
#    # NumPy transformations
    #attribs = np.array(attribs_as_floats_lists, dtype=np.float64)
    #image_ids = np.array(image_id_list, dtype=np.int64)
    #slice_numbers = np.array(slicenum_list, dtype=np.int64)
    #slice_amount = np.array(slice_amount_per_plane, dtype=np.int64)
    
    return  image_id_list, slicenum_list
    #return  attribs, slice_numbers, slice_amount
    

def main(argv):
    ## Testting area
    ids, slices = load_IDs_and_ref_img_from_axis(input_csv)

    print("Image ID \tIndex of axis reference slice")        
    for img_id, slice_num in zip(ids,slices):
        print(str(img_id) + "\t\t" + str(slice_num))
    

if __name__ == "__main__":    
    main(sys.argv)
