#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 18:00:54 2020

@author: rodrigo
"""
import os, csv, sys, re
import numpy as np

__USE_PIT_IMAGES = True
__STEP_TO_PIT = 2
__AXIS = 0

input_csv = "./com_OFFSET_ADNI1_Complete_All_Yr_3T.csv"

#max_row = 70

# OK
def build_csv_dictionary2(csv_file):
    alzheimer_dic = {'CN': 0, 'MCI': 1, 'AD': 2}
    demographics_dictionary = {}

    if os.path.exists(csv_file):
        #import deap_alzheimer as da
        #genders_dic = da.build_gender_to_num_dic()
        try:
            with open(csv_file, 'r') as file:
                #print('CSV File received: ', csv_file)
                reader = csv.reader(file)
                next(reader) 
                for row in reader:
                    image_id = 'I' + row[3]
                    gender = row[6]
                    age = row[7]
                    ref_slices = [row[19],row[20],row[21]]
                    
                    try:
                        age = int(age)
                    except ValueError:
                        print('* Invalid AGE({0}) entry for image ID {1}. CSV file has problems'.format(age, image_id))
                        
                    
                    image_class = alzheimer_dic[row[5]]
                    dic = {'class':image_class, 'gender':gender, 'age':age, 'ref_slices':ref_slices}
                    demographics_dictionary[image_id] = dic
        except os.error:
            print("*** ERROR: The csv file %s can not be readed (os.error in build_classes_dictionary)" % csv_file)    

    else:
        message = str("file %s does not exist!" % csv_file)
        raise ValueError(message)
    return demographics_dictionary

# OK
def get_image_ID2(attributes_filename):
    all_image_id = re.findall(r'I[0-9]+',attributes_filename) # returns a array with all regular exp matches
    if len(all_image_id) > 0:
        return all_image_id[0]
    else:
        return ''
    
def get_image_demographic_data_by_id2(img_id, demographics_dictionary):
    try:
        subject_class_gender_sex_refslices = demographics_dictionary[img_id] # pick up demographics for the first
    except ValueError:
        print('There aren\'t image with this ID ({0})'.format(img_id))
        
    return subject_class_gender_sex_refslices


def get_ref_slice_attributes(attributes_file, csv_dic, target_bplanes=[0]):
    attributes_as_str = []
    
    img_id_str = get_image_ID2(attributes_file)
    
    try:
        img_id = int(img_id_str)
    except ValueError:
        print('This value ({0}) can not be converted to integer'.format(img_id_str))
    
    #csv_dic = build_csv_dictionary(csv_file)
    ref_slices = csv_dic['ref_slices']
    
    for tbp in target_bplanes: # for each target body plane, do it....
        if os.path.exists(attributes_file):
            try:
                with open(attributes_file, 'r') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        bplane = row[0]
                        slicenum = row[1]
                        
                        try:
                            tsn = ref_slices[tbp] # target slice_num will be the ref_slice from this bplane
                        except ValueError:
                            print('*** Invalid target bplane ({0}). rec_slices={1}'.format(tbp,ref_slices))
                            
                        
                        if bplane == tbp and slicenum == tsn:
                            attributes_as_str = row[2:]
                        
            except os.error:
                print("*** ERROR: Attributes file %s can not be readed (os.error in load_attribs function)" % attributes_file)
            except UnicodeDecodeError:
                print('Error processing file ({0})'.format(attributes_file))    
        
    return attributes_as_str
    
    
    attribs_as_floats = []

    for str_attrib in attributes_as_str:
        #print('String attribute: ', str_attrib)
        if str_attrib != '':
            try:
                value = float(str_attrib)
            except ValueError:
                print('*** ERROR: Fail to convert an string attribute ("{0}") to float in load_attribs_and_metadata().\n Attributes File: {1}'.format(str_attrib, attributes_file))
                sys.exit(-1)
            
            attribs_as_floats.append(value)
                    
        
    # NumPy transformations
    attribs = np.array(attribs_as_floats_lists, dtype=np.float64)
    body_plane = np.array(body_plane_list, dtype=np.int64)
    slice_numbers = np.array(slicenum_list, dtype=np.int64)
    slice_amount = np.array(slice_amount_per_plane, dtype=np.int64)
    
    return  attribs, body_plane, slice_numbers, slice_amount

def get_normalized_ref_slices_dic():
    return 0



def load_IDs_and_ref_img_from_axis(csv_file, axis_num=0):
    image_id_column_index = 3
    slice_column_index = 19 + axis_num
    
    image_id_list = []
    slicenum_list = []
    #attributes_list = []
    print('Reading csv file: ' + csv_file)
    
    if os.path.exists(csv_file):
        try:
            with open(csv_file, 'r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    index_ref_slice = row[slice_column_index]
                    
                    if index_ref_slice:
                        try:
                            image_id = row[image_id_column_index]
                            int_index = int(index_ref_slice)

                            slicenum_list.append(int_index)
                            image_id_list.append(image_id)
                            
                        except ValueError:
                            print('* Value of variable int_index ({0}) can not yield a integer (ID={1})!'.format(int_index, image_id))
                        
                        
                    
                    
                    
        except os.error:
            print("*** ERROR: Attributes file %s can not be readed (os.error)" % csv_file)
        except UnicodeDecodeError:
            print('Error processing file ({0})'.format(csv_file))
    


def main(argv):
    ## Testting area
    ids, slices = load_IDs_and_ref_img_from_axis(input_csv)

    print("Image ID \tAxis Ref Slice\t Pit Slice")        
    for img_id, slice_num in zip(ids,slices):
        print(str(img_id) + "\t\t" + str(slice_num) + "\t\t" + str(slice_num))
    
    print('Min slice: ' + str(np.min(slices)) )
    print('Max slice: ' + str(np.max(slices)) )
    print('Average value: ' + str(np.average(slices)) )
    print('Standard Deviation: ' + str(np.std(slices)) )
    
    dic = build_csv_dictionary2(input_csv)
    
    
    #print(dic)
    
    img_demo_dic = get_image_demographic_data_by_id2('I120779',dic)
    print('ref_slices=',img_demo_dic)
    
    
    
    

if __name__ == "__main__":    
    main(sys.argv)
