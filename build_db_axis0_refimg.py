#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 18:00:54 2020

@author: rodrigo
"""
import os, csv, sys, re
import numpy as np
import list_dir

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


#def get_ref_slice_attributes(attributes_file, csv_dic, target_bplanes=[0]):
#    attributes_as_str = []
#    
#    img_id_str = get_image_ID2(attributes_file)
#    
#    try:
#        img_id = int(img_id_str)
#    except ValueError:
#        print('This value ({0}) can not be converted to integer'.format(img_id_str))
#    
#    #csv_dic = build_csv_dictionary(csv_file)
#    ref_slices = csv_dic['ref_slices']
#    
#    for tbp in target_bplanes: # for each target body plane, do it....
#        if os.path.exists(attributes_file):
#            try:
#                with open(attributes_file, 'r') as file:
#                    reader = csv.reader(file)
#                    for row in reader:
#                        bplane = row[0]
#                        slicenum = row[1]
#                        
#                        try:
#                            tsn = ref_slices[tbp] # target slice_num will be the ref_slice from this bplane
#                        except ValueError:
#                            print('*** Invalid target bplane ({0}). rec_slices={1}'.format(tbp,ref_slices))
#                            
#                        
#                        if bplane == tbp and slicenum == tsn:
#                            attributes_as_str = row[2:]
#                        
#            except os.error:
#                print("*** ERROR: Attributes file %s can not be readed (os.error in load_attribs function)" % attributes_file)
#            except UnicodeDecodeError:
#                print('Error processing file ({0})'.format(attributes_file))    
#        
#    return attributes_as_str
    
    
#    attribs_as_floats = []
#
#    for str_attrib in attributes_as_str:
#        #print('String attribute: ', str_attrib)
#        if str_attrib != '':
#            try:
#                value = float(str_attrib)
#            except ValueError:
#                print('*** ERROR: Fail to convert an string attribute ("{0}") to float in load_attribs_and_metadata().\n Attributes File: {1}'.format(str_attrib, attributes_file))
#                sys.exit(-1)
#            
#            attribs_as_floats.append(value)
                    
        
    # NumPy transformations
#    attribs = np.array(attribs_as_floats_lists, dtype=np.float64)
#    body_plane = np.array(body_plane_list, dtype=np.int64)
#    slice_numbers = np.array(slicenum_list, dtype=np.int64)
#    slice_amount = np.array(slice_amount_per_plane, dtype=np.int64)
#    
#    return  attribs, body_plane, slice_numbers, slice_amount

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
    
    return image_id_list, slicenum_list
    

def load_ref_slices_attribs(attributes_file, ref_slices):
    #body_plane_list = []
    #slicenum_list = []
    ref_slices_attributes = []
        
    if os.path.exists(attributes_file):
        for target_bp in range(len(ref_slices)):
            target_refslice = ref_slices[target_bp]

            
            if target_refslice:
                #print('target_bp={0} and target_refslice={1}'.format(target_bp, target_refslice))
                try:
                    with open(attributes_file, 'r') as file:
                        reader = csv.reader(file)
                        for row in reader:
                            bplane = int(row[0])
                            slicenum = int(row[1])
                            #print('bplane={0}, slicenum={1}, target_bp={2}, target_refslice={3}'.format(bplane,slicenum,target_bp,target_refslice))
                            
                            if bplane == int(target_bp) and slicenum == int(target_refslice):
                                attributes = row[2:]
                                ref_slices_attributes.append(attributes)
                                #print('ref slice attributes found: ',attributes)


                except os.error:
                    print("*** ERROR: Attributes file %s can not be readed (os.error in load_attribs function)" % attributes_file)
                except UnicodeDecodeError:
                    print('Error processing file ({0})'.format(attributes_file))
            
                attribs_as_floats_lists = []
                #print("*** Processing attributes file %s)" % attributes_file)
                for attribs_as_string in ref_slices_attributes:
                    #print("Attribs as strings: ", attribs_as_string)
                    a = []
                    for str_attrib in attribs_as_string:
                        #print('String attribute: ', str_attrib)
                        if str_attrib != '':
                            try:
                                value = float(str_attrib)
                            except ValueError:
                                print('*** ERROR: Fail to convert an string attribute ("{0}") to float in load_attribs_and_metadata().\n Attributes File: {1}'.format(str_attrib, attributes_file))
                                sys.exit(-1)
                            
                            a.append(value)
                                
                    attribs_as_floats_lists.append(a)
        
        # Counting slices from each body axis
    #    plane0 = body_plane_list.count('0') - 1 #256 slices are indexed between 0 and 255
    #    plane1 = body_plane_list.count('1') - 1
    #    plane2 = body_plane_list.count('2') - 1
        
        #plane0 = len(body_plane_list) - body_plane_list[::-1].index('0')
        #plane1 = len(body_plane_list) - body_plane_list[::-1].index('1') - plane0
        #plane2 = len(body_plane_list) - body_plane_list[::-1].index('2') - plane0 - plane1
     #   slice_amount_per_plane = [plane0,plane1,plane2]
        
        # NumPy transformations
        attribs = np.array(attribs_as_floats_lists, dtype=np.float64)
    #    body_plane = np.array(body_plane_list, dtype=np.int64)
    #    slice_numbers = np.array(slicenum_list, dtype=np.int64)
    #    slice_amount = np.array(slice_amount_per_plane, dtype=np.int64)
    
    return  attribs


def get_image_ID(attributes_filename):
    all_image_id = re.findall(r'I[0-9]+',attributes_filename) # returns a array with all regular exp matches
    if len(all_image_id) > 0:
        return all_image_id[0]
    else:
        return ''

def load_all_data_using_filters(attributes_dir, input_csv, valid_genders=['M','F'], min_age=0.0, max_age=200.0, debug=False, black_list_id=[]):
    all_ref_attribs = []
    all_ref_slices = []
    #all_body_planes = []
    #all_slice_num = []
    #all_slice_amounts = []
    all_classes = []
    all_genders = []
    all_ages = []
    
    
    #image_id_dictionary  = build_cvs_dictionary(csv_file)
    
    
    dic = build_csv_dictionary2(input_csv)
    
    #img_demo_dic = get_image_demographic_data_by_id2('I120779',dic)
    # img_demo_dic=  {'class': 0, 'gender': 'M', 'age': 82, 'ref_slices': ['37', '', '']}
    
    # Getting all attributes files from attributes directory
    attribs_files = list_dir.list_files(attributes_dir,".txt")
    
    print('black_list_id=',black_list_id)
    
    # Loop which loads attributes, demographics values and slicing info
    for file in attribs_files:
        image_id = get_image_ID(file)
        try:
            black_list_id.index(image_id)
            print('attributes from image_id=\'{0}\' was not loaded because it is blacklisted!'.format(image_id))
            #continue # if image_id was found in blacklist, skip current iteration
        except ValueError:
            
            # image ID was not found in blacklist so we can extract ref_slices attributes from this volume...
            if debug: print('\t* extracting data from file:',file)
            demographics_data = dic[image_id]
            if debug: print('\t* this file demographics: ',demographics_data)
            gender = demographics_data['gender']
            if debug: print('\t* gender loaded: ',gender)
            age = demographics_data['age']
            if debug: print('\t* age loaded: ', age)
            
            age_is_valid = age >= min_age and age <= max_age
            if debug: print('\t* age_is_valid: ', age_is_valid)
            
            try:
                if valid_genders.index(gender) >= 0:
                    gender_is_valid = True
            except ValueError:
                gender_is_valid = False
                
            if debug: print('gender_is_valid: ',gender_is_valid)
            
            if gender_is_valid and age_is_valid:
                if debug: print('Great! Both gender and age are valid!')
                
                
                img_demo_dic = get_image_demographic_data_by_id2(image_id, dic)
                ref_slices = img_demo_dic['ref_slices']
                load_ref_slices_attribs(file, ref_slices)
                
                attribs = load_ref_slices_attribs(file, ref_slices)
                
                all_ref_attribs.append(attribs)
                all_ref_slices.append(ref_slices)
                
                #all_slice_num.append(slice_num)
                #all_slice_amounts.append(slice_amounts)  
            
                #demographics_data = get_image_demographic_data(file,image_id_dictionary)
                
                image_class = img_demo_dic['class']
    
                all_classes.append(image_class)
                all_genders.append(gender)
                all_ages.append(float(age))
        
    array_all_classes = np.array(all_classes, dtype=np.int64)
    array_all_ages = np.array(all_ages, dtype=np.int64)
    
    print('*** {0:04d} attributes files were analysed and {1:04d} atributes were extracted'.format(len(attribs_files),len(all_genders)))
    
    return all_ref_attribs, array_all_classes, all_genders, array_all_ages, img_demo_dic

    # getting partition from 80 to 100th slice from the f-th (f=0) attribs file
    #partition = get_attributes_from_a_range_of_slices(attribs,slice_amounts,p,fs,ls)
    #print('\n-Shape of Attributes Partition of the {0}th data file from the slices {2} and {3}: {1}'.format(f,partition.shape,fs,ls))
    
    #first = fs
    #for at in partition:
    #    print('\n-Attributes from {1}th slice of the {2}th attribs file:\n{0}'.format(at,first,f))
    #    first = first + 1
    
    #print('\nPartition:\n{0}'.format(partition))


# TEMPORARIA (para testes)
def find_attributes_file(target_image_id, csv_dic, attributes_dir='../../attributes2'):
    
    #dic = build_csv_dictionary2(input_csv)
    
    #img_demo_dic = get_image_demographic_data_by_id2('I120779',dic)
    # img_demo_dic=  {'class': 0, 'gender': 'M', 'age': 82, 'ref_slices': ['37', '', '']}
    
    # Getting all attributes files from attributes directory
    attribs_files = list_dir.list_files(attributes_dir,".txt")
    
    for file in attribs_files:
        image_id = get_image_ID(file)
        
        if image_id == target_image_id:
            return file
    
    

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
    img_name = 'I120779'
    img_demo_dic = get_image_demographic_data_by_id2(img_name,dic)
    print('img_demo_dic= ', img_demo_dic)
    ref_slices = img_demo_dic['ref_slices']
    print('ref_slices from image ',img_name,'= ',ref_slices)
    
    attribs_file = find_attributes_file(img_name,dic)
    
    print('target attributes file = ', attribs_file)
    
    ref_attributes = load_ref_slices_attribs(attribs_file, ref_slices)
    
    print('ref_attributes from ref_slices {0} of image {1} are =\n{2}'.format(img_demo_dic['ref_slices'],img_name,ref_attributes))
    
    
    
    
    
    
    
    

if __name__ == "__main__":    
    main(sys.argv)
