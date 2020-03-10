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
attribs_dir = '../../attributes_temp'

max_row = 70

# OK
def build_csv_dictionary2(csv_file):
    alzheimer_dic = {'CN': 0, 'MCI': 1, 'AD': 2}
    demographics_dictionary = {}

    if os.path.exists(csv_file):

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

    
def get_image_demographic_data_by_id2(img_id, demographics_dictionary):
    try:
        subject_class_gender_sex_refslices = demographics_dictionary[img_id] # pick up demographics for the first
    except ValueError:
        print('There aren\'t image with this ID ({0})'.format(img_id))
        
    return subject_class_gender_sex_refslices


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
    
    ref_slices_attributes = []
    attribs_as_floats_lists = []
        
    if os.path.exists(attributes_file):
        for target_bp in range(len(ref_slices)):
            target_refslice = ref_slices[target_bp]

            
            if target_refslice:
                
                try:
                    with open(attributes_file, 'r') as file:
                        reader = csv.reader(file)
                        for row in reader:
                            bplane = int(row[0])
                            slicenum = int(row[1])
                            
                            try:
                                int_target_bp = int(target_bp)
                                int_target_refslice = int(target_refslice)
                            except:
                                break;
                               
                            if bplane == int_target_bp and slicenum == int_target_refslice:
                                attributes = row[2:]
                                ref_slices_attributes.append(attributes)

                except os.error:
                    print("*** ERROR: Attributes file %s can not be readed (os.error in load_attribs function)" % attributes_file)
                except UnicodeDecodeError:
                    print('Error processing file ({0})'.format(attributes_file))
            
                

                for attribs_as_string in ref_slices_attributes:
                    a = []
                    for str_attrib in attribs_as_string:
                        if str_attrib != '':
                            try:
                                value = float(str_attrib)
                            except ValueError:
                                print('*** ERROR: Fail to convert an string attribute ("{0}") to float in load_attribs_and_metadata().\n Attributes File: {1}'.format(str_attrib, attributes_file))
                                sys.exit(-1)
                            
                            a.append(value)
                                
                    attribs_as_floats_lists.append(a)
        
        # NumPy transformations
        attribs = np.array(attribs_as_floats_lists, dtype=np.float64)
    
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

    
    dic = build_csv_dictionary2(input_csv)
    
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
                
               
                image_class = img_demo_dic['class']
    
                all_classes.append(image_class)
                all_genders.append(gender)
                all_ages.append(float(age))
        
    array_all_classes = np.array(all_classes, dtype=np.int64)
    array_all_ages = np.array(all_ages, dtype=np.int64)
    
    print('*** {0:04d} attributes files were analysed and {1:04d} atributes were extracted'.format(len(attribs_files),len(all_genders)))
    
    return all_ref_attribs, array_all_classes, all_genders, array_all_ages, img_demo_dic


# TEMPORARIA (para testes)
def find_attributes_file(target_image_id, csv_dic, attributes_dir='../../attributes2'):
    
    # Getting all attributes files from attributes directory
    attribs_files = list_dir.list_files(attributes_dir,".txt")
    
    for file in attribs_files:
        image_id = get_image_ID(file)
        
        if image_id == target_image_id:
            return file

def get_output_file_header():
    return 'image_id,ref_slice,gender,age,class,attribs...\n'


def append_ref_attribs_to_file(
        img_id,
        output_full_filename, 
        img_dic,
        ref_slice_num,
        ref_attribs, 
        mode="a",
        verbose=False,
        limit_precision=False):
    
    header = ''
    
    output_directory,output_filename = os.path.split(output_full_filename)
    
    print('appending ref attributes from ref_slice_num=',ref_slice_num)
    
    # Check if output file exists
    if not os.path.exists(output_full_filename):
        header = get_output_file_header()
        if not os.path.exists(output_directory):
            try:
                os.makedirs(output_directory)
                # write header of csv file here!!
            except os.error:
                print ('*** ERROR: Output directory (%s) can not be created\n' % output_directory)
                sys.exit(1)
            raise ValueError("*** ERROR: output file can not be found")
            exit(1)
  
    try :
        output_file = open(output_full_filename,mode)
        
        if header:
            output_file.write(header)

        output_file.write('{0},{1},{2},{3},{4}'.format(img_id, ref_slice_num, img_dic['gender'], img_dic['age'], img_dic['class'],ref_attribs))
        
        # writtings attribs one by one
        for attrib in ref_attribs:
            if attrib != '' or attrib != None or attrib != '\n': # or attrib != '\rf':
                if limit_precision:
                    output_file.write(",{0:.8f}".format(attrib))
                else:
                    output_file.write(',{0}'.format(attrib))
            
        output_file.write('\n')
        output_file.close()
    except os.error:
        output_file.close()
        print(" *** ERROR: file %s can not be written" % output_full_filename)
        exit(1)
    
    return 0

#def append_attributes_to_file(
#        all_ref_attributes_file, 
#        ref_attributes,
#        mode="a",
#        verbose=False,
#        limit_precision=True):
#    
#    # build filename structure
##    if not os.path.exists(all_ref_attributes_file):
##        raise ValueError("*** ERROR: input nii image filename not exist or can not be readed")
##        exit(1)
##    
#    #input_file_dir,input_filename = os.path.split(nii_img_full_filename)
#    
#    if output_directory == None:
#        # Use input file dir as output when output dir is None
#        output_directory = input_file_dir
#    elif not os.path.exists(output_directory):
#        # create the output dir whether it doesnt exist
#        try:
#            os.makedirs(output_directory)
#        except os.error:
#            print ('*** ERROR: Output directory (%s) can not be created\n' % output_directory)
#            sys.exit(1)
#    
#    # building txt file name
#    
#    try :
#        output_file = open(all_ref_attributes_file,mode)
#        # writting body axis and slice values
#        #output_file.write('%d,%d,' % (axis_num, slice_num))
#        output_file.write('{0:1d},{1:3d}'.format(image_id,img_dic[])
#        
#        # writtings attribs one by one
#        for attrib in attributes:
#            if attrib != "" or attrib != None or attrib != '\n':
#                if limit_precision:
#                    output_file.write(",{0:.8f}".format(attrib))
#                else:
#                    output_file.write(',{0}'.format(attrib))
#            
#        output_file.write('\n')
#        output_file.close()
#    except os.error:
#        output_file.close()
#        print(" *** ERROR: file %s can not be written" % txt_full_filename)
#        exit(1)
#    
#    return txt_full_filename

    
def save_refslices_attribs(input_csv, 
                           ref_axis, 
                           valid_genders=['M','F'], 
                           min_age=0.0, 
                           max_age=200.0, 
                           debug=False, 
                           black_list_id=[], 
                           o_file='../ref-slices_attributes.csv', 
                           attribs_dir='../../attributes2'):
    #all_ref_attribs = []
    
    dic = build_csv_dictionary2(input_csv)
    
    output_dir,filename = os.path.split(o_file)
    
    output_filename_without_extension, file_extension = os.path.splitext(filename)
    
    new_output_file = "{0}-axis{1}{2}".format(
            output_filename_without_extension, 
            ref_axis,
            file_extension)
    
    output_file = os.path.join(output_dir, new_output_file)
    
    
    
    # Getting all attributes files from attributes directory
    attribs_files = list_dir.list_files(attribs_dir,".txt")
    
    
    print('black_list_id=',black_list_id)
    
# Loop which loads attributes, demographics values and slicing info
    for file in attribs_files:
        image_id = get_image_ID(file)

        
        try:
            black_list_id.index(image_id)
            print('attributes from image_id=\'{0}\' was not loaded because it is blacklisted!'.format(image_id))
            #continue # if image_id was found in blacklist, skip current iteration
        except ValueError:
            
            img_demo_dic = get_image_demographic_data_by_id2(image_id, dic)
    
            ref_slices = img_demo_dic['ref_slices']
            print('ref_slices = ',ref_slices)
            
            int_ref_slice_num = -1
            try:
                int_ref_slice_num = int(ref_slices[ref_axis])
            
                #########
                #######ref_attributes = load_ref_slices_attribs(file, ref_slices)
                #########
                
                # image ID was not found in blacklist so we can extract ref_slices attributes from this volume...
                
                #if debug: print('\t* extracting ref_slices data from file:',file)
                demographics_data = dic[image_id]
                #if debug: print('\t* this file demographics: ',demographics_data)
                gender = demographics_data['gender']
                #if debug: print('\t* gender loaded: ',gender)
                age = demographics_data['age']
                #if debug: print('\t* age loaded: ', age)
                
                
                # Age verification
                age_is_valid = age >= min_age and age <= max_age
                #if debug: print('\t* age_is_valid: ', age_is_valid)
                
                # Gender verification
                try:
                    if valid_genders.index(gender) >= 0:
                        gender_is_valid = True
                except ValueError:
                    gender_is_valid = False
                    
                #if debug: print('gender_is_valid: ',gender_is_valid)
                if gender_is_valid and age_is_valid:
                    #if debug: print('Great! Both gender and age are valid!')
                    
                    img_dic = get_image_demographic_data_by_id2(image_id, dic)
                    ref_slices = img_dic['ref_slices']
                    #load_ref_slices_attribs(file, ref_slices)
                    
                    ref_attribs = load_ref_slices_attribs(file, ref_slices)[ref_axis]
                    
                    
                    append_ref_attribs_to_file(image_id, output_file, img_dic, int_ref_slice_num, ref_attribs, mode='a', verbose=False, limit_precision=False)
                    
                    
                    #all_ref_attribs.append(attribs)
                    #all_ref_slices.append(ref_slices)
                    
                   
                    #image_class = img_demo_dic['class']
        
                    #all_classes.append(image_class)
                    #all_genders.append(gender)
                    #all_ages.append(float(age))
            except:
                print('abnormal ref_slices[ref_axis]=',ref_slices[ref_axis])
        
#    array_all_classes = np.array(all_classes, dtype=np.int64)
#    array_all_ages = np.array(all_ages, dtype=np.int64)
#    
#    print('*** {0:04d} attributes files were analysed and {1:04d} atributes were extracted'.format(len(attribs_files),len(all_genders)))
    
    return 0
#all_ref_attribs, array_all_classes, all_genders, array_all_ages, img_demo_dic


def main(argv):
    ## Testting area
    
#    ids, slices = load_IDs_and_ref_img_from_axis(input_csv)
#
#    print("Image ID \tAxis Ref Slice\t Pit Slice")        
#    for img_id, slice_num in zip(ids,slices):
#        print(str(img_id) + "\t\t" + str(slice_num) + "\t\t" + str(slice_num))
#    
#    print('Min slice: ' + str(np.min(slices)) )
#    print('Max slice: ' + str(np.max(slices)) )
#    print('Average value: ' + str(np.average(slices)) )
#    print('Standard Deviation: ' + str(np.std(slices)) )
    
    dic = build_csv_dictionary2(input_csv)
    
    #outputfile='../axis0_refattributes.csv'
    current_ref_axis = 2
    
    
    
    save_refslices_attribs(input_csv, current_ref_axis, attribs_dir='../../attributes2')
    
    
    
    
    #img_name = 'I120779'
    #img_demo_dic = get_image_demographic_data_by_id2(img_name,dic)
#    print('img_demo_dic= ', img_demo_dic)
#    ref_slices = img_demo_dic['ref_slices']
#    print('ref_slices from image ',img_name,'= ',ref_slices)
#    
#    attribs_file = find_attributes_file(img_name,dic)
#    
#    print('target attributes file = ', attribs_file)
#    
#    ref_attributes = load_ref_slices_attribs(attribs_file, ref_slices)
#    
#    print('ref_attributes from ref_slices {0} of image {1} are =\n{2}'.format(img_demo_dic['ref_slices'],img_name,ref_attributes))
    


if __name__ == "__main__":    
    main(sys.argv)
