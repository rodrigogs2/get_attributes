#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Updates:
# 
# 2019, Jan 17: a função de carregar os dados de todos os arquivos de atributos
# foi modificada para retornar vetores do tipo np.array. Modificação foi 
# testada e está funcionando corretamente

"""
Created on Wed Dec 12 11:23:05 2018

@author: rodrigo
"""

import os, csv, sys, re
import numpy as np
import list_dir


def build_txt_filename_from_3d_image(input_full_filename, output_directory=None):
    # splitting file and directory from input full filename
    input_file_dir,input_filename = os.path.split(input_full_filename)

    # removing extetion from input file
    input_filename_without_extension = os.path.splitext(input_filename)[0]
    
    # putting txt extension to output file
    output_filename = "%s.txt" % input_filename_without_extension
    
    txt_full_filename = os.path.join(output_directory, output_filename)
    
    # ENDED    
    return txt_full_filename


def load_attribs_and_metadata(attributes_file):
    body_plane_list = []
    slicenum_list = []
    attributes_list = []
        
    if os.path.exists(attributes_file):
        try:
            with open(attributes_file, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    attributes_list.append(row[2:])
                    body_plane_list.append(row[0])
                    slicenum_list.append(row[1])
                    
        except os.error:
            print("*** ERROR: Attributes file %s can not be readed (os.error in load_attribs function)" % attributes_file)
        except UnicodeDecodeError:
            print('Error processing file ({0})'.format(attributes_file))
    
    attribs_as_floats_lists = []
    #print("*** Processing attributes file %s)" % attributes_file)
    for attribs_as_string in attributes_list:
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
    plane0 = body_plane_list.count('0') - 1 #256 slices are indexed between 0 and 255
    plane1 = body_plane_list.count('1') - 1
    plane2 = body_plane_list.count('2') - 1
    
    #plane0 = len(body_plane_list) - body_plane_list[::-1].index('0')
    #plane1 = len(body_plane_list) - body_plane_list[::-1].index('1') - plane0
    #plane2 = len(body_plane_list) - body_plane_list[::-1].index('2') - plane0 - plane1
    slice_amount_per_plane = [plane0,plane1,plane2]
    
    # NumPy transformations
    attribs = np.array(attribs_as_floats_lists, dtype=np.float64)
    body_plane = np.array(body_plane_list, dtype=np.int64)
    slice_numbers = np.array(slicenum_list, dtype=np.int64)
    slice_amount = np.array(slice_amount_per_plane, dtype=np.int64)
    
    return  attribs, body_plane, slice_numbers, slice_amount


def build_cvs_dictionary(csv_file):
    alzheimer_dic = {'CN': 0, 'MCI': 1, 'AD': 2}
    demographics_dictionary = {}

    if os.path.exists(csv_file):
        #import deap_alzheimer as da
        #genders_dic = da.build_gender_to_num_dic()
        try:
            with open(csv_file, 'r') as file:
                #print('CSV File received: ', csv_file)
                reader = csv.reader(file)
                headers = next(reader) 
                for row in reader:
                    image_id = 'I' + row[3]
                    gender = row[6]
                    age = row[7]
                    
                    try:
                        age = int(age)
                    except ValueError:
                        print('* Invalid AGE({0}) entry for image ID {1}. CSV file has problems'.format(age, image_id))
                        
                    
                    image_class = alzheimer_dic[row[5]]
                    dic = {'class':image_class, 'gender':gender, 'age':age}
                    demographics_dictionary[image_id] = dic
        except os.error:
            print("*** ERROR: The csv file %s can not be readed (os.error in build_classes_dictionary)" % csv_file)    

    else:
        message = str("file %s does not exist!" % csv_file)
        raise ValueError(message)
    return demographics_dictionary


def get_image_ID(attributes_filename):
    all_image_id = re.findall(r'I[0-9]+',attributes_filename) # returns a array with all regular exp matches
    if len(all_image_id) > 0:
        return all_image_id[0]
    else:
        return ''
   

def get_image_demographic_data(attributes_filename, demographics_dictionary):
    #image_id = re.findall(r'I[0-9]+',attributes_filename) # returns a array with all regular exp matches
    image_id = get_image_ID(attributes_filename)
    
    if image_id != '':
        subject_class_gender_sex = demographics_dictionary[image_id] # pick up demographics for the first
    else:
        raise ValueError('There aren\'t image IDs in this attributes filename ({0})'.format(attributes_filename))
#    if len(image_id) > 0: # if there is at least one match...
#        subject_class_gender_sex = demographics_dictionary[image_id[0]] # pick up demographics for the first
#    else:
#        raise ValueError('There aren\'t image IDs in this attributes filename ({0})'.format(attributes_filename))

    return subject_class_gender_sex

                    
def load_all_data(attributes_dir, csv_file):
    all_attribs = []
    all_body_planes = []
    all_slice_num = []
    all_slice_amounts = []
    all_classes = []
    all_genders = []
    all_ages = []
    
    image_id_dictionary  = build_cvs_dictionary(csv_file)
    
    # Getting all attributes files from attributes directory
    attribs_files = list_dir.list_files(attributes_dir,".txt")
    
    # Loop which loads attributes, demographics values and slicing info
    for file in attribs_files:
        attribs,body_plane,slice_num,slice_amounts = load_attribs_and_metadata(file)
        all_attribs.append(attribs)
        all_body_planes.append(body_plane)
        all_slice_num.append(slice_num)
        all_slice_amounts.append(slice_amounts)  
        
        demographics_data = get_image_demographic_data(file,image_id_dictionary)
        
        image_class = demographics_data['class']
        gender = demographics_data['gender']
        age = demographics_data['age']
        all_classes.append(image_class)
        all_genders.append(gender)
        all_ages.append(age)
        
    array_all_classes = np.array(all_classes, dtype=np.int64)
    array_all_ages = np.array(all_ages, dtype=np.int64)
    
    return all_attribs, all_body_planes, all_slice_num, all_slice_amounts, array_all_classes, all_genders, array_all_ages, image_id_dictionary


def is_black_listed(image_ID, black_list_id=[]):
    black_listed = False
    if len(black_list_id) > 0:
        try:
            position = black_list_id.index(image_ID)
            if position >= 0:
                black_listed = True
        except ValueError:
            black_listed = False
    return black_listed
        

    
def load_all_data_using_filters(attributes_dir, csv_file, valid_genders=['M','F'], min_age=0.0, max_age=200.0, debug=False, black_list_id=[]):
    all_attribs = []
    all_body_planes = []
    all_slice_num = []
    all_slice_amounts = []
    all_classes = []
    all_genders = []
    all_ages = []
    
    image_id_dictionary  = build_cvs_dictionary(csv_file)
    
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
            
            # image ID was not found in blacklist so we can extract attributes from this volume
            if debug: print('\t* extracting data from file:',file)
            demographics_data = get_image_demographic_data(file,image_id_dictionary)
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
                attribs,body_plane,slice_num,slice_amounts = load_attribs_and_metadata(file)
                
                all_attribs.append(attribs)
                all_body_planes.append(body_plane)
                all_slice_num.append(slice_num)
                all_slice_amounts.append(slice_amounts)  
            
                demographics_data = get_image_demographic_data(file,image_id_dictionary)
                
                image_class = demographics_data['class']
    
                all_classes.append(image_class)
                all_genders.append(gender)
                all_ages.append(float(age))
        
    array_all_classes = np.array(all_classes, dtype=np.int64)
    array_all_ages = np.array(all_ages, dtype=np.int64)
    
    print('*** {0:04d} attributes files were analysed and {1:04d} atributes were extracted'.format(len(attribs_files),len(all_genders)))
    
    return all_attribs, all_body_planes, all_slice_num, all_slice_amounts, array_all_classes, all_genders, array_all_ages, image_id_dictionary

    # getting partition from 80 to 100th slice from the f-th (f=0) attribs file
    #partition = get_attributes_from_a_range_of_slices(attribs,slice_amounts,p,fs,ls)
    #print('\n-Shape of Attributes Partition of the {0}th data file from the slices {2} and {3}: {1}'.format(f,partition.shape,fs,ls))
    
    #first = fs
    #for at in partition:
    #    print('\n-Attributes from {1}th slice of the {2}th attribs file:\n{0}'.format(at,first,f))
    #    first = first + 1
    
    #print('\nPartition:\n{0}'.format(partition))


def load_reshaped_attribs_and_metadata(attributes_file):
    attribs,body_plane,slice_numbers,slice_amount = load_attribs_and_metadata(attributes_file)
    reshaped = np.reshape(attribs,-1)
    return reshaped,slice_amount

def get_attributes_from_a_slice(attribs_as_lines, 
                         slice_limits,
                         specific_body_plane, 
                         specific_slice_num):

    #specific_slice_num # value between 0 and 255 typically
    
    valid_bplanes = list(range(len(slice_limits)))
    base = 0
    slice_line = specific_slice_num
    '''
    first_lines_of_each_body_plane = []
    for plane in valid_bplanes:
        first_lines_of_each_body_plane = [0, slice_limits[0]+1, slice_limits[1]+1]
'''
    if specific_body_plane in valid_bplanes:

        if specific_body_plane != 0:
            base = slice_limits[0] + 1 # 255 + 1
            
            if specific_body_plane != 1:
                base = base + slice_limits[1] + 1
            
        
        try:
            attrib = attribs_as_lines[slice_line + base]
            return attrib
        except IndexError:
            print('IndexError at get_attributes_from_a_slice function. Used line={0}'.format(slice_line + base))
            print('slice_limits={0} specific_body_plane={1} base={1}'.format(slice_limits,specific_body_plane,base))
            
    else:
        raise ValueError('specific_body_plane value is invalid!')

    return 0
    
    '''
    if specific_body_plane == 0 and :
        if  plane0_start <= initial_slice_num < end_slice_num and end_slice_num < plane0_end:
                print('It is valid!')
        start_index = initial_slice_num
        end_index = end_slice_num
    elif specific_body_plane == 1:
        start_index = initial_slice_num
        end_index = end_slice_num
    elif specific_body_plane == 2:
        start_index = initial_slice_num
        end_index = end_slice_num
    else:
        raise(ValueError('*** ERROR: Invalid body plane ("{0}") in getAttribsPartitionFromSingleSlicesGrouping().\n'.format(specific_body_plane)))
        sys.exit(-1)
    '''

def get_attributes_from_a_range_of_slices(image_attribs,
                                        slice_amounts,
                                        specific_body_plane,
                                        start_slice,
                                        total_slices):
    attributes_list = []
    for s in range(start_slice, start_slice + total_slices):
        attributes_list.append(get_attributes_from_a_slice(image_attribs,
                                                           slice_amounts,
                                                           specific_body_plane,
                                                           s))
    
    return np.array(attributes_list, dtype=np.float64)


def getSliceLimits(all_slice_amounts):
    size = len(all_slice_amounts[0]) # catching first slice amounts
    max_values = [-1] * size
    min_values = [256] * size
    
    for slice_amount in all_slice_amounts:
        for i in range(size):
            #limit = slice_amount[i] - 1
            value = slice_amount[i]
            if value > max_values[i]: 
                max_values[i] = value
                #print('max slice value {0} was found!'.format(value))
            if value < min_values[i]: 
                min_values[i] = value
                #print('min slice value {0} was found!'.format(value))
    return min_values, max_values


def getBplanes(all_slice_amounts):
    return list(range(len(all_slice_amounts[0])))


def getAttribsPartitionFromSingleSlicesGroupingUsingFilters(all_attribs,
                                                all_slice_amounts,
                                                #all_genders,
                                                #all_ages,
                                                #demographic_dic,
                                                specific_body_plane, 
                                                initial_slice_num, 
                                                total_slices,
                                                valid_genders=['M','F'],
                                                max_age=200,
                                                min_age=0):
    attribs_partition = []
    for attribs,s_amount in zip(all_attribs,all_slice_amounts):
        attribs_partition.append(get_attributes_from_a_range_of_slices(attribs,
                                                                       s_amount,
                                                                       specific_body_plane,
                                                                       initial_slice_num,
                                                                       total_slices))
    
    
    return np.array(attribs_partition, dtype=np.float64)


def getAttribsPartitionFromSingleSlicesGrouping(all_attribs,
                                                all_slice_amounts,
                                                specific_body_plane, 
                                                initial_slice_num, 
                                                total_slices):
    attribs_partition = []
    for attribs,s_amount in zip(all_attribs,all_slice_amounts):
        attribs_partition.append(get_attributes_from_a_range_of_slices(attribs,
                                                                       s_amount,
                                                                       specific_body_plane,
                                                                       initial_slice_num,
                                                                       total_slices))
    
    
    return np.array(attribs_partition, dtype=np.float64)


def getAttribsPartitionFromMultipleSlicesGroupings(all_attribs, 
                         all_slice_amounts,
                         slices_groupings):
    
    if len(slices_groupings) % 3 == 0:
        
        # Testing received arguments
        print('* Testing received arguments...')
        for i in range (len(slices_groupings),3):
            bplane, first_slice, total_slices = slices_groupings[i],slices_groupings[i+1],slices_groupings[i+2]
            print('{0}th grouping is:bplane={1},{2},{3}',i,bplane,first_slice,total_slices)
    return []


def display_help(script_name=None):
    if script_name == None:
        script_name = os.path.split(sys.argv[0])[1]

    print ('Usage:\n    ', script_name, '[Options] -i <inputfile> -o <outputdir> ')
    print ('  Options:')
    print('\t-m, --multicpu\tset on computation over all cores (default: multicore is off)')
    print('\t-v, --verbose\tenables verbose mode (default: disabled)')
    print('\t-r, --resume\tresume extraction: output files are not overwritten (default: resume is off)')

        
def main(argv):
    ## Testting area
    
    # Use this arguments to set the input directory of attributes files
    #attributes_dir = "../../attributes_amostra/"
    attributes_dir = "../../attributes2/"
    csv_file = './ADNI1_Complete_All_Yr_3T.csv' 
    # Getting all files
    
    print('* Loading all attributes data...')
    all_attribs, body_planes, slice_num, slice_amounts, output_classes, all_genders, all_ages, demographics_dic = load_all_data_using_filters(attributes_dir, csv_file, black_list_id=['I288905'])
    print('* ...done')
    
    #print('all_attribs=',all_attribs)
    print('len(all_attribs)=',len(all_attribs))
    print('body_planes=',len(body_planes))
    print('len(slice_num)=',len(slice_num))
    print('len(output_classes)=',len(output_classes))
    print('len(all_genders)=',len(all_genders))
    print('len(all_ages)',len(all_ages))
    print('all_ages array:',np.array(all_ages))
    print('all_genders array: ', np.array(all_genders))
#    
#    print('all_attribs[0].shape=', all_attribs[0].shape)
#    print('body_planes[0].shape=', body_planes[0].shape)
#    print('slice_num[0]=', slice_num[0])
#    print('output_classes[0]=', output_classes[0])
#    print('all_genders[0]=', all_genders[0])
#    print('all_ages[0]=', all_ages[0])
    #print('demographics_dic=',demographics_dic)
    
    bplane = 2
    start_slice = 123
    total_slices = 5
    
    
    data_partition = getAttribsPartitionFromSingleSlicesGrouping(
            all_attribs,
            slice_amounts,
            bplane, 
            start_slice, 
            total_slices)
    
#    print('\n* Testing function to extract a range of slices from a attributes row...')
#    croped_attribs = get_attributes_from_a_range_of_slices(all_attribs[0], 
#                                                           slice_amounts[0], 
#                                                           bplane, 
#                                                           start_slice, 
#                                                           total_slices)
#    
#    print('shape of croped_attribs: ',croped_attribs.shape)
#    print('cropped attribs:\n',croped_attribs)
#    
#    import deap_alzheimer as da
#    possibles_bplanes = getBplanes(slice_amounts)
#    min_slice_limits, max_slice_limits = getSliceLimits(slice_amounts)
#    print('min_slice_limits',min_slice_limits)
#    print('max_slice_limits',max_slice_limits)
#    
#    sg1 = da.buildRandomSliceGrouping(possibles_bplanes,length=20,max_indexes=min_slice_limits,dbug=False)
#    sg2 = da.buildRandomSliceGrouping(possibles_bplanes,length=20,max_indexes=min_slice_limits,dbug=False)
#    all_slices_groupings = list(sg1+sg2)
#    
#    print('sg1 + sg2 concatenation= ',all_slices_groupings)

    
    '''
    attribs_files = list_dir.list_files(attributes_dir,".txt")
    
    # Checking how many files were found
    print("\n* %s attribs files were loaded." % len(attribs_files))

    # Used to check memory usage to load all attributes from all files
    total_memory_usage = 0
    
    # Use these to control how many files will be loaded
    first_file = 0
    total_printed_files = len(attribs_files)
    total_printed_files = 1    
    
    # Loop which loads attributes, checks their shape and gets memory usage
    print("* Loaded Files:")
    for n in range(first_file, total_printed_files):
        file = attribs_files[n]
        memory_size = 0
        attribs,body_plane,slice_num,slice_amounts = load_attribs_and_metadata(file)
        memory_size = attribs.size * attribs.itemsize + body_plane.size * body_plane.itemsize + slice_num.size * slice_num.itemsize
        total_memory_usage += memory_size
        
        print('\t-Attributes from the {0}th data file :\n{1}'.format(n,attribs))
        print('\t-Body planes from the {0}th data file :\n{1}'.format(n,body_plane))
        print('\t-Slice numbers from the {0}th data file :\n{1}'.format(n,slice_num))
        print('\t-Dimensions of each vector: attribs({0}), body_plane({1}) and slices_num({2})'.format(attribs.ndim, body_plane.ndim, slice_num.ndim))
        print('\t-Amount of slices per each body plane of the {0}th data file: {1}'.format(n,slice_amounts))
        print('\t-Memory size usage to load the {0}th data file: {1} bytes'.format(n,memory_size))
        print('\t-Shape of Attributes from the {0}th data file :\n{1}'.format(n,attribs.shape))
    
    print('\t-Total memory usage to load all the {0} data files is:\n\t\t{1} bytes'.format(total_printed_files, total_memory_usage))
    
    f = 0
    attribs,body_plane,slice_num,slice_amounts = load_attribs_and_metadata(attribs_files[f])
    p = 0  # plane of human body (can be 0, 1 or 2)
    fs = 80 # first slice of interval
    ls = 100 # last slice of interval 
    
    # getting partition from 80 to 100th slice from the f-th (f=0) attribs file
    partition = get_attributes_from_a_range_of_slices(attribs,slice_amounts,p,fs,ls)
    print('\n-Shape of Attributes Partition of the {0}th data file from the slices {2} and {3}: {1}'.format(f,partition.shape,fs,ls))
    
    first = fs
    for at in partition:
        print('\n-Attributes from {1}th slice of the {2}th attribs file:\n{0}'.format(at,first,f))
        first = first + 1
    
    print('\nPartition:\n{0}'.format(partition))
    '''
    
if __name__ == "__main__":    
    main(sys.argv)
