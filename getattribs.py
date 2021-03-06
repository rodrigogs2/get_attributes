#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Updates:
# 2019, Jan 17: função de escrever os atributos em cada linha do arquivo 
# (append_attributes_to_file)foi modificada para evitar que cada linha termine 
# com uma virgula sem nenhum dado após ela. 
# obs.: ainda não testado

"""
Created on Sat Nov  3 18:53:17 2018

@author: rodrigo
"""

import os
import getopt
import numpy as np
import csv

import nibabel as nb
import sys
import mahotas as mh

import imageio as iio
import matplotlib.pyplot as plt

import multiprocessing
from multiprocessing import Pool



def check_slice_is_valid(img_data,slicenum,axisnum):
    isValid = False
    if (axisnum >= 0
        and axisnum <= 2 
        and slicenum >= 0 
        and slicenum < img_data.shape[axisnum]):
            isValid = True
    
    return isValid
    

def get_slice_data(img_data, slicenum=0, body_axis=0):
    img_2d_data = None
    
    if not check_slice_is_valid(img_data, slicenum, body_axis):
        # raise exception
        raise ValueError('*** ERROR: invalid image, slice and axis combination')
    else:
        #axis_size = img_data.shape[body_axis]
        #print('axis size: ',axis_size,'\t')
        if body_axis == 0:
            img_2d_data = img_data[slicenum, :, :]
            #print ('returned 2d img data shape: ', img_2d_data.shape)
        elif body_axis == 1:
            img_2d_data = img_data[:, slicenum, :]
            #print ('returned 2d img data shape: ', img_2d_data.shape)
        elif body_axis == 2:
            img_2d_data = img_data[:, :, slicenum]
            #print ('returned 2d img data shape: ', img_2d_data.shape)
    return img_2d_data


def save_image_2d_data_as_png_file(img_data, filename, output):
       
    # Normalization
    max_value = img_data.max()
    img_data = img_data / max_value
    img_data = img_data * 255;
    img_data_uint8 = img_data.astype(np.uint8)
    
    try:
        # Try to save image
        import imageio as iio
        iio.imwrite(filename,img_data_uint8)
    except os.error:
        raise ValueError("Fail to write image 2d data to png file (%s)",filename)
        

def build_png_filename_from_slice(input_full_filename, axisnum=0, slicenum=0, output_directory=None,only_extract_slices=False):
    # splitting file and directory from input full filename
    input_file_dir,input_filename = os.path.split(input_full_filename)

    # removing extetion from input file
    input_filename_without_extension = os.path.splitext(input_filename)[0]
    
    # putting png extension to output file
    output_filename = "Axis%.3d-Slice%.3d-%s.png" % (
            axisnum, 
            slicenum,
            input_filename_without_extension)
    
    
    png_full_filename = os.path.join(output_directory, output_filename)
    
    # ENDED    
    return png_full_filename

def build_txt_filename_from_3d_image(input_image_full_filename, output_directory=None,only_extract_slices=False):
    # splitting file and directory from input full filename
    input_file_dir,input_filename = os.path.split(input_image_full_filename)

    # removing extetion from input file
    input_filename_without_extension = os.path.splitext(input_filename)[0]
    
    # putting txt extension to output file
    if only_extract_slices:
        output_filename = input_filename_without_extension
    else:
        output_filename = "%s.txt" % input_filename_without_extension
    
    txt_full_filename = os.path.join(output_directory, output_filename)
    
    # ENDED    
    return txt_full_filename
    

def check_txt_exists(input_full_filename, output_directory):
    txt_file = build_txt_filename_from_3d_image(input_full_filename, output_directory)
    return os.path.exists(txt_file)

def load_image_attributes(attributes_file):
    attributes_list = []
    if os.path.exists(attributes_file):
        try:
            with open(attributes_file, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    attributes_list.append(row)
#                    number = row[5]
#                    if number:
#                        l.append(number)
        except os.error:
            print("Attributes file %s can not be readed (os.error in load_image_attributes function)" % attributes_file)
    
    else:
        message = str("file %s does not exist!" % attributes_file)
        raise ValueError(message)
    return attributes_list

def load_image_attributes_from_nii_img(nii_img_input_file, attribs_file_dir=None):
    attributes_full_filename = build_txt_filename_from_3d_image(nii_img_input_file, attribs_file_dir)
    return load_image_attributes(attributes_full_filename)


def read_slice_attributes(nii_img_full_filename, axisnum, slicenum, all_images_attributes):
    return "ops"

    
def save_slice_as_png(nii_img_full_filename, slicenum, axisnum, output_directory, show_image=False):
    
    # build filename structure
    if not os.path.exists(nii_img_full_filename):
        raise ValueError("*** ERROR: input nii image filename not exist or can not be readed")
        exit(1)
    
    input_file_dir,input_filename = os.path.split(nii_img_full_filename)
    
    if output_directory == None:
        # Use input file dir as output when output dir is None
        output_directory = input_file_dir
        
    if not os.path.exists(output_directory):
        # create the output dir whether it doesnt exist
        try:
            os.makedirs(output_directory)
        except os.error:
            print ('*** ERROR: Output directory (%s) can not be created\n' % output_directory)
            sys.exit(1)
    
    
    nii_img = nb.load(nii_img_full_filename)
    
    nii_img_data = nii_img.get_fdata()

    # Get a slice and show it
    img_slice = get_slice_data(nii_img_data, slicenum, axisnum)
    
    # Show image
    if show_image:
        plt.imshow(img_slice,cmap="gray",origin="lower")
        #plt.show(img_slice)
        plt.show()
    
    # Normalization
    max_value = img_slice.max()
    if max_value != 0:
        img_slice = img_slice / max_value
    img_slice = img_slice * 255;
    img_slice_uint8 = img_slice.astype(np.uint8)
    
    # saving png file
    png_full_filename = build_png_filename_from_slice(nii_img_full_filename,
                                                      axisnum,
                                                      slicenum,
                                                      output_directory)
    
    iio.imwrite(png_full_filename,img_slice_uint8)
    
    return png_full_filename

def zernike(img_path, rad):
    return mh.features.zernike_moments(mh.imread(img_path, as_grey=True), rad)


def haralick(img_path):
    return mh.features.haralick(mh.imread(img_path))


def attributes(img_path):
    r = mh.bbox(mh.imread(img_path))[1]/2
    return zernike(img_path,r).tolist() + haralick(img_path).flatten('K').tolist()


def get_attributes(png_full_filename):
    return attributes(png_full_filename)


# Updated on 2019, Jan 17: to resolve problem where each file line ends with a comma (,).
    # Resolution are not tested yet
def append_attributes_to_file(
        nii_img_full_filename, 
        attributes, 
        axis_num, 
        slice_num, 
        output_directory, 
        mode="a",
        verbose=False,
        limit_precision=True):
    
    # build filename structure
    if not os.path.exists(nii_img_full_filename):
        raise ValueError("*** ERROR: input nii image filename not exist or can not be readed")
        exit(1)
    
    input_file_dir,input_filename = os.path.split(nii_img_full_filename)
    
    if output_directory == None:
        # Use input file dir as output when output dir is None
        output_directory = input_file_dir
    elif not os.path.exists(output_directory):
        # create the output dir whether it doesnt exist
        try:
            os.makedirs(output_directory)
        except os.error:
            print ('*** ERROR: Output directory (%s) can not be created\n' % output_directory)
            sys.exit(1)
    
    # building txt file name
    txt_full_filename = build_txt_filename_from_3d_image(nii_img_full_filename,
                                                      output_directory)
    try :
        output_file = open(txt_full_filename,mode)
        # writting body axis and slice values
        #output_file.write('%d,%d,' % (axis_num, slice_num))
        output_file.write('{0:1d},{1:3d}'.format(axis_num, slice_num))
        
        # writtings attribs one by one
        for attrib in attributes:
            if attrib != "" or attrib != None or attrib != '\n':
                if limit_precision:
                    output_file.write(",{0:.8f}".format(attrib))
                else:
                    output_file.write(',{0}'.format(attrib))
            
        output_file.write('\n')
        output_file.close()
    except os.error:
        output_file.close()
        print(" *** ERROR: file %s can not be written" % txt_full_filename)
        exit(1)
    
    return txt_full_filename


def remove_attributes_file(
        nii_img_full_filename, 
        output_directory):
    
    attributes_full_filename = build_txt_filename_from_3d_image(
            nii_img_full_filename,
            output_directory)
    
    os.remove(attributes_full_filename)



def extract_attributes_from_file(nii_img_full_filename, 
                          output_directory, 
                          keep_png_cache_files=False, 
                          #extract_attributes=False, 
                          verbose=False,
                          reset_output_file=True,
                          limit_precision=True,
                          testing=False,
                          only_extract_slices=False):
    
    nii_img = nb.load(nii_img_full_filename)
    
    if verbose:
        print ("Initializing attributes extraction from file: %s" % nii_img_full_filename)
    
    if testing:
        total_used_axis = 1
    else:
        total_used_axis = 3
    
    #slices_to_extract = 0
    #all_axis = nii_img.shape
    #for size in all_axis:
        #slices_to_extract = slices_to_extract + size
    
    txt_file = "file.txt"
    
    if not only_extract_slices:
        
        # Building correct filename for txt output file
        txt_file = build_txt_filename_from_3d_image(
                    nii_img_full_filename,output_directory)
        
        if verbose:
            print("    Output file will be: %s" % txt_file)
    
        # Testing if output file should be removed
        if reset_output_file:
            if os.path.exists(txt_file):
                try:
                    os.remove(txt_file)
                except os.error:
                    raise ValueError("*** File %s already exists but can not be removed.", txt_file)
    else:
        # Creates a subfolder from each nii input to save their slices
        output_directory = build_txt_filename_from_3d_image(nii_img_full_filename, output_directory, only_extract_slices=only_extract_slices)
    
    if not os.path.exists(txt_file): # run extraction only when outputfile doesn't exist
        
        # Main loop (core task)
        for axis_number in range(total_used_axis): # picking a axis
            total_slices = nii_img.shape[axis_number] # picking total slices for selected axis

            # Update 14/02/2019: Next line is disabled because the next
            # try/except block was incluced to solve issues with corrupted 
            # nii image files
            #saved_png_full_filename = save_slice_as_png(nii_img_full_filename, slice_number, axis_number, output_directory)
 
            # Updated at 14/02/2019: try/except block included to solve 
            # issues with corrupted nii image files
            try:

                for slice_number in range(total_slices): # picking a valid slice
                    saved_png_full_filename = save_slice_as_png(nii_img_full_filename, slice_number, axis_number, output_directory)
               
                    
                    if not only_extract_slices:
                        # getting image attributes (zernick pm and haralick stats)
                        attribs = get_attributes(saved_png_full_filename)
                    
                        # appending attributes to txt file
                        append_attributes_to_file(nii_img_full_filename, attribs, axis_number, slice_number, output_directory)
                
                    # verbose print
                    if verbose==True:
                        print("Getting slice ", slice_number, "inside body axis " , axis_number)
                        print("\nAttributes (slice=%s,axis=%s): " % (slice_number, axis_number))
                        
                        if not only_extract_slices:
                            print(attribs)
                    #else:
                        #print('.', end='')
                
                    # Check whether cache files must be kept
                    if not keep_png_cache_files:
                        os.remove(saved_png_full_filename)
 
               
            except os.error:
                print("\n * ERROR: File {0} can not be readed fully. Can it be corrupted? (def extract_attributes_from_file())".format(nii_img_full_filename))
                #break
                continue

    else:
        if verbose:
            print("File %s already exist so their attributes will not be extracted" % txt_file)
    
            
def old_list_files(input_dir,extention=".nii"):
    returned_files = []
    dir_files = os.listdir(input_dir)
    for filename in dir_files:
        name,ext = os.path.splitext(filename)
        if ext == extention:
            full_filename = os.path.join(input_dir,filename)
            #print(full_filename)
            if os.path.isfile(full_filename):
                returned_files.append(full_filename)
        
    return returned_files

def list_files(input_dir,extention=".nii"):
    returned_files = []
    for root, dirs, files in os.walk(input_dir,topdown=False):
        for filename in files:
            name,ext = os.path.splitext(filename)
            if ext == extention:
                full_filename = os.path.join(root,filename)
                if os.path.isfile(full_filename):
                    returned_files.append(full_filename)
            
    return returned_files


def extract_attributes_from_dir(input_directory, 
                                output_directory,
                                keep_png_cache_files=False,
                                #extract_attributes=False,
                                verbose=False,
                                reset_output_file=True,
                                limit_precision=True,
                                multi_cpu=False,
                                only_extract_slices=False):

    files = list_files(input_directory,".nii")
    
    if verbose:
        print("Files to be processed: ", files)
        
    reset_output_file
    if not multi_cpu:
        for filename in files:
            extract_attributes_from_file(filename,
                                         output_directory,
                                         keep_png_cache_files,
                                         #extract_attributes,
                                         verbose,
                                         reset_output_file,
                                         limit_precision,
                                         only_extract_slices)
    else:
        cores_num = multiprocessing.cpu_count()
        with Pool(cores_num) as p:
            from functools import partial
            p.map( 
                partial(extract_attributes_from_file,
                           output_directory=output_directory,
                           keep_png_cache_files=keep_png_cache_files,
                           verbose=verbose,
                           reset_output_file=reset_output_file,
                           limit_precision=limit_precision,
                           only_extract_slices=only_extract_slices), 
                files)

        

def extract_attributes(input_path,
                       output_directory,
                       keep_png_cache_files=False,
                       #extract_attributes=False, 
                       verbose=False,
                       reset_output_file=True,
                       limit_precision=True,
                       multi_cpu=False,
                       only_extract_slices=False):
    
    if os.path.exists(input_path):
        
        # Code when input path is a directory
        if os.path.isdir(input_path):
            extract_attributes_from_dir(input_path,
                                        output_directory,
                                        keep_png_cache_files,
                                        #extract_attributes, 
                                        verbose,
                                        reset_output_file,
                                        limit_precision,
                                        multi_cpu,
                                        only_extract_slices)
        # Code when input path is a single file
        else:
            extract_attributes_from_file(input_path,
                                        output_directory,
                                        keep_png_cache_files,
                                        #extract_attributes, 
                                        verbose,
                                        reset_output_file,
                                        limit_precision,
                                        only_extract_slices)                                        

def display_help(script_name=None):
    if script_name == None:
        script_name = os.path.split(sys.argv[0])[1]

    print ('Usage:\n    ', script_name, '[Options] -i <inputfile> -o <outputdir> ')
    print ('  Options:')
    print('\t-m, --multicpu\tset on computation over all cores (default: multicore is off)')
    print('\t-v, --verbose\tenables verbose mode (default: disabled)')
    print('\t-r, --resume\tresume extraction: output files are not overwritten (default: resume is off)')
    print('\t-k, --keep_png\ttemporary png files extracted from MRI volume slices are not deleted (default: keep_png is False)')
    print('\t-s, --only_extract_slices\tattributes from slices are not extracted (default: only_extract_slices is False). Implies --keep_png=True')

def main(argv):
    
    inputfile = ''
    outputdir = ''
    ifile_ok = False
    ofile_ok = False
    verbose_ok = False
    multi_cpu_ok = False
    reset_txt_file_ok = True
    keep_temporary_png = False
    
    only_extract_slices = False # If this is True, keep_temporary_png will be True
    
    try:
        opts, args = getopt.getopt(argv[1:],"hi:o:vmrks",["ifile=","odir=","verbose","multicpu","reset_output_file","keep_png","only_extract_slices"]) 
    except getopt.GetoptError:
        display_help()
        sys.exit(1)
    for opt, arg in opts:
        if opt == '-h':
            display_help()
            sys.exit(0);
        elif opt in ("-i", "--ifile"):
            inputfile = arg
            ifile_ok = True
        elif opt in ("-o", "--odir"):
            outputdir = arg
            ofile_ok = True
        elif opt in ("-v", "--verbose"):
            verbose_ok = True
        elif opt in ("-m", "--multicpu"):
            multi_cpu_ok = True
        elif opt in ("-r", "--resume"):
            reset_txt_file_ok = False
        elif opt in ("-k","--keep_png"):
            keep_temporary_png = True
        elif opt in ("-s","--only_extract_slices"):
            keep_temporary_png = True
            only_extract_slices = True
            
    
    if ifile_ok and ofile_ok:
        print ('Output directory is: ', outputdir)
        print ('Input file is: ', inputfile)
        
        #loadattribs.load_all_data(attribs_dir, csv_file)
        extract_attributes(input_path=inputfile,
                           keep_png_cache_files=keep_temporary_png,
                           output_directory=outputdir,
                           verbose=verbose_ok,
                           multi_cpu=multi_cpu_ok,
                           reset_output_file=reset_txt_file_ok,
                           only_extract_slices=only_extract_slices)
        
    else:
        display_help()
    
if __name__ == "__main__":    
    main(sys.argv)
    

