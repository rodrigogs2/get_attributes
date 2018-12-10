#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 18:53:17 2018

@author: rodrigo
"""

import os
import getopt
import numpy as np

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
        

def build_png_filename_from_slice(input_full_filename, axisnum=0, slicenum=0, output_directory=None):
    # splitting file and directory from input full filename
    input_file_dir,input_filename = os.path.split(input_full_filename)

    # removing extetion from input file
    input_filename_without_extension = os.path.splitext(input_filename)[0]
    
    # putting png extension to output file
    output_filename = "%s-A%.3d-S%.3d.png" % (
            input_filename_without_extension, 
            axisnum, 
            slicenum)
    
    png_full_filename = os.path.join(output_directory, output_filename)
    
    # ENDED    
    return png_full_filename

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
    
    
def save_slice_as_png(nii_img_full_filename, slicenum, axisnum, output_directory, show_image=False):
    
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
        output_file.write('%d,%d,' % (axis_num, slice_num))
        for attrib in attributes:
            if limit_precision:
                output_file.write("{0:.8f}".format(attrib))
            else:
                output_file.write(attrib)
            output_file.write(',')
            
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
                          limit_precision=True):
    
    nii_img = nb.load(nii_img_full_filename)
    
    total_used_axis = 3
    
    #slices_to_extract = 0
    #all_axis = nii_img.shape
    #for size in all_axis:
        #slices_to_extract = slices_to_extract + size
    
    # Testing if output file should be removed
    if reset_output_file:
        txt_file = build_txt_filename_from_3d_image(
                nii_img_full_filename,output_directory)
        if os.path.exists(txt_file):
            try:
                os.remove(txt_file)
            except os.error:
                raise ValueError("*** File %s already exists but can not be removed.", txt_file)
                
    
    # Main loop (core task)
    for axis_number in range(total_used_axis): # picking a axis
        total_slices = nii_img.shape[axis_number] # picking total slices for selected axis
        for slice_number in range(total_slices): # picking a valid slice
            
            saved_png_full_filename = save_slice_as_png(nii_img_full_filename, slice_number, axis_number, output_directory)
            
            # getting image attributes (zernick pm and haralick stats)
            attribs = get_attributes(saved_png_full_filename)
            
            # appending attrbiutes to txt file
            append_attributes_to_file(nii_img_full_filename, attribs, axis_number, slice_number, output_directory)
            
            # verbose print
            if verbose==True:
                print("Getting slice ", slice_number, "inside body axis " , axis_number)
                print("\nAttributes (slice=%s,axis=%s): " % (slice_number, axis_number))
                print(attribs)
            #else:
                #print('.', end='')
            
            # Check whether cache files must be kept
            if not keep_png_cache_files:
                os.remove(saved_png_full_filename)
            
def list_files(input_dir,extention=".nii"):
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

def extract_attributes_from_dir(input_directory, 
                                output_directory,
                                keep_png_cache_files=False,
                                #extract_attributes=False,
                                verbose=False,
                                reset_output_file=True,
                                limit_precision=True,
                                multi_cpu=False):

    files = list_files(input_directory,".nii")
    
    if not multi_cpu:
        for filename in files:
            extract_attributes_from_file(filename,
                                         output_directory,
                                         keep_png_cache_files,
                                         #extract_attributes,
                                         verbose,
                                         reset_output_file,
                                         limit_precision)
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
                           limit_precision=limit_precision), 
                files)

        

def extract_attributes(input_path,
                       output_directory,
                       keep_png_cache_files=False,
                       #extract_attributes=False, 
                       verbose=False,
                       reset_output_file=True,
                       limit_precision=True,
                       multi_cpu=False):
    
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
                                        multi_cpu)
        # Code when input path is a single file
        else:
            extract_attributes_from_file(input_path,
                                        output_directory,
                                        keep_png_cache_files,
                                        #extract_attributes, 
                                        verbose,
                                        reset_output_file,
                                        limit_precision)                                        

def display_help(script_name=None):
    if script_name == None:
        script_name = os.path.split(sys.argv[0])[1]

    print ('Usage:\n    ', script_name, '[Options] -i <inputfile> -o <outputdir> ')
    print ('  Options:')
    print('\t--multicpu=True\tset on computation over all cores (default multicore is off)')
    print('\t--verbose=True\tenables verbose mode (default verbose is disabled)')

def main(argv):
    inputfile = ''
    outputdir = ''
    ifile_ok = False
    ofile_ok = False
    verbose_ok = False
    multi_cpu_ok = False
    
    try:
        opts, args = getopt.getopt(argv[1:],"hi:o:vm",["ifile=","odir=","verbose","multicpu"]) 
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
    
    if ifile_ok and ofile_ok:
        print ('Output directory is: ', outputdir)
        print ('Input file is: ', inputfile)
        extract_attributes(input_path=inputfile,
                           output_directory=outputdir,
                           verbose=verbose_ok,
                           multi_cpu=multi_cpu_ok)
        
    else:
        display_help()
    
if __name__ == "__main__":
    main(sys.argv)
    


"""
# TEST AREA

# Temporary Variables (should be removed later)
img_data_shape = ''
input_full_filename = '/home/rodrigo/Downloads/ADNI_136_S_0184_MR_MPR____N3__Scaled_Br_20090708094745554_S64785_I148265.nii'
input_filepath,input_filename = os.path.split(input_full_filename)
input_filename_without_extension = os.path.splitext(input_filename)[0] 


fake_output_dir = '/home/rodrigo/Downloads/fake_output_dir2/'
built_png_filename = build_png_filename_from_slice(input_full_filename, 0, 80, fake_output_dir)

print  ('*** input_file_path=', input_filepath,
      '\n*** input_filename=', input_filename,
      '\n*** input filename without extension=', input_filename_without_extension,
      '\n*** output png filename=', built_png_filename)

save_slice_as_png(input_full_filename,130,1,fake_output_dir)

extract_slices_as_png(input_full_filename,fake_output_dir,verbose=True)
"""


   








