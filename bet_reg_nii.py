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
#import csv
import subprocess
import shlex

import sys

import multiprocessing
from multiprocessing import Pool

__OUTPUT_PREFIX = "BET_"
__BET_BIN = "bet"
__FLIRT_BIN = "flirt"
__FIXFILE_BIN = "fslchfiletype"
__STD_REF = "$FSLDIR/data/standard/MNI152_T1_1mm_brain"

def run_command(command_str,verbose=False):
    process = subprocess.Popen(shlex.split(command_str), stdout=subprocess.PIPE, encoding='utf8')
    process.stdout.flush()
    lastline = ''
    while True and verbose:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            lastline = output
            print(output.strip())
    rc = process.poll()
    return rc, lastline
    
def run_bet(input_file,output_file,verbose=False):
	global __BET_BIN
	bet_command = "{0} {1} {2}".format(__BET_BIN, input_file, output_file)
	if verbose:
		print("*** Running bet using: {0}".format(bet_command))
	return run_command(bet_command)[0], output_file


def run_flirt(input_file,output_file,ref_file=None,verbose=False,xfm=False):
	global __BET_BIN, __REF_FLIRT
	if not ref_file:
		ref = ref_file
	else:
		ref = __REF_FLIRT
	
	flirt_command = "{0} -in {1} -out {2} -ref {3} {4}".format(
		__FLIRT_BIN, 
		input_file, 
		output_file, 
		ref, 
		xfm)
	if verbose:
		print("*** Running flirt using: {0}".format(flirt_command))
	return run_command(flirt_command)[0],output_file


def run_fixfiletype(filename):
	global __FIXFILE_BIN
	filetype_command = "{1} NIFTI {0}".format(filename,__FIXFILE_BIN)
	fixed_file = run_command(filetype_command)
	return fixed_file[1]

	

def build_output_full_filename(input_image_full_filename, output_directory=None):
    global __OUTPUT_PREFIX

    # splitting file and directory from input full filename
    input_path, input_filename = os.path.split(input_image_full_filename)
    output_filename = str(__OUTPUT_PREFIX + input_filename)
    output_full_filename = os.path.join(input_path, output_filename)
    
    if output_directory:
    	output_full_filename = os.path.join(output_directory, output_filename)
    	
    return output_full_filename


    

def remove_skull_and_normalize_brain_from_file(
			input_full_filename,
			output_directory,
			template,
			verbose=False,
			reset_output_file=False,
			use_xrp=False):
	# Use current directory if there output_directory is null
	if not output_directory:
		output_directory = os.getcwd()

	if os.path.exists(input_full_filename):
		if verbose:
			print ("Initializing skull sttriping from file: %s" % input_full_filename)
		output_full_filename = build_output_full_filename(
			input_full_filename, 
			output_directory)
		if verbose:
			print("Output file will be: %s" % output_full_filename)
		

		# creates output dir when path doesnt exist
		if not os.path.exists(output_directory):
			try:
				os.makedirs(output_directory)
			except os.error:
				print ('*** ERROR: Output directory (%s) can not be created\n' % output_directory)
				sys.exit(1)
		
        if os.path.exists(output_full_filename):
			if reset_output_file:
				try:
					os.remove(output_full_filename)
				except os.error:
					raise ValueError("*** File %s already exists but can not be removed.", nii_file)
				try:
					bet_file = run_bet(input_full_filename,output_full_filename,verbose)[1]
				except OSError as error:
					print("\n * ERROR: Can't strip skull from file {0}".format(input_full_filename))
					print(error)
				try: 
					flirt_file = run_flirt(bet_file,output_file)
				except OSError as error:
					print("\n * ERROR: Can't normalize brain from file {0}".format(bet_file))
					print(error)
				try:
					fixed_file = run_fixfiletype(flirt_file)
				except OSError as error:
					print("\n * ERROR: Can't fix type of file {0}".format(flirt_file))
					print(error)

			else:
				if verbose:
					print("Skipping (file %s already exists)" % output_full_filename)
		else:
			try:
				bet_file = run_bet(input_full_filename,output_full_filename,verbose)[1]
			except OSError as error:
				print("\n * ERROR: Can't strip skull from file {0}".format(input_full_filename))
				print(error)
				sys.exit(1)
            try:
                output_full_filename = str("FLIRT_" + output_full_filename)
                flirt_file = run_flirt(bet_file,output_full_file)
            except OSError as error:
				print("\n * ERROR: Can't normalize brain from file {0}".format(bet_file))
				print(error)
			try:
				fixed_file = run_fixfiletype(flirt_file)
			except OSError as error:
				print("\n * ERROR: Can't fix type of file {0}".format(flirt_file))
				print(error)
            return fixed_file
	else:
		raise ValueError("*** Input path doesn't exist!")
		sys.exit(1)
    

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


def strip_skulls_and_normalize_brains(
		input_path, 
		output_directory,
		ref_file=None,
		verbose=False,
		reset_output_file=False,
		multi_cpu=True,
		file_type=".nii",
		xrp=False):

	if not ref_file:
		template = ref_file
	else:
		global __STD_REF
		template = __STD_REF

	files = list_files(input_path,file_type)
	if files:
		if verbose:
			print("Files to be processed: ", files)
        
		if not multi_cpu:
			for filename in files:
				remove_skull_and_normalize_brain_from_file(
					filename,
					output_directory,
					template,
					verbose,
					reset_output_file,
					xrp)
		else:
		    cores_num = multiprocessing.cpu_count()
		    with Pool(cores_num) as p:
		        from functools import partial
		        p.map( 
		            partial(remove_skull_and_normalize__brain_from_file,
		    			output_directory=output_directory,
		    			verbose=verbose,
					template=emplate,
		    			reset_output_file=reset_output_file,
					use_xrp=xrp),
		            files)

    

def display_help(script_name=None):
    if script_name == None:
        script_name = os.path.split(sys.argv[0])[1]

    print ('Usage:\n    ', script_name, '[Options] -i <input path: file or directory> -o <outputdir> ')
    print ('  Options:')
    print('\t-m, --multicpu\tset on computation over all cores (default: multicore is off)')
    print('\t-v, --verbose\tenables verbose mode (default: disabled)')
    print('\t-r, --resume\tresume extraction: output files are not overwritten (default: resume is off)')


def main(argv):
    
    inputpath = ''
    outputdir = ''
    template = ''
    ifile_ok = False
    ofile_ok = False
    verbose_ok = False
    multi_cpu_ok = False
    reset_output_file_ok = True
    template_ok = False

    try:
        opts, args = getopt.getopt(argv[1:],"hi:o:t:vmr",["ifile=","odir=","template=","verbose","multicpu","reset_output_file"]) 
    except getopt.GetoptError:
        display_help()
        sys.exit(1)
    for opt, arg in opts:
        if opt == '-h':
            display_help()
            sys.exit(0);
        elif opt in ("-i", "--ifile"):
            inputpath = arg
            ifile_ok = True
        elif opt in ("-o", "--odir"):
            outputdir = arg
            ofile_ok = True
        elif opt in ("-t", "--template"):
            template = arg
            template_ok = True
        elif opt in ("-v", "--verbose"):
            verbose_ok = True
        elif opt in ("-m", "--multicpu"):
            multi_cpu_ok = True
        elif opt in ("-r", "--resume"):
            reset_output_file_ok = False
    
    if ifile_ok and ofile_ok:
        print ('Output directory is: ', outputdir)
        print ('Input used: ', inputpath)
    
        if not template_ok:
            global __STD_REF
            template = __STD_REF
        
        strip_skulls_and_normalize_brains(
			input_path=inputpath,
			output_directory=outputdir,
			ref_file=template,
			verbose=verbose_ok,
			multi_cpu=multi_cpu_ok,
			reset_output_file=reset_output_file_ok)
        
    else:
        display_help()
    
if __name__ == "__main__":    
    main(sys.argv)
    

