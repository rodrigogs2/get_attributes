#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Updates:
# 2019, Jan 17: função de escrever os atributos em cada linha do arquivo 
# (append_attributes_to_file)foi modificada para evitar que cada linha termine 
# com uma virgula sem nenhum dado após ela. 
# obs.: ainda não testado

"""
Created on Sun Apr  25 20:53:17 2021

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

#__OUTPUT_PREFIX = "FLIRT_"

#__BET_BIN = "bet"
__FLIRT_PREFIX = "FLIRT_"
__FLIRT_BIN = "flirt"
__FIXFILE_BIN = "fslchfiletype"
__STD_REF = USER = "{0}/data/standard/MNI152_T1_1mm_brain".format(os.getenv('FSLDIR'))

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
    

def run_flirt(input_file,output_file,ref_file=None,verbose=False,xfm=False):
    global __STD_REF
    xfm_option = ''

    if not ref_file:
        ref = ref_file
    else:
        ref = __STD_REF

    if xfm:
        xfm_option = '-applyxfm'
	
    flirt_command = "{0} -in {1} -out {2} -ref {3} {4}".format(
	__FLIRT_BIN, 
	input_file, 
	output_file, 
	ref, 
	xfm_option)
    if verbose:
        print("*** Running flirt using: {0}".format(flirt_command))

    return run_command(flirt_command)[0],output_file


def run_fixfiletype(filename):
	global __FIXFILE_BIN
	filetype_command = "{1} NIFTI {0}".format(filename,__FIXFILE_BIN)
	fixed_file = run_command(filetype_command)
	return fixed_file[1]

	

def build_output_full_filename(input_full_filename, output_directory):
    global __OUTPUT_PREFIX

    # splitting file and directory from input full filename
    input_path, input_filename = os.path.split(input_full_filename)
    output_filename = input_filename


    # to avoid input and output have the same name
    if input_path == output_directory:
        global __FLIRT_PREFIX
        output_filename = str(__FLIRT_PREFIX + output_filename)


    output_full_filename = os.path.join(output_directory, output_filename)

    	
    return output_full_filename


    

def normalize_brain(
        input_full_filename,
        output_directory,
        template,
        verbose=False,
        resume_task=False,
        use_xfm=False):

    # Start task only if input file exists
    if os.path.exists(input_full_filename):
        if verbose:
            print ("Initializing Normalization to file: %s" % input_full_filename)

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

        # Testing whether either output file already exists or not
        if os.path.exists(output_full_filename) and not resume_task:
            try:
                # trying to remove file
                os.remove(output_full_filename)
            except os.error:
                raise ValueError("*** File %s already exists but can not be removed.", nii_file)

        # Try to run flirt to normalize input brain using standard template
        try:
            normalized_brain = run_flirt(input_full_filename, output_full_filename, template, use_xfm)[1]
        except OSError as error:
            print("\n * ERROR: Can't normalize brain from file {0}".format(input_full_filename))
            print(error)

        # Fix file type by forcing to be .nii (NIFTI)
        try:
            fixed_file = run_fixfiletype(normalized_brain)
        except OSError as error:
            print("\n * ERROR: Can't fix type of file {0}".format(normalized_brain))
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


def normalize_all_brains(
		input_path, 
		output_directory,
		ref_file=None,
		verbose=False,
		resume_task=False,
		multi_cpu=False,
		file_type=".nii",
		xfm=False):
    if ref_file:
        if not os.path.exists("{0}.nii.gz".format(ref_file)):
            print("Template file {0} doesnt exist! Leaving now...".format(ref_file))
            sys.exit(1)
        else:
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
                normalize_brain(
			filename,
			output_directory,
			template,
			verbose,
			resume_task,
			xfm)
        else:
            cores_num = multiprocessing.cpu_count()
            with Pool(cores_num) as p:
                from functools import partial
                p.map(
                    partial(normalize_brain,
                            output_directory=output_directory,
                            template=template,
                            verbose=verbose,
                            resume_task=resume_task,
                            use_xfm=xfm),
                    files)

    

def display_help(script_name=None):
    if script_name == None:
        script_name = os.path.split(sys.argv[0])[1]

    print ('Usage:\n    ', script_name, '[Options] -i <input path: file or directory> -o <outputdir>')
    print ('  Options:')
    print('\t-m, --multicpu\tset on computation over all cores (default: multicore is off)')
    print('\t-v, --verbose\tenables verbose mode (default: disabled)')
    print('\t-r, --resume\tresume task: output files are not overwritten (default: resume is off)')
    print('\t-x, --xfm\tapplies xfm on normalization process (default: xfm is off)')



def main(argv):
    
    inputpath = ''
    outputdir = ''
    template = ''
    ifile_ok = False
    ofile_ok = False
    template_ok = False
    verbose_ok = False
    multi_cpu_ok = False
    resume_task_ok = False
    xfm_ok = False

    try:
        opts, args = getopt.getopt(argv[1:],"hi:o:t:vmrx",["ifile=","odir=","template=","verbose","multicpu","resume","xfm"]) 
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
            resume_task_ok = True
        elif opt in ("-x", "--xfm"):
            xfm_ok = True

    
    if ifile_ok and ofile_ok:
        print ('Output directory is: ', outputdir)
        print ('Input used: ', inputpath)
    
        if not template_ok:
            global __STD_REF
            template = __STD_REF
        
        normalize_all_brains(
			input_path=inputpath,
			output_directory=outputdir,
			ref_file=template,
			verbose=verbose_ok,
                        resume_task=resume_task_ok,
			multi_cpu=multi_cpu_ok,
			xfm=xfm_ok)
        
    else:
        display_help()
    
if __name__ == "__main__":    
    main(sys.argv)
    

