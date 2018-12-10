#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 17:11:32 2018

@author: rodrigo
"""
import sys, getopt, os.path


def display_help(script_name=None):
    if script_name == None:
        script_name = os.path.split(sys.argv[0])[1]

    print ('Usage: \n\t', script_name, '-i <inputfile> -o <outputfile>')

def main(argv):
    inputfile = ''
    outputfile = ''
    ifile_ok = False
    ofile_ok = False
    
    try:
        opts, args = getopt.getopt(argv[1:],"hi:o:",["ifile=","ofile="])
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
        elif opt in ("-o", "--ofile"):
            outputfile = arg
            ofile_ok = True
    
    if ifile_ok and ofile_ok:
        print ('Output dir is: ', outputfile)
        print ('Input file is: ', inputfile)
    else:
        display_help()
    
if __name__ == "__main__":
    main(sys.argv)
    

