#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 11:23:05 2018

@author: rodrigo
"""

import os, csv, sys

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
        

def display_help(script_name=None):
    if script_name == None:
        script_name = os.path.split(sys.argv[0])[1]

    print ('Usage:\n    ', script_name, '[Options] -i <inputfile> -o <outputdir> ')
    print ('  Options:')
    print('\t-m, --multicpu\tset on computation over all cores (default: multicore is off)')
    print('\t-v, --verbose\tenables verbose mode (default: disabled)')
    print('\t-r, --resume\tresume extraction: output files are not overwritten (default: resume is off)')

        
def main(argv):
    # lixo
    attributes_dir = "/home/rodrigo/Downloads/fake_dir.nii/"
    nii_file = "/home/rodrigo/Downloads/fake_dir.nii/ADNI_136_S_0184_MR_MPR____N3__Scaled_Br_20090708094745554_S64785_I148265.nii"
    
    attribs = load_image_attributes_from_nii_img(nii_file, attributes_dir)
    print("* Teste de recarga de arquivo de atributos:\n")
    print(".Nii file:", nii_file)
    print("* Directorio do arquivo de atributos:", attributes_dir)

    print("* Origem dos atributos da linha 80: Eixo=%s, Fatia=%s" % (attribs[80][0], attribs[80][1]) )
    print("* Demais atributos dessa linha: ",end='')
    ns = 0
    for n in attribs[80][2:]:
        if ns < 5:
            ns = ns + 1
            print("%s "% n,end='')
    print("")
    
    #fim do lixo

    
if __name__ == "__main__":    
    main(sys.argv)
