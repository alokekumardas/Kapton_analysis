#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# batch mode program to read in a number of Kapton scans, analyze them and upload results to DB. Does the same as the GUI, but w/o a GUI...

#How it works (so far, 14.5.24...)
# execute like this: python KaptonBatchmode.py -f <filename>                      to read a single image file (need to specify full path!)
#                    python KaptonBatchmode.py -d <foldername>                    to read all image files in a folder (of type .jpg). Important: foldername w/o "\" or "/" at end!
#                    python KaptonBatchmode.py -f <filename1> <filename2> ...     to read multiple files
#
#                    additional, important options:
#                                              -k <kaptontype>                    select kapton type for strip numbering. Options long, short, shortR. shortR means rotation of 90deg compared to orientation of long strips
#                                              -u                                 upload to database (not finished yet). Independant from file analysis, i.e. do either analysis or upload. Upload is based on csv file output
#                                              -p                                 do not generate the plots, to save time

#Output: In Results Folder (created in the folder with the images) -> csv File with the Results for each analyzed scan
#        In Plots Folder (created in the folder with the images) -> Labeled image, Filtered image, Template filtered image, Sufficient strips image for each analyzed scan (resolution not as high as in orig image!)

#How to change DPI:
#    in this file in analyze_scans(...), just change the value of the dpi variable
import importlib.metadata
import matplotlib.pyplot as plt
#print('plt.__version__ ', plt.__version__)
import numpy as np
#from shutil import copyfile
import platform
print('platform.__version__ ', platform.__version__)
import subprocess
from os.path import join, isfile
#import xml.etree.cElementTree as ET
#print('ET.__version__ ', ET.__version__)
import datetime
import cv2
print('cv2.__version__ ', cv2.__version__)
#from helper import kapton_SQL
import sys
#print('sys.__version__ ', sys.__version__)
import glob
#print('glob.__version__ ', glob.__version__)
import Kapton_Analyze_batch as kab


def analyze_scans(files,do_fcal,do_upl, savefigs, show_plot,kaptontype):
    dpi=1200
    #kaptontype="short" # this is used to define the way the strips are labeled. long/short -> factor 0.1 i.e. labeling from left to right. shortR for factor 10 i.e. labeling from top to btm
    analyzer=kab.Analyze()
    if do_upl == True:
        print('Upload option selected. Will upload data from already existing csv files to DB')
        #TO DO: simple upload for the long strips. for the short strips however, we must check the orientation! specify in filename?
        analyzer.upload(files)
    else:
        print("Now execute analyze_scans")
        if do_fcal == True:
            for cf in files:
                print("do_fcal set to True: inout files calibration files. Not sure if that works already in a sufficient way... (normally not used)")
                analyzer.doCalibration(cf)
        for i in range(len(files)):
            f=str(files[i])
            print('f= ', str(f))
            analyzer.openScanfile(dpi,f, savefigs, show_plot,kaptontype)

arguments = sys.argv[1:]
print(arguments)

do_filtercalibration = False
do_upload = False
save_plots = True
show_plots = True
kap_type="long" #default option. other options : short, shortR. this is used to define the way the strips are labeled. long/short -> factor 0.1 i.e. labeling from left to right. shortR for factor 10 i.e. labeling from top to btm
if "-h" in arguments or "--help" in arguments or len(arguments) == 0:
    print('Usage of script with single files: python KaptonBatchMode.py --files <filenames>')
    print('                               or: python KaptonBatchMode.py -f <filenames>')
    print('Usage of script with all files in a folder: python KaptonBatchMode.py --dir <foldername>')
    print('                               or: python KaptonBatchMode.py -d <foldername>')
    print(' other options: -c or --calibration -> filter calibration (not to be used).\n --nosave or -n -> only show but do not save the plots.\n -u or --upload -> upload to DB.\n --noplot or -p -> don´t generate plots.\n --kaptontype or -k -> select kapton type for numbering. Options long, short, shortR; Default long.')

else:
    ind_f=-1
    ind_d=-1
    ind_c=-1
    ind_n=-1
    ind_u=-1
    ind_p=-1
    ind_k=-1
    for s in range(len(arguments)):
        if arguments[s] == "-f" or arguments[s] == "--files":
            ind_f=s  
        elif arguments[s] == "-d" or arguments[s] == "--dir":
            ind_d=s
        elif arguments[s] == "-c" or arguments[s] == "--calibration":
            ind_c=s
        elif arguments[s] == "-n" or arguments[s] == "--nosave":
            ind_n=s
        elif arguments[s] == "-u" or arguments[s] == "--upload":
            ind_u=s
        elif arguments[s] == "-p" or arguments[s] == "--noplot":
            ind_p=s
        elif arguments[s] == "-k" or arguments[s] == "--kaptontype":
            ind_k=s

    filenames=[]
    print('ind_f ', ind_f)
    print('ind_c ', ind_c)
    print('ind_u ', ind_u)
    print('ind_n ', ind_n)
    print('ind_p ', ind_p)
    print('ind_k ', ind_k)
    if ind_f !=-1:
        if ind_d !=-1:
            raise RuntimeError('ARGUMENT Error! Specified files and dir mode at the same time! Please use option -h for help. Abort program...')
        list_index_stop=[]
        if ind_c !=-1:
            do_filtercalibration=True
            if ind_c > ind_f:
                list_index_stop.append(ind_c)
        if ind_n !=-1:
            save_plots=False
            if ind_n > ind_f:
                list_index_stop.append(ind_n)
        if ind_u !=-1:
            do_upload=True
            if ind_u > ind_f:
                list_index_stop.append(ind_u)
        if ind_p !=-1:
            show_plots=False
            if ind_p > ind_u:
                list_index_stop.append(ind_p)
        if ind_k !=-1:
            kap_type=str(arguments[ind_k+1])
            if ind_k > ind_u:
                list_index_stop.append(ind_k)
        if not list_index_stop:
            filenames = arguments[ind_f+1:]
        else:
            filenames = arguments[ind_f+1:min(list_index_stop)]

    if ind_d !=-1:
        if ind_f !=-1:
            raise RuntimeError('ARGUMENT Error! Specified files and dir mode at the same time! Please use option -h for help. Abort program...')
        foldername = arguments[ind_d+1]
        if ind_p !=-1:
            show_plots=False
        if ind_c !=-1:
            do_filtercalibration=True
        if ind_n !=-1:
            save_plots=False
        if ind_u !=-1:
            do_upload=True
        if ind_k !=-1:
            kap_type=str(arguments[ind_k+1])
        if platform.system()=="Linux":
            filenames = glob.glob(foldername+"/*.jpg")
        else:
            filenames = glob.glob(foldername+"\*.jpg")

    print('filenames ',filenames)
    analyze_scans(filenames, do_filtercalibration,do_upload,save_plots, show_plots,kap_type)







