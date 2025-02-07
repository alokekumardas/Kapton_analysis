# -*- coding: utf-8 -*-

#small plotting program to display histograms of kapton strip length and width, as well as the scaling factor histograms

#usage: python plot_kapton_results.py -f <filenames>         plot results from multiple results csv files (specify full filename incl. .csv ending)
#                         "           -d <foldername>        plot results from all results csv files in a folder (specify foldername w/o / or \ at end)
#                                     -l <foldernames>       plot results from all results csv files in multiple folders (specify foldername w/o / or \ at end). Careful: it will search for a Results folder within each folder!
# 
# where the data is stored:
# the program creates a folder named "Histos" in the folder with the Results data. 
# !!! the filenames are not unique, if you read in files from same folder,each the histos will be overwritten. 
# !!! Also, when you specify more than 1 input file not being in the same folder, the Histos folder will be created at the location of the first file       

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import platform
import glob
from helper import kaptonconstants

if platform.system()=="Linux":
    delim="/"
else:
    delim="\\"

def plot_histo(data, xlabel, datalabel, savename, plottype,ktype):
    plt.style.use('bmh')
    plt.figure(figsize=(14,8))
    plt.hist(data,bins='auto', label=datalabel+'\nEntries: '+str(len(data))+'\nMean: '+str(np.round(np.mean(data),3))+'\nStd: '+str(np.round(np.std(data),3)))
    if plottype == "length":
        # determine if file for short or long strip data
        if ktype == "long":
        #if np.abs(np.mean(data)-kaptonconstants.nominal_kaptonlength) < np.abs(np.mean(data)-kaptonconstants.nominal_stumpkaptonlength):
            print('strip type long')
            nomlength = kaptonconstants.nominal_kaptonlength
            minlength = kaptonconstants.minimal_kaptonlength
            maxlength = kaptonconstants.maximal_kaptonlength
        else:
            print('strip type short')
            nomlength = kaptonconstants.nominal_stumpkaptonlength
            minlength = kaptonconstants.minimal_stumpkapton_length
            maxlength = kaptonconstants.maximal_stumpkapton_length
        plt.axvline(nomlength, color='k')
        plt.axvline(minlength, color='r')
        if ktype == "long":
            plt.axvline(maxlength, color='r')
    if plottype == "width":
        # determine if file for short or long strip data
        if ktype == "long":
        #if np.abs(np.mean(data)-kaptonconstants.nominal_kaptonwidth) < np.abs(np.mean(data)-kaptonconstants.nominal_stumpkaptonwidth):
            print('strip type long')
            nomwidth = kaptonconstants.nominal_kaptonwidth
            minwidth = kaptonconstants.minimal_kaptonwidth
            maxwidth = kaptonconstants.maximal_kaptonwidth
        else:
            print('strip type short')
            nomwidth = kaptonconstants.nominal_stumpkaptonwidth
            minwidth = kaptonconstants.minimal_stumpkapton_width
            maxwidth = kaptonconstants.maximal_stumpkapton_width
        plt.axvline(nomwidth, color='k')
        plt.axvline(minwidth, color='r')
        if ktype == "long":
            plt.axvline(maxwidth, color='r')
    #if plottype == "short_length":

    #if plottype == "short_width":

    plt.xlabel(xlabel, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Entries / bin', fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig(savename)


def plot_files(filenames, save_folder_base): #input: a list of filenames
    '''
    if platform.system()=="Linux":
        delim="/"
    else:
        delim="\\"
    '''
    #save_folder_base=str(filenames[0]).replace(str(filenames[0]).split(delim)[-1],"")
    save_folder = save_folder_base+"Histos"
    print(save_folder)
    if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    lengths=[]
    widths=[]
    scalings_x=[]
    scalings_y=[]
    lengths_filtered=[] #filter out weird values, especially for short kapton strips, where it is clear that it is an artifact
    widths_filtered=[]
    scalings_x_filtered=[]
    scalings_y_filtered=[]
    limit_low_width=9 #mm
    limit_high_width=11 #mm
    limit_low_length_long=94.5 #mm
    limit_high_length_long=97 #mm
    limit_low_length_short=16.5 #mm
    limit_high_length_short=18.5 #mm
    for i in range(len(filenames)):
        fn = filenames[i]
        f = (pd.read_csv(fn, skiprows=1,sep=',',names=['serial_number','dimensions_check_passed','length','width','area','scaling_x','scaling_y','inner_contour_area','fitmatch','max_residual','max_distance1','max_distance2','max_area_score'])).values
        #print('f = ', f)
        for j in range(len(f)):
            lengths.append(f[j,2])
            widths.append(f[j,3])
            scalings_x.append(f[j,5])
            scalings_y.append(f[j,6])
            #filter weird strips
            if f[j,3] > limit_low_width and f[j,3] < limit_high_width:
                #short
                if f[j,2] < limit_high_length_short and f[j,2] > limit_low_length_short:
                    lengths_filtered.append(f[j,2])
                    widths_filtered.append(f[j,3])
                    scalings_x_filtered.append(f[j,5])
                    scalings_y_filtered.append(f[j,6])
                #long
                elif f[j,2] < limit_high_length_long and f[j,2] > limit_low_length_long:
                    lengths_filtered.append(f[j,2])
                    widths_filtered.append(f[j,3])
                    scalings_x_filtered.append(f[j,5])
                    scalings_y_filtered.append(f[j,6])
                else:
                    print('plot files: j = ', j, " weird strip lengths detected! length ", f[j,2], "mm width ",f[j,3],"mm")
                    print('this is file ', fn)
            else:
                print('plot files: j = ', j, " weird strip widths detected! length ", f[j,2], "mm width ",f[j,3],"mm")
                print('this is file ', fn)
    
    #determine kapton type
    #print('lengths[0] ', lengths[0])
    if lengths[0] > 50:
        kaptontype = "long"
    else:
        kaptontype = "short"

    #Length histogram
    plot_histo(np.array(lengths), 'Length / mm', 'Kapton strip length', save_folder+delim+"hLengths.png","length",kaptontype)
    print("Length       = "+str(np.mean(np.array(lengths)))+" +/- "+str(np.std(np.array(lengths))/np.sqrt(np.mean(np.array(lengths)))))

    #Width histogram
    plot_histo(np.array(widths), 'Width / mm', 'Kapton strip width', save_folder+delim+"hWidths.png", "width",kaptontype)
    print("Width        = "+str(np.mean(np.array(widths)))+" +/- "+str(np.std(np.array(widths))/np.sqrt(np.mean(np.array(widths)))))

    #Scaling factor in x histogram
    plot_histo(np.array(scalings_x), 'Scaling factor', 'Scaling factor along x', save_folder+delim+"hScalingfactor_x.png", "scaling",kaptontype)
    print("Scaling x    = "+str(np.mean(np.array(scalings_x)))+" +/- "+str(np.std(np.array(scalings_x))/np.sqrt(np.mean(np.array(scalings_x)))))

    #Scaling factor in y histogram
    plot_histo(np.array(scalings_y), 'Scaling factor', 'Scaling factor along y', save_folder+delim+"hScalingfactor_y.png", "scaling",kaptontype)
    print("Scaling y    = "+str(np.mean(np.array(scalings_y)))+" +/- "+str(np.std(np.array(scalings_y))/np.sqrt(np.mean(np.array(scalings_y)))))

    ####filtered histos: w/o weird strips
    #Length histogram
    plot_histo(np.array(lengths_filtered), 'Length / mm', 'Kapton strip length', save_folder+delim+"hLengths_filtered.png","length",kaptontype)
    print("Length filtered     = "+str(np.mean(np.array(lengths_filtered)))+" +/- "+str(np.std(np.array(lengths_filtered))/np.sqrt(np.mean(np.array(lengths_filtered)))))

    #Width histogram
    plot_histo(np.array(widths_filtered), 'Width / mm', 'Kapton strip width', save_folder+delim+"hWidths_filtered.png", "width",kaptontype)
    print("Width filtered       = "+str(np.mean(np.array(widths_filtered)))+" +/- "+str(np.std(np.array(widths_filtered))/np.sqrt(np.mean(np.array(widths_filtered)))))

    #Scaling factor in x histogram
    plot_histo(np.array(scalings_x_filtered), 'Scaling factor', 'Scaling factor along x', save_folder+delim+"hScalingfactor_x_filtered.png", "scaling",kaptontype)
    print("Scaling x filtered   = "+str(np.mean(np.array(scalings_x_filtered)))+" +/- "+str(np.std(np.array(scalings_x_filtered))/np.sqrt(np.mean(np.array(scalings_x_filtered)))))

    #Scaling factor in y histogram
    plot_histo(np.array(scalings_y_filtered), 'Scaling factor', 'Scaling factor along y', save_folder+delim+"hScalingfactor_y_filtered.png", "scaling",kaptontype)
    print("Scaling y filtered   = "+str(np.mean(np.array(scalings_y_filtered)))+" +/- "+str(np.std(np.array(scalings_y_filtered))/np.sqrt(np.mean(np.array(scalings_y_filtered)))))
#main part of program
arguments = sys.argv[1:]
the_filenames=[]
ind_f=-1
ind_d=-1
ind_l=-1
savefolder_base=" "
if "-h" in arguments or "--help" in arguments or len(arguments) == 0:
    print('Usage of script with single files: python plot_kapton_results.py --files <filenames>')
    print('                               or: python plot_kapton_results -f <filenames>')
    print('Usage of script with all files in a folder: python plot_kapton_results--dir <foldername>')
    print('                               or: python plot_kapton_results -d <foldername>')
    print('Usage of script with all files in multiple folders ( Careful: it will search for a Results folder within each folder!): python plot_kapton_results--listdir <foldernames>')
    print('                               or: python plot_kapton_results -l <foldernames>')
    print('foldername must be specified without / or \ at end of name')


else:
    for s in range(len(arguments)):
            if arguments[s] == "-f" or arguments[s] == "--files":
                ind_f=s 
            if arguments[s] == "-d" or arguments[s] == "--dir":
                ind_d=s
            if arguments[s] == "-l" or arguments[s] == "--listdir":
                ind_l=s

    if ind_f !=-1:
            if ind_d !=-1 or ind_l !=-1:
                raise RuntimeError('ARGUMENT Error! Specified files and dir or list mode at the same time! Please use option -h for help. Abort program...')
            else:
                the_filenames = arguments[ind_f+1:]
                savefolder_base=str(the_filenames[0]).replace(str(the_filenames[0]).split(delim)[-1],"")

    if ind_d !=-1:
        if ind_f !=-1 or ind_l !=-1:
            raise RuntimeError('ARGUMENT Error! Specified dir and files or list mode at the same time! Please use option -h for help. Abort program...')
        foldername = arguments[ind_d+1]
        if platform.system()=="Linux":
            the_filenames = glob.glob(foldername+"/*.csv")
        else:
            the_filenames = glob.glob(foldername+"\*.csv")
        savefolder_base=str(the_filenames[0]).replace(str(the_filenames[0]).split(delim)[-1],"")
        
    
    if ind_l !=-1:
        if ind_f !=-1 or ind_d !=-1:
            raise RuntimeError('ARGUMENT Error! Specified list and files or dir mode at the same time! Please use option -h for help. Abort program...')
        foldernames = arguments[ind_l+1:]
        print('foldernames ', foldernames)
        savefolder_base=str(foldernames[0]).replace(str(foldernames[0]).split(delim)[-1],"")
        for n in range(len(foldernames)):
            filenames_folder=[]
            if platform.system()=="Linux":
                filenames_folder = glob.glob(foldernames[n]+"/Results/*.csv")
            else:
                filenames_folder = glob.glob(foldernames[n]+"\Results\*.csv")
            print('filenames_folder ', filenames_folder)
            for f in range(len(filenames_folder)):
                the_filenames.append(filenames_folder[f])


    print("the_filenames ", the_filenames)
    print("savefolder_base ", savefolder_base)
    plot_files(the_filenames, savefolder_base)




