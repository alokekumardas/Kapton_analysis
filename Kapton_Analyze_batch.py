# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from helper.kaptondetector import Scan, get_RGB_Distributions, get_optimizedRange
from helper import kaptonconstants
#from shutil import copyfile
import platform
import subprocess
from os.path import join, isfile
#import xml.etree.cElementTree as ET
import datetime
import cv2
from helper import kapton_SQL_batch
import sys
import csv
print('csv.__version__ ', csv.__version__)
import os
import shutil

# analysis and upload functions. Often they call functions from the kaptondetector
class Analyze:
    '''
    def upload(self):
        print("Start databaseupload...")
        if(not isfile("dbConfig.yaml")):
            subprocess.run(['python', 'DBConfCreator.py'], check=True)
            return
        print("Found dbConfig.yaml")
        MyConnector = kapton_SQL.DB_Connector("dbConfig.yaml")
        lastId = MyConnector.getLatestId()[0][0]
        print("Currently latest strip id: {}".format(lastId))
        MyConnector.writeDataToDB(self, self.lineEdit_batch.text(), self.lineEdit_comment.text())
        MyConnector.pushScanToDB(self)
        newId = MyConnector.getLatestId()[0][0]
        print("Now latest strip id: {}".format(newId))
        print("Added {} strips to the database.".format(newId-lastId))
    '''    
    def upload(self,files): # needs a file list
        print("analyzer upload: now do upload of files ", files)
        if(not isfile("dbConfig.yaml")):
            subprocess.run(['python', 'DBConfCreator.py'], check=True)
            return
        print("Found dbConfig.yaml")
        MyConnector = kapton_SQL_batch.DB_Connector("dbConfig.yaml")
        try:
            lastId = MyConnector.getLatestId()[0][0]
        except:
            print("exception: lastid array is empty. lastid is therefore set to 0")
            lastId=0
        print("Currently latest strip id: {}".format(lastId))
        if platform.system()=="Linux":
            delim="/"
        else:
            delim="\\"

        for i in range(len(files)):
            f=str(files[i])
            basefolder= (str(f).replace(str(f).split(delim)[-1],""))+"Results"
            csvname = basefolder+delim+(((str(f).split(delim))[-1])[:-4])+".csv"
            MyConnector.writeDataToDB_kaptontype(csvname)
            MyConnector.pushScanToDB(f)
            newId = MyConnector.getLatestId()[0][0]
            print("Now latest strip id: {}".format(newId))
            if i== len(files)-1:
                print("Added {} strips to the database.".format(newId-lastId))


    def doCalibration(self,calibFileName, pPercentile=0.99, pSimpleRange=True):
        print("Start calibration ...")
        print(calibFileName)
        get_RGB_Distributions(calibFileName, True, "calibSprectrum.pdf", self.scan.rel_path)
        # (R_borders, G_borders, B_borders) = get_optimizedRange(calibFileName, pPercentile, False)
        borders = get_optimizedRange(calibFileName, pPercentile, pSimpleRange)
        R_borders = borders[0]
        G_borders = borders[1]
        B_borders = borders[2]
        print()
        print("-----function doCalibration---------------------------------------------------------")
        print()
        print("Calibration borders detected to be\nR: {} -- {}\nG: {} -- {}\nB: {} -- {}".format(R_borders[0], R_borders[1], G_borders[0], G_borders[1], B_borders[0], B_borders[1]))
        print()
        print("------------------------------------------------------------------------------------")
        print()
        self.scan.get_filtered(pFilterR=R_borders, pFilterG=G_borders, pFilterB=B_borders)
        try:
            self.scan.contours, self.scan.hierarchy = self.scan.get_contourandhierarchy()
            self.scan.contoursAvailable = True
        except:
            print("Change percentile in calibration")
            if(pPercentile<=1):
                self.doCalibration(pPercentile+0.002)
        self.scan.kaptonstrips
        #self.fill_table() #stattdessen in File schreiben
        #self.tag_table()

    def openScanfile(self,dpi,fileName,save_figs=True, show_plot=False,kapton_type="long"):
        print('open scanfile: dpi = ', dpi)
        print('open scanfile: fileName = ', fileName)
        self.scan = Scan(fileName,DPI=dpi,kaptonType=kapton_type)
        #self.pushButton_UploadToDatabase.setStyleSheet("background-color : white")
        self.strips = self.scan.kaptonstrips
        self.kapcontours = self.scan.kaptoncontours
        #print('Analyze.openscanfile: self.strips.')

        self.scalingfactors = self.scan.scalingFactors
        doCalib = False
        if(len(self.strips)==0 or len(self.strips)>140): #klappt das überhaupt??? oder einfach raus???
            doCalib = True
            print()
            print("----Function openScanfile-----------------------------------------------------------")
            print("Analysis error: Empty or falsely interpreted image. Start RGB analysis for cut tuning! Need to specify calibration filename first:")
            calibrationImage=input() #need to specify calibration filename in this case
            print("Calinration image is: ",calibrationImage)
            print("Analysis may take a while...")
            R, G, B = self.scan.get_RGB_Distributions(True, calibrationImage)
            calibrationBorders = self.scan.get_extremaFromImage(calibrationImage)
            print("Calibration borders detected to be\nR: {} -- {}\nG: {} -- {}\nB: {} -- {}".format(calibrationBorders[0][0], calibrationBorders[0][1], calibrationBorders[1][0], calibrationBorders[1][1], calibrationBorders[2][0], calibrationBorders[2][1]))
            print()
            print("Now do scan again")
            self.scan = Scan(fileName, DPI=dpi, pFilterR=[calibrationBorders[0][0], calibrationBorders[0][1]], pFilterG=[calibrationBorders[1][0], calibrationBorders[1][1]], pFilterB=[calibrationBorders[2][0], calibrationBorders[2][1]])
            
        self.ScanLoaded = True
 
        self.write_table(fileName)
        if show_plot == True:
            self.show_pic(fileName, save_figs)
        '''
        if(not doCalib):
            self.fill_table()
        self.show_pic()
        if(not doCalib):
            self.tag_table()
        '''

    def show_pic(self, file_name, save_fig=True):

        if platform.system()=="Linux":
            delim="/"
        else:
            delim="\\"
        savename = ((str(file_name).split(delim))[-1])[:-4]
        print("savename ",savename)
        save_folder_base=str(file_name).replace(str(file_name).split(delim)[-1],"")
        save_folder = save_folder_base+"PlotsShape"
        print('save_folder ', save_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

       
        plt.style.use('bmh')

        plt.figure()
        plt.imshow(self.scan.rank_strips(rankingtype='stripnumber'))
        plt.tight_layout()
        if save_fig==True:
            plt.savefig(save_folder+delim+savename+"_labeledStrips.jpg")
        plt.clf()

        #plot sufficient strips
        plt.figure()
        plt.imshow(self.scan.mark_insufficient_strips()) # Set mark_insufficient_strips(False) if you want to set any inner contour > 0 sufficient
        plt.tight_layout()
        if save_fig==True:
            plt.savefig(save_folder+delim+savename+"_sufficientStrips.jpg")
        plt.clf()

        # filtered image
        plt.figure()
        plt.imshow(self.scan.filtered)           # Set mark_insufficient_strips(False) if you want to set any inner contour > 0 sufficient
        plt.tight_layout()
        if save_fig==True:
            plt.savefig(save_folder+delim+savename+"_filteredImage.jpg")
        plt.clf()

        # template filtered image
        plt.figure()
        plt.imshow(self.scan.templateFiltered)           # Set mark_insufficient_strips(False) if you want to set any inner contour > 0 sufficient
        plt.tight_layout()
        if save_fig==True:
            plt.savefig(save_folder+delim+savename+"_templateFilteredImage.jpg")
        plt.clf()

        #translate images from curved contours to save folder with correct naming
        temp_folder_curved = 'temp'
        new_folder_curved = save_folder+delim+'shape'
        if not os.path.exists(new_folder_curved):
            os.makedirs(new_folder_curved)
        if os.listdir(temp_folder_curved):
            for index, tempfile in enumerate(os.listdir(temp_folder_curved)):
                temp_filepath = os.path.join(temp_folder_curved,tempfile)
                #print('show_pic: tempfile ', tempfile)
                #print('self.scan.kaptoncontours ', self.scan.kaptoncontours)
                #strip_nb = self.scan.kaptoncontours[int(tempfile[-5])]+1
                strip_nb = self.scan.kaptoncontours.index(int(tempfile[-6:-4]))+1
                #print('strip_nb ', strip_nb)
                new_curved_path = new_folder_curved+delim+savename+'_'+tempfile[:-5]+'strip'+str(strip_nb)+'.png'
                shutil.move(temp_filepath,new_curved_path)
                print('Moved ',temp_filepath,' to ',new_curved_path)





    def write_table(self,filename): #specify the filename -> it will be used as savename. the ending however (.jpg etc) is replaced by .csv   
        #write a table, print in a nice way and save. print in extra window should be de-activatable
        #it also does the strip grading and writes result in table
        
        if platform.system()=="Linux":
            delim="/"
        else:
            delim="\\"

        savename = ((str(filename).split(delim))[-1])[:-4]+".csv"
        print("savename ",savename)
        save_folder_base=str(filename).replace(str(filename).split(delim)[-1],"")
        save_folder = save_folder_base+"ResultsShape"
        print('save_folder ', save_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        #header: values in brackets indicate limits/requirements
        header=["Strip nb","length and width ok?", "Length / mm", "Width / mm", "area / mm^2", "scaling x / mm/px", "scaling y / mm/px", "fitmatch (>=0.98)"," contour left curved ", "contour right curved "]

        # finding non-sufficient strips: sets passes criteria? to Yes if everything is ok, to No if length/width out of specs and to Questionable if requirements for other fields are not fulfilled, but lengt/width ok

        print("function write_table: Start finding non sufficient strips!")
        if self.strips[0].get_Type()=='long':
            print("Minimum width: {}\nMinimum length: {}".format(kaptonconstants.minimal_kaptonwidth, kaptonconstants.minimal_kaptonlength))
            print("Maximum width: {}\nMaximum length: {}".format(kaptonconstants.maximal_kaptonwidth, kaptonconstants.maximal_kaptonlength))
        else:
            print("Minimum width: {}\nMinimum length: {}".format(kaptonconstants.minimal_stumpkapton_width, kaptonconstants.minimal_stumpkapton_length))
            print("Maximum width: {}\nMaximum length: {}".format(kaptonconstants.maximal_stumpkapton_width, kaptonconstants.maximal_stumpkapton_length))
        with open(save_folder+delim+savename, "w", encoding="UTF8", newline="") as csvfile:
            writer=csv.writer(csvfile)
            writer.writerow(header)
            for i in range(self.scan.get_NumStrips()):
            #for i in self.kapcontours:
                strip = self.strips[i]
                print('write table: strip.contournb ', strip.contour_nb)
                print('write table: strip.position ', strip.position)
                print('write table: strip.dimensions ', strip.dimensions)
                print('write table: strip.edgecurved ', strip.edgecurved)
                #tag strips
                strip_ok="Yes"
                if strip.sufficient == False:
                    strip_ok="No"
                #if strip.get_Type()=='long':
                #    if strip.dimensions[0]<kaptonconstants.minimal_kaptonwidth or strip.dimensions[1]<kaptonconstants.minimal_kaptonlength:
                #        strip_ok="No"
                #else:
                #    if strip.dimensions[0]<kaptonconstants.minimal_stumpkapton_width or strip.dimensions[1]<kaptonconstants.minimal_stumpkapton_length:
                #        strip_ok="No"
                #if strip.innerContour_Area()>0 or strip.fitmatch<0.98 or strip.max_residual>=2 or strip.max_distance>=10 or strip.max_area_score>=1.25:
                #    strip_ok="Questionable"
                
                #write table WHAT TO DO FOR SHORT STRIPS WHERE EDGE CURVED STUFF IS NOT APPLIED? IS IT THEN SET TO -1?
                #row=[i+1, strip_ok, round(strip.dimensions[1],3), round(strip.dimensions[0],3), round(strip.area,3), self.scalingfactors[0], self.scalingfactors[1], round(strip.innerContour_Area(),3) , round(strip.fitmatch,3), strip.max_residual_text, strip.max_distance_text, round(strip.max_area_score,3),strip.edgecurved[0],strip.edgecurved[1]]
                #define strings for curved edges
                str_edge_left="None"
                str_edge_right="None"
                if strip.edgecurved[0] == 0:
                    str_edge_left="No"
                elif strip.edgecurved[0] ==1:
                    str_edge_left="Yes"
                elif strip.edgecurved[0] == 2:
                    str_edge_left="Strange"
                else:
                    str_edge_left="None"
                if strip.edgecurved[1] == 0:
                    str_edge_right="No"
                elif strip.edgecurved[1] ==1:
                    str_edge_right="Yes"
                elif strip.edgecurved[1] == 2:
                    str_edge_right="Strange"
                else:
                    str_edge_right="None"
                
                row=[i+1,strip_ok,round(strip.dimensions[1],3), round(strip.dimensions[0],3), round(strip.area,3), self.scalingfactors[0], self.scalingfactors[1], round(strip.fitmatch,3), str_edge_left,str_edge_right] #new 30.01.25 (Nina)
                writer.writerow(row)
        

            
     
