#!/usr/bin/env python3

#from unittest.loader import _SortComparisonMethod
import paramiko
print('paramiko.__version__ ', paramiko.__version__)
import datetime
import yaml
print('yaml.__version__ ', yaml.__version__)
import numpy as np
#from .. import Kapton_Analyze_Batch
import ntpath
import pandas as pd
print('pd.__version__ ', pd.__version__)
import matplotlib.pyplot as plt
from os.path import join
from pathlib import Path
import platform

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def path_base(path):
	head, tail = ntpath.split(path)
	return head

class DB_Connector:
    """
    This class is used to upload the results to our DB
    The dictionary self.connectConf is loaded from a yaml file holding the attributes: db_username, db_password, db_host, db_port, db_database.
    These are mandatory to connect to a postgres SQL database.
    """
    def __init__(self, pConfigFile, pSkipSQL=False, pSkipSSH=False):
        self.loggedSQL = False
        self.loggedSSH = False
        with open(pConfigFile) as config:
            self.connectConf = yaml.load(config, Loader=yaml.FullLoader)
        if(not pSkipSQL):
            if(self.connectConf["db_type"]=="postgres"):
                import psycopg2 as sql
            else:
                import mysql.connector as sql
            try:
                self.sqlConn = sql.connect(host=self.connectConf["db_host"],
                                                    port=self.connectConf["db_port"],
                                                    dbname=self.connectConf["db_database"],
                                                    user=self.connectConf["db_username"],
                                                    password=self.connectConf["db_password"])
                self.sqlCursor = self.sqlConn.cursor()
                self.loggedSQL = True
            except sql.OperationalError:
                print("Wrong database information in YAML File")
        if(not pSkipSSH):
            self.sshConn = paramiko.SSHClient()
            self.sshConn.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                lPKey = paramiko.Ed25519Key(filename=self.connectConf["ssh_locPKey"])
                self.sshConn.connect(hostname=self.connectConf["ssh_host"],
                    username=self.connectConf["ssh_user"],
                    pkey=lPKey
                    )
                self.sftp = self.sshConn.open_sftp()
                self.loggedSSH = True
            except paramiko.AuthenticationException:
                print("Wrong ssh information in YAML File")

    def __del__(self):
        try:
            self.sqlCursor.close()
            self.sqlConn.close()
        except:
            pass

    def query(self, pQuery):
        if(self.loggedSQL):
            self.sqlCursor.execute(pQuery)
            return self.sqlCursor.fetchall()
        else:
            return
    
    def getLatestId(self):
        return self.query("""SELECT "id" FROM "Tools_kapton_metrology" ORDER BY "id" DESC LIMIT 1""")
    
    
    def writeDataToDB(self,filename,pScanComment): #filename of the csv file with the analysis results. striptype long, short_y, short_x -> if short_y only add length, otherwise only add width ? 
        #updates made to load less stuff in the DB (30.01.25,Nina)
        if platform.system()=="Linux":
            delim="/"
        else:
            delim="\\"
        
        if(self.loggedSQL):
            scanresult = pd.read_csv(filename, header=1, sep=",")
            for i in range(len(scanresult[:,0])):
                print('scanresult[i,0] ', scanresult[i,0])
                data = (scanresult[i,0],
                        ((str(filename).split(delim))[-1])[:-4],
                        datetime.datetime.now(),
                        str(pScanComment),
                        str(scanresult[i,1]), #strip ok (in size)
                        float(scanresult[i,2]), #length
                        float(scanresult[i,3]), #width
                        float(scanresult[i,4]), #area
                        float(scanresult[i,5]), #scaling x
                        float(scanresult[i,6]), #scaling y
                        float(scanresult[i,7]), #fitmatch
                        str(scanresult[i,8]), # result new edge curved check, right edge
                        str(scanresult[i,9]) # result new edge curved check, right edge
                        )
                sqlQuery = 'INSERT into "Tools_kapton_metrology" ("serial_number", "batch_number", "timestamp", "comment", "dimensions ok?", "length [mm]", "width [mm]", "area [mm^2]", "scaling x [mm/px]", "scaling y [mm/px]", "fitmatch",  "curved left edge", "curved right edge") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
                #sqlQuery = 'INSERT into "Tools_kapton_metrology" ("serial_number", "batch_number", "timestamp") VALUES (%s,%s,%s)'
                self.sqlCursor.execute(
                    sqlQuery,
                    data)
                self.sqlConn.commit()
                #counter+=1
        else:
            return

    def writeDataToDB_kaptontype(self,filename): #filename of the csv file with the analysis results. striptype long, short_y, short_x -> if short_y only add length, otherwise only add width ? 
        if platform.system()=="Linux":
            delim="/"
        else:
            delim="\\"
        if(self.loggedSQL):
            scanresult = pd.read_csv(filename, skiprows=1,sep=',',names=['serial_number','dimensions_check_passed','length','width','area','scaling_x','scaling_y','inner_contour_area','fitmatch','max_residual','max_distance1','max_distance2','max_area_score'])
            #scanresult = pd.read_csv(filename, skiprows=0,sep=',')
            print('scanresult:\n', scanresult)
            #print('scanresult.values ', scanresult.values)
            #print('test',scanresult.iat[0,0])
            #print(scanresult[:,0])
            kaptontype=None
            if scanresult.iat[0,2] >90:
            #if scanresult.iat[0,1] >90:
                kaptontype="long"
            elif scanresult.iat[0,2] >12 and scanresult.iat[0,2] < 90:
            #elif scanresult.iat[0,1] >12 and scanresult.iat[0,1] < 90:
                kaptontype="short"
            elif scanresult.iat[0,2] <12 and scanresult.iat[0,3] <90 :
            #elif scanresult.iat[0,1] <12 and scanresult.iat[0,2] <90 :
                kaptontype="short_r" #scan rotated by 90 deg. 
            elif scanresult.iat[0,2] <12 and scanresult.iat[0,3] >90 :
            #elif scanresult.iat[0,1] <12 and scanresult.iat[0,2] >90 :
                kaptontype="long_r" #scan rotated by 90 deg. 
            print(kaptontype)
            for i in range(len(scanresult.index)):
                if kaptontype == "long" or kaptontype=="short":
                    data = (int(scanresult.iat[i,0]),
                            ((str(filename).split(delim))[-1])[:-4],
                            datetime.datetime.now(),
                            str(kaptontype),
                            str(scanresult.iat[i,1]),
                            float(scanresult.iat[i,2]), #length
                            float(scanresult.iat[i,3]), #width
                            float(scanresult.iat[i,4]), #area
                            float(scanresult.iat[i,5]), #scaling x
                            float(scanresult.iat[i,6]), #scaling y
                            float(scanresult.iat[i,7]), #inner contour
                            float(scanresult.iat[i,8]), #fitmatch
                            #str(scanresult.iat[i,9]), #max residual
                            #str(scanresult.iat[i,10]),#max distance
                            #float(scanresult.iat[i,11]),#max area score,
                            ((str(filename).split(delim))[-1])[:-4]+".jpg" #image file name
                            )
                else:
                    data = (scanresult.iat[i,0],
                            ((str(filename).split(delim))[-1])[:-4],
                            datetime.datetime.now(),
                            kaptontype,
                            str(scanresult.iat[i,1]),
                            float(scanresult.iat[i,3]), #length
                            float(scanresult.iat[i,2]), #width
                            float(scanresult.iat[i,4]), #area
                            float(scanresult.iat[i,5]), #scaling x
                            float(scanresult.iat[i,6]), #scaling y
                            float(scanresult.iat[i,7]), #inner contour
                            float(scanresult.iat[i,8]), #fitmatch
                            #str(scanresult.iat[i,9]), #max residual
                            #str(scanresult.iat[i,10]),#max distance
                            #float(scanresult.iat[i,11])#max area score
                            ((str(filename).split(delim))[-1])[:-4]+".jpg" #image file name
                            )
                #sqlQuery = 'INSERT into "Tools_kapton_metrology" ("serial_number", "batch_number", "timestamp", "kapton_type", "dimension_check_passed", "length", "width", "area", "scaling_x", "scaling_y", "inner_contour_area", "fitmatch",  "max_residual", "max_distance", "max_area_score", "image_filename") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
                #sqlQuery = 'INSERT into "Tools_kapton_metrology" ("serial_number", "batch_number", "timestamp") VALUES (%s,%s,%s)'
                sqlQuery = 'INSERT into "Tools_kapton_metrology" ("serial_number", "batch_number", "timestamp", "kapton_type", "dimension_check_passed", "length", "width", "area", "scaling_x", "scaling_y", "inner_contour_area", "fitmatch", "image_filename") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'

                self.sqlCursor.execute(
                    sqlQuery,
                    data)
                self.sqlConn.commit()
                #counter+=1
        else:
            return

    #def pushScanToDB(self, pApplication: KaptonGUI.Ui_MainWindow):  TO DO!!!
    def pushScanToDB(self, filename):
        if platform.system()=="Linux":
            delim="/"
        else:
            delim="\\"
        if(self.loggedSSH):
            self.sftp.put(filename, self.connectConf["ssh_db_imagePath"]+path_leaf(filename))
            self.sftp.put(path_base(filename)+delim+"Plots"+delim+path_leaf(filename)[:-4]+"_labeledStrips.jpg", self.connectConf["ssh_db_imagePath"]+path_leaf(filename)[:-4]+"_label.jpg")
            self.sftp.put(path_base(filename)+delim+"Plots"+delim+path_leaf(filename)[:-4]+"_filteredImage.jpg", self.connectConf["ssh_db_imagePath"]+path_leaf(filename)[:-4]+"_filtered.jpg")
            #self.sftp.put(pApplication.scan.fileName, self.connectConf["ssh_db_imagePath"]+path_leaf(pApplication.scan.fileName))
            #plt.imshow(pApplication.scan.rank_strips(rankingtype='stripnumber'))
            #plt.imshow(filename[:-4]+"_labeledStrips.jpg")
            #plt.tight_layout()
            #plt.savefig(join('.tmpFiles','tmp_img.jpg'), dpi=150)
            #self.sftp.put(join('.tmpGUI','tmp_img.jpg'), join(self.connectConf["ssh_db_imagePath"], "{}_label.jpg".format(Path(pApplication.scan.fileName).stem)))		   		   
        else:
            print('pushscantodb failed :/')
            return
        

