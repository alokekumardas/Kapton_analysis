import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import array
import math
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import Circle

from helper import kaptonconstants
import sys
import copy
from PIL import Image

sdf = 1

# Kapton strips and template must have the same orientation!

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def gauss(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2./(2.*sigma**2.))

def get_RGB_Distributions(pImageFile, doPlot=False, saveFileName="", pSecondImage=""):
    R_arr, G_arr, B_arr = get_RGB_PIL(pImageFile)
    if doPlot:
        plt.plot(range(0,256), R_arr, color='r', alpha=0.5, label='R')
        plt.plot(range(0,256), G_arr, color='g', alpha=0.5, label='G')
        plt.plot(range(0,256), B_arr, color='b', alpha=0.5, label='B')
        if(pSecondImage!=""):
            R_sec, G_sec, B_sec = get_RGB_PIL(pSecondImage)
            plt.plot(range(0,256), R_sec, color='darkred', alpha=0.5, label='R total')
            plt.plot(range(0,256), G_sec, color='darkgreen', alpha=0.5, label='G total')
            plt.plot(range(0,256), B_sec, color='darkblue', alpha=0.5, label='B total')
        plt.xlabel('rgb values')
        plt.ylabel('entries')
        plt.yscale("log")
        plt.legend(loc=0)
        if saveFileName!="":
            plt.tight_layout()
            plt.savefig(saveFileName)
        plt.show()
    return [R_arr, G_arr, B_arr]

def get_RGB_PIL(pImageFile):
    """
        returns RGB arrays as R_arr, G_arr, B_arr
    """
    Image.MAX_IMAGE_PIXELS = None
    image   = Image.open(pImageFile)
    RGB_arr = image.histogram()
    R_arr   = RGB_arr[:256]
    G_arr   = RGB_arr[256:256+256]
    B_arr   = RGB_arr[-256:]
    return R_arr, G_arr, B_arr

def get_extremaFromImage(pImageFile):
    """
        returns extrema as ((R_min, R_max), (G_min, G_max), (B_min, B_max))
    """
    Image.MAX_IMAGE_PIXELS  = None
    image                   = Image.open(pImageFile)
    extrema                 = image.getextrema()
    return extrema

def get_optimizedRange(pImageFile, pPercentile=0.99, pOnlySimpleBorders=True):
    R_arr, G_arr, B_arr = get_RGB_PIL(pImageFile)
    Npixel              = np.sum(R_arr)
    curPointer          = 0
    curSum              = 0.
    curPointerB         = 0
    while(curSum/Npixel<pPercentile):
        curSum      += B_arr[curPointer]
        curPointer  += 1
        if(curSum/Npixel<1-pPercentile):
            curPointerB += 1
    B_pointer       = curPointer
    B_pointer_start = curPointerB

    curPointer  = -1
    curSum      = 0.
    curPointerB = -1
    while(curSum/Npixel<pPercentile):
        curSum      += G_arr[curPointer]
        curPointer  -= 1
        if(curSum/Npixel<1-pPercentile):
            curPointerB -= 1
    G_pointer       = curPointer +256
    G_pointer_end   = curPointerB+256

    curPointer  = -1
    curSum      = 0.
    curPointerB = -1
    while(curSum/Npixel<pPercentile):
        curSum      += R_arr[curPointer]
        curPointer  -= 1
        if(curSum/Npixel<1-pPercentile):
            curPointerB -= 1
    R_pointer       = curPointer +256
    R_pointer_end   = curPointerB+256

    if(pOnlySimpleBorders):
        return [R_pointer, 255], [G_pointer, 255], [0,B_pointer]
    else:
        return [R_pointer, R_pointer_end], [G_pointer, G_pointer_end], [B_pointer_start, B_pointer]



def get_FilterBorders(pColArr):
    lColVal = np.arange(256)
    lColArr = np.array(pColArr)

def grouper(iterable, limit):
    prev = None
    group = []
    for item in iterable:
        if prev is None or item - prev <= limit:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group

class KaptonStrip:
#this class contains the functions for Kaptonstrip analysis
    def __init__(self, pic, cnt_nb):
        self.Scan           = pic
        self.contour_nb     = cnt_nb
        print('\n')
        print('self.contour_nb ', self.contour_nb)
        self.contour        = pic.contours[cnt_nb]
        self.area           = cv2.contourArea(self.contour)
        self.position, self.dimensions, self.rotation = cv2.minAreaRect(self.contour)
        self.dimensions     = (min(self.dimensions),max(self.dimensions))
        if self.dimensions[1] >80.0: #shape determination only for long strips
            self.edgecurved = self.determine_shape() # new 13.1.25 (Nina) to find banana shape strips.
        else:
            self.edgecurved=[-1,-1]
        if(self.dimensions[0]*self.dimensions[1]!=0):
            self.fitmatch = self.area/(self.dimensions[0]*self.dimensions[1])
        else:
            self.fitmatch = 0
        print('self.fitmatch ',self.fitmatch)
        if self.fitmatch!=0:
            self.angle              = self.rotation if(self.dimensions[0]<self.dimensions[1]) else 90.-self.rotation
            self.contour_copy   = copy.deepcopy(self.contour[:,0,:])
            self.Ktype              = self.get_Type()
            self.inner_contours     = self.get_innerContours()
            self.sufficient         = self.check_size()
            self.max_residual       = 0
            self.max_residual_text  = ""
            self.max_distance       = 0
            self.max_distance_text  = ""
            self.OCpassed           = False
            self.max_area_score     = 0
            
            if self.sufficient:
                self.edges = self.determine_edges()
                self.edge_fit()
                self.hull_area_coeff()

            print('self.position ', self.position)
            print('self.dimensions ', self.dimensions)
            print(' ')

    def determine_shape(self):    
        """
        this function checks the shape of the long Kapton strips for curved shape fromthe strip contour. Using 2 points on the long side, one near each end (more precisely: mean of several points in this region),
        a line through these point is put and all other points on this Kapton strip edge are compared with this line. If the distance is too large (see limits below) the points are stored. 
        If there are too many points (see limits below), plots of the contour are made. If the points cluster near the center of the strips (see limits below), strip is labeled as curved.
        """
        #variables for edge shape
        is_curved_edge = [0,0] # left and right edge

        ######## analysis of edge shape #########################################
        # Referenz- Koordinaten oben und unten an den Streifen-Seiten finden
        limit_x=1.0
        limit_y=3.0
        range_x=50
        index_x1=0
        index_x2=0
        index_o=np.zeros(2)
        index_u=np.zeros(2)
        
        for c in range(range_x, len(self.contour)-range_x,1):
            if np.abs(self.contour[c][0][1] - self.contour[c+range_x][0][1]) < limit_x and np.abs(self.contour[c][0][1] - self.contour[c-range_x][0][1]) < limit_x and self.contour[c][0][1] > 90:
                print("c =", c , "self.contour[c][0][0] = ", self.contour[c][0][0], " self.contour[c][0][1] = ", self.contour[c][0][1])
                index_x1=c
                break
        for cn in range(len(self.contour)-range_x-1,range_x,-1):
            if np.abs(self.contour[cn][0][1] - self.contour[cn+range_x][0][1]) < limit_x and np.abs(self.contour[cn][0][1] - self.contour[cn-range_x][0][1]) < limit_x and self.contour[cn][0][1] < 90:
                print("cn =", cn , "self.contour[cn][0][0] = ", self.contour[cn][0][0], " self.contour[cn][0][1] = ", self.contour[cn][0][1])
                index_x2=cn
                break
        for ind in range(0,len(self.contour),1):
            if (self.contour[index_x1][0][1] - self.contour[ind][0][1]) >limit_y:
                if ind!=0 and (self.contour[index_x1][0][1] - self.contour[ind-1][0][1]) <= limit_y:
                    index_o[0]=ind
                elif ind!=len(self.contour)-1 and (self.contour[index_x1][0][1] - self.contour[ind+1][0][1])<=limit_y:
                    index_o[1]=ind
            if (self.contour[ind][0][1] - self.contour[index_x2][0][1]) >limit_y:
                if ind!=len(self.contour)-1 and (self.contour[ind+1][0][1] - self.contour[index_x2][0][1]) <= limit_y:
                    index_u[0]=ind
                elif ind!=0 and (self.contour[ind-1][0][1] - self.contour[index_x2][0][1]) <= limit_y:
                    index_u[1]=ind
                
        print("index_o", index_o, " index_u ", index_u)
        if self.contour[int(index_o[0])][0][0] < self.contour[int(index_o[1])][0][0]:
            index_o1=int(index_o[0])
            index_o2=int(index_o[1])
        else:
            index_o1=int(index_o[1])
            index_o2=int(index_o[0])
        if self.contour[int(index_u[0])][0][0] < self.contour[int(index_u[1])][0][0]:
            index_u1=int(index_u[0])
            index_u2=int(index_u[1])
        else:
            index_u1=int(index_u[1])
            index_u2=int(index_u[0])
        
        ######### calculate the straight lines
        #determine the reference points
        xo1 =0.0;xo2=0.0;xu1=0.0;xu2=0.0
        yo1 =0.0;yo2=0.0;yu1=0.0;yu2=0.0
        Np_dir=5
        for o1 in range(max(0,index_o1-Np_dir),min(index_o1+Np_dir+1,len(self.contour))):
            xo1+=self.contour[o1][0][0]/len(range(max(0,index_o1-Np_dir),min(index_o1+Np_dir+1,len(self.contour))))
            yo1+=self.contour[o1][0][1]/len(range(max(0,index_o1-Np_dir),min(index_o1+Np_dir+1,len(self.contour))))
        for o2 in range(max(0,index_o2-Np_dir),min(index_o2+Np_dir+1,len(self.contour))):
            xo2+=self.contour[o2][0][0]/len( range(max(0,index_o2-Np_dir),min(index_o2+Np_dir+1,len(self.contour))))
            yo2+=self.contour[o2][0][1]/len( range(max(0,index_o2-Np_dir),min(index_o2+Np_dir+1,len(self.contour))))
        for u1 in range(max(0,index_u1-Np_dir),min(index_u1+Np_dir+1,len(self.contour))):
            xu1+=self.contour[u1][0][0]/len(range(max(0,index_u1-Np_dir),min(index_u1+Np_dir+1,len(self.contour))))
            yu1+=self.contour[u1][0][1]/len(range(max(0,index_u1-Np_dir),min(index_u1+Np_dir+1,len(self.contour))))
        for u2 in range(max(0,index_u2-Np_dir),min(index_u2+Np_dir+1,len(self.contour))):
            xu2+=self.contour[u2][0][0]/len( range(max(0,index_u2-Np_dir),min(index_u2+Np_dir+1,len(self.contour))))
            yu2+=self.contour[u2][0][1]/len( range(max(0,index_u2-Np_dir),min(index_u2+Np_dir+1,len(self.contour))))
        
        print('xo1 ', xo1, 'yo1 ', yo1)
        print('xo2 ', xo2, 'yo2 ', yo2)
        print('xu1 ', xu1, 'yu1 ', yu1)
        print('xu2 ', xu2, 'yu2 ', yu2)

        # left line
        ml = (xo1-xu1)/(yo1-yu1)
        x0l= xu1-ml*yu1
        #right line
        mr = (xo2-xu2)/(yo2-yu2)
        x0r= xu2-mr*yu2
        #print('ml ',ml, ' x0l ',x0l)
        #print('mr ',mr, ' x0r ',x0r)

        ######################################################################################
        #limits for the deviations to detect curved contours
        lim_dist_max = 0.1 #mm max. allowed distance to straight line /abso
        min_px_above_lim=0.05 #relative
        std_y_lim=0.1 #relative
        diffMeanCenter_lim = 0.2 #relative
        ######################################################################################

        #find the distances of each contour point at the long sides relative to the straight lines
        distances_l = []
        distances_r = []
        x_l = []
        x_r = []
        y_l = []
        y_r = []
        distances_l_lim=[] #distances above the set limit
        distances_r_lim=[] #distances above the set limit
        x_l_lim=[] # the corresponding x values
        x_r_lim=[] # the corresponding x values
        y_l_lim=[] # the corresponding y values
        y_r_lim=[] # the corresponding y values

        for il in range(min(index_u1,index_o1),max(index_u1,index_o1)):
            #dl = self.contour[il][0][0] - (self.contour[il][0][1]-y0l)/ml
            dl = self.contour[il][0][0] - (ml*self.contour[il][0][1]+x0l)
            distances_l.append(dl)
            x_l.append(self.contour[il][0][0])
            y_l.append(self.contour[il][0][1])
            if np.abs(dl) >= lim_dist_max:
                distances_l_lim.append(dl)
                x_l_lim.append(self.contour[il][0][0])
                y_l_lim.append(self.contour[il][0][1])
        
        for ir in range(min(index_u2,index_o2),max(index_u2,index_o2)):
            #dr = self.contour[ir][0][0] - (self.contour[ir][0][1]-y0r)/mr
            dr = self.contour[ir][0][0] - (mr*self.contour[ir][0][1]+x0r)
            distances_r.append(dr)
            x_r.append(self.contour[ir][0][0])
            y_r.append(self.contour[ir][0][1])
            if np.abs(dr) >= lim_dist_max:
                distances_r_lim.append(dr)
                x_r_lim.append(self.contour[ir][0][0])
                y_r_lim.append(self.contour[ir][0][1])
        print('left:  max distance ', np.max(np.array(distances_l)),' min distance ',np.min(np.array(distances_l)))
        print('right:  max distance ', np.max(np.array(distances_r)),' min distance ',np.min(np.array(distances_r)))
        print(' ')
        print('now determine shape of Kapton strip')
        print('len(distances_l_lim)/len(distances_l) ', len(distances_l_lim)/len(distances_l))
        print('len(distances_r_lim)/len(distances_r) ', len(distances_r_lim)/len(distances_r))

        if len(distances_l_lim)/len(distances_l) >= min_px_above_lim or len(distances_r_lim)/len(distances_r) >= min_px_above_lim:
            if len(distances_l_lim)/len(distances_l) >= min_px_above_lim:# or len(distances_r_lim)/len(distances_r) >= min_px_above_lim:
                print('More than a fraction of ',min_px_above_lim,' outliers found for the left side. Now look if this is because of banana or pillow shape or sth else')
                #check if the outliers cluster somewhere, and if this is near the center of the strip
                mean_y_l_lim = np.mean(np.array(y_l_lim))
                print(' mean_y_l_lim ', mean_y_l_lim)
                groups_y_l_lim = dict(enumerate(grouper(y_l_lim,2), 1))

                if len(groups_y_l_lim) == 1:
                    if np.abs(mean_y_l_lim - np.mean(np.array(y_l)))/np.mean(np.array(y_l))< diffMeanCenter_lim:
                        print('Curved left contour detected')
                        plot_title_l='Curved left contour detected'
                        is_curved_edge[0] = 1
                    else:
                        print('Left contour : Outliers clustered but not centered. So not identified as curved outline.')
                        plot_title_l='Left contour outliers clustered but not centered'
                        is_curved_edge[0] = 2
                else:
                    print('left contour: no clear clustering. number of groups is  ', len(groups_y_l_lim))
                    plot_title_l='Left contour: no clear clustering'
                    is_curved_edge[0] = 2

                
                #plot
                plt.figure('left1')
                plt.plot(x_l,y_l,'b.',label='contour left')
                plt.plot(x_l_lim,y_l_lim,'r.',label='points above limit')
                plt.plot(np.array(y_l,dtype=np.float64)*ml+x0l,y_l,'c-',label='left line')
                plt.title(plot_title_l)
                plt.xlabel('x coordinate / mm')
                plt.ylabel('y coordinate / mm')
                plt.legend()
                if self.contour_nb < 10:
                    plt.savefig('temp'+'\\'+'curvature_l_c_0'+str(self.contour_nb)+'.png')
                else:
                    plt.savefig('temp'+'\\'+'curvature_l_c_'+str(self.contour_nb)+'.png')
                plt.clf()
                plt.close()

                plt.figure('left2')
                plt.plot(x_l,distances_l,'b.',label='distances left')
                plt.xlabel('x coordinate / mm')
                plt.ylabel('Distance to line / mm')
                plt.title(plot_title_l)
                plt.legend()
                if self.contour_nb < 10:
                    plt.savefig('temp'+'\\'+'curvature_l_d_0'+str(self.contour_nb)+'.png')
                else:
                    plt.savefig('temp'+'\\'+'curvature_l_d_'+str(self.contour_nb)+'.png')
                plt.clf()
                plt.close()

                plt.figure('contour')
                for icl in range(len(self.contour)):
                    plt.plot(self.contour[icl][0][0],self.contour[icl][0][1],'.')
                plt.plot(xo1,yo1,'bd',label='o1')
                plt.plot(xo2,yo2,'bH',label='o2')
                plt.plot(xu1,yu1,'rd',label='u1')
                plt.plot(xu2,yu2,'rH',label='u2')
                plt.xlabel('x / mm')
                plt.ylabel('y / mm')
                plt.title('Full contour: '+plot_title_l)
                plt.legend()
                if self.contour_nb < 10:
                    plt.savefig('temp'+'\\'+'fullcontour_l_0'+str(self.contour_nb)+'.png')
                else:
                    plt.savefig('temp'+'\\'+'fullcontour_l_'+str(self.contour_nb)+'.png')
                plt.clf()
                plt.close()
            
            if len(distances_r_lim)/len(distances_r) >= min_px_above_lim:
                print('More than a fraction of ',min_px_above_lim,' outliers found for the right side. Now look if this is because of banana or pillow shape or sth else')
                mean_y_r_lim = np.mean(np.array(y_r_lim))
                print(' mean_y_r_lim ',mean_y_r_lim)
                groups_y_r_lim = dict(enumerate(grouper(y_r_lim,2), 1))
                if len(groups_y_r_lim) == 1:
                    if np.abs(mean_y_r_lim - np.mean(np.array(y_r)))/np.mean(np.array(y_r))< diffMeanCenter_lim:
                        print('Curved right contour detected')
                        plot_title_r='Curved right contour detected'
                        is_curved_edge[1] = 1
                    else:
                        print('Right contour: Outliers clustered but not centered. So not identified as curved outline.')
                        plot_title_r='Right contour outliers clustered but not centered'
                        is_curved_edge[1] = 2
                else:
                    print('right contour: no clear clustering. number of groups is  ', len(groups_y_r_lim)) 
                    plot_title_r='Right contour: no clear clustering'  
                    is_curved_edge[1] = 2
                
                
                #plot
                plt.figure('right1')
                plt.plot(x_r,y_r,'b.',label='contour right')
                plt.plot(x_r_lim,y_r_lim,'r.',label='points above limit')
                plt.plot(np.array(y_r,dtype=np.float64)*mr+x0r,y_r,'c-',label='right line')
                plt.xlabel('x coordinate / mm')
                plt.ylabel('y coordinate / mm')
                plt.title(plot_title_r)
                plt.legend()
                if self.contour_nb<10:
                    plt.savefig('temp'+'\\'+'curvature_r_c_0'+str(self.contour_nb)+'.png')
                else:
                    plt.savefig('temp'+'\\'+'curvature_r_c_'+str(self.contour_nb)+'.png')
                plt.clf()
                plt.close()

                plt.figure('right2')
                plt.plot(x_r,distances_r,'b.',label='distances right')
                plt.xlabel('x coordinate / mm')
                plt.ylabel('Distance to line / mm')
                plt.title(plot_title_r)
                plt.legend()
                if self.contour_nb<10:
                    plt.savefig('temp'+'\\'+'curvature_r_d_0'+str(self.contour_nb)+'.png')
                else:
                    plt.savefig('temp'+'\\'+'curvature_r_d_'+str(self.contour_nb)+'.png')
                plt.clf()
                plt.close()

                plt.figure('contour')
                for icr in range(len(self.contour)):
                    plt.plot(self.contour[icr][0][0],self.contour[icr][0][1],'.')
                plt.plot(xo1,yo1,'bd',label='o1')
                plt.plot(xo2,yo2,'bH',label='o2')
                plt.plot(xu1,yu1,'rd',label='u1')
                plt.plot(xu2,yu2,'rH',label='u2')
                plt.xlabel('x / mm')
                plt.ylabel('y / mm')
                plt.title('Full contour: '+plot_title_r)
                plt.legend()
                if self.contour_nb < 10:
                    plt.savefig('temp'+'\\'+'fullcontour_r_0'+str(self.contour_nb)+'.png')
                else:
                    plt.savefig('temp'+'\\'+'fullcontour_r_'+str(self.contour_nb)+'.png')
                plt.clf()
                plt.close()
            #plt.show()
        else:
            print('Shape: Not enough outlier pixels to determine a shape. If anything, probably just dust or sth... ')
        print('is_curved_edge = ', is_curved_edge)
        return is_curved_edge
        


    def determine_edges(self):
        """
        function discerns to which edge of the strip points belong.

        1. Calculate corner points of the fitted rectangle
        2. Calculate angles of the vectors pointing to the corners from the center of the strip
        3. If a point lies within the angle region between two corners, it is assumed to belong to the given edge

        returns array of edges containing points
        """
        cont = self.contour_copy
        def angle_between(p1, p2):
            """
            returns angle between two points
            """
            return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

        points              = self.contour_copy
        center, size, angle = self.position, self.dimensions, self.rotation
        height, width       = size
        #print('determine_edges: height,width ', height,' ',width)

        rect_angle_rad  = np.radians(angle)
        rect_cos        = np.cos(rect_angle_rad)
        rect_sin        = np.sin(rect_angle_rad)

        half_width  = width / 2
        half_height = height / 2

        center_x, center_y = center

        corners = np.array([
            [center_x + half_width * rect_cos - half_height * rect_sin, center_y + half_width * rect_sin + half_height * rect_cos],
            [center_x - half_width * rect_cos - half_height * rect_sin, center_y - half_width * rect_sin + half_height * rect_cos],
            [center_x - half_width * rect_cos + half_height * rect_sin, center_y - half_width * rect_sin - half_height * rect_cos],
            [center_x + half_width * rect_cos + half_height * rect_sin, center_y + half_width * rect_sin - half_height * rect_cos]
        ])


        corner_angles = np.array([angle_between(corner, center) for corner in corners])
        corner_angles.sort()

        sides = [[] for _ in range(4)]

        point_angles = np.array([angle_between(point, center) for point in points])

        for i in range(4):
            if (corner_angles[i] < corner_angles[(i + 1) % 4]):
                condition = np.where(np.logical_and(point_angles >= corner_angles[i], point_angles<corner_angles[(i + 1) % 4]))
            else:
                condition = np.where(np.logical_and(point_angles >= corner_angles[i], point_angles<(corner_angles[(i + 1) % 4]+2*np.pi)))

            sides[i].extend(points[condition])
            sides[i] = np.array(sides[i])
        #print('sides ',sides)
        return sides

    def edge_fit(self):
        """
        fits a linear function to each edge and determines the greatest residual and distances of points to the function. Can also be used for shape check, see Tim Ziemons phd thesis.

        1. Fit each edge and save the residuals to a list
        2. Determine which edge has the greatest residual
        3. Determine which side has the greatest distance (outlier)
        """

        residuals_list          = np.empty(4)
        max_distances           = np.empty(4)
        self.residual_points    = list()
        #plt.figure(figsize=(12,12))
        #print('edges ',self.edges)
        for id_edge, edge in enumerate(self.edges):
            if edge.size>1:
                #plt.plot(edge[:,0],edge[:,1],'.',label='id '+str(id_edge))
                #print('id_edge ', id_edge, ' edge[:,0] ', edge[:,0], ' edge[:,1] ', edge[:,1]) #Nina 2.12.24
                #if id_edge%2!=0:
                #    edge[:,0], edge[:,1] = edge[:,1], edge[:,0]
                #print('id_edge ', id_edge, ' edge[:,0] ', edge[:,0], ' edge[:,1] ', edge[:,1]) #Nina 2.12.24
                if id_edge%2==0:
                    p, residuals, rank, singular_values, rcond = np.polyfit(edge[:,0], edge[:,1], 1,rcond=None, full=True, w=None, cov=False)
                else:
                    p, residuals, rank, singular_values, rcond = np.polyfit(edge[:,1], edge[:,0], 1,rcond=None, full=True, w=None, cov=False)
               # print('p, residuals, rank, singular_values, rcond: ', p, residuals, rank, singular_values, rcond)
                
                #plt.plot(edge[:,0],edge[:,1],'.',label='id '+str(id_edge))
                
                edge_max_distance       = np.max(np.abs((edge[:,0]*p[0]+p[1])-edge[:,1]))
                max_distances[id_edge]  = edge_max_distance

                residuals_list[id_edge] = residuals/(edge[:,0].size-1)
                self.residual_points.append(np.array([edge[:,0],edge[:,1]-p[0]*edge[:,0]-p[1]]))
            else:
                residuals_list[id_edge] = np.inf
                self.residual_points.append(np.array([0,0]))
        residuals_list = np.array(residuals_list)
        #plt.legend()
        #plt.show()

        if np.argmax(residuals_list) == 0 and np.max(residuals_list) < np.inf:
            self.max_residual_text  = "Upper: " + str(round(np.max(residuals_list),5))
            self.max_residual       = np.max(residuals_list)
        elif np.argmax(residuals_list) == 2 and np.max(residuals_list) < np.inf:
            self.max_residual_text  = "Lower: " + str(round(np.max(residuals_list),1))
            self.max_residual       = np.max(residuals_list)
        else:
            self.max_residual_text  = "Short side: " + str(round(np.max(residuals_list),1))
            self.max_residual       = np.max(residuals_list)
        if np.argmax(max_distances) == 0:
            self.max_distance_text  = "Upper: " + str(round(np.max(max_distances),1))
            self.max_distance       = np.max(max_distances)
            self.greatest_dist_edge = 0
        elif np.argmax(max_distances) == 2:
            self.max_distance_text  = "Lower: " + str(round(np.max(max_distances),1))
            self.max_distance       = np.max(max_distances)
            self.greatest_dist_edge = 2
        else:
            self.max_distance_text = "Short side: " + str(round(np.max(max_distances),1))
            self.max_distance = np.max(max_distances)
            if max_distances[0]> max_distances[1]:
                self.greatest_dist_edge = 0
            else:
                self.greatest_dist_edge = 2

    def hull_area_coeff(self):

        """
        function determines convex hull area of each edge and assigns a score
        maximum score is passed kapton strip as attribute
        NOT OPTIMIZED, DO NOT USE
        """
        def filter_outliers(edge):
            """
            Using a clustering algorithm, outliers are filtered out.
            Outliers the software detects are usually due to dust particles.
            Small edge imperfections like nicks or "tails" are taken care of in optical control

            returns arrays of points that do and do not belong to the cluster
            """

            def compress_input(in_arr):
                """
                accounting for digitization
                Distance between individual points no the same in all directions.
                Therefore the scan in compressed in one direction.
                """
                out_arr = copy.deepcopy(in_arr)
                clenching_factor = 1/(np.max(out_arr[:,0])-np.min(out_arr[:,0]))
                mean = np.mean(out_arr[:,0])
                out_arr[:,0] = (out_arr[:,0]-mean)*clenching_factor
                return out_arr

            db          = DBSCAN(eps=100, min_samples=int(edge.size*0.2)).fit(edge)#(compress_input(edge))
            labels      = db.labels_
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_    = list(labels).count(-1)
            #print("Estimated number of clusters: %d" % n_clusters_)
            #print("Estimated number of noise points: %d" % n_noise_)
            cluster         = edge[np.where(np.heaviside(labels, 1).astype(int))]
            not_cluster     = edge[np.where(np.logical_not(np.heaviside(labels, 1)).astype(int))]
            return cluster, not_cluster

        def PolyArea(x,y):
            """
            returns area of given polygon using shoelace formula
            """
            area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
            return area

        hulls = list()
        areas = list()
        area_scores = list()
        for id_edge, edge in enumerate(self.residual_points):
            if edge.shape[1] != 2:
                tmpedge = np.transpose(edge)
            else:
                tmpedge = edge
            edge_, edge_issues = filter_outliers(tmpedge)
            if np.max(edge[:,0])-np.min(edge[:,0]) > 15:
                hull = cv2.convexHull(edge_.astype(np.float32), False)
            else:
                hull = None
                
            hulls.append(hull)

            if self.Ktype =="long":
                height = kaptonconstants.nominal_kaptonwidth
                length = kaptonconstants.nominal_kaptonlength
            else:
                height = kaptonconstants.nominal_stumpkaptonwidth
                length = kaptonconstants.nominal_stumpkaptonlength

            scoring_constant = (25.4/self.Scan.DPI) * 2

            if hull is None:
                areas.append(0)
                area_scores.append(0)
            else:
                areas.append(PolyArea(hull[:,:,0][:,0],hull[:,:,1][:,0]))
                area_scores.append(areas[-1]/(length * scoring_constant))
                
                """
                mean_xpos   = np.mean(edge_[0])
                mean_ypos   = np.mean(edge_[1])
                xticks      = [(mean_xpos-length/2)+(j*(length/5)) for j in range(6)]
                yticks      = [(mean_ypos-height/2)+(k*(height/3)) for k in range(4)]
                plt.clf()
                
                if np.max(hull[:,:,0])-np.min(hull[:,:,0])>100:
                    plt.xlim(np.min(hull[:,:,0])-length/5, np.max(hull[:,:,0])+length/5)
                else:
                    plt.xlim(np.min(hull[:,:,0])-100, np.max(hull[:,:,0])+100)
                if np.max(hull[:,:,1])-np.min(hull[:,:,1])>10:
                    plt.ylim(np.min(hull[:,:,1])-2, np.max(hull[:,:,1])+2)
                else:
                    plt.ylim(np.min(hull[:,:,1]-3), np.max(hull[:,:,1])+3)
                
                print(50*"*")
                print(edge_issues.shape)
                plt.scatter(edge_[:,0], edge_[:,1], label="cluster")
                plt.scatter(edge_issues[:,0], edge_issues[:,1], label="not cluster")
                plt.legend()
                plt.grid()
                plt.title("Area = "+str(round(areas[id_edge],1)))
                plt.xlabel("x-position")
                plt.ylabel("residue")
                plt.fill(hull[:,:,0],hull[:,:,1], edgecolor='r', fill=False)
                print("saving "+r"D:\KaptonMeasurement\KaptonControl\1200_Long\HullDebug\hull"+str(sdf)+"_"+str(id_edge)+"_"+str(float(area_scores[id_edge]))+".png")
                plt.savefig(r"D:\KaptonMeasurement\KaptonControl\1200_Long\HullDebug\hull"+str(sdf)+"_"+str(id_edge)+"_"+str(float(area_scores[id_edge]))+".png")
                plt.close()
                sdf+=1
                """    
        self.max_area_score = np.max(area_scores)


    def show_KaptonStrip(self):
        plt.clf()
        tmp_copy = cv2.imread(self.Scan.rel_path, cv2.IMREAD_COLOR)
        tmp_copy = cv2.cvtColor(tmp_copy, cv2.COLOR_BGR2RGB)
        cv2.drawContours(tmp_copy, [self.contour], 0, (255,0,0), 7)
        plt.imshow(tmp_copy)
        plt.show()
        return True

    def get_CalculatedArea(self):
        return self.dimensions[0]*self.dimensions[1]

    def check_size(self):
        if(self.Ktype =='long'):
            return ((self.dimensions[0]>kaptonconstants.minimal_kaptonwidth)
                    and (self.dimensions[1]>kaptonconstants.minimal_kaptonlength)
                    and (self.dimensions[0]<kaptonconstants.maximal_kaptonwidth)
                    and (self.dimensions[1]<kaptonconstants.maximal_kaptonlength))
        elif(self.Ktype =='short'):
            return ((self.dimensions[0]>kaptonconstants.minimal_stumpkapton_width)
                    and (self.dimensions[1]>kaptonconstants.minimal_stumpkapton_length)
                    and (self.dimensions[0]<kaptonconstants.maximal_stumpkapton_width)
                    and (self.dimensions[1]<kaptonconstants.maximal_stumpkapton_length))

    def get_innerContours(self):
        # NOT OPTIMIZED
        inner_contours = []
        for i in range(len(self.Scan.hierarchy)):
            if(self.Scan.hierarchy[i][3]==self.contour_nb):
                inner_contours.append(i)
        return inner_contours

    def innerContour_Area(self):
        # NOT OPTIMIZED
        iC_area = 0.
        for ic in self.inner_contours:
            iC_area += cv2.contourArea(self.Scan.contours[ic])*self.Scan.scale**2
        return iC_area

    def get_Type(self):
        #check Kapton strip type
        Ktype = ''
        if(self.dimensions[1]<50.):
            Ktype = 'short'
        else:
            Ktype = 'long'
        return Ktype


class Scan:

    def __init__(self, Scanfilename, DPI=600, pFilterR = kaptonconstants.defaultFilter_R, pFilterG = kaptonconstants.defaultFilter_G, pFilterB = kaptonconstants.defaultFilter_B,kaptonType="long"):
        """
        Takes Scan image
        Applies Filters to it. Filters give output 255 if color of pixel is in range, otherwise 0
        Contours and hierarchy: contours are outermost pixels of the shapes marked with 255 by the filter. If there are pixels inside shape, that do not correpsond to the color range, so called innerContours will be found.
        """
        print("Open "+Scanfilename+" for Kapton strip analysis with DPI setting "+str(DPI)+"...")
        self.fileName   = Scanfilename
        self.img        = cv2.imread(Scanfilename, cv2.IMREAD_COLOR)
        print("Convert coloring...")
        self.img        = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.imgShape   = [self.img.shape[1], self.img.shape[0]]  # [width, height] in pixels
        self.DPI        = DPI
        self.scale      = 25.4/DPI
        self.scalingFactors = np.array([0.02107728337236534, 0.021168824476098913])
        self.contoursAvailable = False
        print("Filter photo...")
        self.imgcopy            = copy.deepcopy(self.img)
        self.filtered           = self.get_filtered(pFilterR, pFilterG, pFilterB)
        self.yTemplateFiltered  = cv2.inRange(self.imgcopy, np.array([kaptonconstants.yTemplateFilter_R[0], kaptonconstants.yTemplateFilter_G[0], kaptonconstants.yTemplateFilter_B[0]]), np.array([kaptonconstants.yTemplateFilter_R[1], kaptonconstants.yTemplateFilter_G[1], kaptonconstants.yTemplateFilter_B[1]]))
        self.xTemplateFiltered  = cv2.inRange(self.imgcopy, np.array([kaptonconstants.xTemplateFilter_R[0], kaptonconstants.xTemplateFilter_G[0], kaptonconstants.xTemplateFilter_B[0]]), np.array([kaptonconstants.xTemplateFilter_R[1], kaptonconstants.xTemplateFilter_G[1], kaptonconstants.xTemplateFilter_B[1]]))
        self.templateFiltered   = np.concatenate((self.xTemplateFiltered[:1500],self.yTemplateFiltered[1500:7500], self.xTemplateFiltered[7500:]), axis=0)
        print("Get contours...")
        #try:
        self.contours, self.hierarchy, self.templateContours, self.templateHierarchy = self.get_contourandhierarchy()
        self.contours, self.hierarchy = self.filtered_contours()
        print("Found Contours and Hierarchy!")
        self.templateContoursAvailable  = (self.templateContours != False)
        self.contoursAvailable          = True
        #except:
        #    print("Contours could not be fitted. Get RGB distributions now.")
        #    self.get_RGB_Distributions(True)
        #    self.contoursAvailable          = False
        #    self.templateContoursAvailable  = False
        
        if self.templateContoursAvailable:
            print("joined routine!")
            foundTemplateInCountours = False
            n_templates_found = 0
            self.template_parameters = list()

            
            for i in range(len(self.templateContours)):
                if((self.templateHierarchy[i][3]==-1) and (cv2.contourArea(self.templateContours[i])>900000)):
                    templatePosition_px, templateDimensions_px, templateRotation = cv2.minAreaRect(self.templateContours[i])
                    templateDimensions_px = (max(templateDimensions_px), min(templateDimensions_px))
                    self.template_parameters.append([templatePosition_px, templateDimensions_px, templateRotation])
                    n_templates_found+=1
                if n_templates_found==4:
                    foundTemplateInCountours = True
                    break
            if(foundTemplateInCountours):
                self.scalingFactors = self.get_corrected_scalingFactor()
                print("Templates found! Applying scaling factors {} along x and {} along y.".format(self.scalingFactors[0], self.scalingFactors[1]))
            else:
                print("Did not find the template! Default scaling factor of 0.02116667 mm/dot is applied")
            
            self.contours = [arr.astype(np.float32) * self.scalingFactors.astype(np.float32) for arr in self.contours]
        


        self.rel_path = Scanfilename
        if(self.contoursAvailable):
            self.kaptoncontours, self.kaptonstrips = self.get_Kaptoncontour_indices(kapton_type=kaptonType)
            print("Found {} strips...".format(len(self.kaptoncontours)))

            #self.circles = self.get_issue_circles()
            #self.show_single_residuals()
        
        #self.reference_strip_index = self.identify_reference()

            
    def PolyArea(self,x,y):
            """
            returns area of given polygon using shoelace formula
            """
            area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
            return area

    def filtered_contours(self):
        # Used to filter out contours, that are too small to be strips. NECESSARY, otherwise analysis takes too much time

        contours = []
        hierarchy = []
        for i, contour in enumerate(self.contours):
            if contour.size > 1000:
                contours.append(contour)
                hierarchy.append(self.hierarchy[i])

        return contours, hierarchy
    

    def template_order(self):
        #Uses Predefined position, which is usual position of templates to determine which template is which. Distance to the usual position has to be less than 1000

        """pre 27.03.24
        initial_position1 = (712, 3696)
        initial_position2 = (4556, 855)
        initial_position3 = (4598, 6856)
        initial_position4 = (7871, 3886)
        """
        initial_position1 = (5377, 435)
        initial_position3 = (5049, 8121)
        initial_position4 = (9153, 4393)
        initial_position5 = (513, 4745)


        def distBetween(p1, p2):
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        

        for i in range(4):
            """pre 27.03.24
            if distBetween(initial_position1, self.template_parameters[i][0]) < 1000:
                first = i
            elif distBetween(initial_position2, self.template_parameters[i][0]) < 1000:
                second = i
            elif distBetween(initial_position3, self.template_parameters[i][0]) < 1000:
                third = i
            elif distBetween(initial_position4, self.template_parameters[i][0]) < 1000:
                forth = i
            """
            if distBetween(initial_position1, self.template_parameters[i][0]) < 1000:
                first = i
            elif distBetween(initial_position5, self.template_parameters[i][0]) < 1000:
                fifth = i
            elif distBetween(initial_position3, self.template_parameters[i][0]) < 1000:
                third = i
            elif distBetween(initial_position4, self.template_parameters[i][0]) < 1000:
                forth = i

        indices = (first, third, forth, fifth)

        return indices


    def get_corrected_scalingFactor(self):
        # Calculate Scaling factors for scan using templates
        scalingFactors1 = np.array(kaptonconstants.templateDimensions1)/self.template_parameters[self.template_order()[0]][1]
        scalingFactors3 = np.array(kaptonconstants.templateDimensions3)/self.template_parameters[self.template_order()[1]][1]
        scalingFactors4 = np.array(kaptonconstants.templateDimensions4)/self.template_parameters[self.template_order()[2]][1]
        scalingFactors5 = np.array(kaptonconstants.templateDimensions5)/self.template_parameters[self.template_order()[3]][1]

        print(str(kaptonconstants.templateDimensions1)+" : "+str(self.template_parameters[self.template_order()[0]][1])+" = "+str(scalingFactors1))
        print(str(kaptonconstants.templateDimensions3)+" : "+str(self.template_parameters[self.template_order()[1]][1])+" = "+str(scalingFactors3))
        print(str(kaptonconstants.templateDimensions4)+" : "+str(self.template_parameters[self.template_order()[2]][1])+" = "+str(scalingFactors4))
        print(str(kaptonconstants.templateDimensions5)+" : "+str(self.template_parameters[self.template_order()[3]][1])+" = "+str(scalingFactors5))

        scaling_x = (scalingFactors3[0])#+scalingFactors1[0])/2
        scaling_y = (scalingFactors5[0]+scalingFactors4[0])/2

        scalingFactors = np.array([scaling_x, scaling_y])
        

        return scalingFactors
       


    def split_gradients_into_lines(self):
        pass

    def change_DPI(self, act_DPI):
        self.DPI = act_DPI
        self.scale = 25.4/act_DPI
        return True

    def get_Size(self):
        return np.shape(self.img)

    def get_filtered(self, pFilterR = kaptonconstants.defaultFilter_R, pFilterG = kaptonconstants.defaultFilter_G, pFilterB = kaptonconstants.defaultFilter_B):
        filtered = cv2.inRange(self.img, np.array([pFilterR[0], pFilterG[0], pFilterB[0]]), np.array([pFilterR[1], pFilterG[1], pFilterB[1]]))
        return filtered

    def get_RGB_Distributions(self, doPlot=False, saveFileName=""):
        R_arr, G_arr, B_arr = get_RGB_PIL(self.fileName)
        if doPlot:
            plt.plot(range(0,256), R_arr, color='r', alpha=0.5, label='R')
            plt.plot(range(0,256), G_arr, color='g', alpha=0.5, label='G')
            plt.plot(range(0,256), B_arr, color='b', alpha=0.5, label='B')
            plt.xlabel('rgb values')
            plt.ylabel('entries')
            plt.yscale("log")
            plt.legend(loc=0)
            if saveFileName!="":
                plt.savefig(saveFileName)
            plt.show()
        return [R_arr, G_arr, B_arr]


    def get_contourandhierarchy(self):
        contours, hierarchy = cv2.findContours(self.filtered, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        try:
            templateContours, templateHierarchy = cv2.findContours(self.templateFiltered, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            return contours, hierarchy[0], templateContours, templateHierarchy[0]
        except:
            print("Template contours could not be fitted. Returning strip contours only.")
            return contours, hierarchy[0], False, False

    def get_NumStrips(self):
        return len(self.kaptoncontours)
    
    def ordered_numbering(self, contours, strips,ktype="long"):

        """
        Basically uses an inclined plane and takes the strip that is lowest on the plane
        use 10*strips[i].position[0]-strips[i].position[1] < 10*x-y for top to bottom (ktype shortR) and 0.1 for left to right numbering (ktype long, short)
        """

        ordered_contours, ordered_strips = [], []

        smallest_number = 0
        while len(strips)>0:
            smallest = np.inf
            x, y = 100000, 100000
            for i in range(len(strips)):
                if ktype=="shortR":
                    if 10*strips[i].position[0]-strips[i].position[1] < 10*x-y:  #if 0.1*strips[i].position[0]+strips[i].position[1] < 0.1*x+y:
                        x, y = strips[i].position[0], strips[i].position[1]
                        smallest = i
                else:
                    if 0.1*strips[i].position[0]+strips[i].position[1] < 0.1*x+y:
                        x, y = strips[i].position[0], strips[i].position[1]
                        smallest = i
            ordered_contours.append(int(contours.pop(int(smallest))))
            ordered_strips.append(strips.pop(int(smallest)))
        print('ordered_contours ', ordered_contours)
        return ordered_contours, ordered_strips



    def get_Kaptoncontour_indices(self,kapton_type="long"):
        kaptoncontours  = []
        kaptonstrips    = []
        for i in range(len(self.hierarchy)):
            if(self.hierarchy[i][3]==-1):
                tmpStrip = KaptonStrip(self, i)
                if(tmpStrip.fitmatch>0.2 and tmpStrip.dimensions[1]>kaptonconstants.nominal_stumpkaptonlength*0.5):
                    kaptoncontours.append(i)
                    kaptonstrips.append(tmpStrip)

        return self.ordered_numbering(kaptoncontours, kaptonstrips,kapton_type)


    def get_Strips(self) -> list[KaptonStrip]:
        strips = []
        for i in self.kaptoncontours:
            strips.append(KaptonStrip(self,i))
        return strips

    def get_SizeRanking(self, BigToSmall=True):
        strips      = self.kaptonstrips
        ranking     = []
        areas       = []
        n_strips    = self.get_NumStrips()

        for i in range(n_strips):
            areas.append(strips[i].area)
        areas = np.asarray(areas)
        areas = array.array('f',areas)

        if(BigToSmall):
            for i in range(n_strips):
                tmp_index = areas.index(max(areas))
                ranking.append(tmp_index)
                areas[tmp_index] = 0
        else:
            for i in range(n_strips):
                tmp_index = areas.index(min(areas))
                ranking.append(tmp_index)
                areas[tmp_index] = 100000
        return ranking

    def get_CalcSizeRanking(self, BigToSmall=True):
        strips      = self.kaptonstrips
        ranking     = []
        areas       = []
        n_strips    = self.get_NumStrips()

        for i in range(n_strips):
            areas.append(strips[i].get_CalculatedArea())
        areas = np.asarray(areas)
        areas=array.array('f',areas)

        if(BigToSmall):
            for i in range(n_strips):
                tmp_index = areas.index(max(areas))
                ranking.append(tmp_index)
                areas[tmp_index] = 0
        else:
            for i in range(n_strips):
                tmp_index = areas.index(min(areas))
                ranking.append(tmp_index)
                areas[tmp_index] = 100000
        return ranking

    def get_LengthRanking(self,BigToSmall=True):
        strips      = self.kaptonstrips
        ranking     = []
        lengths     = []
        n_strips    = self.get_NumStrips()

        for i in range(n_strips):
            lengths.append(strips[i].dimensions[1])
        lengths = np.asarray(lengths)
        lengths = array.array('f',lengths)

        if(BigToSmall):
            for i in range(n_strips):
                tmp_index = lengths.index(max(lengths))
                ranking.append(tmp_index)
                lengths[tmp_index] = 0
        else:
            for i in range(n_strips):
                tmp_index = lengths.index(min(lengths))
                ranking.append(tmp_index)
                lengths[tmp_index] = 100000
        return ranking

    def get_WidthRanking(self,BigToSmall=True):
        strips      = self.kaptonstrips
        ranking     = []
        widths      = []
        n_strips    = self.get_NumStrips()

        for i in range(n_strips):
            widths.append(strips[i].dimensions[0])
        widths = np.asarray(widths)
        widths = array.array('f',widths)

        if(BigToSmall):
            for i in range(n_strips):
                tmp_index = widths.index(max(widths))
                ranking.append(tmp_index)
                widths[tmp_index] = 0
        else:
            for i in range(n_strips):
                tmp_index = widths.index(min(widths))
                ranking.append(tmp_index)
                widths[tmp_index] = 100000
        return ranking

    def rank_strips(self, rankingtype='size'):
        if(rankingtype      =='size'):
            rank_indices = self.get_SizeRanking()
        elif(rankingtype    =='calcsize'):
            rank_indices = self.get_CalcSizeRanking()
        elif(rankingtype    =='length'):
            rank_indices = self.get_LengthRanking()
        elif(rankingtype    == 'width'):
            rank_indices = self.get_WidthRanking()
        elif(rankingtype    == 'stripnumber'):
            rank_indices = range(self.get_NumStrips())
        else:
            raise NameError('Invalid Rankingtype')

        tmpimg = copy.deepcopy(self.img)
        strips=self.kaptonstrips
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 5
        fontColor              = (0,0,0)
        lineType               = cv2.LINE_AA

        for i in range(self.get_NumStrips()):
            txtsize, baseline = cv2.getTextSize(str(i+1),font, fontScale, 1)
            cv2.putText(tmpimg,str(i+1),(int(strips[rank_indices[i]].position[0]/self.scalingFactors[0])-int(txtsize[0]/2), int(strips[rank_indices[i]].position[1]/self.scalingFactors[1])+int(txtsize[1]/2)), font,fontScale,fontColor,lineType)
        return cv2.resize(tmpimg, (int(self.imgShape[0]*kaptonconstants.plottingScale), int(self.imgShape[1]*kaptonconstants.plottingScale)))

    def show_greatest_residuals(self):
        # Selection by greatest distance
        strips  = self.kaptonstrips
        nstrips = len(strips)
        if nstrips < 60:
            ncolumns = 4
        else:
            ncolumns = 10
        nrows = int(np.ceil(self.get_NumStrips()/ncolumns))

        matplotlib.rc('xtick', labelsize=2)
        matplotlib.rc('ytick', labelsize=2)
        fig, axs = plt.subplots(nrows, ncolumns)

        for ids, strip in enumerate(strips):
            height = 10
            if strip.Ktype =="long":
                length = 5000
            else:
                length = 1000

            mean_xpos   = np.mean(strip.residual_points[strip.greatest_dist_edge][0])
            mean_ypos   = np.mean(strip.residual_points[strip.greatest_dist_edge][1])
            xticks      = [(mean_xpos-length/2)+(j*(length/5)) for j in range(6)]
            yticks      = [(mean_ypos-height/2)+(k*(height/3)) for k in range(4)]

            ax_index1 = ids // ncolumns
            ax_index2 = ids %  ncolumns
            try:
                axs[ax_index1, ax_index2].scatter(strip.residual_points[strip.greatest_dist_edge][0], strip.residual_points[strip.greatest_dist_edge][1], s=2)
                title_font = {'fontsize': 5,
                            'fontweight' : matplotlib.rcParams['axes.titleweight'],
                            'verticalalignment': 'bottom',
                            'horizontalalignment': 'center'}
                axs[ax_index1, ax_index2].set_title(str(ids+1), fontdict=title_font, y=0.5)
                #axs[ax_index1, ax_index2].set_xticks(xticks)
                #axs[ax_index1, ax_index2].set_xticks(yticks)
                axs[ax_index1, ax_index2].set_xlim(xticks[0], xticks[-1])
                axs[ax_index1, ax_index2].set_ylim(yticks[0], yticks[-1])
            except:
                print("Error while attempting to show residuals!")

    def show_single_residuals(self):
        strips = self.kaptonstrips
        for idstrip, strip in enumerate(strips):
            fig, axs = plt.subplots(2,2)

            height = 10
            if strip.Ktype =="long":
                length = 5000
            else:
                length = 1000

            for idedge, edge in enumerate(strip.residual_points):
                a, b        = idedge//2, idedge%2
                mean_xpos   = np.mean(edge[0])
                mean_ypos   = np.mean(edge[1])
                xticks      = [(mean_xpos-length/2)+(j*(length/5)) for j in range(6)]
                yticks      = [(mean_ypos-height/2)+(k*(height/3)) for k in range(4)]
                #axs[idedge].set_xlim(xticks[0], xticks[-1])
                #axs[idedge].set_ylim(yticks[0], yticks[-1])
                axs[a,b].scatter(edge[0], edge[1], s=2)
            plt.savefig("D:\KaptonMeasurement\KaptonTest\strip_residual_strip"+str(idstrip+1)+".png")
            plt.close()

    def mark_insufficient_strips(self, ignore_holes=True):
        tmpimg      = copy.deepcopy(self.img)
        strips      = self.kaptonstrips
        font        = cv2.FONT_HERSHEY_SIMPLEX
        fontScale   = 20
        fontColor   = (255,0,0)
        lineType    = cv2.LINE_AA

        for i in range(self.get_NumStrips()):
            txtsize, baseline = cv2.getTextSize('X',font, fontScale, 1)
            if(not strips[i].sufficient):
                cv2.putText(tmpimg,'X',(int(strips[i].position[0]/self.scalingFactors[0])-int(txtsize[0]/2), int(strips[i].position[1]/self.scalingFactors[1])+int(txtsize[1]/2.)), font,fontScale,fontColor,lineType)
            elif(ignore_holes == False and len(strips[i].inner_contours)>0):
                cv2.putText(tmpimg,'X',(int(strips[i].position[0]/self.scalingFactors[0])-int(txtsize[0]/2), int(strips[i].position[1]/self.scalingFactors[1])+int(txtsize[1]/2.)), font,fontScale,fontColor,lineType)

        return cv2.resize(tmpimg, (int(self.imgShape[0]*kaptonconstants.plottingScale), int(self.imgShape[1]*kaptonconstants.plottingScale)))


    def show_template_filter_coverage(self):
        tmpimg       = copy.deepcopy(self.img)
        templateBool = np.not_equal(self.templateFiltered, np.zeros(self.templateFiltered.shape))
        tmpimg[templateBool] = np.array([255,0,0])
        return cv2.resize(tmpimg, (int(self.imgShape[0]*kaptonconstants.plottingScale), int(self.imgShape[1]*kaptonconstants.plottingScale)))

    def show_strip_filter_coverage(self):
        tmpimg       = copy.deepcopy(self.img)
        stripBool = np.not_equal(self.filtered, np.zeros(self.filtered.shape))
        tmpimg[stripBool] = np.array([0,0,255])
        return cv2.resize(tmpimg, (int(self.imgShape[0]*kaptonconstants.plottingScale), int(self.imgShape[1]*kaptonconstants.plottingScale)))
