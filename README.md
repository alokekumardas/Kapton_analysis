# Kaptonstrip analysis

Analysis of Kaptonstrip scan images w.r.t. size and shape.

## Description
This software is designed to analyse scanner images of the long and short kapton strips.
The scans are done in batches of 10 long (1 row) and up to 32 (4 rows with 8 strips each) short kapton strips, at 1200 DPI resolution.
Besides the Kapton strips, 4 templates are used to calibrate the scanner pixel-to-mm conversion.
The length and width of the Kaptonstrips are determined, as well as their area.
Also, an edge fit with a line is done and the fitmatch is stored.
Then, the edge shape is also evaluated for curvature, as described in my e-mail.

## Getting Started

Please clone the repository. 
Check the packages.txt file to see which python version and additional packages are used in the software. Install the missing ones. (Maybe also other versions work, but I do not guarantee for this...)


### Executing program

The code is currently always executed on Windows10 using the VSCode terminal. It should however in principle also work on Linux.

Please execute 
```
python3 KaptonBatchmode.py -f <filename> -k <kaptontype (long, short)>
```
to analyse a single file/scan
or
```
python3 KaptonBatchmode.py -d <foldername> -k <kaptontype (long, short)>
```
to analyse all files/scans in the specified folder. Please have a look into the KaptonBatchmode.py for more options and details.

In the scan folder, 2 folders will be created, "ResultsShape" and "PlotsShape". The first includes csv files with the results (size etc), the other one plots of the kaptonstrip numbering, the filtered images, and an additional folder called "shape" where plots of contours are stored in case the curvature detection algorithm detects anything suspicious. 

Also a DB upload is possible, however this is optimized for our DB and will need changes/modifications from your side, if you want to use it.

### Structure of the software
* KaptonBatchmode.py : Main program, calls especially Kapton_Analyze_Batch.py
* Kapton_Analyze_batch.py: Here, the analysis class is defined. I would say the most important methods are the openScanfile and the write_table methods. The first opens the file and calls analysis methods defined in the kaptondetector.py script. The latter defines which results are written to the csv results file.
* *Helper folder:*
* kaptondetector.py : Contains class definitions + methods for the Scan and the Kaptonstrip analysis.
* kaptonconstants.py: Here, the nominal kaptonsizes and the acceptable tolerance window as well as the RGB filter settings used by the kaptondetector.py script are defined. HERE, YOU WILL DEFINITELY HAVE TO OPTIMIZE FOR YOUR USE CASE! Be also aware that the tolerance window used is no official specification, but chosen by us in a way we thought it makes sense. 

### Analysis method
To get the Kaptonstrip length, the image is filtered by applying RGB filters. There are separate ones for the templates and the strips. Using OpenCV methods, a contour around each strip is found and the smallest rectangle around this contour is found. This is then used to get the strip size and width in pixels. Using the known template sizes, one can calibrate from pixels to mm. For details please have a look at the PhD thesis of Tim Ziemons (add link!!).
For the curvature detection for long Kapton strips, on each long edge of the strip contour, two points are selected: one close to each end of the strip (actually, it is the average of 5 neighbouring points or so...). Then, a straight line is defined in between these 2 points. For all points on the contour in between the 2 starting points, the horizontal distance to the line is calculated. If it is above 0.1mm, the point is stored as outlier. Afterwards, the amount of outliers is analyzed. If there are more than 5% outliers, plots are generated from the contour and the distances to the line. These plots are stored in the PlotsShape/shape folder. Then, it is checked if the outliers cluster and if the cluster lies near the middle of the strip. If yes, the strip is marked as "curved". If not, it is marked as "strange". This label is written into the results csv file, for each strip for each of the 2 long edges. To check or change the limits, please have a look at the kaptondetector.py file, method KaptonStrip::determine_shape().
The curvature detection is not implemented for short strips.


## Authors
Main work done by Tim Ziemons.
Additions by Philippe Clement and Nina Höflich. (all RWTH)

Ask Nina Höflich in case of questions/issues/...: nina.hoeflich@rwth-aachen.de

