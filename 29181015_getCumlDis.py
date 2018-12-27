#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 01:36:42 2018

@author: aman
"""

import cv2
import os
import numpy as np
import re
from datetime import datetime
import Tkinter as tk
import tkFileDialog as tkd
import multiprocessing as mp
import time
import glob
import random
import csv
from PIL import ImageTk, Image
from scipy.spatial import distance
#import itertools
from matplotlib import pyplot as plt
from scipy import stats


def present_time():
        now = datetime.now()
        return now.strftime('%Y%m%d_%H%M%S')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def getFolder(initialDir):
    '''
    GUI funciton for browsing and selecting the folder
    '''    
    root = tk.Tk()
    initialDir = tkd.askdirectory(parent=root,
                initialdir = initialDir, title='Please select a directory')
    root.destroy()
    return initialDir+'/'

def getDirList(folder):
    return natural_sort([os.path.join(folder, name) for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))])

def getFiles(dirname, extList):
    filesList = []
    for ext in extList:
        filesList.extend(glob.glob(os.path.join(dirname, ext)))
    return natural_sort(filesList)

def random_color():
    levels = range(32,256,2)
    return tuple(random.choice(levels) for _ in range(3))

#colors = [random_color() for i in xrange(20)]
def readCsv(csvFname):
    rows = []
    with open(csvFname, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        for row in csvreader: 
            rows.append(row) 
    return rows
def rejectOutliers(data, m=2):
    return data[abs(data - np.nanmean(data)) < m * np.nanstd(data)]


pxSize = 80/1250.75     #CALCULCATED BY CALIBRATING THE FLYbOWL IMAGES (mm/px)
    

dirName = '/media/aman/data/flyBowl/trackedCSVs_FlyTracker/'
#dirName = '/media/aman/data/flyBowl/trackedCSVs_FlyTracker/2KCNJ10SS--flyBowl/'

corrVal=[8,10]


allDis = []
baseDirs = getDirList(dirName)
for md,bDs in enumerate(baseDirs):
    csvFiles = []
    dirs = getDirList(bDs)
    for _,d in enumerate(dirs):
        csvFiles.extend(getFiles(d, ['*.csv']))
    totDis = []
    correc = 1
    for f,flyDataFname in enumerate(csvFiles):
        if f>=corrVal[md]:
            correc=3
        flyData = np.asarray(readCsv(flyDataFname))
        flyCents = np.asarray(flyData[1:,:2], dtype=np.float64)
        dis = np.zeros((flyCents.shape[0]-correc))
        x=0
        for i in xrange(1, flyCents.shape[0]-correc):
            if np.isnan(flyCents[i,0]):
                print i, '--', flyCents[i], flyCents[x]
                if x<=(i):
                    print x
                    x=i-1
                pass
            else:
                if x==i-1:
                    x=i
                    dis[i] = distance.euclidean(flyCents[i-1], flyCents[x])
                else:
                    x=i
                    pass
        totDis.append(dis*pxSize)
    allDis.append(np.array(totDis).T)
    csvOutFile = bDs+'totalEucDis_mm_flyBowl.csv'
    with open(csvOutFile, "wb") as f:
        writer = csv.writer(f)
        writer.writerow(['fly'+str(i+1) for i in xrange(len(csvFiles))])
        writer.writerows(np.array(totDis).T)


#
#fps = 15
#minuteBin = 1
#
#frameBin = fps*minuteBin*60-1
#AllSumDis = []
#for p, totalDistances in enumerate(allDis):
#    sumDis = []
#    for i in xrange(0, len(totalDistances), frameBin):
#        sumDis.append(np.sum(totalDistances[i:i+frameBin,:], axis=0))
#    AllSumDis.append(np.array(sumDis[:-1]))
##    csvOutFile = baseDirs[p]+'_BinnedEucDis_mm_flyBowl.csv'
##    with open(csvOutFile, "wb") as f:
##        writer = csv.writer(f)
##        writer.writerow(['fly'+str(i+1) for i in xrange(len(sumDis[0]))])
##        writer.writerows(AllSumDis[-1])
#plt.plot(np.sum(AllSumDis[0], axis=1))
#plt.plot(np.sum(AllSumDis[1], axis=1))
#plt.show()
#
alfa = 0.71
div = 255.0
colors = [(230/div,218/div,66/div,alfa),(0/div,114/div,178/div,alfa)]   # for 30 minutes, KCNJ10 data
labels = [r'W$^1$$^1$$^1$$^8$', 'KCNJ10SS-']

markers = ['8']
markerSize = 10
lineWidth = 3

#-------Remove Outliers-----

outlierSD = 2.5
fps = 15
minuteBin = 1

frameBin = fps*minuteBin*60-1
AllSumDis = []
for p, totalDistances in enumerate(allDis):
    sumDis = []
    for i in xrange(0, len(totalDistances), frameBin):
        sumDis.append(np.sum(totalDistances[i:i+frameBin,:], axis=0))
    AllSumDis.append([rejectOutliers(x, outlierSD) for _,x in enumerate(sumDis[:-1])])
#    csvOutFile = baseDirs[p]+'_BinnedEucDis_mm_flyBowl.csv'
#    with open(csvOutFile, "wb") as f:
#        writer = csv.writer(f)
#        writer.writerow(['fly'+str(i+1) for i in xrange(len(sumDis[0]))])
#        writer.writerows(AllSumDis[-1])


fig, ax = plt.subplots(1, figsize=(5,3), tight_layout = True)
for i in xrange(len (AllSumDis)):
    ave = [np.average(x/(minuteBin*60)) for _,x in enumerate(AllSumDis[i])]
    sd = [stats.sem(x/(minuteBin*60)) for _,x in enumerate(AllSumDis[i])]
    ax.errorbar(np.arange(len(ave)),ave, yerr=sd, color = colors[i], label = labels[i],\
                    fmt='-'+markers[0], markersize=markerSize, linewidth = lineWidth)
ax.legend(loc=1, ncol=i+1)
ax.set_ylim(0,10)
plt.title('Average Speed per minute')
plt.rcParams['font.size'] = 12
plt.xticks(np.arange(len(ave)), np.arange(1,len(ave)+1))
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
plt.show()



showMeans = False
showMedians = True
showExtrema = False
medianColor = 'Orange'
vPlotLineShow = 'cmedians'
vAlpha = 0.5


bwMethod = 'silverman'
boxLineWidth = 3
boxprops = dict(linestyle='-', linewidth=boxLineWidth)
whiskerprops = dict(linestyle='-', linewidth=boxLineWidth)
capprops = dict(linestyle='--', linewidth=boxLineWidth)
medianprops = dict(linestyle = None, linewidth=0)
boxPro = dict(boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops)



sWidth = 0.15   #0.012
sSize = 150
sMarker = 'o'
sAlpha = 0.6
sLinewidth = 0.2
sEdgCol = (0,0,0)
scatterDataWidth = 0.012
sCol = (0,0,0)

def plotScatter(axis, data, scatterX, scatterWidth = sWidth, \
                scatterRadius = sSize , scatterColor = sCol,\
                scatterMarker = sMarker, scatterAlpha = sAlpha, \
                scatterLineWidth = sLinewidth, scatterEdgeColor = sEdgCol, zOrder=0):
    '''
    Takes the data and outputs the scatter plot on the given axis.
    
    Returns the axis with scatter plot
    '''
    return axis.scatter(np.linspace(scatterWidth+scatterX, -scatterWidth+scatterX,len(data)), data,\
            s=scatterRadius, color = scatterColor, marker=scatterMarker,\
            alpha=scatterAlpha, linewidths=scatterLineWidth, edgecolors=scatterEdgeColor, zorder=zOrder )




outlierSD = 2.5
bxPltData = []
for i, d in enumerate(allDis):
    bxPltData.append([(np.sum(rejectOutliers(d[:,x], outlierSD))/len(rejectOutliers(d[:,x], outlierSD)))*fps for x in xrange(d.shape[1])])

fig, ax = plt.subplots(1, figsize=(5,10), tight_layout = True)
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5, linewidth=3)
bp = ax.boxplot(bxPltData, sym='', medianprops = medianprops, boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops, zorder=1)
vp = ax.violinplot(bxPltData, showmeans=showMeans, showmedians=showMedians, showextrema=showExtrema, bw_method=bwMethod)
for i in xrange(len(bxPltData)):
    plotScatter(ax, bxPltData[i],i+1, scatterRadius = sSize,scatterAlpha = sAlpha, zOrder=3)

vp[vPlotLineShow].set_color(medianColor)
vp[vPlotLineShow].set_zorder(1)
for patch, color in zip(vp['bodies'], colors):
    patch.set_color(color)
    patch.set_edgecolor(None)
    patch.set_alpha(vAlpha)
#for patch in (vp['cmedians']):
vp['cmedians'].set_linewidth(boxLineWidth)
vp['cmedians'].zorder=4

plt.show()














