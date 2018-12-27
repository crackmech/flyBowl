#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 15:36:29 2018

@author: aman
"""

import cv2
import os
import glob
import numpy as np
import re
from datetime import datetime
import Tkinter as tk
import tkFileDialog as tkd
import time
import subprocess as sp
import csv
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

baseDir = '/media/pointgrey/data/test/'
downSampleSize = 4
 
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def present_time():
        return datetime.now().strftime('%Y%m%d_%H%M%S')

def getFolder(initialDir):
    '''
    GUI funciton for browsing and selecting the folder
    '''    
    root = tk.Tk()
    initialDir = tkd.askdirectory(parent=root,
                initialdir = initialDir, title='Please select a directory')
    root.destroy()
    return initialDir+'/'

def getDirList(parentDir):
    return natural_sort([ os.path.join(parentDir, name) for name in os.listdir(parentDir)\
                        if os.path.isdir(os.path.join(parentDir, name)) ])

def getFiles(dirname, extList):
    filesList = []
    for ext in extList:
        filesList.extend(glob.glob(os.path.join(dirname, '*'+ext)))
    return natural_sort(filesList)

def readCSV(csvFName):
    '''
    returns the data of the csv file as a list of 
    '''
    with open(csvFName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csvData = []
        count = 0
        for row in csv_reader:
            if count==0:
                for _,x in enumerate(row):
                    csvData.append([x])
            else:
                for i,x in enumerate(row):
                    csvData[i].append(x)
            count+=1
    return csvData

def calcEuDis(xyArray, stepSize):
    '''
    returns an array of eucledian distance between the 
    points in xyArray with consequent xy points seperated by stepSize
    '''
    euDisArray = np.zeros(((xyArray.shape[1]/stepSize)+1))
    for i in xrange(0, xyArray.shape[1]-stepSize, stepSize):
        euDisArray[i/stepSize]=np.sqrt(np.square(xyArray[:,i+stepSize]-xyArray[:,i]).sum())
    return euDisArray


sWidth = 0.012
sSize = 15
sMarker = 'o'
sAlpha = 0.6
sLinewidth = 0.2
sEdgCol = (0,0,0)
scatterDataWidth = 0.012
sCol = (0.34,1,1)
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




initDir = '/media/aman/data/flyBowl'
initDir = '/media/aman/easystore/20180926/KCNJ10Data/flyBowl/'
csvFolderExt = '-trackfeat'

baseDir = getFolder(initDir)
csvExt = ['*.csv']

euDisStep = 2
fps = 15
pixelSize = 70.0/1024.0 #size of the flyBowl (in mm) fitting into the 1024x1024 frame
avMaxFlySpeed = 45.0 #mm per second


ylimit = avMaxFlySpeed/(pixelSize * fps)# based on approx max speed of a fly, i.e 45mm per second

colors = ['orange','b']
i=0
alpha = 0.4
labels = [r'W$^1$$^1$$^1$$^8$', r'KCNJ10$^S$$^S$$^-$']

nXticks = 5
nYticks = 5
dirList = getDirList(baseDir)

totalEuDis1 = []
for d in dirList:
    dateFolders = getDirList(d)
    csvList = []
    for _,x in enumerate(dateFolders):
        if csvFolderExt in x:
            csvList.extend(getFiles(os.path.join(d,x), csvExt))
    euDisList = []
    for idx,csvName in enumerate(csvList):
        csvData = readCSV(csvName)
        inArray = np.array((csvData[0][1:],csvData[1][1:]), dtype=np.float32)
        euDisList.append(calcEuDis(inArray, euDisStep))
    totalEuDis1.append(euDisList)

totalEuDis = []
for genotype in xrange(len(totalEuDis1)):
    euDis = np.zeros((len(totalEuDis1[genotype]), len(totalEuDis1[genotype][0])), dtype=np.float32)
    for i in xrange(euDis.shape[0]):
        euDis[i] = totalEuDis1[genotype][i][:euDis[i].shape[0]]
        print i
    totalEuDis.append(euDis)

avspeed = [(np.sum(x, axis=1)/900)*pixelSize for _,x in enumerate(totalEuDis)]
#avspeed.reverse()
avSpeedBox = [x[~np.isnan(x)] for x in avspeed]

sWidth = 0.05
sSize = 25

#plotScatter(axis, data, scatterX, scatterWidth = sWidth, \
#                scatterRadius = sSize , scatterColor = sCol,\
#                scatterMarker = sMarker, scatterAlpha = sAlpha, \
#                scatterLineWidth = sLinewidth, scatterEdgeColor = sEdgCol, zOrder=0):
fig, ax = plt.subplots(1)
for s,scatterPlotData in enumerate(avspeed):
    plotScatter(ax, scatterPlotData, scatterX = s+1, scatterRadius = sSize , scatterWidth = sWidth, scatterColor = colors[s], zOrder=2)
ax.boxplot(avSpeedBox)
plt.xticks([1,2], labels)
plt.ylabel('Average Speed (mm/s)')
plt.suptitle('Average speed of flies in a horizontal disk in 15 minutes\n(3 Batches, n>30)')
plt.show()



from scipy import stats

print stats.ttest_ind(avSpeedBox[0],avSpeedBox[1])


nFrames = totalEuDis[0].shape[1]
imageingTimeDuration = (nFrames*euDisStep)/(60*fps)     # in minutes

xTickLabels = np.arange(0,imageingTimeDuration+1,imageingTimeDuration/nXticks)
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)          # controls default text sizes
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels

ylimit = 23.0
for i,dis in enumerate(totalEuDis):
    mean = savgol_filter(np.nanmean(dis, axis=0), 101,3)
    sd = savgol_filter(np.nanstd(dis, axis=0)/np.sqrt(dis.shape[0]), 101,3)
    plt.fill_between(np.arange(mean.shape[0]), mean+sd,  mean-sd, facecolor=colors[i], alpha=alpha*0.7)
    plt.plot(np.arange(mean.shape[0]), mean, color = colors[i], alpha=alpha, label = labels[i])
plt.xticks(np.arange(0,nFrames+(nFrames*0.01), nFrames/nXticks ), xTickLabels)
plt.xlim(-10,nFrames+(nFrames*0.01))
plt.ylim(0,ylimit)
plt.yticks(np.arange(0,ylimit,ylimit/nYticks), np.round(np.arange(0,ylimit,ylimit/nYticks)*pixelSize, 2), rotation=90)
plt.ylabel('distance traveled\n(mm)')
plt.xlabel('minutes')
plt.legend(loc='upper right').draggable()
plt.suptitle('Overall Movement of flies in a horizontal disk in 15 minutes\n(3 Batches, n>30)')
plt.show()








#totalEuDis = []
#for d in dirList:
#    csvList = getFiles(os.path.join(d,d.split(os.sep)[-1]+csvFolderExt), csvExt)
#    euDisList = []
#    for idx,csvName in enumerate(csvList):
#        csvData = readCSV(csvName)
#        inArray = np.array((csvData[0][1:],csvData[1][1:]), dtype=np.float32)
#        euDisList.append(calcEuDis(inArray, euDisStep))
#    totalEuDis.append(np.array(euDisList))
#
#
#
#fig, ax = plt.subplots(len(totalEuDis)+1)
#for i,dis in enumerate(totalEuDis):
#    mean = savgol_filter(np.nanmean(dis, axis=0), 101,3)
#    sd = savgol_filter(np.nanstd(dis, axis=0)/np.sqrt(dis.shape[0]), 101,3)
#    ax[i].fill_between(np.arange(mean.shape[0]), mean+sd,  mean-sd, facecolor=colors[i], alpha=alpha*0.3)
#    ax[i].set_xticks(np.arange(0,nFrames+(nFrames*0.01), nFrames/nXticks ))
#    ax[i].set_xticklabels(xTickLabels)
#    ax[i].set_yticks(np.arange(0,ylimit,ylimit/nYticks))
#    ax[i].set_yticklabels(np.round(np.arange(0,ylimit,ylimit/nYticks)*pixelSize, 2), rotation=90)
#    #plt.plot(savgol_filter(dis, 101,3).T, color = colors[i], alpha=alpha*0.1)
#    ax[i].plot(np.arange(mean.shape[0]), mean, color = colors[i], alpha=alpha, label = labels[i])
#    ax[i].set_xlim(-10,nFrames+(nFrames*0.01))
#    ax[i].set_ylim(0,ylimit)
#    ax[i].set_ylabel('distance traveled\n(mm)')
#    ax[i].set_xlabel('minutes')
#    plt.plot(np.arange(mean.shape[0]), mean, color = colors[i], label = labels[i])
#plt.xticks(np.arange(0,nFrames+(nFrames*0.01), nFrames/nXticks ), xTickLabels)
#plt.xlim(-10,nFrames+(nFrames*0.01))
#plt.ylim(0,ylimit)
#plt.yticks(np.arange(0,ylimit,ylimit/nYticks), np.round(np.arange(0,ylimit,ylimit/nYticks)*pixelSize, 2), rotation=90)
#plt.ylabel('distance traveled\n(mm)')
#plt.xlabel('minutes')
#plt.legend(loc='upper right').draggable()
#plt.suptitle('Overall Movement of flies in a horizontal disk in 15 minutes\n(Second batch, 12-18 flies per genotype)')
#plt.show()
#





