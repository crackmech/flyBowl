#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 12:04:24 2018

@author: pointgrey

1) Select the folder with all AVI files to be combined
2) Set the DownSampling Rate (the fraction of the original FPS)
3) The output file is stored by the foldername_FPS.avi
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
import shutil

baseDir = '/media/pointgrey/data/test/'
downSampleSize = 4
 
def move(src, dest):
    shutil.move(src, dest)

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
        filesList.extend(glob.glob(os.path.join(dirname, ext)))
    return natural_sort(filesList)

def getFramesFomVid(vidName):
    '''
    return a an array of all frames of the video 
    '''
    vidObj = cv2.VideoCapture(vidName) 
    success = 1# checks whether frames were extracted 
    imgs = [] 
    while success: 
        success, image = vidObj.read()
        if success:
            imgs.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    vidObj.release()
    imgStack = np.zeros((len(imgs),\
                            imgs[0].shape[0],\
                            imgs[0].shape[1],),dtype=np.uint8)
    for idx,img in enumerate(imgs):
        imgStack[idx]=img
    return imgStack

def writeFrames(inDir, vidExt, imExt, downSampleSize):
    '''
    '''
    flist = getFiles(inDir, vidExt)
    if flist[0].split('/')[-1]!='.':
        outDir = inDir.rstrip('/')+'_frames_'+str(downSampleSize)+'X-downSampledFPS/'
        try:
            os.mkdir(outDir)
        except:
            pass
        imCount = 0
        for i, vid in enumerate(flist):
            start = time.time()
            for j, img in enumerate(getFramesFomVid(vid)[::downSampleSize].copy()):
                cv2.imwrite(os.path.join(outDir, str(imCount)+imExt), img)
                imCount+=1
            print("Extraced %d frames in %.05s Seconds from video #%d (%s) "\
                    %(j,time.time()-start, i+1, vid.split('/')[-1]))
    else:
        print('No videos present in  %s'%inDir)


dirName = '/media/pointgrey/Hungry_mate/13_September_2018/'
vidExt = ['*.avi']
imExt = '.png'
baseDir = getFolder(dirName)
rawdirs = getDirList(baseDir)

for _,rawDir in enumerate(rawdirs):
    rawDirs = getDirList(rawDir)
    for _,d in enumerate(rawDirs):
        print('Started processing %s at %s----------------------'%(d, present_time()))
        writeFrames(d, vidExt, imExt, downSampleSize)



#flist = getFiles(baseDir, vidExt)
#outDir = baseDir.rstrip('/')+'_frames_'+str(downSampleSize)+'X-downSampledFPS/'
#try:
#    os.mkdir(outDir)
#except:
#    pass
#imCount = 0
#for i, vid in enumerate(flist):
#    start = time.time()
#    for j, img in enumerate(getFramesFomVid(vid)[::downSampleSize].copy()):
#        cv2.imwrite(os.path.join(outDir, str(imCount)+imExt), img)
#        imCount+=1
#    print("aviRead time for %d frames: %s Seconds in %d vid"%(j,time.time()-start, i+1 ))



#frames = []
#for i, vid in enumerate(flist[:26]):
#    start = time.time()
#    frames.append(getFramesFomVid(vid)[::downSampleSize].copy())
#    print("aviRead time for %d frames: %s Seconds in %d vid"%(len(frames[-1]),time.time()-start, i+1 ))
#
#
#
#
#
#
#frames1 = np.vstack(frames)
#for _,img in enumerate(frames1):
#    cv2.imshow('frame',img)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#cv2.destroyAllWindows()
























































