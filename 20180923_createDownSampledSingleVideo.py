#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 07:03:12 2018

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

def getFrameStack(inDir, vidExt, downSampleSize):
    '''
    '''
    flist = getFiles(inDir, vidExt)
    if flist[0].split('/')[-1]!='.':
        images = []
        for i, vid in enumerate(flist[:10]):
            start = time.time()
            images.append(getFramesFomVid(vid)[::downSampleSize].copy())
            print("Extraced %d frames in %.05s Seconds from video #%d/%d (%s) "\
                    %(len(images[i]),time.time()-start, i+1, len(flist)+1, vid.split('/')[-1]))
        return np.vstack(images)
    else:
        print('No videos present in %s'%inDir)
        return []

<<<<<<< HEAD

dirName = '/media/aman/Hungry_mate/'
=======
def writeVidCommand(vidfname):
    return [ 'ffmpeg',
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-s', '1024x1024', # size of one frame
            '-pix_fmt', 'gray',
            '-i', 'pipe:0', # The imput comes from a pipe
            '-an', # Tells FFMPEG not to expect any audio
            '-vcodec', 'rawvideo',
            '-y',
            vidfname ]
    

dirName = '/media/aman/Hungry_mate/'
dirName = '/media/pointgrey/Hungry_mate/'
dirName = '/media/pointgrey/DCshared/Hungrymating/quantification_to_be_done/NoFood/'
>>>>>>> cda8107dec0f7315ff29dd51e4fa0305fbf99778
#dirName = '/media/aman/data/Dhananjay/FlyBowl/flyCourtship/test/'
vidExt = ['.avi']
getString = 0        # set to '0' to write line by line as string, set to '1' to write the whole imgStack in one go as a string
baseDir = getFolder(dirName)
rawdirs = getDirList(baseDir)

for _,rawDir in enumerate(rawdirs):
    rawDirs = getDirList(rawDir)
    for _,d in enumerate(rawDirs):
        start = time.time()
        print('Started processing %s at %s----------------------'%(d, present_time()))
        imgStack = getFrameStack(d, vidExt, downSampleSize)
<<<<<<< HEAD
        vidFName = d+'_'+str(downSampleSize)+'X-downSampledFPS'+vidExt[0]
        command = [ 'ffmpeg',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-s', '1024x1024', # size of one frame
                '-pix_fmt', 'gray',
                '-i', 'pipe:0', # The imput comes from a pipe
                '-an', # Tells FFMPEG not to expect any audio
                '-vcodec', 'rawvideo',
                '-y',
                vidFName ]
        pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE)
        pipe.communicate(input=imgStack.tostring() )
=======
        vidFName = d+'_'+str(downSampleSize)+'X-downSampledFrames'+vidExt[0]
        command = writeVidCommand(vidFName)
        if getString==1:
            imgStack = imgStack.tostring()
            pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE)
            pipe.communicate(input=imgStack)
            
        else:
            pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE)
            for i in xrange(imgStack.shape[0]):
                print >>pipe.stdin, imgStack[i].tostring()
        del imgStack
>>>>>>> cda8107dec0f7315ff29dd51e4fa0305fbf99778
        print('\n------%s Seconds taken to downSample frames to\n ==>%s'%(time.time()-start, vidFName))




