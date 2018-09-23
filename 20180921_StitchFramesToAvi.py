#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:53:01 2018

@author: pointgrey
taking cues from
http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/

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

def readImStack(flist):
    '''
    returns a numpy array of all the images with extension 'imExt' in folder "imFolder"
    '''
    img = cv2.imread(flist[0], cv2.IMREAD_GRAYSCALE)
    imStack = np.zeros((len(flist), img.shape[0], img.shape[1]), dtype=np.uint8)
    for idx, f in enumerate(flist):
        imStack[idx] = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    return imStack


initDir = '/media/aman/Hungry_mate/13_September_2018_frames/'

vidExt = '.avi'
imExt = '.png'
baseDir = getFolder(initDir)
outFile = baseDir.rstrip('/')+vidExt

vidFName = '..\temp.avi'
print('Started stitching frames to AVI on %s'%present_time())
rawdirs = getDirList(baseDir)
for _,rawDir in enumerate(rawdirs):
    rawDirs = getDirList(rawDir)
    for _,d in enumerate(rawDirs):
        vidFName = d+vidExt
        start = time.time()
        imgStack = readImStack(getFiles(d,[imExt]))
        ##http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/        
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
        print('\n------%s Seconds taken to convert\n ==>%s'%(time.time()-start, vidFName))
        
#baseDir = getFolder(initDir)
#outFile = baseDir.rstrip('/')+vidExt
#
#vidFName = '..\temp.avi'
#
#print('Started stitching frames to AVI on %s'%present_time())
#rawdirs = getDirList(baseDir)
#for _,rawDir in enumerate(rawdirs):
#    rawDirs = getDirList(rawDir)
#    start = time.time()
#    for _,d in enumerate(rawDirs):
#        vidFName = d+vidExt
#        cmdline = ['ffmpeg',
#                   '-i',
#                   d+'/%d'+imExt,
#                   '-an',
#                   '-vcodec',
#                   'rawvideo',
#                   '-y',
#                   vidFName]
#        sp.call(cmdline,\
#                        stdin=open(os.devnull, 'wb'),\
#                        stdout=open(os.devnull, 'wb'),\
#                        stderr=open(os.devnull, 'wb')\
#                        )
#        print('%s Seconds taken to convert %s'%(time.time()-start, vidFName))


#







