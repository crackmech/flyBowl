#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:53:01 2018

@author: pointgrey
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

def getFiles(dirname, extList):
    filesList = []
    for ext in extList:
        filesList.extend(glob.glob(os.path.join(dirname, ext)))
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



vidExt = '11.avi'
imExt = ['*.png']
baseDir = getFolder('/media/pointgrey/')
flist = getFiles(baseDir, imExt)

outFile = baseDir.rstrip('/')+vidExt

fourcc = cv2.cv.CV_FOURCC(*'DIB ')
img = cv2.imread(flist[0])
imSize = img.shape[1], img.shape[0]
out = cv2.VideoWriter(outFile, fourcc, 40.0/downSampleSize, imSize)

for _, f in enumerate(flist):
    img = cv2.imread(f)
    out.write(img)
out.release()






















