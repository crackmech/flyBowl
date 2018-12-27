#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:46:57 2017

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
#from tracker_cm import Tracker
##import sys
#import matplotlib.pyplot as plt
#import trackpy as tp
import random
import csv
import itertools




flyParams = cv2.SimpleBlobDetector_Params()
flyParams.blobColor = 0
flyParams.minThreshold = 5
flyParams.maxThreshold = 240#120   120 for original image, 250 for bg subtracted images
flyParams.filterByArea = True
flyParams.filterByCircularity = True
flyParams.minCircularity = 0
flyParams.filterByConvexity = False
flyParams.filterByInertia = False
flyParams.minArea = 200# 200 for flyClimbing, 1000 for fly walking
flyParams.maxArea = 8000

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(flyParams)
else : 
	detector = cv2.SimpleBlobDetector_create(flyParams)



nImThreshold = 0# if number of images in a folder is less than this, then the folder is not processed
imgDatafolder = 'imageData'



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

def random_color():
    levels = range(0,255,32)
    return tuple(random.choice(levels) for _ in range(3))

colors = [(0,200,200),(200,0,200),(200,200,0),(150,0,0),(0,0,200),(200,200,255)]
colors = [random_color() for x in xrange(1000)]
#csvOutFile = '/media/aman/data/thesis/colorPalette.csv'
#with open(csvOutFile, "wb") as f:
#    writer = csv.writer(f)
#    writer.writerows(colors)

def createTrack(trackData, img):
    '''
    input:
        create an image of shape 'imgShape' with the x,y coordiates of the track from the array 'trackData
    returns:
        an np.array with the cv2 image array, which can be saved or viewed independently of this function
    '''
    #img = np.ones((imgShape[0], imgShape[1], 3), dtype = 'uint8')
    blue = np.hstack((np.linspace(0, 255, num = len(trackData)/2),np.linspace(255, 0, num = (len(trackData)/2)+1)))
    green = np.linspace(255, 0, num = len(trackData))
    red = np.linspace(0, 255, num = len(trackData))
    cv2.putText(img,'Total frames: '+str(len(trackData)), (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255))
    for i in xrange(1,len(trackData)):
        cv2.circle(img,(int(trackData[i,0]), int(trackData[i,1])), 2, (blue[i], green[i], red[i]), thickness=2)#draw a circle on the detected body blobs
    for i in xrange(1,len(trackData)):
        if i%100==0:
            cv2.putText(img,'^'+str(i), (int(trackData[i,0]), int(trackData[i,1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255))
    #cv2.imshow('track', img); cv2.waitKey(); cv2.destroyAllWindows()
    return img



def getTrackData(imStack, Blobparams, blurParams):
    '''
    returns the numpy array of coordinates of the centroid of blob in the stack of images provided as input numpy array 'imStack'
    
    '''
    nFrames = imStack.shape[0]
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        	detector = cv2.SimpleBlobDetector(Blobparams)
    else : 
        	detector = cv2.SimpleBlobDetector_create(Blobparams)
    trackData = np.zeros((nFrames,2))
    kernel, sigma = blurParams
    for f in xrange(nFrames):
        im = imStack[f]
        keypoints = detector.detect(cv2.GaussianBlur(im, (kernel, kernel), sigma))
        kp = None
        try:
            for kp in keypoints:
                trackData[f] = (kp.pt[0],kp.pt[1])
        except:
            pass
    return trackData

def getContours((idx, im, contourParams, blurParams)):
    kernel, sigma = blurParams
    ret,th = cv2.threshold(cv2.GaussianBlur(im, (kernel,kernel), sigma), contourParams['threshLow'], contourParams['threshHigh'],cv2.THRESH_BINARY)
    th = cv2.bitwise_not(th)
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        contours, hierarchy = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    else : 
        im2, contours, hierarchy = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    try:
        contours = max(contours, key = cv2.contourArea)
#        if not contourParams['minCntArea']<=cv2.contourArea(x)<=contourParams['maxCntArea']:
#            contours = []
    except:
        contours = []
    return [idx, contours]

def getContourData(imStack, fList, contourParams, blurParams, pool):
    '''
    returns the ellipse fit data of the fly in the stack of images provided as input numpy array 'imStack'  
    '''
#    imgStack = np.array(pool.map(imRead, flist), dtype=np.uint8)
#    poolArgList = itertools.izip(flist, itertools.repeat(params), np.arange(len(flist)))
#    imgWithCnt = pool.map(imReadNCnt, poolArgList)
    poolArgList = itertools.izip(fList, imStack, itertools.repeat(contourParams), itertools.repeat(blurParams))
    contours = pool.map(getContours, poolArgList)
    trackData = []
    for idx,cnt in enumerate(contours):
        if len(cnt[1])!=0:
            try:
                trackData.append([cnt[0], cv2.fitEllipse(cnt[1])])
            except:
                print ('no contour detected in frame# %s'%cnt[0])
                trackData.append([cnt[0], 'noContourDetected'])
        else:
            print ('no contour detected in frame# %s'%cnt[0])
            trackData.append([cnt[0], 'noContourDetected'])
    return trackData
#    trackData = []
#    for idx, im in enumerate(imStack):
#        frId = flist[idx]
#        contours = getContours((frId, im, contourParams, blurParams))
#        if len(contours[1])!=0:
#            trackData.append([contours[0], cv2.fitEllipse(contours[1])])
#        else:
#            print ('no contour detected in frame# %d'%frId)
#            trackData.append([contours[0], 'noContourDetected'])
#    cv2.destroyAllWindows()
#    return trackData

def cropImstack(imStack, trackData, heightCropbox, widthCropbox, blurParams, ratTailParams):
    '''
    returns a list of all images, cropped as per cropBox dimensions
    '''
    kernel, sigma = blurParams
    thresh, nIterations, erodeKernel = ratTailParams
    ims = []
    for i in xrange(imStack.shape[0]):
        im = imStack[i]
        try:
            x,y = trackData[i]
            if (heightCropbox<=y<=imStack.shape[1]-heightCropbox and widthCropbox<=x<=imStack.shape[2]-widthCropbox):
                pts = [int(y)-heightCropbox, int(y)+heightCropbox, int(x)-widthCropbox,int(x)+widthCropbox]
                im_cropped = im[pts[0]:pts[1], pts[2]:pts[3]]
                _,th = cv2.threshold(cv2.GaussianBlur(im_cropped, (kernelSize,kernelSize), sigma), thresh, 255,cv2.THRESH_BINARY)
                th = cv2.bitwise_not(th)
                erosion = cv2.erode(th,erodeKernel,iterations = nIterations)
                dilation = cv2.dilate(erosion, erodeKernel, iterations = nIterations)
                ims.append([i,np.bitwise_xor(th, dilation)])
            else:
                 ims.append([i, 'NoCroppedImage'])
        except:
            pass
    return ims

def cropImstackGray(imStack, trackData, heightCropbox, widthCropbox):
    '''
    returns a list of all images, cropped as per cropBox dimensions
    '''
    ims = []
    for i in xrange(imStack.shape[0]):
        im = imStack[i]
        try:
            x,y = trackData[i]
            if (heightCropbox<=y<=imStack.shape[1]-heightCropbox and widthCropbox<=x<=imStack.shape[2]-widthCropbox):
                pts = [int(y)-heightCropbox, int(y)+heightCropbox, int(x)-widthCropbox,int(x)+widthCropbox]
                im_cropped = im[pts[0]:pts[1], pts[2]:pts[3]]
                ims.append([i,im_cropped])
            else:
                 ims.append([i, 'NoCroppedImage'])
        except:
            pass
    return ims

def saveCroppedIms(croppedStack, ImStack, saveDir, extension, hCropbox):
    '''
    saves the output of the tracked flies in the given format (specifice by 'extension') in the given directory.
    If a fly is not detected in a continous frame, new folder is created to save the next sequence
    '''
    ext = extension
    outDir = saveDir
    cropDir = outDir+'_cropped/'
    imDir = outDir+'_original_subIms/'
    os.mkdir(imDir)
    os.mkdir(cropDir)
    for i in xrange(len(croppedStack)):
        if 'NoCroppedImage' not in croppedStack[i][1]:
            cv2.imwrite(cropDir+str(i)+ext, croppedStack[i][1])
            cv2.imwrite(imDir+str(i)+ext, ImStack[i])
        else:
            print i, croppedStack[i][1]
    return cropDir, imDir

def getFiles(dirname, extList):
    filesList = []
    for ext in extList:
        filesList.extend(glob.glob(os.path.join(dirname, ext)))
    return natural_sort(filesList)


def displayImgs(imgs, fps):
    for _, img in enumerate(imgs):
        cv2.imshow('123',img)
        cv2.waitKey(1000/fps)
    cv2.destroyAllWindows()


def imRead(x):
    return cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    #return cv2.rotate(cv2.imread(x, cv2.IMREAD_GRAYSCALE), cv2.ROTATE_90_COUNTERCLOCKWISE)

def getBgIm(imgs):
    '''
    returns a background Image for subtraction from all the images using weighted average
    '''
    avg = np.array((np.median(imgs, axis=0)))
    return cv2.convertScaleAbs(avg)

def getBgSubImStack((inImgstack, bgIm)):
    '''
    returns the stack of images after subtracting the background image from the input imagestack
    '''
    subIms = np.zeros(np.shape(inImgstack), dtype=np.uint8)
    for f in range(0, len(inImgstack)):
        subIms[f] = cv2.bitwise_not(cv2.absdiff(inImgstack[f], bgIm))
    return subIms
    
def getBgSubIm((inImg, bgIm)):
    '''
    returns the stack of images after subtracting the background image from the input imagestack
    '''
    return cv2.bitwise_not(cv2.absdiff(inImg, bgIm))    
    
def getSubIms(dirname, imExts, pool, workers):
    '''
    tracks the fly using cv2.SimpleBlobDetector method and saves the tracked flies in folders
    '''
    flist = getFiles(dirname, imExts)[:1024]
    nImsToProcess = len(flist)
    print 'processing %i frames in\n==> %s'%(nImsToProcess, dirname)
    startTime = time.time()
    imgStack = np.array(pool.map(imRead, flist), dtype=np.uint8)
    t1 = time.time()-startTime
    print("imRead time for %d frames: %s Seconds at %f FPS\n"%(len(flist),t1 ,len(flist)/float(t1)))
    displayImgs(imgStack, 200)
    t1 = time.time()
    imStackChunks = np.array_split(imgStack, 4*workers, axis=1)
    #imStackChunks = [x.copy() for x in imStackChunks if x.size > 0]
    bgImChunks = pool.map(getBgIm, imStackChunks)
    bgIm = np.array(np.vstack((bgImChunks)), dtype=np.uint8)
    t2 = time.time()-t1
    print("parallel bg calculation time for %d frames: %s Seconds at %f FPS\n"%(len(flist),t2 ,len(flist)/float(t2)))
    t2 = time.time()
    subIms = np.array(pool.map(getBgSubIm, itertools.izip(imgStack, itertools.repeat(bgIm))), dtype=np.uint8)
    t = time.time()-t2
    print("bg Subtraction time for %d frames: %s Seconds at %f FPS\n"%(len(flist),t ,len(flist)/float(t)))
    t = time.time()-t1
    print("imRead and bgSub time for %d frames: %s Seconds at %f FPS\n"%(len(flist),t ,len(flist)/float(t)))
    displayImgs(subIms, 200)
    return imgStack, subIms, flist


 



imExtensions = ['*.png', '*.jpeg']
imDataFolder = 'imageData'
statsfName = 'contoursStats_threshBinary'
statsFileHeader = ['frameDetails','x-coord','y-coord','minorAxis (px)','majorAxis (px)','angle','area (px)']

nThreads = 4
pool = mp.Pool(processes=nThreads)

imFolder = '/media/aman/data/KCNJ10/KCNJ10_labeledVideos/fullRes/20180927_000154_20180922_2230_KCNJ10SS--flyBowl/imageData/20180927_000433'
flyContours = getSubIms(imFolder, imExtensions, pool, nThreads)






""" 

initialDir = '/media/pointgrey/data/flywalk/'
initialDir = '/media/aman/data/flyWalk_data/climbingData/gait/data/tmp/pythonTmp/'
initialDir = '/media/aman/data/flyWalk_data/climbingData/gait/data/copiedLegTrackingTrackData/'
baseDir = getFolder(initialDir)

outDir = '/media/aman/data/flyWalk_data/climbingData/gait/data/tmp/'

imExtensions = ['*.png', '*.jpeg']
heightCrop = 80
widthCrop = 100
legCntThresh = 0.31

nThreads = 4
kernelSize = 5
gauBlurParams = (kernelSize,1)

threshVal = 250
nIterations = 2
kernel = np.ones((kernelSize,kernelSize),np.uint8)

pxMvdByLegBwFrm = 50
legTipFrmSkipthresh = 40

rattailparams = (threshVal, nIterations, kernel)
#baseDir = initialDir
print baseDir
cntParams = {'maxCntArea'   :   7000,\
             'minCntArea'   :   1000,\
             'threshLow'    :   230,\
             'threshHigh'   :   255}


trackparams = [imExtensions, heightCrop, widthCrop, cntParams, flyParams,\
               nImThreshold, gauBlurParams, rattailparams]


rawDirs = getDirList(baseDir)
pool = mp.Pool(processes=nThreads)
procStartTime = time.time()
totalNFrames = 0
print "Started processing directories at "+present_time()
for _,rawDir in enumerate(rawDirs):
    d = os.path.join(rawDir, imgDatafolder)
    print rawDir
    imdirs = getDirList(d)
    for imdir in imdirs:
        startTime = time.time()
        nFrames = len(getFiles(imdir, imExtensions))
        fname = imdir.rstrip(os.sep)+'_legTips-Climbing_allPts_'
        legTipLocs = getAllLocs(imdir, trackparams, legCntThresh, fname, pool, nThreads)
        print('Processed %i frames in %0.3f seconds\nAverage total processing speed: %05f FPS'\
        %(nFrames, time.time()-startTime, (nFrames/(time.time()-startTime)))) 
        totalNFrames +=nFrames
pool.close()
totSecs = time.time()-procStartTime
print('Processing finished at: %05s, in $sSeconds, total processing speed: %05f FPS'\
      %(present_time(),totSecs , totalNFrames/totSecs))




"""























