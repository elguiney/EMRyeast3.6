'''
Script that emulates measure_GFP_wRFPmarker
use for testing
'''

import YMPy
import pickle
import numpy as np
import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd

import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display,clear_output

import random


folderPath = r"C:\Users\elgui\Documents\Emr Lab Post Doc\microscopy\Art1Quant"

#%% local function parameters
parameterDict = dict(
        expIDloc=[0,6], #position of machine readable experiment ID
        imageExtension='R3D_D3D.dv', #images have this extension
        rolloff = 64, #ignore outermost n pixels
        nChannels = 3,
        redChannel = 0,
        greenChannel = 1,            
        bwChannel = 2,
        zFirst = True, #True if stack is saved zcxy
        bfAnomalyShiftVector = [2,-1], #correction for scope 
            #bf vs fluorescence slight misalignment
        minAngle = 22, #angle of curvature at bud neck
        minLength = 5, #integration length for curvature measurement
        closeRadius = 3, #closing structure element size
        minBudSize = 75,
        cortexWidth = 10,
        bufferSize = 5, #buffer region outside of bf border estimate
        showProgress = True,
        mkrChannel = 0,
        markerRadius = 7, #size radius that defines typical marker foci
        markerCircleIterations = 5, #number of iterations to convert
            #raw marker mask to foci centered circles
        nHistBins = 1000, #bins (across global min-max range)
        measuredProteinName = 'Art1-mNG',
        measuredProteinChannel = 1,
        measuredProteinZ = 3, #measure in this z-slice
        markerName = 'golgi',
        rgbScalingFactors = [0.05,0.1], #semi empirically determend factors
            #to make qc images at a reasonable brightness and contrast
        gryScalingFactors = [0.2,0.2], #same
        measureFields = [0,1] #either 'all', to measure entire folder, or
            #a slice. [5,10] would measure fields 5,6,7,8 and 9 as they
            #appear in folderData['pathList'] from YMPy.batchParse()
        )

#%% start measurment
# shorten paramterDict name
p = parameterDict

print('beginning analysis of \n',folderPath,'\n at ',
      datetime.datetime.now())

# initialize experiment variables
totalResults = []
totalQC = []
fieldsAnalyzed = []
totalMcl = []
    
resultsDirectory = folderPath + '/results/'
if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory)

folderData = YMPy.batchParse(
        folderPath, p['expIDloc'], p['imageExtension'])
nFields = folderData['nFields']
imageNameList = folderData['imagenameList']
pathList = folderData['pathlist']
expIDlist = folderData['expIDlist']
#%% measure global values (slow for big datasets)
globalExtremaG = YMPy.batchIntensityScale(
        folderData, p['greenChannel'], p['showProgress'])
globalExtremaR = YMPy.batchIntensityScale(
        folderData, p['redChannel'], p['showProgress'])
globalMinG = globalExtremaG['globalmin']
globalMaxG = globalExtremaG['globalmax']
globalMinR = globalExtremaR['globalmin']
globalMaxR = globalExtremaR['globalmax']
#%% main loop
if p['measureFields'] == 'all':
    start = 0
    stop = nFields
else:
    start = p['measureFields'][0]
    stop = p['measureFields'][1]
for field in range(start,stop):
    # read image
    dvImage = YMPy.basicDVreader(
            pathList[field], p['rolloff'], p['nChannels'], p['zFirst'])
    #%% find cells and cleanup morphology
    # find cells from brightfield step 1
    bwCellZstack = YMPy.makeCellzStack(
            dvImage, p['bwChannel'], p['showProgress'])
    # find cells from brightfield step 2
    nZslices = dvImage.shape[1]
    for z in range(nZslices):
        bwCellZstack[z,:,:] = YMPy.helpers.correctBFanomaly(
            bwCellZstack[z,:,:], p['bfAnomalyShiftVector'])
    # find cells from brightfield step 3
    rawMcl = YMPy.cellsFromZstack(bwCellZstack, p['showProgress'])[0]
    # find cells from brightfield step 4
    unbufferedMcl = YMPy.bfCellMorphCleanup(
            rawMcl, p['showProgress'], p['minAngle'], 
            p['minLength'], p['closeRadius'], p['minBudSize'])
    #%% define measurment masks
    # unbufferedMcl is the best guess at the 'true outside edge' of the 
    # cells; use it as the starting point to find a 10pixel thick cortex
    unbufferedCortexMcl = YMPy.labelCortex_mcl(
            unbufferedMcl, p['cortexWidth'])
    # because the bright field and fluorescence are not perfectly aligned,
    # and to handle inaccuracies in edge finding, also buffer out from the
    # outside edge
    buffer = YMPy.buffer_mcl(
            unbufferedMcl, p['bufferSize'], p['showProgress'])
    # merge this buffer onto the unbufferedMcl and the cortexMcl
    masterCellLabel = YMPy.merge_labelMcl(unbufferedMcl, buffer)
    cortexMcl = YMPy.merge_labelMcl(unbufferedCortexMcl, buffer)
    
    # use Otsu thresholding on the max projection of RFPmarker
    markerMclOtsu = YMPy.labelMaxproj(
            masterCellLabel, dvImage, p['mkrChannel'])
    # then use centroidCircles to uniformly mask peri-golgi regions
    markerCirclesMcl = YMPy.centroidCirclesMcl(
            markerMclOtsu.astype('bool'), masterCellLabel,
            p['markerRadius'], p['markerCircleIterations'])
    # subtract so that marker localization has precedence over cortical
    # localization
    cortexMinusMarker = YMPy.subtract_labelMcl(cortexMcl, markerCirclesMcl)
    # finally, compute mask for remaining cytoplasmic regions
    cytoplasmMcl =YMPy.subtract_labelMcl(masterCellLabel,
            YMPy.merge_labelMcl(markerCirclesMcl, cortexMinusMarker))
    #%% measure
    # measure Art1-mNG in the middle z-slice
    primaryImage = {p['measuredProteinName']:
        dvImage[p['measuredProteinChannel'],p['measuredProteinZ'],:,:]}
    # measure against buffered cortex (minus marker mask), marker, and  
    # cytoplasm
    refMclDict = {'cortex(non-' + p['markerName'] + ')':cortexMinusMarker,
                  p['markerName'] + '(circles)':markerCirclesMcl,
                  'cytoplasm':cytoplasmMcl,
                  }
    # also record field wide information
    expID = expIDlist[field]
    imageName = imageNameList[field]
    # measurement function
    results = YMPy.measure_cells(
            primaryImage, masterCellLabel, refMclDict,
            imageName, expID, field,
            globalMinG, globalMaxG, p['nHistBins'], p['showProgress'])
    # add measurements from each field to total results
    totalResults = list(np.concatenate((totalResults,results)))
    #%% quality control prep
    print('preparing quality control information')
    greenFluor = dvImage[p['measuredProteinChannel'],
                         p['measuredProteinZ'],:,:].astype(float)
    greenFluorScaled = ((greenFluor.astype('float')-globalMinG)
                        /(globalMaxG-globalMinG))
    redFluor = np.amax(dvImage[0,:,:,:],axis=0).astype(float)
    redFluorScaled = ((redFluor.astype('float')-globalMinR)
                      /(globalMaxR-globalMinR))
    qcMclList = [cortexMinusMarker, markerCirclesMcl]
    rgbQC = YMPy.prep_rgbQCImage(
            greenFluorScaled, redFluorScaled,
            qcMclList, p['rgbScalingFactors'])
    
    # add qcStack to totalQC
    totalQC.append(rgbQC)
    # record field as analyzed
    fieldsAnalyzed.append(field)
    # save unbuffered masterCellLabel (it is the most computationally
    # intensive to produce; other masks can be relatively quickly
    # drived from it)
    totalMcl.append(unbufferedMcl)
