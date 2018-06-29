'''
Script that emulates measure_GFP_wRFPmarker
use for testing
'''

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

import ympy
folderPath = ('C:/Users/elgui/Documents/Emr Lab Post Doc/microscopy/'
              '2018-06-29_Art1wt-YPD-timcourse_exp1')
p = ympy.pipelines.GFPwMarkerPipeline(
        measureFields=[0,1])
p.initialize(folderPath)

print('beginning analysis of \n', p.folderPath,'\n at ',
  datetime.datetime.now())
# initialize experiment variables
totalResults = []
totalQC = []
fieldsAnalyzed = []
totalMcl = []
    
resultsDirectory = p.folderPath + '/results/'
if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory)

#%% measure global values (slow for big datasets)
globalExtremaG = ympy.batchIntensityScale(
        p.folderData, p.greenChannel, p.showProgress)
globalExtremaR = ympy.batchIntensityScale(
        p.folderData, p.redChannel, p.showProgress)
globalMinG = globalExtremaG['globalmin']
globalMaxG = globalExtremaG['globalmax']
globalMinR = globalExtremaR['globalmin']
globalMaxR = globalExtremaR['globalmax']
#%% main loop
if p.measureFields == 'all':
    start = 0
    stop = p.nFields
else:
    start = p.measureFields[0]
    stop = p.measureFields[1]
field = start
# read image
dvImage = ympy.basicDVreader(
        p.pathList[field],
        p.rolloff,
        p.nChannels,
        p.zFirst)
#%% find cells and cleanup morphology
# find cells from brightfield step 1
bwCellZstack = ympy.makeCellzStack(
        dvImage, p.bwChannel, p.showProgress)
# find cells from brightfield step 2
nZslices = dvImage.shape[1]
for z in range(nZslices):
    bwCellZstack[z,:,:] = ympy.helpers.correctBFanomaly(
        bwCellZstack[z,:,:], p.bfAnomalyShiftVector)
# find cells from brightfield step 3
rawMcl = ympy.cellsFromZstack(bwCellZstack, p.showProgress)[0]
# find cells from brightfield step 4
unbufferedMcl = ympy.bfCellMorphCleanup(
        rawMcl, p.showProgress, p.minAngle, 
        p.minLength, p.closeRadius, p.minBudSize)
#%% define measurment masks
# unbufferedMcl is the best guess at the 'true outside edge' of 
# the cells; use it as the starting point to find a 10pixel thick 
# cortex
unbufferedCortexMcl = ympy.labelCortex_mcl(
        unbufferedMcl, p.cortexWidth)
# because the bright field and fluorescence are not perfectly 
# aligned, and to handle inaccuracies in edge finding, also buffer 
# out from the outside edge
buffer = ympy.buffer_mcl(
        unbufferedMcl, p.bufferSize, p.showProgress)
# merge this buffer onto the unbufferedMcl and the cortexMcl
masterCellLabel = ympy.merge_labelMcl(unbufferedMcl, buffer)
cortexMcl = ympy.merge_labelMcl(unbufferedCortexMcl, buffer)

# use Otsu thresholding on the max projection of RFPmarker
markerMclOtsu = ympy.labelMaxproj(
        masterCellLabel, dvImage, p.mkrChannel)
# then use centroidCircles to uniformly mask peri-golgi regions
markerCirclesMcl = ympy.centroidCirclesMcl(
        markerMclOtsu.astype('bool'), masterCellLabel,
        p.markerRadius, p.markerCircleIterations)
# subtract so that marker localization has precedence over cortical
# localization
cortexMinusMarker = ympy.subtract_labelMcl(
        cortexMcl, markerCirclesMcl)
# finally, compute mask for remaining cytoplasmic regions
cytoplasmMcl =ympy.subtract_labelMcl(masterCellLabel,
        ympy.merge_labelMcl(markerCirclesMcl, cortexMinusMarker))
#%% measure
# measure Art1-mNG in the middle z-slice
primaryImage = {p.measuredProteinName:
    dvImage[p.measuredProteinChannel,
            p.measuredProteinZ, :, :]}
# measure against buffered cortex (minus marker mask), marker, and  
# cytoplasm
refMclDict = {
        'cortex(non-' + p.markerName + ')': cortexMinusMarker,
        p.markerName + '(circles)': markerCirclesMcl,
        'cytoplasm': cytoplasmMcl,
              }
# also record field wide information
# measurement function
results = ympy.measure_cells(
        primaryImage, masterCellLabel, refMclDict,
        p.imageNameList[field], p.expIDlist[field], field,
        globalMinG, globalMaxG, p.nHistBins, p.showProgress)
# add measurements from each field to total results
totalResults = list(np.concatenate((totalResults,results)))
#%% quality control prep
print('preparing quality control information')
greenFluor = dvImage[p.measuredProteinChannel,
                     p.measuredProteinZ,:,:].astype(float)
greenFluorScaled = ((greenFluor.astype('float')-globalMinG)
                    /(globalMaxG-globalMinG))
redFluor = np.amax(
        dvImage[p.mkrChannel,:,:,:],axis=0).astype(float)
redFluorScaled = ((redFluor.astype('float')-globalMinR)
                  /(globalMaxR-globalMinR))
qcMclList = [cortexMinusMarker, markerCirclesMcl]
rgbQC = ympy.prep_rgbQCImage(
        greenFluorScaled, redFluorScaled,
        qcMclList, p.rgbScalingFactors)

# add qcStack to totalQC
totalQC.append(rgbQC)
# record field as analyzed
fieldsAnalyzed.append(field)
# save unbuffered masterCellLabel (it is the most computationally
# intensive to produce; other masks can be relatively quickly
# drived from it)
totalMcl.append(unbufferedMcl)
#%% pool and save
print('saving progress')
resultsDic = {'totalResults': totalResults,
           'totalQC': totalQC,
           'fieldsAnalyzed': fieldsAnalyzed,
           'totalMcl': totalMcl,
           'parameters': p.p
           }