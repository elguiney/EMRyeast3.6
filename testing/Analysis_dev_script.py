'''
analysis dev script
'''
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from ipywidgets import Layout
import pickle
import random
import seaborn as sns
from skimage.measure import regionprops

import ympy

completedPath = ('C:/Users/elgui/Documents/Emr Lab Post Doc/microscopy/'
                 'Art1Quant/results/2018-06-26_mereged_completed.p')
resultsDfFinal = pickle.load(open(completedPath, 'rb'))

lookup = {
        'AA2800':('Art1','SCD'),
        'AA2801':('Art1','YPD 1hr'),
        'AA2802':('Art1','YPD 2hr'),
        'AA2810':('Art1(ΔN,1,2,C)','SCD'),
        'AA2811':('Art1(ΔN,1,2,C)','YPD 1hr'),
        'AA2812':('Art1(ΔN,1,2,C)','YPD 2hr'),
        'AA2820':('Art1(ΔN)','SCD'),
        'AA2821':('Art1(ΔN)','YPD 1hr'),
        'AA2822':('Art1(ΔN)','YPD 2hr'),
        'AA2830':('Art1(Δ1)','SCD'),
        'AA2831':('Art1(Δ1)','YPD 1hr'),
        'AA2832':('Art1(Δ1)','YPD 2hr'),
        'AA2840':('Art1(Δ2)','SCD'),
        'AA2841':('Art1(Δ2)','YPD 1hr'),
        'AA2842':('Art1(Δ2)','YPD 2hr'),
        'AA2850':('Art1(Δ3)','SCD'),
        'AA2851':('Art1(Δ3)','YPD 1hr'),
        'AA2852':('Art1(Δ3)','YPD 2hr'),
        'AA2860':('Art1(ΔC)','SCD'),
        'AA2861':('Art1(ΔC)','YPD 1hr'),
        'AA2862':('Art1(ΔC)','YPD 2hr'),
        }

# check that rejected list contains bad cells
qcAutosavePath = ('C:/Users/elgui/Documents/Emr Lab Post Doc/microscopy/'
                  'Art1Quant/results/qcAutosave.p')
resultsPath = ('C:/Users/elgui/Documents/Emr Lab Post Doc/microscopy/'
               'Art1Quant/results/2018-06-25_analysis.p')
qcAutosave = pickle.load(open(qcAutosavePath, 'rb'))
qcStatus = pd.Series(qcAutosave['statusList'], name='qcStatus')
resultsDf = ympy.helpers.loadResultsDf(resultsPath)
resultsDf = resultsDf[resultsDf['expID'].str.contains('AA')]
resultsDf['qcStatus'] = qcStatus
resultsDfRejected = resultsDf[resultsDf['qcStatus']=='rejected']
totalResults = pickle.load(open(resultsPath, 'rb'))

p = dict(
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
        measureFields = [10,11], #either 'all', to measure entire folder, or
            #a slice. [5,10] would measure fields 5,6,7,8 and 9 as they
            #appear in folderData['pathList'] from ympy.batchParse()
        smoothingKernelWidth = 5, #smoothing kernel for histogram calibrated
            #foci-brightness and foci-area measurments
        qcBorderSize = 10 #border around cells for qc
        )

def checkPanel(startIdx, df, totalQC, totalMcl, p, size=[8,10]):
    fields = [df.iloc[n]['fieldIdx'] 
              for n in range(startIdx, startIdx + np.prod(size))]
    cellLbls = [df.iloc[n]['localLbl'] 
                for n in range(startIdx, startIdx + np.prod(size))]
    fig, axs = plt.subplots(size[0], size[1])
    for ax in axs.flatten():
        ax.set_xticklabels('')
        ax.set_yticklabels('')
    for fieldIdx, cellLbl, ax in zip(fields, cellLbls, axs.flatten()):
        rgbQC = totalQC[fieldIdx]
        masterCellLabel = totalMcl[fieldIdx]
        mask = np.zeros(masterCellLabel.shape, dtype='uint8')
        mask[np.abs(masterCellLabel) == cellLbl] = 1
        cellProps = regionprops(mask)
        nRows,nCols = masterCellLabel.shape
        ymin,xmin,ymax,xmax = cellProps[0].bbox
        height = ymax-ymin
        width = xmax-xmin
    
        squareSide = np.max([width,height]) + p['qcBorderSize']
        centerY = int(ymax - height/2)
        centerX = int(xmax - width/2)
        ytop = int(min(max(squareSide/2, centerY), nRows - squareSide/2)
                   - squareSide/2)
        xtop = int(min(max(squareSide/2, centerX), nCols - squareSide/2)
                   - squareSide/2)
        grayscaleQC = np.max(rgbQC, axis=2)
        grayQCz3 = np.concatenate(
                3*[grayscaleQC.reshape(nRows, nCols, 1)], axis=2)
        maskz3 = np.concatenate(3*[mask.reshape(nRows,nCols,1)], axis=2)
        flatMask = np.ndarray.flatten(maskz3)
        flatDisplay = np.ndarray.flatten(grayQCz3).astype('uint8')
        flatDisplay[flatMask==1] = np.ndarray.flatten(rgbQC)[flatMask == 1]
        display = flatDisplay.reshape((nRows, nCols, 3))
        qcFrame = display[ytop : ytop+squareSide, xtop : xtop+squareSide, :]
        ax.imshow(qcFrame)
        ympy.helpers.progressSpinner_text(cellLbl, 'drawing', 'cell')
        
class PipelineParameter:
    done=0        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        