''' 
Script to run EMRyeast36 on Art1-mNG, mCh-Sec7 microscopy.
will eventually incorporate into a full jupyternotebook workflow for record
keeping; runs here as a script for better interactive debugging in sypder
'''

import EMRyeast36
import pandas as pd
import pickle
import numpy as np
import datetime


targetFolder = r"C:\Users\elgui\Documents\Emr Lab Post Doc\microscopy\2018-06-12_Art1Quant_exp1"

print('beginning analysis of \n',targetFolder,'\n at ', datetime.datetime.now())
folderData = EMRyeast36.batchParse(targetFolder, expIDloc=[0,6])
nFields = folderData['nFields']
imageNameList = folderData['imagenameList']
pathList = folderData['pathlist']
expIDlist = folderData['expIDlist']

# experiment variables
startIdx = 0
globalExtrema = EMRyeast36.batchIntensityScale(
        folderData, channel=1, showProgress=True)
globalMin = globalExtrema['globalmin']
globalMax = globalExtrema['globalmax']
totalResults = []
totalQC = []
fieldsAnalyzed = []
totalMcl = []

# local function parameters
rolloff = 64
nChannels = 3
zFirst = True
bwChannel = 2
bfAnomalyShiftVector = [2,-1]
minAngle = 22
minLength = 5
closeRadius = 3
minBudSize = 75
cortexWidth = 10
bufferSize = 5
showProgress = True
mkrChannel = 0
borderSize = 10
rgbScalingFactors = [0.4,0.4]
gryScalingFactors = [0.2,0.2]


for field in range(nFields):
    print('starting image: ', imageNameList[field])
    # read image
    dvImage = EMRyeast36.basicDVreader(pathList[field], rolloff, nChannels,
                                       zFirst)
    # find cells from brightfield step 1
    bwCellZstack = EMRyeast36.makeCellzStack(dvImage, bwChannel, showProgress)
    # find cells from brightfield step 2
    nZslices = dvImage.shape[1]        
    for z in range(nZslices):
        bwCellZstack[z,:,:] = EMRyeast36.correctBFanomaly(bwCellZstack[z,:,:],
                                       bfAnomalyShiftVector)
    # find cells from brightfield step 3
    rawMcl = EMRyeast36.cellsFromZstack(bwCellZstack, showProgress)[0]
    # find cells from brightfield step 4
    unbufferedMcl = EMRyeast36.bfCellMorphCleanup(rawMcl, showProgress,
                                                  minAngle, minLength,
                                                  closeRadius, minBudSize)
    # unbufferedMcl is the best guess at the 'true outside edge' of the cells;
    # use it as the starting point to find a 10pixel thick cortex
    cortexMcl = EMRyeast36.labelCortex_mcl(unbufferedMcl, cortexWidth)
    # because the bright field and fluorescence are not perfectly aligned, and
    # to handle inaccuracies in edge finding, also buffer out from the outside
    # edge
    buffer = EMRyeast36.buffer_mcl(unbufferedMcl, bufferSize, showProgress)
    # merge this buffer onto the unbufferedMcl and the cortexMcl
    masterCellLabel = EMRyeast36.merge_labelMcl(unbufferedMcl, buffer)
    cortexMclBuffered = EMRyeast36.merge_labelMcl(cortexMcl, buffer)
    # use Otsu thresholding on the max projection of mCh-Sec7 to find golgi
    golgiMcl = EMRyeast36.labelMaxproj(masterCellLabel, dvImage, mkrChannel)
    # subtract so that golgi localization has precedence over cortical
    # localization
    cortexMinusGolgi = EMRyeast36.subtract_labelMcl(cortexMclBuffered,golgiMcl)
    # prepare for measuring:
    # measure Art1-mNG in the middle z-slice
    primaryImage = {'Art1-mNG':dvImage[1,3,:,:]}
    # measure against buffered cortex, golgi, and non-golgi buffered cortex
    refMclDict = {'cortex(buffered)':cortexMclBuffered,
                  'golgi':golgiMcl,
                  'nonGolgicortex(buffered)':cortexMinusGolgi
                  }
    # also record field wide information
    expID = expIDlist[field]
    imageName = imageNameList[field]
    # measure
    results,startIdx = EMRyeast36.measure_cells(primaryImage, masterCellLabel,
                                 refMclDict, imageName, expID, startIdx,
                                 globalMin, globalMax, showProgress)
    # add measurements from each field to total results
    totalResults = np.concatenate((totalResults,results))
    # quality control prep
    print('preparing quality control information')
    greenFluor = dvImage[1,3,:,:].astype(float)
    redFluor = np.amax(dvImage[0,:,:,:],axis=0).astype(float)
    qcMclList = [cortexMinusGolgi,golgiMcl]
    rgbQC = EMRyeast36.prep_rgbQCImage(greenFluor, redFluor,
                                       qcMclList, rgbScalingFactors)
    qcStack = EMRyeast36.prep_qcStack(rgbQC, masterCellLabel,
                                      greenFluor, redFluor, gryScalingFactors,
                                      startIdx, borderSize)
    # add qcStack to totalQC
    totalQC = np.concatenate((totalQC,qcStack))
    # record field as analyzed
    fieldsAnalyzed.append(field)
    # save unbufferedMcl
    totalMcl.append(unbufferedMcl)
    # pool and save
    print('saving progress')
    results = {'totalResults':totalResults,
               'totalQC':totalQC,
               'totalMcl':totalMcl,
               'fieldsAnalyzed':fieldsAnalyzed
               }
    pickle.dump(results, open(targetFolder +
            '\\results\\pooled_measurements.p', 'wb'))
    print(imageNameList[field],' complete at ',datetime.datetime.now())
    
