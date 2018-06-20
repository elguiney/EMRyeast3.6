import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import pickle
import pandas as pd
import random

import EMRyeast36


def initializeQC(resultsDataPath, randomSeed):
    resultsData = pickle.load(resultsDataPath, 'rb') 
    df = pd.DataFrame(list(resultsData['totalResults']))
    randomIdx = list(range(len(df)))
    random.shuffle(randomIdx,random.seed(randomSeed))
    df = df.assign(randomIdx = randomIdx, qcStatus = 'unassigned')
    resultsData['totalResults'] = df
    pickle.dump(resultsData, open(resultsDataPath, 'wb'))
    return df
    
def syncQCstate(qcAutosavePath, resultsDataPath):
    qcAutosave = pickle.load(qcAutosavePath, 'rb')
    resultsData = pickle.load(resultsDataPath, 'rb') 
    qcStatus = pd.Series(qcAutosave['statusList'], name='qcStatus')
    resultsData['totalResults']['qcStatus'] = qcStatus
    pickle.dump(resultsData, open(resultsDataPath, 'wb')) 
    
def loadQCdataframe(resultsDataPath):
    resultsData = pickle.load(resultsDataPath, 'rb')
    df = resultsData['totalResults']
    return df
    
def makeQCbuttons():
    b1 = widgets.Button(
        description=' accept',
        icon='check-square',
        tooltip='accept cell and continute; will autosave')
    b2 = widgets.Button(
        description=' reject',
        icon='minus-square',
        tooltip='reject cell and continue; will autosave')
    b3 = widgets.Button(
        description=' previous',
        icon='chevron-left',
        tooltip='go back to previously reviewed cell')
    b4 = widgets.Button(
        description=' next',
        icon='chevron-right',
        tooltip='go to next cell without changing status')
    return(b1,b2,b3,b4)

def makeQCoutputs():
    out1 = widgets.Output(layout=Layout(
            height='400px', width = '600px', border='solid'))
    out2 = widgets.Output(layout=Layout(
            border='solid'))
    return out1, out2

def makeQCframe(randLookup, resultsData, df, pathList,
                scalingFactors=[0.2, 0.3], borderSize=10):
    fieldIdx = int(df.loc[df['randomIdx'] == randLookup,'fieldIdx'])
    cellLbl = int(df.loc[df['randomIdx'] == randLookup,'localLbl'])
    rgbQC = resultsData['totalQC'][fieldIdx]
    masterCellLabel = resultsData['totalMcl'][fieldIdx]
    dvImage = EMRyeast36.basicDVreader(pathList[fieldIdx],rolloff=64)
    greenFluor = dvImage[1,3,:,:].astype(float)
    redFluor = np.amax(dvImage[0,:,:,:],axis=0).astype(float)
    mask = np.zeros(masterCellLabel.shape, dtype='uint8')
    mask[np.abs(masterCellLabel) == cellLbl] = 1
    cellProps = regionprops(mask)
    nRows,nCols = masterCellLabel.shape
    ymin,xmin,ymax,xmax = cellProps[0].bbox
    height = ymax-ymin
    width = xmax-xmin

    squareSide = np.max([width,height]) + borderSize
    centerY = int(ymax - height/2)
    centerX = int(xmax - width/2)
    ytop = int(min(max(squareSide / 2, centerY), nRows - squareSide / 2)
               - squareSide/2)
    xtop = int(min(max(squareSide / 2, centerX), nCols - squareSide / 2)
               - squareSide/2)

    greenFluor = greenFluor.astype('float')
    redFluor = redFluor.astype('float')
    grayscaleQC = np.max(rgbQC, axis=2)
    grayQCz3 = np.concatenate(3*[grayscaleQC.reshape(nRows,nCols,1)],axis=2)
    maskz3 = np.concatenate(3*[mask.reshape(nRows,nCols,1)],axis=2)
    redScaled = ((redFluor-redFluor.min())
                 / (scalingFactors[0]*(redFluor.max()-redFluor.min())))
    redScaled[redScaled > 1] = 1
    redInv = (255*(1 - redScaled)).astype('uint8')
    greenScaled = ((greenFluor-greenFluor.min())
                   / (scalingFactors[0]*(greenFluor.max()-greenFluor.min())))
    greenScaled[greenScaled > 1] = 1
    greenInv = (255*(1 - greenScaled)).astype('uint8')
    flatMask = np.ndarray.flatten(maskz3)
    flatDisplay = np.ndarray.flatten(grayQCz3).astype('uint8')
    flatDisplay[flatMask==1] = np.ndarray.flatten(rgbQC)[flatMask==1]
    display = flatDisplay.reshape((nRows,nCols,3))
    qcFrame = display[ytop:ytop+squareSide,xtop:xtop+squareSide,:]
    redInvFrame = redInv[ytop:ytop+squareSide,xtop:xtop+squareSide]
    greenInvFrame = greenInv[ytop:ytop+squareSide,xtop:xtop+squareSide]
    qcDict = {'qcFrame':qcFrame,
              'redInvFrame':redInvFrame,
              'greenInvFrame':greenInvFrame}
    return(qcDict)
    
def frameDisplay(qcDict,frameTitles):
    '''display a qcFrame object, for use with an output widget'''
    qcFrame = qcDict['qcFrame']
    redInvFrame = qcDict['redInvFrame']
    greenInvFrame = qcDict['greenInvFrame']
    frameSize = len(redInvFrame)
    fig = plt.figure(figsize=(12,8))
    qcAx = plt.subplot2grid((2,3), (0,0), rowspan=2, colspan=2)
    qcAx.imshow(qcFrame)
    qcAx.axis('off')
    plt.title('Main display: blinded cell idx = '+str(randLookup))
    redAx = plt.subplot2grid((2,3), (0,2))
    redAx.imshow(redInvFrame, cmap='gray')
    redAx.xaxis.set_ticks(np.linspace(0,frameSize,5))
    redAx.yaxis.set_ticks(np.linspace(0,frameSize,5))
    redAx.xaxis.set_ticklabels([])
    redAx.yaxis.set_ticklabels([])
    redAx.grid()
    plt.title(frameTitles[0])
    grnAx = plt.subplot2grid((2,3), (1,2))
    grnAx.imshow(greenInvFrame, cmap='gray')
    grnAx.xaxis.set_ticks(np.linspace(0,frameSize,5))
    grnAx.yaxis.set_ticks(np.linspace(0,frameSize,5))
    grnAx.xaxis.set_ticklabels([])
    grnAx.yaxis.set_ticklabels([])
    grnAx.grid()
    plt.title(frameTitles[1])
    clear_output(wait=True)
    plt.show()
    
def get_context(randLookup, indexArray, statusList):
    pastRandIdces = list(range(max(0,randLookup-5),randLookup+1))
    pastStatus = [statusList[pastloc] for 
                  pastloc in indexArray[max(0,randLookup-5):randLookup+1]]
    return(pastRandIdces, pastStatus)
    
def print_context(currentlocation, randLookup, statusList,
                  pastRandIdces, pastStatus, autosave):
    print('current cell (blind Idx):',
          str(randLookup),
          '\nstatus: ',str(statusList[currentlocation])),'\n'
    if autosave: print('\nautosave enabled')
    print('\n',str(total),' cells total',
          '\n\nprevious cells:\n')
    for pair in zip(reversed(pastRandIdces),reversed(pastStatus)):
        print(pair)

def saveQCstate(randLookup, statusList, resultsDirectory):
    stateDict = {'randLookup':randLookup,
                 'statusList':statusList}
    pickle.dump(stateDict,open(resultsDirectory+'/qcAutosave.p','wb'))  
        
def makeQC_clickfunctions(randLookupStart, resultsData, df, pathList,
                          frameTitles, resultsDirectory, autosave=True):
    '''
    builds click functions for b1, b2, b3, and b4 generated by makeQCbuttons()
    each funtion updates the qc statusList based on the button press
    (b1 = accept and advance, b2 = reject and advance,
     b3 = go back, b4 = advance)
    
    '''
    
    global randLookup
    global total
    total = len(df)
    randLookup = randLookupStart
    randArray = list(df['randomIdx'])
    indexArray = [randArray.index(i) for i in range(len(randArray))]
    statusList = list(df['qcStatus'])
    out1,out2 = makeQCoutputs()

           
    def click_b1(b):
        global randLookup
        location = indexArray[randLookup]
        statusList[location] = 'accepted'
        if autosave: saveQCstate(randLookup, statusList, resultsDirectory)
        pastRandIdces, pastStatus = get_context(
                randLookup, indexArray, statusList)
        if randLookup == total-1:
            with out2:
                out2.clear_output()
            with out2:
                print('finshed')
        else:
            randLookup += 1
            currentlocation = indexArray[randLookup]
            qcDict = makeQCframe(randLookup, resultsData, df, pathList)
            with out1:
                frameDisplay(qcDict,frameTitles)
            with out2:
                out2.clear_output()
            with out2:
                print_context(currentlocation, randLookup, statusList,
                              pastRandIdces, pastStatus, autosave)
            return(statusList)
                
    def click_b2(b):
        global randLookup
        location = indexArray[randLookup]
        statusList[location] = 'rejected'
        if autosave: saveQCstate(randLookup, statusList, resultsDirectory)
        pastRandIdces, pastStatus = get_context(
                randLookup, indexArray, statusList)
        if randLookup == total-1:
            with out2:
                out2.clear_output()
            with out2:
                print('finshed')
        else:
            randLookup += 1
            currentlocation = indexArray[randLookup]
            qcDict = makeQCframe(randLookup, resultsData, df, pathList)
            with out1:
                frameDisplay(qcDict,frameTitles)
            with out2:
                out2.clear_output()
            with out2:
                print_context(currentlocation, randLookup, statusList,
                              pastRandIdces, pastStatus, autosave)
            return(statusList)
        
    def click_b3(b):
        global randLookup
        randLookup -= 1
        currentlocation = indexArray[randLookup]
        pastRandIdces, pastStatus = get_context(
                randLookup-1, indexArray, statusList)
        qcDict = makeQCframe(randLookup, resultsData, df, pathList)
        with out1:
            frameDisplay(qcDict,frameTitles)
        with out2:
            out2.clear_output()
        with out2:
            print_context(currentlocation, randLookup, statusList,
                          pastRandIdces, pastStatus, autosave)

    def click_b4(b):
        global randLookup
        randLookup += 1
        currentlocation = indexArray[randLookup]
        pastRandIdces, pastStatus = get_context(
                randLookup-1, indexArray, statusList)
        qcDict = makeQCframe(randLookup, resultsData, df, pathList)
        with out1:
            frameDisplay(qcDict,frameTitles)
        with out2:
            out2.clear_output()
        with out2:
            print_context(
                        currentlocation, randLookup, pastRandIdces, pastStatus)
             
    return(click_b1, click_b2, click_b3, click_b4, out1, out2, statusList)