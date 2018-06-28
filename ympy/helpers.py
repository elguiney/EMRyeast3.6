import sys
import numpy as np
import scipy.ndimage as ndimage
from skimage import draw
import matplotlib.pyplot as plt
import pickle
import pandas as pd

def mergeForTiff(imageList):
    ''' helper function to format a list of 2d-images, all with matching x,y
    dimensions, into a single multichannel image for exporting as a .tiff
    bare bones
    expects a list like [img1,img2,img3]
    '''
    imageArrayList = [np.array(img) for img in imageList]
    image8bit = [img.astype(float)-img.min()
                 for img in imageArrayList]
    image8bitb = []
    for img in image8bit:
        if img.max() != 0:
            img = (img*(255/img.max())).astype('uint8')
        else:
            img = img.astype('uint8')
        image8bitb.append(img)
    nRows = np.shape(image8bit[0])[0]
    nCols = np.shape(image8bit[0])[1]
    tiffStack = np.concatenate([img.reshape(1,nRows,nCols)
                                for img in image8bitb])
    return tiffStack

def mergeForImshow(imageList):
    imageArrayList = [np.array(img) for img in imageList]
    image8bit = [img.astype(float)-img.min()
                 for img in imageArrayList]
    image8bitb = []
    for img in image8bit:
        if img.max() != 0:
            img = (img*(255/img.max())).astype('uint8')
        else:
            img = img.astype('uint8')
        image8bitb.append(img)
    nRows = np.shape(image8bit[0])[0]
    nCols = np.shape(image8bit[0])[1]
    imshowStack = np.concatenate([img.reshape(nRows,nCols,1)
                                  for img in image8bitb], axis=2)
    return imshowStack


def correctBFanomaly(binaryCellMask, shiftVector):
    ''' correct for systematic optical bias between brightfield image and
    fluorescence images. On the Emr lab DV scope this abberation manifests as
    the top/right part of the bright field image being slightly under focused,
    and the bottom/left being slightly overfocused. This results in a generally
    sharper border on the bottom, and a stretched out border on the top, after
    findLogZeros edgefinding.
    (Not currently known if this shift drifts over time, but it seems somewhat
    stable between experiments.) A shiftVector of [2,-1] seems to provide an
    adequate correction; convolving with a kernel derived from the shift vector
    then taking only pixels where the convlution is at its maximal value
    effectively erodes only the top/right edge of cell outlines.
    '''
    kernelMidpt = np.abs(shiftVector).max()
    kernelSize = 2*kernelMidpt+1
    shiftKernel = np.zeros([kernelSize,kernelSize], dtype=int)
    rr,cc = draw.line(kernelMidpt, kernelMidpt,
                      kernelMidpt+shiftVector[0], kernelMidpt+shiftVector[1])
    shiftKernel[rr,cc] = 1
    convolved = ndimage.convolve(binaryCellMask, shiftKernel)
    correctedCellMask = np.zeros(binaryCellMask.shape, dtype=int)
    correctedCellMask[convolved==shiftKernel.sum()] = 1
    correctedCellMask = ndimage.binary_opening(correctedCellMask)
    return correctedCellMask


def progressBar_text(index,nIndices,processName):
    dispText = (processName + '- progress',
                '-'*int(20*(index+1)/nIndices),
                int(100*(index+1)/nIndices))
    sys.stdout.write('\r')
    sys.stdout.write("%s:[%-20s] %d%%" % dispText)
    sys.stdout.flush()

def progressSpinner_text(index,processName,indexType):
    dispText = (processName + '- progress',
                r'—\|/'[index%4],
                index,
                indexType)
    sys.stdout.write('\r')
    sys.stdout.write("%s:[%s] %d %s" % dispText)
    sys.stdout.flush()
    
def plt_qcFrame(qcDict,frameTitles):
    ''' for displaying a qcFrame object with plt'''
    qcFrame = qcDict['qcFrame']
    redInvFrame = qcDict['redInvFrame']
    greenInvFrame = qcDict['greenInvFrame']
    frameSize = len(redInvFrame)
    fig = plt.figure(figsize=(12,8))
    qcAx = plt.subplot2grid((2,3), (0,0), rowspan=2, colspan=2)
    plt.imshow(qcFrame)
    qcAx.axis('off')
    plt.title('Main display')
    #qcAx.xaxis.set_major_locator(plt.NullLocator())
    #qcAx.yaxis.set_major_locator(plt.NullLocator())
    redAx = plt.subplot2grid((2,3), (0,2))
    plt.imshow(redInvFrame, cmap='gray')
    redAx.xaxis.set_ticks(np.linspace(0,frameSize,5))
    redAx.yaxis.set_ticks(np.linspace(0,frameSize,5))
    redAx.xaxis.set_ticklabels([])
    redAx.yaxis.set_ticklabels([])
    plt.grid()
    plt.title(frameTitles[0])
    grnAx = plt.subplot2grid((2,3), (1,2))
    plt.imshow(greenInvFrame, cmap='gray')
    grnAx.xaxis.set_ticks(np.linspace(0,frameSize,5))
    grnAx.yaxis.set_ticks(np.linspace(0,frameSize,5))
    grnAx.xaxis.set_ticklabels([])
    grnAx.yaxis.set_ticklabels([])
    plt.grid()
    plt.title(frameTitles[1])
    
def loadResultsDf(resultsPath):
    resultsPackage = pickle.load(open(resultsPath, 'rb'))
    resultsDf = resultsPackage['totalResults']
    resultsDf = pd.DataFrame(resultsPackage['totalResults'])
    return resultsDf