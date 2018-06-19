"""
EMRyeast36_v0.1

Emr lab image analysis module
adapted for python 3.6
contains endocytosis profiling modules (vacmorph, vacmorph advanced) -not yet
    tested
extends methods to quantify flourescence localization against two markers
    A) higher precision cortex localization from bright field image
    B) punctate RFP colocalization marker (tested with mCh-Sec7)
    and evaluate punctate, membrane coloaclization (tested with Art1-mNG)
"""

import os
import numpy as np
import scipy as sp
import mrcfile
import warnings
import scipy.ndimage as ndimage
from skimage.measure import regionprops
from skimage import draw
from skimage.filters import threshold_otsu
import skimage.morphology as morph
import matplotlib as mpl
import matplotlib.pyplot as plt
import tifffile
from timeit import default_timer as timer
import sys
import basicText2im

def batchParse(targetFolder, expIDloc, imageExtension='R3D_D3D.dv'):
    '''updated parser based on EMRyeastv2 parser

    targetFolder: path to the top level analysis folder
    imageExtension: terminating characters of images to includ in analysis.
        defaults to "R3D_D3D.dv", the flag and extension deltavision software

    '''
    pathlist = []
    imagenameList = []
    expIDlist = []
    extensionLen = len(imageExtension)
    for root, dirs, files in os.walk(targetFolder):
        for item in files:
            if item[-extensionLen::] == imageExtension:
                pathlist.append(os.path.join(root, item))
                imagenameList.append(item)
                expID = item[expIDloc[0]:expIDloc[1]]
                expIDlist.append(expID)
    folderData = {'imagenameList' : imagenameList,
                   'pathlist' : pathlist,
                   'nFields': len(pathlist),
                   'expIDlist': expIDlist}
    return folderData

def batchIntensityScale(folderData, channel, showProgress=True):
    nFields = folderData['nFields']
    pathList = folderData['pathlist']
    maxlist = []
    minlist = []
    for field in range(nFields):
        dvImage = basicDVreader(
                pathList[field],rolloff=64,nChannels=3,zFirst=True)
        fieldMax = np.max(dvImage[channel,:,:,:])
        maxlist.append(fieldMax)
        fieldMin = np.min(dvImage[channel,:,:,:])
        minlist.append(fieldMin)
        if showProgress: progressBar_text(field,nFields,'scaling intensities')
    if showProgress: print()
    globalmax = np.max(maxlist)
    globalmin = np.min(minlist)
    result = {'globalmax':globalmax, 'globalmin':globalmin}
    return result

def basicDVreader(imagePath, rolloff, nChannels=3, zFirst=True):
    ''' very simple function to read .dv files as formatted by deltavision
    microscopes.

    imagePath is the complete file path of the image
    for deconvolved images, rolloff specifies the width of the border in pixels
        to be cropped before further processing
    nChannels (defualt 3) is the number of fluorescence and bright field
        channels
    zFirst (default True); boolean, is the the order of the .dv tiff stack z-
        first or channel-first (often, zFirst = True for R3D_D3D, zFirst = False
        for R3D)
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with mrcfile.open(imagePath,permissive=True) as dv:
            dvData = dv.data[:]
            dvShape = dvData.shape
            nZslices = int(dvShape[0]/nChannels)
            dvImage = np.zeros([nChannels,nZslices,dvShape[1],dvShape[2]],
                               dtype='uint16')
            if zFirst:
                for channel in range(nChannels):
                    dvImage[channel,:,:,:] = dvData[channel*nZslices
                                                    :channel*nZslices+nZslices,
                                                    :,:]
            else:
                for channel in range(nChannels):
                    dvImage[channel,:,:,:] = dvData[channel::nChannels,:,:]
    dvImage = dvImage[:,:,rolloff:-rolloff,rolloff:-rolloff]
    return dvImage

def findLogZeros(image, logKernelSize, gradKernelSize, gradstrength,
                 gradmethod='any'):
    ''' improved implementation of find log zeros.
    use np.diff of np.signbit to detect sign changes.
    about 20 times faster than a loop like:
        for y in range(ySize):
            for x in range(xSize):

    Also, keep track of positive gradient and negative gradient, and shift
    appropriately so all edges are on the negative side of a zero contour,
    rather than always appearing below/right of a zero, as is default behavior
    for np.diff and the old for-loop logZeros method.

    to find all zeros, set gradstrength to 0.

    gradmethod:
        'any' returns all edges where any pixel is greater than gradstrength
        'mean' returns all edges where the average gradient is greater than
            gradsthrength. 'mean' may be computationally quite taxing.

    to find typical cell edges from a brightfield image,
    logKernelSize = 4
    gradKernelSize = 2
    gradstrength = 0.05
    gradmethod = 'any'

    to find typical fluorescence features,
    logKernelSize = 2.5
    gradKernelSize = 2.5
    gradstrength = 0.05
    gradmethod = 'mean'
    '''
    # find raw zeros of the laplace of gaussian convolution of the input image
    scaledImage = ((image.astype('float64') - image.min()) /
                   (image.max() - image.min()))
    #laplace of gaussian
    log = ndimage.gaussian_laplace(scaledImage, logKernelSize)
    # initialize for zeros of laplace of gaussian
    logZeros = np.zeros(log.shape, np.bool)
    xZerosRaw = np.diff(np.signbit(log).astype(int),axis=1) # row zeros
    yZerosRaw = np.diff(np.signbit(log).astype(int),axis=0) # column zeros
    # find the indices for left, right, top and bottom edges
    leftZerosIdx = np.where(xZerosRaw==1)
    rightZerosIdx = np.where(xZerosRaw==-1)
    topZerosIdx = np.where(yZerosRaw==1)
    bottomZerosIdx = np.where(yZerosRaw==-1)
    # left and top zeros must be shifted by one column/row respectively
    logZeros[:,1:][leftZerosIdx] = True
    logZeros[1:,:][topZerosIdx] = True
    # right and bottom zeros can be added directly
    logZeros[rightZerosIdx] = True
    logZeros[bottomZerosIdx] = True
    # filter by gradient, treating connected pixels as a single edge, and
    # discarding any edge as specified by gradmethod
    grad = ndimage.gaussian_gradient_magnitude(scaledImage, gradKernelSize)
    lbl_logZeros, nEdges = ndimage.label(logZeros,
                                         [[1, 1, 1],
                                          [1, 1, 1],
                                          [1, 1, 1]])
    if gradmethod == 'mean':
        for edge in range(nEdges):
            if grad[lbl_logZeros==edge].mean() < gradstrength:
                logZeros[lbl_logZeros==edge] = 0
    elif gradmethod == 'any':
        logZeros = np.zeros(log.shape, np.bool)
        cutoffEdgeLabels = list(set(lbl_logZeros[grad > gradstrength]))[1:]
        for edge in cutoffEdgeLabels:
            logZeros[lbl_logZeros==edge] = 1
    return logZeros

def fillLogZeros(logZeros,perimCutoff=5,minAreaCutoff=500,maxAreaCutoff=15000):
    ''' recieves input from findLogZeros, and performs basic morphology checks
    before returning a filled image, where background = 0 and cells = 1
    '''
    filledLogZeros = ndimage.binary_fill_holes(logZeros)
    # Remove poorly segmented cells
    filledLabels, nObj = ndimage.label(filledLogZeros)
    filledRegionprops = regionprops(filledLabels)
    for obj in range(nObj):
        objPerim = filledRegionprops[obj].perimeter
        objArea = filledRegionprops[obj].area
        if objPerim/(objArea**(0.5)) > perimCutoff:
            filledLogZeros[filledLabels == obj+1] = 0
        elif objArea < minAreaCutoff:
            filledLogZeros[filledLabels == obj+1] = 0
        elif objArea > maxAreaCutoff:
            filledLogZeros[filledLabels == obj+1] = 0
    return filledLogZeros

def makeCellzStack(dvImage, bwChannel=2, showProgress=True):
    ''' chain multiple calls to findLogZeros and fillLogZeros to find cells
    across a multiple z-field image'''
    nZslices = dvImage.shape[1]
    bwCellZstack = np.zeros(dvImage.shape[1:])
    for z in range(nZslices):
        logZeros = findLogZeros(dvImage[bwChannel,z,:,:], logKernelSize=4,
                                gradKernelSize=2, gradstrength=0.05)
        filledLogZeros = fillLogZeros(logZeros)
        bwCellZstack[z,:,:] = filledLogZeros
        if showProgress: progressBar_text(z,nZslices,'processing edges')
    if showProgress: print()
    return bwCellZstack



def cellsFromZstack(bwCellZstack, showProgress=True):
    '''find the best, non duplicate, non overlapping cell outlines from a
    z-stack of binary cell objects, eg as found with EMRyeast.bfFlipFlop'''
    nZstacks = bwCellZstack.shape[0]
    stackData = {'nObj': np.zeros(nZstacks,dtype='int16'),
                 'bwLabel': [], 'props': []}
    for z in range(nZstacks):
        bwLabel, nObj = ndimage.label(bwCellZstack[z, :, :])
        props = regionprops(bwLabel)
        stackData['nObj'][z] = nObj
        stackData['bwLabel'].append(bwLabel)
        stackData['props'].append(props)
    masterCellLabel = np.zeros(bwCellZstack.shape[1:], dtype=int)
    lookupList = []
    obj = 0
    source = np.copy(stackData['bwLabel'])
    while source.max():
        alreadyWritten = False
        nextTuple = np.nonzero(source)
        nextZ = nextTuple[0][0]
        nextY = nextTuple[1][0]
        nextX = nextTuple[2][0]
        nextObj = source[nextZ, nextY, nextX]
        obj += 1
        ref = stackData['bwLabel'][nextZ]
        candidates = np.zeros([nZstacks, 2], dtype='int32')
        for z in range(nZstacks):
            query = source[z]
            ans = query[ref == nextObj]
            if len(ans) != 0: ansID = sp.stats.mode(ans)[0][0]
            if ansID != 0 and len(ans) != 0:
                candidates[z, 0] = stackData['props'][z][ansID-1].area
                candidates[z, 1] = ansID
                source[z][np.where(query == ansID)] = 0
        # and check masterCellLabel
        query = np.array(masterCellLabel, dtype=int)
        ans = query[ref == nextObj]
        if len(ans) != 0: ansID = sp.stats.mode(ans)[0][0]
        if ansID != 0 and len(ans) != 0:
            alreadyWritten = True
            obj -= 1
        if not alreadyWritten:
            bestField = candidates[:, 0].argmax()
            bestID = candidates[bestField, 1]
            lookupList.append((bestField, bestID))
            masterCellLabel[stackData['bwLabel'][bestField] == bestID] = obj
        if showProgress: progressSpinner_text(obj,'finding cells','cells')
    stackData['masterCellLabel'] = masterCellLabel.astype('uint16')
    stackData['lookupList'] = lookupList
    if showProgress: print()
    return masterCellLabel.astype('uint16'), stackData

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

def edgeLabel(cellMask):
    ''' takes a single identified cell, cellMask, and returns a list of pixels
    along the perimiter, ordered going clockwise around the cell.

    in the cell below, with perimiter marked with letters, and inside #'s,
    search begins for starting point (c) and proceeds clockwise from *
    .*......
    ..cde...    =>       0 1 2
    ab###fg.             7 c 3
    #######h             6 5 4

    identifying d as the next point along the perimiter. the search continues
    iteratively, always clockwise from immediately after the previous starting
    point

    y,x coordinates are stored as a list of (y,x) tuples with the ith tuple
    marking the ith point encountered clockwise around the cell.

    '''

    cellMask_nz = np.nonzero(cellMask)
    # set initial y and x search position
    yInt = cellMask_nz[0][0]
    xInt = cellMask_nz[1][0]
    yS = yInt
    xS = xInt

    # setup search variables
    go = True
    startPos = 0
    pattern = [ 0, 1, 2, 5, 8, 7, 6, 3]
    deltaY =  [-1,-1,-1, 0, 0, 0, 1, 1, 1]
    deltaX =  [-1, 0, 1,-1, 0, 1,-1, 0, 1]
    newStart =[ 5, 6, 7, 4, 8, 0, 3, 2, 1]
    # initialize edge label
    edgeLabels = []


    while go:
        tempPattern = pattern[startPos:]+pattern[:startPos]
        edgeLabels.append((yS,xS)) # record indices
        # temporarily store old indices
        yL = yS
        xL = xS
        # find next indices:
        # get the neighborhood of the search position
        nbhd = cellMask[yS-1:yS+2,xS-1:xS+2]
        nbhd_flat = np.ndarray.flatten(nbhd)
        # convert to clockwise list
        nbhd_flat_cw = [nbhd_flat[i] for i in tempPattern]
        next_idx = tempPattern[np.nonzero(nbhd_flat_cw)[0][0]]
        # update yS and xS
        yS = yL + deltaY[next_idx]
        xS = xL + deltaX[next_idx]
        # update startPos
        startPos = newStart[next_idx]
        if (yS == yInt) & (xS == xInt):
            go = False
    return edgeLabels

def angleFinder(cellMask,edgeLabels,lineLength=[4,7]):
    ''' takes a single identified cell (cellMask), along with a clockwise
    perimiter map from edgeLabel. Finds the local curvature along the
    perimiter.lineLength is the distance range from each point at which
    curvature is calculated.
    in the example below, where "." is external space and "#" the inside of a
    cell, with the default lineLength = [4,7], the angle at g is the average
    of a\g/m, b\g/l and c\g/k; while with a lineLength = [2,3] it is e\g/i.

    ................
    ..........klmn..
    ab.......j####op
    ##cde..hi#######
    #####fg#########

    curvature sign is defined so that convex regions of the perimiter have
    positive curvature'''

    perimLength = len(edgeLabels)
    localThetas = []
    for idx in range(perimLength):
        lkpFor = np.mod(range(idx+lineLength[0],idx+lineLength[1]),perimLength)
        lkpRev = np.mod(range(idx-lineLength[1],idx-lineLength[0]),perimLength)
        y = edgeLabels[idx][0]
        x = edgeLabels[idx][1]
        yFor = [edgeLabels[i][0]-y for i in lkpFor]
        xFor = [edgeLabels[i][1]-x for i in lkpFor]
        yRev = [edgeLabels[i][0]-y for i in reversed(lkpRev)]
        xRev = [edgeLabels[i][1]-x for i in reversed(lkpRev)]
        thetaList = np.array([np.arctan2(yRev[j],xRev[j])
                             -np.arctan2(yFor[j],xFor[j])
                             for j in range(len(lkpFor))])
        thetaList = np.pi - np.mod(thetaList,2*np.pi)
        newTheta = np.mean(thetaList)
        if newTheta == 0:
            if idx == 0:
                oldSign = 1
            else:
                oldSign = np.sign(localThetas[idx-1])
                newTheta = 0.000001 * oldSign
        localThetas.append(newTheta)
    return localThetas

def bfCellMorphCleanup(mcl, showProgress,
                       minAngle=22, minLength=5,
                       closeRadius=3, minBudSize = 75):
    '''
    process mcl, a labeled master cell list np-array from
    EMRyeast.cellsFromZstack().
    measures morphology and identifies buds, cleans up cell outlines
    (1) Cleans up mcl with a binary closing to remove edge-holes, using
        closeRadius to create a disk structuring element
    (2) Uses EMRyeast.edgeLabel() and EMRyeast.angleFinder() to identify
    regions of negative curvature
        - regions of negative curvature are checked for the following
        a) minimum length (default minLength = 5 pixels)
        b) minimum curvature (averaged across the region, default minAngle =
        22 degrees)
    (3) a) cells with no regions of negative curvature are left as is (usually,
        these are unbudded cells))
        b) cells with one region of substantial negative curvature usually
        result from bright field optical artifacts where two cells are closely
        apposed (at a late bud neck, or between closely clumped cells). These
        are repaired by a convex hull.
        c) For cells with exactly two negative curvature regions, segment
        between to produce a bud and mother
        d) For cells with more than two negative curvature regions, segment
        between two most negative regions (usually these are the neck), then
        convex hull the remaining.
    (4) Outputs:
        resegmented cleanedMcl, with new labels.
        does not match labels from input mcl.
        mother cells are identified with a positive integer label
        corresponding buds are labeled with -1 * mother_label
    '''

    minAngleRads = (float(minAngle)/180*np.pi)
    closeStruct = morph.disk(closeRadius,dtype='float')
    cleanedMcl = np.zeros(np.shape(mcl),dtype='int16')
    #resegment mcl
    flatMcl = mcl.astype('bool')
    relabeledMcl, nCells = ndimage.label(flatMcl)
    idxOffset = 0
    for cell in range(nCells):
        cellIdx = cell + 1
        masterCellMask = np.zeros(np.shape(mcl), dtype='bool')
        masterCellMask[relabeledMcl==cellIdx] = 1
        try:
            cellIdx = cellIdx + idxOffset
            cellProps = regionprops(masterCellMask.astype('int'))
            box = cellProps[0].bbox
            cellMask = masterCellMask[box[0]-1:box[2]+1,box[1]-1:box[3]+1]
            cellMask = ndimage.binary_fill_holes(ndimage.binary_closing
                                                 (cellMask,closeStruct))
            cmsk_lbls, nlbls = ndimage.label(cellMask, structure = np.ones((3,3)))
            sizes = [len(cmsk_lbls[cmsk_lbls==idx+1]) for idx in range(nlbls)]
            cellMask[cmsk_lbls!=np.argmax(sizes)+1]=0
            edgeLabels = edgeLabel(cellMask)
            localThetas = np.array(angleFinder(cellMask,edgeLabels))
            ''' prep localThetas for finding curves by
            (1) wrapping the end and (2) replacing zeros with small values '''
            curveZeros = np.where(np.diff(np.sign(np.append(localThetas,
                                                            localThetas[0]))))[0]
            ''' assume to always be starting at regions of positive curvature,
            based on geometry of edgeLabel()'''
            negCurves = np.reshape(curveZeros,(-1,2))
            negCurveLengths = np.diff(negCurves)
            nNegCurves = int(len(curveZeros)/2)
            negCurveAngles = np.zeros(nNegCurves)
            # check morphology
            for idx in range(nNegCurves):
                negCurveAngles[idx] = np.mean([localThetas[i+1]
                                               for i in range(negCurves[idx,0],
                                                              negCurves[idx,1])])
            # find goodCurves, convex regions with stronger curvature than minAngle
            # these are either the edges of the bud neck, or bright field artifacts
            goodCurves = np.where(((negCurveLengths > minLength).T
                                   & (negCurveAngles < -minAngleRads))[0])[0]
            nCurves = len(goodCurves)
            if nCurves == 0:
                # write to cleanedMcl without alteration
                cleanedMcl[box[0]-1:box[2]+1,
                           box[1]-1:box[3]+1][cellMask==1]=cellIdx
            elif nCurves == 1:
                # one concave region is typically not a bud neck, instead usually
                # results from bright field artifacts.
                cellMask[1:-1,1:-1] = cellProps[0].convex_image
                # write to cleanedMcl
                cleanedMcl[box[0]-1:box[2]+1,
                           box[1]-1:box[3]+1][cellMask==1]=cellIdx
            elif nCurves == 2:
                neck_linIdx_1 = (negCurves[goodCurves][0,0] +
                                 np.argmin(localThetas[negCurves[goodCurves][0,0]:
                                           negCurves[goodCurves][0,1]:1]))
                neck_linIdx_2 = (negCurves[goodCurves][1,0] +
                                 np.argmin(localThetas[negCurves[goodCurves][1,0]:
                                           negCurves[goodCurves][1,1]:1]))
                neck_yx1 = edgeLabels[neck_linIdx_1]
                neck_yx2 = edgeLabels[neck_linIdx_2]
                rr, cc = draw.line(neck_yx1[0],neck_yx1[1],neck_yx2[0],neck_yx2[1])
                cellMask[rr,cc]=0
                newCells, nlbl = ndimage.label(cellMask)
                # if segmentation produced a 'bud' below the budsize cutoff, it is
                # likely an error. take the convex hull instead.
                sizes = np.array([len(newCells[newCells==idx+1])
                                  for idx in range(nlbl)])
                largest = np.argmax(sizes)+1
                if sum(newCells.astype('bool')[newCells!=largest]) < minBudSize:
                    cellMask[1:-1,1:-1] = cellProps[0].convex_image
                    # write to cleanedMcl
                    cleanedMcl[box[0]-1:box[2]+1,
                               box[1]-1:box[3]+1][cellMask==1]=cellIdx
                else:
                    #check for second largest cell
                    second = np.where(
                            sizes==np.max(np.delete(sizes,largest-1)))[0][0]+1
                    # recut neck to assign neck to largest cell
                    newCells[rr,cc]=largest
                    # write to cleanedMcl with -idx for bud cell
                    cleanedMcl[box[0]-1:box[2]+1,
                               box[1]-1:box[3]+1][newCells==largest]=cellIdx
                    cleanedMcl[box[0]-1:box[2]+1,
                               box[1]-1:box[3]+1][newCells==second]=-cellIdx
            elif nCurves > 2:
                goodCurveAngles = negCurveAngles[goodCurves]
                edge1L = negCurves[goodCurves[np.argsort(goodCurveAngles)[0]]][0]
                edge1R = negCurves[goodCurves[np.argsort(goodCurveAngles)[0]]][1]
                edge2L = negCurves[goodCurves[np.argsort(goodCurveAngles)[1]]][0]
                edge2R = negCurves[goodCurves[np.argsort(goodCurveAngles)[1]]][1]
                neck_linIdx_1 = edge1L + np.argmin(localThetas[edge1L:edge1R])
                neck_linIdx_2 = edge2L + np.argmin(localThetas[edge2L:edge2R])
                neck_yx1 = edgeLabels[neck_linIdx_1]
                neck_yx2 = edgeLabels[neck_linIdx_2]
                rr, cc = draw.line(neck_yx1[0],neck_yx1[1],neck_yx2[0],neck_yx2[1])
                cellMask[rr,cc]=0
                newCells, nlbl = ndimage.label(cellMask)
                # if segmentation produced a 'bud' below the budsize cutoff, it is
                # likely an error. take the convex hull instead.
                sizes = np.array([len(newCells[newCells==idx+1])
                                  for idx in range(nlbl)])
                largest = np.argmax(sizes)+1
                if sum(newCells.astype('bool')[newCells!=largest]) < minBudSize:
                    cellMask[1:-1,1:-1] = cellProps[0].convex_image
                    # write to cleanedMcl
                    cleanedMcl[box[0]-1:box[2]+1,
                               box[1]-1:box[3]+1][cellMask==1]=cellIdx
                # otherwise, remaining minor negative curvatures are likely bright-
                # field artifacts. Take convex hull of each instead.
                else:
                    #check for second largest cell
                    second = np.where(
                            sizes==np.max(np.delete(sizes,largest-1)))[0][0]+1
                    cleanup = np.zeros(newCells.shape,dtype=newCells.dtype)
                    subCellProps = regionprops(newCells)
                    for subCell in [second,largest]:
                        subBox = subCellProps[subCell-1].bbox
                        sub_cvx_hull = subCellProps[subCell-1].convex_image
                        cleanup[subBox[0]:subBox[2],
                                subBox[1]:subBox[3]][sub_cvx_hull==1] = subCell
                    # recut neck to assign neck to largest cell
                    cleanup[rr,cc]=largest
                    # write to cleanedMcl with -idx for bud cell
                    cleanedMcl[box[0]-1:box[2]+1,
                               box[1]-1:box[3]+1][cleanup==largest]=cellIdx
                    cleanedMcl[box[0]-1:box[2]+1,
                               box[1]-1:box[3]+1][cleanup==second]=-cellIdx
            # check for stranded buds, remove them

        # if nothing works, just delete the cell
        except:
            cleanedMcl[masterCellMask] = 0
            idxOffset -= 1
            print('cleanup error at cellIdx=%s, dropped from mcl' % cellIdx)
        if showProgress: progressBar_text(cell,nCells,
                                          'processing cell outlines')
    if showProgress: print()
    return cleanedMcl

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

def labelCortex_mcl(masterCellLabel, cortexWidth):
    '''
    Generate labeled mask of cell cortical region, using masterCellLabel (with
    numbered cells, buds as -1*[mother_label])
    '''
    erosion = ndimage.binary_erosion(masterCellLabel,morph.disk(cortexWidth))
    cortexMcl = (masterCellLabel
                 - np.multiply(erosion.astype(int), masterCellLabel))
    return cortexMcl

def labelMaxproj(masterCellLabel, image, mkrChannel):
    '''
    Early version of colocalization marker label;
    uses Otsu's threshold of maximum projection of marker stack; assumes
    insubstantial photobleaching
    '''
    mkrZstk = image[mkrChannel,:,:,:]
    mkrMaxProj = np.amax(mkrZstk,axis=0)
    threshold = threshold_otsu(mkrMaxProj)
    mkrOtsu = np.zeros(mkrMaxProj.shape,dtype='bool')
    mkrOtsu[mkrMaxProj>threshold] = True
    mkrOtsu = ndimage.binary_opening(mkrOtsu)
    mkrOtsu = ndimage.binary_closing(mkrOtsu)
    mkrMcl = np.zeros(mkrOtsu.shape,dtype='int16')
    mkrMcl[mkrOtsu] = masterCellLabel[mkrOtsu]
    return mkrMcl

def subtract_labelMcl(labelMcl_one, labelMcl_two):
    subtractedMcl = np.array(labelMcl_one,dtype=labelMcl_one.dtype)
    subtractedMcl[labelMcl_one==labelMcl_two] = 0
    return subtractedMcl

def merge_labelMcl(labelMcl_one, labelMcl_two):
    mergedMcl = np.array(labelMcl_one,dtype=labelMcl_one.dtype)
    mergedMcl[labelMcl_one == 0] = labelMcl_two[labelMcl_one == 0]
    return mergedMcl

def buffer_mcl(unbufferedMcl, bufferSize, showProgress):
    '''
    add a buffer to cells on the unbufferedMcl to compensate for minor errors
    in registration between brightfield derived outlines and fluorescence
    image. Intelligently split borders between closely touching cells so that
    nearby fluorescent signal is not misassigned.
    '''
    #initialize workspaces
    tablet = np.zeros(unbufferedMcl.shape, unbufferedMcl.dtype)
    mergeTab = np.zeros(unbufferedMcl.shape, unbufferedMcl.dtype)
    overlaps = np.zeros(unbufferedMcl.shape, unbufferedMcl.dtype)
    #get number of cells (treating buds as distinct)
    uniqueLabels = np.unique(unbufferedMcl)
    index = np.argwhere(uniqueLabels == 0)
    uniqueLabels = np.delete(uniqueLabels,index)
    #build borders in loop to maintain overlaps
    for cellIdx in uniqueLabels:
        tablet[unbufferedMcl == cellIdx] = 1
        #dilate each cell by the bufferSize, then add two more dilations to
        #ensure segmentation of regions that touch but do not overlap at the
        #requested buffer size
        tablet = ndimage.binary_dilation(tablet, iterations=bufferSize +2)
        mergeTab = mergeTab+tablet
        tablet[tablet == 1] = 0
        if showProgress: progressSpinner_text(cellIdx,
                                              'generating measurement masks',
                                              '')
    overlaps[mergeTab > 1] = 1
    bufferedCells = ndimage.binary_dilation(
            unbufferedMcl, iterations=bufferSize)
    #divide overlapping regions with skeletonize
    overlapsSkeleton = morph.skeletonize(overlaps).astype('int')
    #extend skeleton so it always completely divides cells
    #first find edges of skeleton (points with only one 8-connected neighbor)
    kernel = [[1,1,1],
              [1,0,1],
              [1,1,1]]
    overlapsSklt_conn = ndimage.convolve(overlapsSkeleton,
                                         kernel,mode='constant')
    overlapsSklt_conn[overlapsSkeleton == 0] = 0
    ovlpSklt_tipY,ovlpSklt_tipX = np.where(overlapsSklt_conn == 1)
    distBuffCells = ndimage.morphology.distance_transform_cdt(
            ndimage.binary_dilation(bufferedCells).astype('int'))
    #second, loop through edges and find nearest descending distance_transform
    #point, update skeleton, and continue until new edge is at a distance == 1
    yPattern = np.array([-1,-1,-1, 0, 1, 1, 1, 0],dtype=int)
    xPattern = np.array([-1, 0, 1, 1, 1, 0,-1,-1],dtype=int)
    for point in zip(ovlpSklt_tipY,ovlpSklt_tipX):
        dist = distBuffCells[point]
        while dist != 0:
            neighbors = (yPattern+point[0],xPattern+point[1])
            #descend gradient at midmost point
            lowerLinIdx = np.where(distBuffCells[neighbors] == dist - 1)[0]
            lowerLinIdxMid = int(len(lowerLinIdx)/2)
            point = (point[0] + yPattern[lowerLinIdx[lowerLinIdxMid]],
                     point[1] + xPattern[lowerLinIdx[lowerLinIdxMid]])
            overlapsSkeleton[point] = 1
            dist = distBuffCells[point]
    bufferedCells[overlapsSkeleton == 1] = 0
    labeledBuffer, nLblBuffer = ndimage.label(bufferedCells)
    #relabel to match mcl
    mclCentroids = np.array(
            ndimage.measurements.center_of_mass(
                    unbufferedMcl,unbufferedMcl,uniqueLabels),
            dtype=int)
    for y, x, cellLbl in zip(mclCentroids[:,0],mclCentroids[:,1],uniqueLabels):
        centroid = (y,x)
        bufferLbl = labeledBuffer[centroid]
        labeledBuffer[labeledBuffer == bufferLbl] = cellLbl
    labeledBuffer[unbufferedMcl != 0] = 0
    uniqueNewLabels = np.unique(labeledBuffer)
    diffLabels = np.setdiff1d(uniqueNewLabels,uniqueLabels)
    for diff in diffLabels:
        labeledBuffer[labeledBuffer==diff]=0
    if showProgress: print('finsished')
    return labeledBuffer

def centroidCirclesMcl(mask, masterCellLabel, radius, iterations=1):
    maskLabels, nlbl = ndimage.label(mask)
    maskProps = regionprops(maskLabels)
    newMask = np.zeros(mask.shape, dtype='bool')
    newMcl = np.zeros(mask.shape, dtype='int')
    nRows,nCols = mask.shape
    for lbl in range(nlbl):
        y,x = maskProps[lbl].centroid
        # fix y,x if previous iterations have yielded a centroid point that is
        # not inside the mask region
        if not mask[int(y),int(x)]:
            dist = np.zeros(mask.shape, dtype='int')
            dist[maskLabels == lbl+1] = 1
            dist = ndimage.binary_erosion(dist)
            if np.sum(dist) > 1:
                dist = ndimage.morphology.distance_transform_edt(dist)
                y,x = np.unravel_index(np.argmax(dist),mask.shape)
        rr,cc = draw.circle(y, x, radius)
        oobPixels = np.where((rr>=nRows) | (rr<0) | (cc>=nCols) | (cc<0))
        rr = np.delete(rr,oobPixels)
        cc = np.delete(cc,oobPixels)
        newMask[rr,cc] = 1
    if iterations > 1:
        for i in range(iterations-1):
            residual = np.copy(mask)
            residual[newMask] = 0
            residualMask = centroidCirclesMcl(
                    residual, masterCellLabel,
                    radius, iterations=1).astype('bool')
            newMask[residualMask] = 1
    newMcl[newMask] = masterCellLabel[newMask]
    return newMcl


def measure_cells(primaryImage, masterCellLabel, refMclDict,
                  imageName, expID, fieldIdx,
                  globalMin, globalMax, showProgress):
    '''
    measurement function
    measures fluorescence intensity in the primaryImage for each cell in the
    masterCellLabel image, and measures subcellular fluorescence as specified
    in any number of refMcls specified in the refMclDict

    primary image is a dictionary consisting of the name of the image,
    (ie, Art1-mNG) and a single z-section fluorescence image. The name should
    not change across an experiment (ie, don't use Art1-mNG for one set of
    images and Art1(K486R)-mNG for another)

    masterCellLabel/refMcl format is an extension of the
    skdimage.measure.regionprops labeled image; background regions are 0, each
    cell is a unique ascending integer. Mother/bud pairs are positive/negative
    respectively. refMcl's should have the same numbering scheme as the
    masterCellLabel. refMclDict is a standard dictionary with the names of each
    refMcl to be used.

    Images are rescaled to experiment wide min,max values; background is
    the modal fluorescence intensity after binning into 256 bins between global
    min and max.

    Outputs to a list of dictionaries for easy import into pandas; optionally
    provide imageName and expID to populate these values. For experiment wide
    indexing, provide startIdx to continue labeling cells sequentially across
    the entire experiment.
    '''

    results = []
    nCells = np.max(masterCellLabel)
    #unpack names
    fluorName = str(*primaryImage)
    fluorImage = primaryImage[fluorName]
    fluorScaled = (fluorImage.astype('float')-globalMin)/(globalMax-globalMin)
    flatScaleduint8 = np.ndarray.flatten((fluorScaled*255).astype('uint8'))
    bkg = (sp.stats.mode(flatScaleduint8).mode.astype('float'))/255
    fluorBkgCorr = 1000*(fluorScaled-bkg)
    refNames = [key for key in refMclDict]
    #measurment loop
    for cellidx in range(nCells):
        cellLbl = cellidx + 1
        # write image specific identifiers
        measurements = {'imageName':imageName,
                        'expID':expID,
                        'localLbl':cellLbl,
                        'fieldIdx':fieldIdx,
                        'qcStatus':'unreviewed',
                        'qcTimestamp':[]}
        # define basic cell wide measurements
        totalIntDen = (fluorBkgCorr[np.abs(masterCellLabel) == cellLbl]).sum()
        totalArea = (np.abs(masterCellLabel) == cellLbl).sum()
        totalBrightness = totalIntDen / totalArea
        # measure fluorescence at bud, if present
        if -cellLbl in masterCellLabel:
            budFound = True
            budIntDen = (fluorBkgCorr[masterCellLabel == -cellLbl]).sum()
            budArea = (masterCellLabel == -cellLbl).sum()
            budBrightness = budIntDen / budArea
        else:
            budFound = False
            budIntDen = budArea = budBrightness = np.nan
        # measure fluorescen at masks
        for refKey in refNames:
            refMcl = refMclDict[refKey]
            refIntDen = (fluorBkgCorr[np.abs(refMcl) == cellLbl]).sum()
            refArea = np.sum(np.abs(refMcl) == cellLbl)
            if refArea != 0:
                refBrightness = refIntDen / refArea
            else:
                refBrightness = 0
            if budFound:
                budrefIntDen = (fluorBkgCorr[refMcl == cellLbl]).sum()
                budrefArea = (refMcl == -cellLbl).sum()
                if budrefArea != 0:
                    budrefBrightness = budrefIntDen / budrefArea
                else:
                    budrefBrightness = 0
            else:
                budrefIntDen = budrefArea = budrefBrightness = np.nan
            measurements[fluorName + '_intDensity_at_' + refKey] = refIntDen
            measurements[refKey + '_area'] = refArea
            measurements[fluorName + '_brightness_at_' + refKey] = (
                    refBrightness)
            measurements['bud_' + fluorName + '_intDensity_at_' + refKey] = (
                    budrefIntDen)
            measurements['bud_' + refKey + '_area'] = budrefArea
            measurements['bud_' + fluorName + '_brightness_at_' + refKey] = (
                    budrefBrightness)
            measurements['total_' + fluorName + '_intDensity'] = totalIntDen
        measurements['total_cell_area'] = totalArea
        measurements['total_' + fluorName + '_brightness'] = totalBrightness
        measurements['bud_' + fluorName + '_intDensity'] = budIntDen
        measurements['bud_area'] = budArea
        measurements['bud_' + fluorName + '_brightness'] = budBrightness
        results.append(measurements)
        if showProgress: progressBar_text(cellLbl,nCells,'measuring')
    if showProgress: print()
    return results

#prepare qc image
def prep_rgbQCImage(greenFluor, redFluor, qcMclList, scalingFactors):
    rgbList=[0,0,0]
    ySize,xSize = greenFluor.shape
    mask_one = qcMclList[0].astype('bool')
    mask_one_edge = mask_one ^ ndimage.binary_erosion(mask_one)
    mask_two = qcMclList[1].astype('bool')
    mask_two_edge = mask_two ^ ndimage.binary_erosion(mask_two)
    blue = np.zeros(greenFluor.shape,dtype='uint8')
    blue[mask_one_edge + mask_two_edge] = 255
    blue = blue.reshape(ySize,xSize,1)
    for idx in range(2):
        fluor = [redFluor,greenFluor][idx]
        fluor = fluor.astype('float')
        scaled = 255*((fluor-fluor.min())
                  / (scalingFactors[idx]*(fluor.max()-fluor.min())))
        scaled[scaled>255] = 255
        scaled = scaled.astype('uint8').reshape(ySize,xSize,1)
        rgbList[idx] = scaled
    rgbList[2] = blue
    rgbList[0][mask_one_edge.reshape(ySize,xSize,1)]=255
    rgbList[0][mask_two_edge.reshape(ySize,xSize,1)]=255
    rgbList[1][mask_two_edge.reshape(ySize,xSize,1)]=255
    rgbQC = np.concatenate(rgbList, axis=2)
    return(rgbQC)

def make_qcFrame(rgbQC, greenFluor, redFluor, masterCellLabel,
                 cellLbl, scalingFactors, borderSize):
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
    return qcDict

def display_qcFrame(qcDict,frameTitles):
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
