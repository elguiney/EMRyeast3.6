"""
EMRyeast36_v0.1

Emr lab image analysis module
adapted for python 3.6
contains endocytosis profiling modules (vacmorph, vacmorph advanced) -not yet 
    tested
extends methods to quantify flourescence localization against two markers
    A) higher precision cortext localization from bright field image
    B) punctate RFP colocalization marker (tested with mCh-Sec7)
    and evaluate punctate, membrane coloaclization (tested with Art1-mNG)
"""

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
        if showProgress:
            dispText = ('finding edges: progress',
                        '='*int(20*(z+1)/nZslices),
                        int(100*(z+1)/nZslices))
            sys.stdout.write('\r')
            sys.stdout.write("%s:[%-20s] %d%%" % dispText)
            sys.stdout.flush()
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
        if showProgress:
            dispText = ('finding cells: progress',
                        '='*obj,
                         obj)
            sys.stdout.write('\r')
            sys.stdout.write("%s:[%s] %d cells" % dispText)
            sys.stdout.flush()
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
    image16bit = [(255*(img.astype(float)-img.min())/
                   (img.max()-img.min())).astype('uint8')
                    for img in imageArrayList]
    nRows = np.shape(image16bit[0])[0]
    nCols = np.shape(image16bit[0])[1]
    tiffStack = np.concatenate([img.reshape(1,nRows,nCols)
                                for img in image16bit])
    return tiffStack

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
    for cell in range(nCells):
        cellIdx = cell + 1    
        masterCellMask = np.zeros(np.shape(mcl), dtype='bool')
        masterCellMask[relabeledMcl==cellIdx] = 1
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
                # recut neck to assign neck to largest cell
                newCells[rr,cc]=largest
                # write to cleanedMcl with -idx for bud cell
                cleanedMcl[box[0]-1:box[2]+1,
                           box[1]-1:box[3]+1][newCells==largest]=cellIdx
                cleanedMcl[box[0]-1:box[2]+1,
                           box[1]-1:box[3]+1][newCells==largest%2+1]=-cellIdx
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
        if showProgress:
            dispText = ('processing cell outlines: progress',
                        '='*int(20*(cell+1)/nCells),
                        int(100*(cell+1)/nCells))
            sys.stdout.write('\r')
            sys.stdout.write("%s:[%-20s] %d%%" % dispText)
            sys.stdout.flush()
    return cleanedMcl

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

def buffer_mcl(masterCellLabel, bufferSize):
    '''
    add a buffer to cells on the masterCellLabel to compensate for minor errors
    in registration between brightfield derived outlines and fluorescence
    image. Intelligently split borders between closely touching cells so that
    nearby fluorescent signal is not misassigned.
    '''
    #initialize workspaces
    tablet = np.zeros(masterCellLabel.shape, masterCellLabel.dtype)
    mergeTab = np.zeros(masterCellLabel.shape, masterCellLabel.dtype)
    overlaps = np.zeros(masterCellLabel.shape, masterCellLabel.dtype)
    #get number of cells (treating buds as distinct)
    uniqueLabels = np.unique(masterCellLabel)
    index = np.argwhere(uniqueLabels == 0)
    uniqueLabels = np.delete(uniqueLabels,index)
    #build borders in loop to maintain overlaps
    for cellIdx in uniqueLabels:
        tablet[masterCellLabel == cellIdx] = 1
        #dilate each cell by the bufferSize, then add two more dilations to 
        #ensure segmentation of regions that touch but do not overlap at the
        #requested buffer size
        tablet = ndimage.binary_dilation(tablet, iterations=bufferSize +2)
        mergeTab = mergeTab+tablet
        tablet[tablet == 1] = 0
    overlaps[mergeTab > 1] = 1
    bufferedCells = ndimage.binary_dilation(
            masterCellLabel, iterations=bufferSize)
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
            bufferedCells.astype('int'))
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
                    masterCellLabel,masterCellLabel,uniqueLabels),
            dtype=int)
    for y, x, cellLbl in zip(mclCentroids[:,0],mclCentroids[:,1],uniqueLabels):
        centroid = (y,x)
        bufferLbl = labeledBuffer[centroid]
        labeledBuffer[labeledBuffer == bufferLbl] = cellLbl
    labeledBuffer[masterCellLabel != 0] = 0
    return labeledBuffer


def measure_cells(primaryImage, masterCellLabel, refMclList, refMclNames,
                  imageName, expID, startIdx):
    '''
    measurement function
    measures fluorescence intensity in the primaryImage for each cell in the 
    masterCellLabel image, and measures subcellular fluorescence as specified
    in any number of refMcls specified in the refMclList and refMclNames
    
    primary image is a single z-section fluorescence image
    
    masterCellLabel/refMcl format is an extension of the 
    skdimage.measure.regionprops labeled image; background regions are 0, each 
    cell is a unique ascending integer. Mother/bud pairs are positive/negative
    respectively. refMcl's should have the same numbering scheme as the 
    masterCellLabel.
    
    Outputs to a list of dictionaries for easy import into pandas; optionally 
    provide imageName and expID to populate these values. For experiment wide
    indexing, provide startIdx to continue labeling cells sequentially across
    the entire experiment.
    '''
    #testing values for script-mode
    primaryImage = dvImage[1,3,:,:]
    
    refMclList = [cortexMcl,golgiMcl,cortexMinusGolgiMcl]
    refMclNames = ['cortexMcl','golgiMcl','cortexMinusGolgiMcl']
    expID = 'xx0001'
    
    nCells = np.max(masterCellLabel)
    for cellidx in range(nCells):
        #measure things
        a = 1
    return #stuff
    
    
'''
below: script for developing new Art1 localization measurements.
'''
folderPath = ("C:/Users/elgui/Documents/Emr Lab Post Doc/microscopy/"
              "/2018-05-15_YEG280_exposure_tests/")
imageName = "YEG280_SCD_mCh100T25ms_mNG100T250ms-06_R3D_D3D.dv"
rolloff = 64
bfAnomalyShiftVector = [2,-1]

#def processImage(folderPath,imageName):
showProgress=True
print('starting image', imageName)
imagePath = folderPath + imageName
#read image to memory
dvImage = basicDVreader(imagePath,rolloff)
#find edges in brightfields
bwCellZstack = makeCellzStack(dvImage,showProgress=True)
#correct brightfield anamolies
nZslices = dvImage.shape[1]
for z in range(nZslices):
    bwCellZstack[z,:,:] = correctBFanomaly(bwCellZstack[z,:,:],
                                           bfAnomalyShiftVector)
#identify and merge cell borders
rawMcl = cellsFromZstack(bwCellZstack,showProgress=True)[0]
masterCellLabel = bfCellMorphCleanup(rawMcl, showProgress=True,)

#generate masks (with mcl consistent labels)
print('\ngenerating measurement masks')
cortexMcl = labelCortex_mcl(masterCellLabel,cortexWidth=8)
buffer = buffer_mcl(masterCellLabel, bufferSize=5)
masterCellLabelBuffered = merge_labelMcl(masterCellLabel, buffer)
cortexMclBuffered = merge_labelMcl(cortexMcl, buffer)
golgiMcl = labelMaxproj(masterCellLabelBuffered,image=dvImage,mkrChannel=0)
cortexMinusGolgiMcl = subtract_labelMcl(cortexMcl,golgiMcl)
cortexBufferedMinusGolgi = subtract_labelMcl(cortexMclBuffered,golgiMcl)

#save test image
golgiSlice = dvImage[0,3,:,:]
art1Slice = dvImage[1,3,:,:]
bfSlice = dvImage[2,3,:,:]
testImage = mergeForTiff([golgiSlice,
                          art1Slice,
                          masterCellLabel,
                          masterCellLabelBuffered,
                          cortexMcl,
                          cortexMclBuffered,
                          cortexMinusGolgiMcl])
tifffile.imsave(folderPath+'test.tiff',testImage)



'''
bw = np.zeros(cleanMcl.shape)
bw = np.zeros(cleanMcl.shape,dtype='uint8')
bw[cleanMcl!=0]=255
plt.imshow(bw)
gfp = dvImage[1,3,:,:]
bf = dvImage[2,3,:,:]
merge = mergeForTiff([bw,gfp,bf])
tifffile.imsave(folderPath+'merge.tiff',merge)
'''



'''
xSize,ySize = np.shape(mcl)
cell = 37
bWidth = 10



cellIdx = cell + 1
masterCellMask = np.zeros([ySize,xSize], dtype='int')
masterCellMask[mcl==cellIdx] = 1
cellProps = regionprops(masterCellMask.astype('int'))
box1 = cellProps[0].bbox
stamp = ~basicText2im.numstr2im(cell)
bbf = 2*bWidth
boxEdges = (bWidth,bWidth,ySize-bWidth,xSize-bWidth)
box = [0,0,0,0]
for i in range(0,2):
    box[i] = max(box1[i],boxEdges[i])

for i in range(2,4):
    box[i] = min(box1[i],boxEdges[i])

top = min(box[0], mcl.shape[0] - stamp.shape[0])
left = max(0, box[1] - stamp.shape[1])

roi = []
cellMask = masterCellMask[box[0]-bWidth:box[2]+bWidth,box[1]-bWidth:box[3]+bWidth]
cellEdge = cellMask - ndimage.binary_erosion(cellMask)
roi.append(cellMask)
roi.append(dvImage[1,4,box[0]-bWidth:box[2]+bWidth,box[1]-bWidth:box[3]+bWidth])
roi.append(dvImage[2,4,box[0]-bWidth:box[2]+bWidth,box[1]-bWidth:box[3]+bWidth])
'''

'''
def buffer_outlines(masterCellLabel, bufferSize):


    #initialize workspaces
    tablet = np.zeros(masterCellLabel.shape, masterCellLabel.dtype)
    mergeTab = np.zeros(masterCellLabel.shape, masterCellLabel.dtype)
    overlaps = np.zeros(masterCellLabel.shape, masterCellLabel.dtype)
    border = np.zeros(masterCellLabel.shape, masterCellLabel.dtype)
    #get number of cells
    nCells = np.max(masterCellLabel)
    #merge buds with mothers
    absMcl = np.abs(masterCellLabel)
    #build borders in loop to maintain overlaps
    for cellIdx in range(1, nCells+1):
        tablet[absMcl == cellIdx] = 1
        #dilate each cell by the bufferSize, then add two more dilations to 
        #ensure segmentation of regions that touch but do not overlap at the
        #requested buffer size
        tablet = ndimage.binary_dilation(tablet, iterations=bufferSize +2)
        mergeTab = mergeTab+tablet
        tablet[tablet == 1] = 0
    overlaps[mergeTab > 1] = 1
    overlaps[masterCellLabel != 0] = 0
    bufferedCells = ndimage.binary_dilation(absMcl, iterations=bufferSize)
    border = bufferedCells.astype('bool') ^ absMcl.astype('bool').astype('int')                
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
            bufferedCells.astype('int'))
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
    bordersForAssignment = np.array(border,dtype=int)
    bordersForAssignment[overlapsSkeleton == 1] = 0
    labeledBforAssg, nBforAssg = ndimage.label(bordersForAssignment)
    #loop through and assign to appropriate cell
    '''