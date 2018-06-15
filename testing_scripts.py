''' 
testing scripts

test test testf
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

rolloff=64
bfAnomalyShiftVector = [2,-1]
startIdx = 0
globalExtrema = EMRyeast36.batchIntensityScale(
        folderData, channel=1, showProgress=True)
globalMin = globalExtrema['globalmin']
globalMax = globalExtrema['globalmax']
totalResults = []
totalQC = []
fieldsAnalyzed = []
totalMcl = []
rgbScalingFactors = [0.4,0.4]
gryScalingFactors = [0.2,0.2]

for field in range(nFields):
    print('starting image: ', imageNameList[field])
    # read image
    dvImage = EMRyeast36.basicDVreader(pathList[field],
                                       rolloff=rolloff,
                                       nChannels=3,
                                       zFirst=True)
    # find cells from brightfield step 1
    bwCellZstack = EMRyeast36.makeCellzStack(dvImage,showProgress=True)
    # find cells from brightfield step 2
    nZslices = dvImage.shape[1]        
    for z in range(nZslices):
        bwCellZstack[z,:,:] = EMRyeast36.correctBFanomaly(bwCellZstack[z,:,:],
                                       bfAnomalyShiftVector)
    # find cells from brightfield step 3
    rawMcl = EMRyeast36.cellsFromZstack(bwCellZstack,showProgress=True)[0]
    # find cells from brightfield step 4
    unbufferedMcl = EMRyeast36.bfCellMorphCleanup(rawMcl, showProgress=True,)
    # unbufferedMcl is the best guess at the 'true outside edge' of the cells;
    # use it as the starting point to find a 10pixel thick cortex
    cortexMcl = EMRyeast36.labelCortex_mcl(unbufferedMcl,cortexWidth=10)
    # because the bright field and fluorescence are not perfectly aligned, and
    # to handle inaccuracies in edge finding, also buffer out from the outside
    # edge
    buffer = EMRyeast36.buffer_mcl(unbufferedMcl, 
                                   bufferSize=5, showProgress=True)
    # merge this buffer onto the unbufferedMcl and the cortexMcl
    masterCellLabel = EMRyeast36.merge_labelMcl(unbufferedMcl, buffer)
    cortexMclBuffered = EMRyeast36.merge_labelMcl(cortexMcl, buffer)
    # use Otsu thresholding on the max projection of mCh-Sec7 to find golgi
    golgiMcl = EMRyeast36.labelMaxproj(masterCellLabel,
                                       image=dvImage,mkrChannel=0)
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
                                 globalMin, globalMax, showProgress=True)
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
                                      startIdx, borderSize=10)
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
#print('\ngenerating measurement masks')
cortexMcl = labelCortex_mcl(masterCellLabel,cortexWidth=10)
buffer = buffer_mcl(masterCellLabel, bufferSize=5, showProgress=True)

#combine masks
masterCellLabelBuffered = merge_labelMcl(masterCellLabel, buffer)
cortexMclBuffered = merge_labelMcl(cortexMcl, buffer)
golgiMcl = labelMaxproj(masterCellLabelBuffered,image=dvImage,mkrChannel=0)
cortexMinusGolgiMcl = subtract_labelMcl(cortexMcl,golgiMcl)
cortexBufferedMinusGolgi = subtract_labelMcl(cortexMclBuffered,golgiMcl)

#testing values for script-mode measure_cells()
primaryImage = {'Art1-mNG':dvImage[1,3,:,:]}

refMclDict = {'cortex(buffered)':cortexMclBuffered,
              'golgi':golgiMcl,
              'nonGolgicortex(buffered)':cortexBufferedMinusGolgi
              }
expID = 'xx0001'
masterCellLabelUnbuffered = np.copy(masterCellLabel)
masterCellLabel = np.copy(masterCellLabelBuffered)
startIdx = 0

globalMin = primaryImage['Art1-mNG'].min()
globalMax = primaryImage['Art1-mNG'].max()

results,startIdx = measure_cells(primaryImage, masterCellLabel, refMclDict,
                                 imageName, expID, startIdx,
                                 globalMin, globalMax, showProgress=True)


scalingFactors = [0.2,0.2]
grnScl = 0.4
greenFluor = dvImage[1,3,:,:].astype(float)
redScl = 0.4
redFluor = np.amax(dvImage[0,:,:,:],axis=0).astype(float)
qcMclList = [cortexBufferedMinusGolgi,golgiMcl]

rgbQC = prep_rgbQCImage(greenFluor, redFluor, qcMclList, scalingFactors)
plt.imshow(rgbQC)
#greenFluor = (greenFluor-greenFluor.min())/(grnScl*greenFluor.max()-greenFluor.min())
#greenInv = -1*(greenFluor-1)
#greenFluor[greenFluor>1]=1
#redFluor = (redFluor-redFluor.min())/(redScl*redFluor.max()-redFluor.min())
#redFluor[redFluor>1]=1
#rgb = np.concatenate([img.reshape(1920,1920,1) for img in [redFluor,greenFluor,]],axis=2)
#plt.imshow(rgb)
'''
'''
#save test image
golgiSlice = dvImage[0,3,:,:]
art1Slice = dvImage[1,3,:,:]
bfSlice = dvImage[2,3,:,:]
testImage = mergeForTiff([golgiSlice,
                          art1Slice,
                          masterCellLabel,
                          masterCellLabelBuffered,
                          cortexMclBuffered,
                          cortexBufferedMinusGolgi,
                          golgiMcl])
tifffile.imsave(folderPath+'test.tiff',testImage)
'''


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