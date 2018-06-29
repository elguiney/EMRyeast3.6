''' 
ympy bespoke measurement pipelines.
Follow the examples here to create your own
'''

import ympy
import pickle
import numpy as np
import datetime
import os

GFP_wRFPmarker_params = dict(
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
        measureFields = 'all', #either 'all', to measure entire folder, or
            #a slice. [5,10] would measure fields 5,6,7,8 and 9 as they
            #appear in folderData['pathList'] from ympy.batchParse()
        smoothingKernelWidth = 5, #smoothing kernel for histogram calibrated
            #foci-brightness and foci-area measurments
        qcBorderSize = 10, #border around cells for qc
        returnTotalResults = False #whether measure_GFP_wRFPmarker should
            #return results as a variable in addition to saving them
            #(it always saves)
        )

class GFPwMarkerPipeline():
    def __init__(self, **kwargs):
        # defaults
        self.p = GFP_wRFPmarker_params
        self.p.update(kwargs)
        for k,v in self.p.items():
            setattr(self, k, v)
    def help(self):
        print(' ympy pipeline object\n',
              'usage:\n',
              'reference and set parameters as class attributes.\n',
              'initialize(folderPath) to prepare a folder for analysis\n',
              'runPipeline() to proccess all files in targetFolder')
    def initialize(self, folderPath):
        # run batchParse to initiailze on folder
        self.folderPath = folderPath
        self.folderData = ympy.batchParse(
                folderPath, self.expIDloc, self.imageExtension)
        self.imageNameList = self.folderData['imagenameList']
        self.pathList = self.folderData['pathlist']
        self.nFields = self.folderData['nFields']
        self.expIDlist = self.folderData['expIDlist']
    def runPipeline(self):
        print('beginning analysis of \n', self.folderPath,'\n at ',
          datetime.datetime.now())
        # initialize experiment variables
        totalResults = []
        totalQC = []
        fieldsAnalyzed = []
        totalMcl = []
            
        resultsDirectory = self.folderPath + '/results/'
        if not os.path.exists(resultsDirectory):
            os.makedirs(resultsDirectory)
        
        #%% measure global values (slow for big datasets)
        globalExtremaG = ympy.batchIntensityScale(
                self.folderData, self.greenChannel, self.showProgress)
        globalExtremaR = ympy.batchIntensityScale(
                self.folderData, self.redChannel, self.showProgress)
        globalMinG = globalExtremaG['globalmin']
        globalMaxG = globalExtremaG['globalmax']
        globalMinR = globalExtremaR['globalmin']
        globalMaxR = globalExtremaR['globalmax']
        #%% main loop
        if self.measureFields == 'all':
            start = 0
            stop = self.nFields
        else:
            start = self.measureFields[0]
            stop = self.measureFields[1]
        for field in range(start,stop):
            # read image
            dvImage = ympy.basicDVreader(
                    self.pathList[field],
                    self.rolloff,
                    self.nChannels,
                    self.zFirst)
            #%% find cells and cleanup morphology
            # find cells from brightfield step 1
            bwCellZstack = ympy.makeCellzStack(
                    dvImage, self.bwChannel, self.showProgress)
            # find cells from brightfield step 2
            nZslices = dvImage.shape[1]
            for z in range(nZslices):
                bwCellZstack[z,:,:] = ympy.helpers.correctBFanomaly(
                    bwCellZstack[z,:,:], self.bfAnomalyShiftVector)
            # find cells from brightfield step 3
            rawMcl = ympy.cellsFromZstack(bwCellZstack, self.showProgress)[0]
            # find cells from brightfield step 4
            unbufferedMcl = ympy.bfCellMorphCleanup(
                    rawMcl, self.showProgress, self.minAngle, 
                    self.minLength, self.closeRadius, self.minBudSize)
            #%% define measurment masks
            # unbufferedMcl is the best guess at the 'true outside edge' of 
            # the cells; use it as the starting point to find a 10pixel thick 
            # cortex
            unbufferedCortexMcl = ympy.labelCortex_mcl(
                    unbufferedMcl, self.cortexWidth)
            # because the bright field and fluorescence are not perfectly 
            # aligned, and to handle inaccuracies in edge finding, also buffer 
            # out from the outside edge
            buffer = ympy.buffer_mcl(
                    unbufferedMcl, self.bufferSize, self.showProgress)
            # merge this buffer onto the unbufferedMcl and the cortexMcl
            masterCellLabel = ympy.merge_labelMcl(unbufferedMcl, buffer)
            cortexMcl = ympy.merge_labelMcl(unbufferedCortexMcl, buffer)
            
            # use Otsu thresholding on the max projection of RFPmarker
            markerMclOtsu = ympy.labelMaxproj(
                    masterCellLabel, dvImage, self.mkrChannel)
            # then use centroidCircles to uniformly mask peri-golgi regions
            markerCirclesMcl = ympy.centroidCirclesMcl(
                    markerMclOtsu.astype('bool'), masterCellLabel,
                    self.markerRadius, self.markerCircleIterations)
            # subtract so that marker localization has precedence over cortical
            # localization
            cortexMinusMarker = ympy.subtract_labelMcl(
                    cortexMcl, markerCirclesMcl)
            # finally, compute mask for remaining cytoplasmic regions
            cytoplasmMcl =ympy.subtract_labelMcl(masterCellLabel,
                    ympy.merge_labelMcl(markerCirclesMcl, cortexMinusMarker))
            #%% measure
            # measure Art1-mNG in the middle z-slice
            primaryImage = {self.measuredProteinName:
                dvImage[self.measuredProteinChannel,
                        self.measuredProteinZ, :, :]}
            # measure against buffered cortex (minus marker mask), marker, and  
            # cytoplasm
            refMclDict = {
                    'cortex(non-' + self.markerName + ')': cortexMinusMarker,
                    self.markerName + '(circles)': markerCirclesMcl,
                    'cytoplasm': cytoplasmMcl,
                          }
            # also record field wide information
            # measurement function
            results = ympy.measure_cells(
                    primaryImage, masterCellLabel, refMclDict,
                    self.imageNameList[field], self.expIDlist[field], field,
                    globalMinG, globalMaxG, self.nHistBins, self.showProgress)
            # add measurements from each field to total results
            totalResults = list(np.concatenate((totalResults,results)))
            #%% quality control prep
            print('preparing quality control information')
            greenFluor = dvImage[self.measuredProteinChannel,
                                 self.measuredProteinZ,:,:].astype(float)
            greenFluorScaled = ((greenFluor.astype('float')-globalMinG)
                                /(globalMaxG-globalMinG))
            redFluor = np.amax(
                    dvImage[self.mkrChannel,:,:,:],axis=0).astype(float)
            redFluorScaled = ((redFluor.astype('float')-globalMinR)
                              /(globalMaxR-globalMinR))
            qcMclList = [cortexMinusMarker, markerCirclesMcl]
            rgbQC = ympy.prep_rgbQCImage(
                    greenFluorScaled, redFluorScaled,
                    qcMclList, self.rgbScalingFactors)
            
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
                       'parameters': self.p
                       }
            pickle.dump(resultsDic,
                        open(self.folderPath
                               + '/results/'
                               + str(datetime.datetime.now().date())
                               + '_analysis.p', 'wb'))
            print(self.imageNameList[field],
                  ' complete at ', datetime.datetime.now())
            if self.returnTotalResults:
                return(resultsDic)

def measure_GFP_wRFPmarker(folderPath, parameterDict):
    '''        
    finds borders of cells and corrects morphology with yeast-based heuristics,
    then measures fluorescence at the medial plane of a zstack.
    returns total, cortical, and marker overlapping/adjacent fluorescence.
    displays progress to console output, and autosaves.
    
    main results are foci intensity at cortex and at marker;
        calculated by multiplying the [0,1] scaled cumsum of the cytoplasmic 
        intenstiy histogram with the intensity histograms for cortex and marker
        fluorescence, summing, then dividing by cell area. (roughly, this 
        yeilds the normalized intensity of foci, with foci as regions brighter
        than typical cytoplasmic fluorescence). 
    also returns a wide variety of related measures.
    
    Parameters:
    
    folderPath: aboslute path to folder for analysis
    parameterDict: dictionary of parameters. The following are required, with
    example values that worked for Art1-mNG vs mCh-Sec7 measurement.
    
    this dict can be imported from the ympy.pipelines module; 
    use: 
    import ympy.pipelines.GFP_wRFPmarker_params as p
    
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
            measureFields = 'all' #either 'all', to measure entire folder, or
                #a slice. [5,10] would measure fields 5,6,7,8 and 9 as they
                #appear in folderData['pathList'] from ympy.batchParse()
            )
    '''
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
    
    folderData = ympy.batchParse(
            folderPath, p['expIDloc'], p['imageExtension'])
    nFields = folderData['nFields']
    imageNameList = folderData['imagenameList']
    pathList = folderData['pathlist']
    expIDlist = folderData['expIDlist']
    #%% measure global values (slow for big datasets)
    globalExtremaG = ympy.batchIntensityScale(
            folderData, p['greenChannel'], p['showProgress'])
    globalExtremaR = ympy.batchIntensityScale(
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
        dvImage = ympy.basicDVreader(
                pathList[field], p['rolloff'], p['nChannels'], p['zFirst'])
        #%% find cells and cleanup morphology
        # find cells from brightfield step 1
        bwCellZstack = ympy.makeCellzStack(
                dvImage, p['bwChannel'], p['showProgress'])
        # find cells from brightfield step 2
        nZslices = dvImage.shape[1]
        for z in range(nZslices):
            bwCellZstack[z,:,:] = ympy.helpers.correctBFanomaly(
                bwCellZstack[z,:,:], p['bfAnomalyShiftVector'])
        # find cells from brightfield step 3
        rawMcl = ympy.cellsFromZstack(bwCellZstack, p['showProgress'])[0]
        # find cells from brightfield step 4
        unbufferedMcl = ympy.bfCellMorphCleanup(
                rawMcl, p['showProgress'], p['minAngle'], 
                p['minLength'], p['closeRadius'], p['minBudSize'])
        #%% define measurment masks
        # unbufferedMcl is the best guess at the 'true outside edge' of the 
        # cells; use it as the starting point to find a 10pixel thick cortex
        unbufferedCortexMcl = ympy.labelCortex_mcl(
                unbufferedMcl, p['cortexWidth'])
        # because the bright field and fluorescence are not perfectly aligned,
        # and to handle inaccuracies in edge finding, also buffer out from the
        # outside edge
        buffer = ympy.buffer_mcl(
                unbufferedMcl, p['bufferSize'], p['showProgress'])
        # merge this buffer onto the unbufferedMcl and the cortexMcl
        masterCellLabel = ympy.merge_labelMcl(unbufferedMcl, buffer)
        cortexMcl = ympy.merge_labelMcl(unbufferedCortexMcl, buffer)
        
        # use Otsu thresholding on the max projection of RFPmarker
        markerMclOtsu = ympy.labelMaxproj(
                masterCellLabel, dvImage, p['mkrChannel'])
        # then use centroidCircles to uniformly mask peri-golgi regions
        markerCirclesMcl = ympy.centroidCirclesMcl(
                markerMclOtsu.astype('bool'), masterCellLabel,
                p['markerRadius'], p['markerCircleIterations'])
        # subtract so that marker localization has precedence over cortical
        # localization
        cortexMinusMarker = ympy.subtract_labelMcl(cortexMcl, markerCirclesMcl)
        # finally, compute mask for remaining cytoplasmic regions
        cytoplasmMcl =ympy.subtract_labelMcl(masterCellLabel,
                ympy.merge_labelMcl(markerCirclesMcl, cortexMinusMarker))
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
        results = ympy.measure_cells(
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
        redFluor = np.amax(dvImage[p['mkrChannel'],:,:,:],axis=0).astype(float)
        redFluorScaled = ((redFluor.astype('float')-globalMinR)
                          /(globalMaxR-globalMinR))
        qcMclList = [cortexMinusMarker, markerCirclesMcl]
        rgbQC = ympy.prep_rgbQCImage(
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
        #%% pool and save
        print('saving progress')
        resultsDic = {'totalResults':totalResults,
                   'totalQC':totalQC,
                   'fieldsAnalyzed':fieldsAnalyzed,
                   'totalMcl':totalMcl,
                   'parameters':parameterDict
                   }
        pickle.dump(resultsDic,
                    open(folderPath
                           + '/results/'
                           + str(datetime.datetime.now().date())
                           + '_analysis.p', 'wb'))
        print(imageNameList[field],' complete at ',datetime.datetime.now())
        if p['returnTotalResults']:
            return(resultsDic)