''' 
ympy bespoke measurement pipelines.
Follow the examples here to create your own
'''

import pickle
import numpy as np
import datetime
import os

import ympy
from ympy.parameterSets import YmpyParam

class GFPwMarkerPipeline():
    def __init__(self, experiment_parameter_dict):
        # setup parameters. Pulls default parameters from ympy.parameterSets
        # during initiation of YmpyParam object. Pass a dictionary of 
        # experiment specific parameters following template_GFPwMarker_params
        self.Param = YmpyParam(experiment_parameter_dict)
        self.folder_data = ympy.batchParse(
                self.Param.folder_path, self.Param.exp_ID_loc,
                self.Param.image_extension)
    def help(self):
        print(' ---ympy pipeline object---\n',
              'Usage:\n',
              'Reference and set parameters from Param attribute.\n',
              'GFPmarkerPipeline.Param.listParamters() to',
              'view current parameters\n',
              'GFPmarkerPipeline.runPipeline() to proccess all files in\n',
              '    GFPmarkerPipeline.Param.target_folder')
    def runPipeline(self):
        # unpack for easier debugging
        p = self.Param
        fd = self.folder_data
        
        print('beginning analysis of \n', p.folder_path,'\n at ',
          datetime.datetime.now())
        # initialize experiment variables
        totalResults = []
        totalQC = []
        fieldsAnalyzed = []
        totalMcl = []
            
        resultsDirectory = p.folder_path + '/results/'
        if not os.path.exists(resultsDirectory):
            os.makedirs(resultsDirectory)
        
        #%% measure global values (slow for big datasets)
        globalExtremaG = ympy.batchIntensityScale(
                fd, p.image_reader, p.reader_args,
                p.green_channel, p.show_progress)
        globalExtremaR = ympy.batchIntensityScale(
                fd, p.image_reader, p.reader_args,
                p.red_channel, p.show_progress)
        globalMinG = globalExtremaG['globalmin']
        globalMaxG = globalExtremaG['globalmax']
        globalMinR = globalExtremaR['globalmin']
        globalMaxR = globalExtremaR['globalmax']
        #%% main loop
        if p.measure_fields == 'all':
            start = 0
            stop = fd['n_fields']
        else:
            start = p.measure_fields[0]
            stop = p.measure_fields[1]
        for field in range(start,stop):
            print(field)
            # read image
            field_path = fd['path_list'][field]
            dvImage = p.image_reader(
                    **ympy.helpers.readerHelper(
                            p.image_reader, field_path, p.reader_args))
            dvImage = ympy.helpers.cropRolloff(dvImage, p.image_rolloff)
            #%% find cells and cleanup morphology
            # find cells from brightfield step 1
            bwCellZstack = ympy.makeCellzStack(
                    dvImage, p.bf_channel, p.show_progress)
            # find cells from brightfield step 2
            nZslices = dvImage.shape[1]
            for z in range(nZslices):
                bwCellZstack[z,:,:] = ympy.helpers.correctBFanomaly(
                    bwCellZstack[z,:,:], p.bf_offest_vector)
            # find cells from brightfield step 3
            rawMcl = ympy.cellsFromZstack(bwCellZstack, p.show_progress)[0]
            # find cells from brightfield step 4
            unbufferedMcl = ympy.bfCellMorphCleanup(
                    rawMcl, p.show_progress, p.min_angle, 
                    p.min_length, p.closing_radius, p.min_bud_size)
            #%% define measurment masks
            # unbufferedMcl is the best guess at the 'true outside edge' of 
            # the cells; use it as the starting point to find a 10pixel thick 
            # cortex
            unbufferedCortexMcl = ympy.labelCortex_mcl(
                    unbufferedMcl, p.cortex_width)
            # because the bright field and fluorescence are not perfectly 
            # aligned, and to handle inaccuracies in edge finding, also buffer 
            # out from the outside edge
            buffer = ympy.buffer_mcl(
                    unbufferedMcl, p.buffer_size, p.show_progress)
            # merge this buffer onto the unbufferedMcl and the cortexMcl
            masterCellLabel = ympy.merge_labelMcl(unbufferedMcl, buffer)
            cortexMcl = ympy.merge_labelMcl(unbufferedCortexMcl, buffer)
            
            # use Otsu thresholding on the max projection of RFPmarker
            markerMclOtsu = ympy.labelMaxproj(
                    masterCellLabel, dvImage, p.marker_channel)
            # then use centroidCircles to uniformly mask peri-golgi regions
            markerCirclesMcl = ympy.centroidCirclesMcl(
                    markerMclOtsu.astype('bool'), masterCellLabel,
                    p.marker_radius, p.marker_circle_iterations)
            # subtract so that marker localization has precedence over cortical
            # localization
            cortexMinusMarker = ympy.subtract_labelMcl(
                    cortexMcl, markerCirclesMcl)
            # finally, compute mask for remaining cytoplasmic regions
            cytoplasmMcl =ympy.subtract_labelMcl(masterCellLabel,
                    ympy.merge_labelMcl(markerCirclesMcl, cortexMinusMarker))
            #%% measure
            # measure Art1-mNG in the middle z-slice
            primaryImage = {p.measured_protein_name:
                dvImage[p.measured_protein_channel,
                        p.measured_protein_z, :, :]}
            # measure against buffered cortex (minus marker mask), marker, and  
            # cytoplasm
            refMclDict = {
                    'cortex(non-' + p.marker_name + ')': cortexMinusMarker,
                    p.marker_name + '(circles)': markerCirclesMcl,
                    'cytoplasm': cytoplasmMcl,
                          }
            # also record field wide information
            # measurement function
            results = ympy.measure_cells(
                    primaryImage, masterCellLabel, refMclDict,
                    fd['imagename_list'][field],
                    fd['expID_list'][field],
                    field, globalMinG, globalMaxG, p.n_hist_bins,
                    p.show_progress)
            # add measurements from each field to total results
            totalResults = list(np.concatenate((totalResults,results)))
            #%% quality control prep
            print('preparing quality control information')
            greenFluor = dvImage[p.measured_protein_channel, #TODO function
                                 p.measured_protein_z,:,:].astype(float)
            greenFluorScaled = ((greenFluor.astype('float')-globalMinG)
                                /(globalMaxG-globalMinG))
            redFluor = np.amax(
                    dvImage[p.marker_channel,:,:,:],axis=0).astype(float)
            redFluorScaled = ((redFluor.astype('float')-globalMinR)
                              /(globalMaxR-globalMinR))
            qcMclList = [cortexMinusMarker, markerCirclesMcl]
            rgbQC = ympy.prep_rgbQCImage(
                    greenFluorScaled, redFluorScaled,
                    qcMclList, p.rgb_scaling_factors) #TODO function return
            
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
                       'parameters': p.listParameters()
                       }
            pickle.dump(resultsDic,
                        open(p.folder_path
                               + '/results/'
                               + str(datetime.datetime.now().date())
                               + '_analysis.p', 'wb'))
            print(fd['imagename_list'][field],
                  ' complete at ', datetime.datetime.now())
        if p.return_total_results:
            return(resultsDic)
