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
    
    #%% run the pipeline main function
    
    def runPipeline(self):
        self.history.append('ran pipeline at ' + str(datetime.datetime.now()))
        self.setupResultsfolder()
        self.scaleBrightness()
        total = self.analyzeRange()[1] - self.analyzeRange()[0]
        self.history.append('entered main loop')
        for field in range(*self.analyzeRange()):
            self.field_log = [self.history.pop()]
            self.current_field = field
            print('field #{} of {} total'.format(field, total))
            image = self.readCurrentField()
            master_cell_label = self.segmentImage(image)
            ref_mcl_dict = self.buildMeasurementMasks(
                    master_cell_label,
                    image)
            self.measureSingleField(
                    ref_mcl_dict,
                    image)
            self.buildQCfield(image)
            self.fieldsAnalyzed.append(field)
            self.totalMcl.append(master_cell_label)
            self.saveState()

    def __init__(self, experiment_parameter_dict):
        # setup parameters. Pulls default parameters from ympy.parameterSets
        # during initiation of YmpyParam object. Pass a dictionary of 
        # experiment specific parameters following template_GFPwMarker_params
        self.Param = YmpyParam(experiment_parameter_dict)
        self.folder_data = ympy.batchParse(
                self.Param.folder_path, self.Param.exp_ID_loc,
                self.Param.image_extension)
                # initialize experiment variables
        self.totalResults = []
        self.totalQC = []
        self.fieldsAnalyzed = []
        self.totalMcl = []
        self.history = ['initialized']
        self.current_field = []
        self.field_log = []
    
    def help(self):
        print(' ---ympy pipeline object---\n',
              'Usage:\n',
              'Reference and set parameters from Param attribute.\n',
              'GFPmarkerPipeline.Param.listParamters() to',
              'view current parameters\n',
              'GFPmarkerPipeline.runPipeline() to proccess all files in\n',
              '    GFPmarkerPipeline.Param.target_folder')
                
    #%% setup folder    
    
    def setupResultsfolder(self):    
        print('beginning analysis of \n{}\n at {}'.format(
                self.Param.folder_path, datetime.datetime.now()))
        resultsDirectory = self.Param.folder_path + '/results/'
        if not os.path.exists(resultsDirectory):
            self.history.append('created results folder')
            os.makedirs(resultsDirectory)

    #%% measure global values (slow for big datasets)
    
    def scaleBrightness(self):
        self.history.append('scaled experiment brightness values')
        self.global_extrema = {}
        self.global_extrema['green'] = ympy.batchIntensityScale(
                self.folder_data,
                self.Param.image_reader,
                self.Param.reader_args,
                self.Param.green_channel,
                self.Param.show_progress)
        self.global_extrema['red'] = ympy.batchIntensityScale(
                self.folder_data,
                self.Param.image_reader,
                self.Param.reader_args,
                self.Param.red_channel,
                self.Param.show_progress)
    
    #%% select range to analyze based on YmpyParam values
    
    def analyzeRange(self):
        if self.Param.measure_fields == 'all':
            start = 0
            stop = self.folder_data['n_fields']
        else:
            start = self.Param.measure_fields[0]
            stop = self.Param.measure_fields[1]
        return(start, stop)
    
    #%% read image with rederHelper and image_reader
    
    def readCurrentField(self):    
        # read image
        field_path = self.folder_data['path_list'][self.current_field]
        image = self.Param.image_reader(
                **ympy.helpers.readerHelper(
                        self.Param.image_reader,
                        field_path,
                        self.Param.reader_args))
        image = ympy.helpers.cropRolloff(image, self.Param.image_rolloff)
        self.field_log.append('read image for field #' 
                              + str(self.current_field))
        return(image)
        
    #%% find cells and cleanup morphology
        
    def segmentImage(self, image):
        # find cells from brightfield step 1
        bw_cell_zstack = ympy.makeCellzStack( 
                image,
                self.Param.bf_channel,
                self.Param.show_progress)
        # find cells from brightfield step 2
        nZslices = image.shape[1]
        for z in range(nZslices):
            bw_cell_zstack[z, :, :] = ympy.helpers.correctBFanomaly(
                    bw_cell_zstack[z, :, :],
                    self.Param.bf_offest_vector)
        # find cells from brightfield step 3
        raw_mcl = ympy.cellsFromZstack(
                bw_cell_zstack,
                self.Param.show_progress)[0]
        # find cells from brightfield step 4
        master_cell_label = ympy.bfCellMorphCleanup( 
                raw_mcl, 
                self.Param.show_progress, 
                self.Param.min_angle, 
                self.Param.min_length, 
                self.Param.closing_radius,
                self.Param.min_bud_size)
        self.field_log.append('segmented image for field #'
                              + str(self.current_field))
        self.ncells = np.max(master_cell_label)
        self.field_log.append(
                'found {} cells in field #{}'.format(
                        self.ncells,
                        self.current_field))
        return(master_cell_label)
        
    #%% define measurment masks
    
    def buildMeasurementMasks(self, master_cell_label, image):
        # unbufferedMcl is the best guess at the 'true outside edge' of 
        # the cells; use it as the starting point to find a 10pixel thick 
        # cortex
        inner_cortex_mcl = ympy.labelCortex_mcl( 
                master_cell_label,
                self.Param.cortex_width)
        # because the bright field and fluorescence are not perfectly 
        # aligned, and to handle inaccuracies in edge finding, also buffer 
        # out from the outside edge
        buffer = ympy.buffer_mcl(
                master_cell_label,
                self.Param.buffer_size,
                self.Param.show_progress)
        # merge this buffer onto the master_cell_label and the 
        # inner_cortex_mcl
        self.buffered_master_cell_label = ympy.merge_labelMcl(
                master_cell_label,
                buffer) 
        full_cortex_mcl = ympy.merge_labelMcl(
                inner_cortex_mcl,
                buffer)
        # use Otsu thresholding on the max projection of RFPmarker
        marker_mcl_otsu = ympy.labelMaxproj(
                self.buffered_master_cell_label,
                image,
                self.Param.marker_channel)
        # then use centroidCircles to uniformly mask peri-golgi regions
        self.marker_mcl_ccadjusted = ympy.centroidCirclesMcl( 
                marker_mcl_otsu.astype('bool'), 
                self.buffered_master_cell_label,
                self.Param.marker_radius, 
                self.Param.marker_circle_iterations)
        # subtract so that marker localization has precedence over cortical
        # localization
        self.cortex_mcl_nonmarker = ympy.subtract_labelMcl( 
                full_cortex_mcl, 
                self.marker_mcl_ccadjusted)
        # finally, compute mask for remaining cytoplasmic regions
        cytoplasm_mcl = ympy.subtract_labelMcl(
                self.buffered_master_cell_label,
                ympy.merge_labelMcl(
                        self.marker_mcl_ccadjusted,
                        self.cortex_mcl_nonmarker))
        ref_mcl_dict = {
                'cortex(non-{})'.format(self.Param.marker_name):
                        self.cortex_mcl_nonmarker,
                '{}(circles)'.format(self.Param.marker_name):
                        self.marker_mcl_ccadjusted,
                'cytoplasm': cytoplasm_mcl}
        self.field_log.append('built measurement masks for field #'
                              + str(self.current_field))
        return(ref_mcl_dict)
        
        #%% measure
        
    def measureSingleField(self, ref_mcl_dict, image):           
        # measure Art1-mNG in the middle z-slice
        primaryImage = {
                self.Param.measured_protein_name:
                        image[self.Param.measured_protein_channel,
                              self.Param.measured_protein_z, :, :]}
        # measure against buffered cortex (minus marker mask), marker, and  
        # cytoplasm
        # also record field wide information
        # measurement function
        results = ympy.measure_cells(
                primaryImage,
                self.buffered_master_cell_label,
                ref_mcl_dict,
                self.folder_data['imagename_list'][self.current_field],
                self.folder_data['expID_list'][self.current_field],
                self.current_field,
                self.global_extrema['green']['globalmin'],
                self.global_extrema['green']['globalmax'],
                self.Param.n_hist_bins,
                self.Param.show_progress)
        for cell in range(self.ncells):
            hist_scores = ympy.cortex_marker_histScore(
                    results[cell],
                    self.Param)
            results[cell].update(hist_scores)
        # add measurements from each field to total results
        self.field_log.append('measured fluorescence for field #'
                              + str(self.current_field))
        self.totalResults = list(np.concatenate((self.totalResults, results)))
        
        #%% quality control prep
        
    def buildQCfield(self, image):
        print('preparing quality control information')
        green_fluor = image[self.Param.measured_protein_channel, 
                            self.Param.measured_protein_z, :, :].astype(float)
        green_fluor_scaled = ((green_fluor.astype('float') 
                               - self.global_extrema['green']['globalmin'])
                              / (self.global_extrema['green']['globalmax']
                                 - self.global_extrema['green']['globalmin']))
        red_fluor = np.amax(
                image[self.Param.marker_channel,:,:,:], axis=0).astype(float)
        red_fluor_scaled = ((red_fluor.astype('float') 
                               - self.global_extrema['red']['globalmin'])
                              / (self.global_extrema['red']['globalmax']
                                 - self.global_extrema['red']['globalmin']))
        qc_mcl_list = [self.cortex_mcl_nonmarker, self.marker_mcl_ccadjusted]
        rgb_qc = ympy.prep_rgbQCImage(
                green_fluor_scaled,
                red_fluor_scaled,
                qc_mcl_list,
                self.Param.rgb_scaling_factors)
        self.field_log.append('built quality control image for field #'
                              + str(self.current_field))
        self.totalQC.append(rgb_qc)

    #%% pool and save

    def saveState(self):
        print('saving progress')
        self.field_log.append('saved state after analysis of field #'
                              + str(self.current_field))
        self.history.append(self.field_log)
        resultsDic = {
                'totalResults': self.totalResults,
                'totalQC': self.totalQC,
                'fieldsAnalyzed': self.fieldsAnalyzed,
                'totalMcl': self.totalMcl,
                'parameters': self.Param.listParameters(),
                'object_history': self.history
                }
        date_today = str(datetime.datetime.now().date())
        save_path = '{}/results/{}_analysis.p'.format(
                self.Param.folder_path, date_today)
        pickle.dump(resultsDic, open(save_path, 'wb'))
        print(self.folder_data['imagename_list'][self.current_field],
              ' complete at ', datetime.datetime.now())

