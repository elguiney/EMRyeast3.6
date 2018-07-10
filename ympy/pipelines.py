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
        self.history.append(['entered main loop'])
        for field in range(*self.analyzeRange()):
            self.current_field = field
            print('field #{} of {} total'.format(field, total))
            self.readCurrentField()
            self.segmentImage()
            self.buildMeasurementMasks()
            self.measureSingleField()

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
        self.total_bool_masks = []
        self.fieldsAnalyzed = []
        self.totalMcl = []
        self.history = ['initialized']
        self.current_field = []
        self.field_log = []
        self.image = []
        self.unbuffered_master_cell_label = []
        self.buffered_master_cell_label = []
        # status flags
        self._found_results_folder = False
        self._scaled_brightness = False
        self._read_field = False
        self._segmented_image = False
        self._made_masks = False
        self._measured = False
        self._saved = False
        # order to run functions in
        self.order = [
              'setupResultsfolder',
              'scaleBrightness',
              'readCurrentField',
              'segmentImage',
              'buildMeasurementMasks',
              'measureSingleField',
              'saveState'
             ]
    
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
        self._found_results_folder = True

    #%% measure global values (slow for big datasets)
    
    def scaleBrightness(self):
        self.checkState('scaleBrightness')
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
        self._scaled_brightness = True
    

    #%% read image with rederHelper and image_reader
    
    def readCurrentField(self):
        # begin tracking current field
        self.field_log = self.history.pop()
        self.checkState('readCurrentField')
        # read image
        field_path = self.folder_data['path_list'][self.current_field]
        self.image = self.Param.image_reader(
                **ympy.helpers.readerHelper(
                        self.Param.image_reader,
                        field_path,
                        self.Param.reader_args))
        self.image = ympy.helpers.cropRolloff(
                self.image,
                self.Param.image_rolloff)
        self.field_log.append('read image for field #'+str(self.current_field))
        self._read_field = True
        # must resegment, mask, and measure after calling readCurrentField to
        # avoid measurement/masking mismatch
        self._segmented_image = False
        self._made_masks = False
        self._measured = False
        self._saved = False
        
    #%% find cells and cleanup morphology
        
    def segmentImage(self):
        self.checkState('segmentImage')
        # find cells from brightfield step 1
        bw_cell_zstack = ympy.makeCellzStack( 
                self.image,
                self.Param.bf_channel,
                self.Param.show_progress)
        # find cells from brightfield step 2
        nZslices = self.image.shape[1]
        for z in range(nZslices):
            bw_cell_zstack[z, :, :] = ympy.helpers.correctBFanomaly(
                    bw_cell_zstack[z, :, :],
                    self.Param.bf_offest_vector)
        # find cells from brightfield step 3
        raw_mcl = ympy.cellsFromZstack(
                bw_cell_zstack,
                self.Param.show_progress)[0]
        # find cells from brightfield step 4
        self.unbuffered_master_cell_label = ympy.bfCellMorphCleanup( 
                raw_mcl, 
                self.Param.show_progress, 
                self.Param.min_angle, 
                self.Param.min_length, 
                self.Param.closing_radius,
                self.Param.min_bud_size)
        self.field_log.append('segmented image for field #'
                              + str(self.current_field))
        self.ncells = np.max(self.unbuffered_master_cell_label)
        self.field_log.append(
                'found {} cells in field #{}'.format(
                        self.ncells,
                        self.current_field))
        self._segmented_image = True

    #%% define measurment masks
    
    def buildMeasurementMasks(self):
        self.checkState('buildMeasurementMasks')
        # unbufferedMcl is the best guess at the 'true outside edge' of 
        # the cells; use it as the starting point to find a 10pixel thick 
        # cortex
        inner_cortex_mcl = ympy.labelCortex_mcl( 
                self.unbuffered_master_cell_label,
                self.Param.cortex_width)
        # because the bright field and fluorescence are not perfectly 
        # aligned, and to handle inaccuracies in edge finding, also buffer 
        # out from the outside edge
        buffer = ympy.buffer_mcl(
                self.unbuffered_master_cell_label,
                self.Param.buffer_size,
                self.Param.show_progress)
        # merge this buffer onto the unbuffered_master_cell_label and the 
        # inner_cortex_mcl
        self.buffered_master_cell_label = ympy.merge_labelMcl(
                self.unbuffered_master_cell_label,
                buffer) 
        full_cortex_mcl = ympy.merge_labelMcl(
                inner_cortex_mcl,
                buffer)
        # use Otsu thresholding on the max projection of RFPmarker
        marker_mcl_otsu = ympy.labelMaxproj(
                self.buffered_master_cell_label,
                self.image,
                self.Param.marker_channel)
        # then use centroidCircles to uniformly mask peri-golgi regions
        marker_mcl_ccadjusted = ympy.centroidCirclesMcl( 
                marker_mcl_otsu.astype('bool'), 
                self.buffered_master_cell_label,
                self.Param.marker_radius, 
                self.Param.marker_circle_iterations)
        # subtract so that marker localization has precedence over cortical
        # localization
        cortex_mcl_nonmarker = ympy.subtract_labelMcl( 
                full_cortex_mcl, 
                marker_mcl_ccadjusted)
        # finally, compute mask for remaining cytoplasmic regions
        cytoplasm_mcl = ympy.subtract_labelMcl(
                self.buffered_master_cell_label,
                ympy.merge_labelMcl(
                        marker_mcl_ccadjusted,
                        cortex_mcl_nonmarker))
        self.ref_mcl_dict = {
                'cortex(non-{})'.format(self.Param.marker_name):
                        cortex_mcl_nonmarker,
                '{}(circles)'.format(self.Param.marker_name):
                        marker_mcl_ccadjusted,
                'cytoplasm': cytoplasm_mcl}
        self.field_log.append('built measurement masks for field #'
                              + str(self.current_field))
        self.bool_masks = {
                'cortex(non-{})_mask'.format(self.Param.marker_name):
                        cortex_mcl_nonmarker.astype(bool),
                '{}(circles)_mask'.format(self.Param.marker_name):
                        marker_mcl_ccadjusted.astype(bool),
                'unbuffered_mask':
                        self.unbuffered_master_cell_label.astype(bool)
                }
        self._made_masks = True
        
        #%% measure
        
    def measureSingleField(self):
        self.checkState('measureSingleField')
        # measure Art1-mNG in the middle z-slice
        primaryImage = {
                self.Param.measured_protein_name:
                        self.image[self.Param.measured_protein_channel,
                              self.Param.measured_protein_z, :, :]}
        # measure against buffered cortex (minus marker mask), marker, and  
        # cytoplasm
        # also record field wide information
        # measurement function
        results = ympy.measure_cells(
                primaryImage,
                self.buffered_master_cell_label,
                self.ref_mcl_dict,
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
        self._measured = True
        
    #%% pool and save

    def saveState(self):
        self.checkState('saveState')
        print('saving progress')
        self.fieldsAnalyzed.append(self.current_field)
        self.totalMcl.append(self.buffered_master_cell_label)
        resultsDic = {
                'totalResults': self.totalResults,
                'fieldsAnalyzed': self.fieldsAnalyzed,
                'totalMcl': self.totalMcl,
                'parameters': self.Param.listParameters(),
                'object_history': self.history,
                'total_bool_masks': self.bool_masks
                }
        date_today = str(datetime.datetime.now().date())
        save_path = '{}/results/{}_analysis.p'.format(
                self.Param.folder_path, date_today)
        pickle.dump(resultsDic, open(save_path, 'wb'))
        print(self.folder_data['imagename_list'][self.current_field],
              ' complete at ', datetime.datetime.now())
        self.field_log.append('saved state after analysis of field #'
                      + str(self.current_field))
        
        self.history.append(self.field_log)
        self._saved = True
    
    #%% helper methods
    def checkState(self, state_function_name):
        state = [self._found_results_folder,
                 self._scaled_brightness,
                 self._read_field,
                 self._segmented_image,
                 self._made_masks,
                 self._measured,
                 self._saved
                ]
        position = self.order.index(state_function_name)
        error_text_1 = ('\nmust call runPipeline,\nor call main pipeline'
                      'functions in order:\n')
        error_text_2 = ',\n'.join('{}: {}'.format(num +1, val)
                               for num, val in enumerate(self.order))
        error_text_3 = '\nattempted to call {} before calling {}'.format(
                state_function_name, ', '.join(
                        np.array(self.order[0:position])[~np.array(
                                state[0:position])]))
        error_text = error_text_1 + error_text_2 + error_text_3
        if not all(state[0:position]):
            raise Exception(error_text)
        if self._saved:
            if state_function_name is not 'readCurrentField':
                error_text = ('analysis finished for field {}, use '
                              'current_field and readCurrentField to' 
                              'initialize analysis of a new field'.format(
                                      self.current_field))
                raise Exception(error_text)
                
    
    def analyzeRange(self):
        if self.Param.measure_fields == 'all':
            start = 0
            stop = self.folder_data['n_fields']
        else:
            start = self.Param.measure_fields[0]
            stop = self.Param.measure_fields[1]
        return(start, stop)
    