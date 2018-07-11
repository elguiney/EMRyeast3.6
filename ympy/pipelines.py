''' 
ympy bespoke measurement pipelines.
Follow the examples here to create your own
'''

import pickle
import numpy as np
import datetime
import os
import pandas as pd
import random
import scipy.ndimage as ndimage
import cv2

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
        self.total_bool_masks.append(self.bool_masks)
        self.field_log.append('saved state after analysis of field #'
                              + str(self.current_field))        
        self.history.append(self.field_log)
        resultsDic = {
                'totalResults': self.totalResults,
                'fieldsAnalyzed': self.fieldsAnalyzed,
                'totalMcl': self.totalMcl,
                'parameters': self.Param.listParameters(),
                'object_history': self.history,
                'total_bool_masks': self.total_bool_masks
                }
        date_today = str(datetime.datetime.now().date())
        save_path = '{}/results/{}_analysis.p'.format(
                self.Param.folder_path, date_today)
        pickle.dump(resultsDic, open(save_path, 'wb'))
        print(self.folder_data['imagename_list'][self.current_field],
              ' complete at ', datetime.datetime.now())
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

class QualityControlPipeline():
    def __init__(self, results_path, experiment_parameter_dict):
        total_results_dict = pickle.load(open(results_path, 'rb'))
        self.total_mcl = total_results_dict['totalMcl']
        self.total_bool_masks = total_results_dict['total_bool_masks']
        self.Param = YmpyParam(experiment_parameter_dict)
        self.history = total_results_dict['object_history']
        self.folder_data = ympy.batchParse(
                self.Param.folder_path, self.Param.exp_ID_loc,
                self.Param.image_extension)
        self.history.append('entered QualityControlPipeline')
        # assemble dataframes
        # results_df is the top level dataframe
        self.results_df = pd.DataFrame(total_results_dict['totalResults'])
        random_idx = list(range(len(self.results_df)))
        random.shuffle(random_idx, random.seed(results_path))        
        self.results_df = self.results_df.assign(randomIdx = random_idx)
        # working, accepted and rejected for sorting cells during qc
        self.working_df = pd.DataFrame({})
        self.accepted_df = pd.DataFrame({})
        self.rejected_df = pd.DataFrame({})
        self.sortQCDF()
    def prefilterQC(self):
        no_cytoplasm_mask = self.results_df['cytoplasm_area'] == 0  
        self.results_df.loc[no_cytoplasm_mask, 'qcStatus'] = 'autorejected'
        mrk_column_name = '{}(circles)_area'.format(self.Param.marker_name)
        mrk_noarea_mask = self.results_df[mrk_column_name] == 0
        self.results_df.loc[mrk_noarea_mask, 'qcStatus'] = 'autorejected'
        n_rejected = (self.results_df['qcStatus']
                      .str.contains('autorejected').sum())
        self.history.append('autorejected {} cells with zero area marker or'
                            ' cytoplasm masks'.format(n_rejected))
        self.sortQCDF()
    def sortQCDF(self):
        work_locs = self.results_df['qcStatus'].str.contains('unreviewed')
        accepted_locs = self.results_df['qcStatus'].str.contains('accepted')
        rejected_locs = self.results_df['qcStatus'].str.contains('rejected')
        autorejected_locs = self.results_df['qcStatus'].str.contains(
                'autorejected')
        self.working_df = self.results_df[work_locs]
        self.accepted_df = self.results_df[accepted_locs]
        self.rejected_df = self.results_df[rejected_locs | autorejected_locs]
        
class QCframeLib():
    def __init__(self, QualityControlPipeline, array_size=(5,5), frame_size=225):
        
        self.array_size = array_size
        self.frame_size = frame_size
        self.QCpipe = QualityControlPipeline
        self._ctx_key = 'cortex(non-{})_mask'.format(
                self.QCpipe.Param.marker_name)
        self._mk_key = '{}(circles)_mask'.format(
                self.QCpipe.Param.marker_name)
        self.initializeLibrary()
        
    def initializeLibrary(self):
        abscounter = 0
        p = self.QCpipe.Param
        index_series = self.QCpipe.working_df.index
        self.framelib = pd.DataFrame(index=index_series,
                                     columns=['red', 'green', 'colormask',
                                              'markermask','cortexmask',],
                                     dtype=object)
        current_field_idx = -1
        previous_field_idx = -1
        for df_idx in index_series:
            current_field_idx = self.QCpipe.results_df.loc[df_idx, 'fieldIdx']
            if current_field_idx is not previous_field_idx:
                field_path = self.QCpipe.folder_data[
                        'path_list'][current_field_idx]
                expanded_args = ympy.helpers.readerHelper(
                        p.image_reader, field_path, p.reader_args)
                current_image = p.image_reader(**expanded_args)
                current_image = ympy.helpers.cropRolloff(
                        current_image, p.image_rolloff)
                current_green = current_image[p.measured_protein_channel,
                                              p.measured_protein_z,::
                                              ].astype(float)
                current_red = np.amax(current_image[p.marker_channel,:,:,:
                                                    ], axis=0).astype(float)
                current_mcl = self.QCpipe.total_mcl[current_field_idx]
                current_marker_bool = self.QCpipe.total_bool_masks[
                        current_field_idx][self._mk_key]
                current_cortex_bool = self.QCpipe.total_bool_masks[
                        current_field_idx][self._ctx_key]
            cell_lbl = self.QCpipe.results_df.loc[df_idx, 'localLbl']
            cell_tablet = np.zeros(current_mcl.shape, dtype=bool)
            cell_tablet[np.abs(current_mcl) == cell_lbl] = 1
            #TODO functionailze this:
            #start
            bounds = np.where(cell_tablet)
            cell_top, cell_bottom, cell_left, cell_right = (
                    bounds[0].min(), bounds[0].max(),
                    bounds[1].min(), bounds[1].max())
            mid_y = cell_top + (cell_bottom-cell_top)/2
            mid_x = cell_left + (cell_right-cell_left)/2
            lower_bound = self.frame_size/2
            upper_bound_y = current_green.shape[0] -  lower_bound
            upper_bound_x = current_green.shape[1] -  lower_bound
            use_y = min(max(lower_bound, mid_y), upper_bound_y)
            use_x =  min(max(lower_bound, mid_x), upper_bound_x)
            top = int(use_y-self.frame_size/2)
            bottom =  int(use_y + self.frame_size/2)
            left = int(use_x-self.frame_size/2)
            right =  int(use_x + self.frame_size/2)
            #end
            mkr_tablet = np.zeros(current_mcl.shape, dtype=bool)
            mkr_tablet[cell_tablet & current_marker_bool] = 1
            mkr_tablet = mkr_tablet ^ ndimage.binary_erosion(mkr_tablet)
            ctx_tablet = np.zeros(current_mcl.shape, dtype=bool)
            ctx_tablet[cell_tablet & current_cortex_bool] = 1
            ctx_tablet = ctx_tablet ^ ndimage.binary_erosion(ctx_tablet)
            self.framelib.at[df_idx,'red'] = current_red[top:bottom,
                                                          left:right]
            self.framelib.at[df_idx,'green'] = current_green[top:bottom,
                                                              left:right]
            self.framelib.at[df_idx,'colormask'] = cell_tablet[top:bottom,
                                                                left:right]
            self.framelib.at[df_idx,'markermask'] = mkr_tablet[top:bottom,
                                                                left:right]
            self.framelib.at[df_idx,'cortexmask'] = ctx_tablet[top:bottom,
                                                                left:right]
            if p.show_progress:
                ympy.helpers.progressBar_text(abscounter,
                                              len(index_series),
                                              'building frame library')
                abscounter += 1
    def assembleNewQCPage(self, indexSeries):
        #make blank page according to array size
        row_size = self.array_size[0]*self.frame_size
        col_size = self.array_size[1]*self.frame_size
        self.qcpage = np.zeros((row_size,col_size,3))
        self.idxpage = np.zeros((row_size,col_size), dtype=int)
        for idx, df_idx in enumerate(indexSeries):
            col = idx % self.array_size[0]
            row = idx // self.array_size[1]
            qc_frame = self.assembleColoredFrame(df_idx)
            idx_frame = np.ones(2*[self.frame_size], dtype=int)*int(idx)
            self.qcpage = self.placeFrame(row, col, qc_frame, self.qcpage)
            self.idxpage = self.placeFrame(row, col, idx_frame, self.idxpage)
    def assembleBaseFrame(self, df_idx):
        cortexmask = self.framelib.loc[df_idx,'cortexmask']
        markermask = self.framelib.loc[df_idx,'markermask']
        red = self.scaleFrame(self.framelib.loc[df_idx,'red'], 1.2)
        red[cortexmask + markermask] = 1
        green = self.scaleFrame(self.framelib.loc[df_idx,'green'], 1.2)
        green[markermask] = 1
        blue = np.zeros((self.frame_size, self.frame_size))
        blue[cortexmask + markermask] = 1
        red = red.reshape(self.frame_size, self.frame_size, 1)
        green = green.reshape(self.frame_size, self.frame_size, 1)
        blue = blue.reshape(self.frame_size, self.frame_size, 1)
        base_image = np.concatenate((blue, green, red), axis=2)
        return base_image
    def assembleGrayFrame(self, df_idx):
        base_image = self.assembleBaseFrame(df_idx)
        gray_frame = np.max(base_image, axis=2)
        gray = gray_frame.reshape(self.frame_size, self.frame_size, 1)
        gray_image = np.concatenate(3*[gray], axis=2)
        return gray_image
    def assembleColoredFrame(self, df_idx):
        base_image = self.assembleBaseFrame(df_idx)
        gray_image = self.assembleGrayFrame(df_idx)
        colormask = self.framelib.loc[df_idx, 'colormask']
        colormask_image = np.concatenate(3*[colormask.reshape(
                self.frame_size, self.frame_size, 1)], axis=2)
        colored_cell_image = np.array(gray_image)
        colored_cell_image[colormask_image] = base_image[colormask_image]
        return colored_cell_image        
    def placeFrame(self, row, col, frame, page):
        if len(page.shape) == 3: is_rgb = True
        else: is_rgb = False
        top = row*self.frame_size
        bottom = (row+1)*self.frame_size
        left = col*self.frame_size
        right = (col+1)*self.frame_size
        if is_rgb:
            for z in range(page.shape[2]):
                page[top:bottom, left:right, z] = frame[:, :, z]
        else:
            page[top:bottom, left:right] = frame
        return page
    def assembleDetails(self):
        foo
    def scaleFrame(self, frame, factor):
        frame = (frame-np.min(frame)) * factor/(np.max(frame)-np.min(frame))
        frame[frame > 1] = 1
        return frame
    def click_cv2(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            idx_click = self.idxpage[y,x]
            df_idx_click = self.qcIdxSeries.iloc[idx_click]
            self._clickerstatus[idx_click] = ~self._clickerstatus[idx_click]
            print(df_idx_click)
            col_click = idx_click % self.array_size[0]
            row_click = idx_click // self.array_size[1]
            if self._clickerstatus[idx_click]:
                frame = self.assembleGrayFrame(df_idx_click)
                self.placeFrame(row_click, col_click, frame, self.qcpage)
            else:
                frame = self.assembleColoredFrame(df_idx_click)
                self.placeFrame(row_click, col_click, frame, self.qcpage)                
    def displayForQC(self):
        cv2.namedWindow('qc_window', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('qc_window', self.click_cv2)
        self.qcIdxSeries = pd.Series(self.QCpipe.working_df.iloc[0:25].index)
        self.assembleNewQCPage(self.qcIdxSeries)
        self._clickerstatus = np.zeros(25, dtype=bool)
        while True:
            cv2.imshow('qc_window', self.qcpage)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            
        
    