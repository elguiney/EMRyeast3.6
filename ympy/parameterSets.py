"""
parameter sets
"""
import ympy

default_parameter_dict = dict(
        exp_ID_loc=[0,6], #position of machine readable experiment ID
        image_rolloff = 64, #ignore border of n pixels
        red_channel = 0,
        green_channel = 1,
        bf_channel = 2,
        min_angle = 22, #angle of curvature at bud neck
        min_length = 5, #integration length for curvature measurement
        closing_radius = 3, #closing structure element size
        min_bud_size = 75,
        cortex_width = 10,
        buffer_size = 5, #buffer region outside of bf border estimate
        show_progress = True,
        marker_radius = 7, #size radius that defines typical marker foci
        marker_circle_iterations = 5, #number of iterations to convert
            #raw marker mask to foci centered circles
        n_hist_bins = 1000, #bins (across global min-max range)
        rgb_scaling_factors = [0.05,0.1], #semi empirically determend factors
            #to make qc images at a reasonable brightness and contrast
        gray_scaling_factors = [0.2,0.2], #same
        measure_fields = 'all', #either 'all', to measure entire folder, or
            #a slice. [5,10] would measure fields 5,6,7,8 and 9 as they
            #appear in folderData['pathList'] from ympy.batchParse()
        smoothing_kernel_width = 5, #smoothing kernel for histogram calibrated
            #foci-brightness and foci-area measurments
        qc_border_size = 10, #border around cells for qc
        return_total_results = True #whether measure_GFP_wRFPmarker should
            #return results as a variable in addition to saving them
            #(it always saves)
        )

template_GFPwMarker_params = dict(
        folder_path = ('home/user/microscopy/experiment'), # replace this with
            # a real folder path
        image_extension = 'R3D_D3D.dv', # deltavision deconvolved extension
            # -- replace with your extension
        image_reader = ympy.helpers.basicDVreader,
#            reader for deltavision format imges
#            -- replace with an appropriate reader for your format;
#            imageio.imread() uses the same syntax as basicDVreader and reads
#            many common formats.
#            -- Must be a function that accepts a filepath string as its first
#            argument.
#            -- Image must be in the order czxy; you may need to create
#            a wrapper function around imread() and call the numpy transpose
#            method if your microscope saves in a different order
        reader_args = {'n_channels': 3, 'z_first': True}, # additional args for
#            reader function. keyword format only. leave blank if necessary.
        measured_protein_name = 'Yfg1-GFP', # replace with your protein name
        marker_name = 'RFP-marker', # replace with your marker name
        measured_protein_channel = 1,
        measured_protein_z = 3, #measure in this z-slice
        marker_channel = 0,
        bf_offest_vector = [0, 0], #correction for scope bf vs fluorescence
            # slight misalignment; default is no correction
        )

class YmpyParam():
    def __init__(self, experiment_parameter_dict):
        for k, v in default_parameter_dict.items():
            setattr(self, k, v)
        for k, v in experiment_parameter_dict.items():
            setattr(self, k, v)
    def update(self, new_parameter_dict):
        for k, v in new_parameter_dict.items():
            setattr(self, k, v)
    def printParameters(self):
        for k, v in self.__dict__.items():
            print('{}: {}'.format(k, v))
    def listParameters(self):
        printDict = dict(self.__dict__)
        printDict['image_reader'] = str(self.image_reader)
        return printDict
