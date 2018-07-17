"""Core functions and operations for the ccb library.
"""
import pandas as _pd
import multiprocessing as _mp
from psutil import virtual_memory as _vm

# get info on the cpu for setting memory/thread limits
_ncpu = _mp.cpu_count()
_mems = _vm.virtual_memory()
_memb = _mems.total


class maxent:
    def __init__(self, samples=None, layers=None, outdir=None, projection=None):
        """
        """
        # set up a bunch of the parameters for running maxent from the command line
        
        # boolean for creating graphs showing predicted relative probabilities
        self.response_curves = response_curves
        
        # boolean for creaing output pictures of predicted spatial distributions
        self.pictures = pictures
        
        # boolean to measure variable importance
        self.jackknife
        
        # set the format for output data reporting
        formats = ['cloglog', 'logistic', 'cumulative', 'raw']
        formats_default = 'logistic'
        if outformat.lower() in formats:
            self.outformat = outformat
        else:
            print("[ ERROR! ]: incorrect output format specified: {}".format(outformat))
            print("[ ERROR! ]: must be one of: {}".format(', '.join(formats)))
            print("[ ERROR! ]: using default: {}".format(formats_default))
            self.outformat = formats_default
            
        # set the output file type
        types = ['asc', 'bil', 'grd', 'mxe']
        types_default = 'bil'
        if outtype.lower() in types:
            self.outtype = outtype
        else:
            print("[ ERROR! ]: incorrect output data type specified: {}".format(outtype))
            print("[ ERROR! ]: must be one of: {}".format(', '.join(types)))
            print("[ ERROR! ]: using default: {}".format(types_default))
            self.outtype = types_default
            
        # set the output directory
        self.outdir = outdir
        
        # set the directories or files for projection
        self.projection = projection
        
        # set the sample file
        self.samples = samples
        
        # boolean for log-scaling output pictures
        self.logscale = logscale
        
        # booleans for displaying warnings/tooltips
        self.warnings = warnings
        self.tooltips = tooltips
        
        # booleans for overwriting/skipping existing files
        self.overwrite = overwrite
        self.skip_exists = skip_exists
        
        # boolean to remove duplicate data points in the same grid cell
        self.remove_duplicates = remove_duplicates
        
        # booleans to write certain outputs
        self.write_clamp_grid = write_clamp_grid
        self.write_mess = write_mess
        self.write_plot = write_plot
        self.write_grids = write_grids
        self.write_plots = write_plots
        
        # parameters for sampling test data
        self.test_samples = test_sample_file
        
        # set test percentage to an integer if a float is passed
        test_pct_default = 30
        if type(test_pct) is float:
            self.test_pct = int(100 * test_pct)
        else:
            try:
                self.test_pct = int(test_pct)
            except:
                print("[ ERROR! ]: incorrect test percent specified: {}".format(test_pct))
                print("[ ERROR! ]: must be an integer between 0-100")
                print("[ ERROR! ]: using default: {}".format(test_pct_default))
                self.test_pct = test_pct_default
                
        # set the beta multiplier
        self.beta_multiplier = beta_multiplier
        
        # the number of background points
        self.n_background = n_background
        
        # set replicate parameters
        self.n_replicates = n_replicates
        
        # sample bias file (should be raster with 0 < values < 100)
        self.bias_file = bias_file
        
        # set how replicates are handled
        replicate_types = ['crossvalidate', 'bootstrap', 'subsample']
        replicate_types_default = 'crossvalidate'
        if replicate_type.lower() in replicate_types:
            self.replicate_type = replicate_type
        else:
            print("[ ERROR! ]: incorrect replicate type specified: {}".format(replicate_type))
            print("[ ERROR! ]: must be one of: {}".format(', '.join(replicate_types)))
            print("[ ERROR! ]: using default: {}".format(replicate_types_default))
            self.replicate_type = replicate_types_default
            
        # booleans for writing additional output files
        self.per_species_results = per_species_results
        self.write_background = write_background
        
        # set options for the features to use
        features_types = ['linear', 'quadratic', 'product', 'threshold', 'hinge', 'auto']
        features_default = ['hinge']
        if features in features_types:
            self.features = features
        else:
            print("[ ERROR! ]: incorrect features specified: {}".format(', '.join(features)))
            print("[ ERROR! ]: must be one of: {}".format(', '.join(features_types)))
            print("[ ERROR! ]: using default: {}".format(', '.join(features_default)))
            
        # set options for adding background samples
        self.add_samples_background = add_samples_background
        self.add_all_background = add_all_background
        
        # set clamping options
        self.fade_by_clamping = fade_by_clamping
        self.clamp = clamp
        
        # enable extrapolation to novel conditions
        self.extrapolate = self.extrapolate
        
        # set a dummy variable to state this object has not yet been initialized 
        #  (i.e., the sample file parsed for species)
        self.initialized = False
        
        # finally, set the memory and threads for running maxent
        self.memory = memory
        self.threads = threads
        
    def initialize(self):
        """
        """
        # check that the bare minimum data have been set
        if self.samples is None:
            print("[ ERROR! ]: no sample file has been set. Unable to initialize.")
            return -1
        
        if self.layers is None:
            print("[ ERROR! ]: no layers have been set. Unable to initialize.")
            return -1
    
    def set_layers(self, directory, layers=None):
        """
        """
        
    def build_string(self):
        """
        """
        
    def run(self):
        """
        """
        # check that the object has been initialized to check on
        if not self.initialized:
            print("[ ERROR! ]: unable to run maxent. run {}.initialize() first".format(self.__name__))
            return -1
            
        # fist build the string to create the 