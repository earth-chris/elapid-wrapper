"""Core functions and operations for the ccb library.
"""
import pandas as _pd
import multiprocessing as _mp
from psutil import virtual_memory as _vm

# get info on the cpu for setting memory/thread limits
_ncpu = _mp.cpu_count()
_mems = _vm().total / (1024 * 1024)


class maxent:
    def __init__(self, samples=None, env_layers=None, model_dir=None, predict_layers=None,
                 bias_file=None, test_samples=None, tau=0.5, n_test_points=0, n_background=10000,
                 n_replicates=1, replicate_type='bootstrap', features=None, write_grids=False,
                 logfile=None, cache=True, n_threads=_ncpu-1, mem=_mems/2):
        """
        """
        
        # assign the passed values to a parameters dictionary
        self.parameters_ = {
            'samples': samples,
            'model_dir': model_dir,
            'env_layers': env_layers,
            'predict_layers': predict_layers,
            'bias_file': bias_file,
            'test_samples': test_samples,
            'n_test_points': n_test_points,
            'n_background': n_background,
            'n_replicates': n_replicates,
            'replicate_type': replicate_type,
            'features': features,
            'write_grids': write_grids,
            'logfile': logfile,
            'cache': cache,
            'tau': tau,
            # and set a bunch of misc parameters that are usually too dumb to try and set
            'response_curves': False,
            'pictures': False,
            'jackknife': False,
            'output_format': 'logistic',
            'output_type': 'bil',
            'random_seed': False,
            'log_scale': True,
            'warnings': False,
            'tooltips': False,
            'ask_overwrite': False,
            'skip_if_exists': False,
            'remove_deuplicates': True,
            'write_clampgrid': True,
            'write_mess': True,
            'beta_multiplier': 1.0,
            'per_species_results': True,
            'write_background_predictions': True,
            'response_curve_exponent': False,
            'add_samples_to_background': False,
            'add_all_samples_to_background': False,
            'write_plot_data': False,
            'fade_by_clamping': False,
            'extrapolate': True,
            'visible': False,
            'auto_feature': False,
            'do_clamp': False,
            'plots': False,
            'append_to_results_file': False,
            'maximum_iterations': 500,
            'convergence_threshold': 1e-5,
            'adjust_sample_radius': 0,
            'n_threads': _ncpu-1,
            'min_samples_threshold_product': 80,
            'min_samples_quadratic': 10,
            'min_samples_hinge': 15,
            'beta_threshold': -1.,
            'beta_categorical': -1.,
            'beta_lqp': -1.,
            'beta_hinge': -1.,
            'verbose': True,
            'allow_partial_data': False,
            'nodata': -9999,
            'prefixes': True
            }
        
        # set a dummy variable to state this object has not yet been initialized 
        #  (i.e., the sample file parsed for species)
        self.initialized = False
        
    def update_parameters(self, **kwargs):
        """
        """
        keys = kwargs.keys()
        for param in keys:
            self.parameters_[param] = kwargs[param]
            
    def list_parameters(self):
        """
        """
        keys = self.parameters_.keys()
        keys.sort()
        return keys
        
    def initialize(self, **kwargs):
        """
        """
        # update the parameters dictionary to 
        
        # check that the bare minimum parameters have been set
        if self.parameters_['samples'] is None:
            print("[ ERROR! ]: no sample file has been set. Unable to initialize.")
            return -1
        
        if self.parameters_['env_layers'] is None:
            print("[ ERROR! ]: no layers have been set. Unable to initialize.")
            return -1
    
    def set_layers(self, directory, layers=None):
        """
        """
        
    def build_string(self):
        """
        """
        
        # set options for the features to use
        features_types = ['linear', 'quadratic', 'product', 'threshold', 'hinge', 'auto']
        features_default = ['hinge']
        if features in features_types:
            self.features = features
        else:
            print("[ ERROR! ]: incorrect features specified: {}".format(', '.join(features)))
            print("[ ERROR! ]: must be one of: {}".format(', '.join(features_types)))
            print("[ ERROR! ]: using default: {}".format(', '.join(features_default)))
            
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
        
    def fit(self):
        """
        """
        # check that the object has been initialized to check on
        if not self.initialized:
            print("[ ERROR! ]: unable to run maxent. run {}.initialize() first".format(self.__name__))
            return -1
            
        # fist build the string to create the 