"""Core functions and operations for the ccb library.
"""
import os as _os
import pandas as _pd
import subprocess as _sp
import multiprocessing as _mp
from psutil import virtual_memory as _vm

# get info on the cpu for setting memory/thread limits
_ncpu = _mp.cpu_count()
_mems = _vm().total / (1024 * 1024)


# set up a function to run external commands and return stdout/stderr
def run(cmd, stderr=True):
    """
    """
    # set whether or not stderr is included in return or just stdout
    if stderr:
        se = _sp.STDOUT
    else:
        se = None
    
    # run the command, and return stdout as a list
    try:
        proc = _sp.check_output(cmd, shell = True,
          stderr = se)
        return proc.split("\n")
        
    except _sp.CalledProcessError, e:
        output = e.output.strip()
        sp = output.find(":") + 2
        prnt.error(output[sp:])
        return e.output.strip().split("\n")


# test whether a file exists
def test_file(path, file_name='file'):
    """
    """
    try:
        if _os.path.isfile(path):
            return True
        else:
            prnt.error('{} does not exist: {}'.format(file_name, path))
            return False
    except:
        prnt.error('no {} path set'.format(file_name))
        return False
        

# test whether a directory exists
def test_dir(path, directory_name='directory'):
    """
    """
    try:
        if _os.path.isdir(path):
            return True
        else:
            prnt.error('{} does not exist: {}'.format(directory_name, path))
            return False
    except:
        prnt.error('no {} path set'.format(directory_name))
        return False

# set up a class to consistently print status/errors
class prnt:
    def __init__(self):
        pass
    
    @staticmethod
    def error(message):
        if type(message) is str:
            print("[ ERROR! ]: {}".format(message))
        elif type(message) is list:
            for item in list:
                print("[ ERROR! ]: {}".format(item))
        elif isinstance(message, _num.Number):
            print("[ ERROR! ]: {}".format(message))
        else:
            pass
        
    @staticmethod
    def status(message):
        if type(message) is str:
            print("[ STATUS ]: {}".format(message))
        elif type(message) is list:
            for item in list:
                print("[ STATUS ]: {}".format(item))
        elif isinstance(message, _num.Number):
            print("[ STATUS ]: {}".format(message))
        else:
            pass

class maxent:
    def __init__(self, samples=None, env_layers=None, model_dir=None, predict_layers=None,
                 bias_file=None, test_samples=None, tau=0.5, pct_test_points=0, n_background=10000,
                 n_replicates=1, replicate_type='bootstrap', features=['hinge'], write_grids=False,
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
            'pct_test_points': pct_test_points,
            'n_background': n_background,
            'n_replicates': n_replicates,
            'replicate_type': replicate_type,
            'features': features,
            'write_grids': write_grids,
            'logfile': logfile,
            'cache': cache,
            'tau': tau,
            # set a few properties for which species and layers to map
            'species_list': None,
            'all_species': True,
            'layers_list': None,
            'layers_original': None,
            'all_layers': True,
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
            'mem': mem,
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
            'prefixes': True,
            'path_maxent': 'maxent.jar',
            'path_java': 'java'
            }
        
        # set a dummy variable to state this object has not yet been initialized 
        #  (i.e., the sample file parsed for species)
        self.initialized_ = False
        
    def set_parameters(self, **kwargs):
        """
        """
        keys = kwargs.keys()
        for param in keys:
            self.parameters_[param] = kwargs[param]
            
    def get_parameter_keys(self):
        """
        """
        keys = self.parameters_.keys()
        keys.sort()
        return keys
        
    def get_parameter(self, *args):
        """
        """
        params = {}
        for item in args:
            params[item] = self.parameters_[item]

        return params
        
    def initialize(self, **kwargs):
        """
        """
        # get ready for just so, so many if statements.
        
        # set a flag to track progress on whether the settings are correct
        flag = True
        
        # check that the bare minimum parameters have been set
        if self.parameters_['samples'] is None:
            prnt.error("no sample file has been set.")
            flag = False
        
        if self.parameters_['env_layers'] is None:
            prnt.error("no environmental layers have been set.")
            flag = False
            
        # check the input/output paths exist
        if not test_file(self.parameters_['samples'], 'samples file'):
            flag = False
        
        if not test_dir(self.parameters_['env_layers'], 'environmental layers directory'):
            flag = False
            
        if not test_dir(self.parameters_['model_dir'], 'model output directory'):
            flag = False
            
        if self.parameters_['bias_file'] is not None:
            if not test_file(self.parameters_['bias_file'], 'bias file'):
                flag = False
                
        if self.parameters_['test_samples'] is not None:
            if not test_file(self.parameters_['test_samples'], 'test samples'):
                flag = False
                
        if self.parameters_['predict_layers'] is not None:
            if not test_dir(self.parameters_['predict_layers'], 'prediction directory'):
                flag = False
        
        # check correct formatting for several options    
        # set options for the features to use
        features_types = ['linear', 'quadratic', 'product', 'threshold', 'hinge', 'auto']
        features_default = ['hinge']
        for feature in self.parameters_['features']:
            if feature.lower() not in features_types:
                prnt.error("incorrect feature specified: {}".format(', '.join(feature)))
                prnt.error("must be one of: {}".format(', '.join(features_types)))
                prnt.error("using default: {}".format(', '.join(features_default)))
                continue
            
        # set how replicates are handled
        replicate_types = ['crossvalidate', 'bootstrap', 'subsample']
        replicate_types_default = 'crossvalidate'
        if self.parameters_['replicate_type'].lower() not in replicate_types:
            prnt.error("incorrect replicate type specified: {}".format(replicate_type))
            prnt.error("must be one of: {}".format(', '.join(replicate_types)))
            prnt.error("using default: {}".format(replicate_types_default))
            self.parameters_['replicate_type'] = replicate_types_default
            
        # set test percentage to an integer if a float is passed
        test_pct_default = 30
        if type(self.parameters_['pct_test_points']) is float:
            self.parameters_['pct_test_points'] = int(100 * self.parameters_['pct_test_points'])
        else:
            try:
                self.parameters_['pct_test_points'] = int(test_pct)
            except:
                prnt.error("incorrect test percent specified: {}".format(self.parameters_['pct_test_points']))
                prnt.error("must be an integer between 0-100")
                prnt.error("using default: {}".format(test_pct_default))
                self.parameters_['pct_test_points'] = test_pct_default

        # set the format for output data reporting
        formats = ['cloglog', 'logistic', 'cumulative', 'raw']
        formats_default = 'logistic'
        if self.parameters_['output_format'].lower() not in formats:
            prnt.error("incorrect output format specified: {}".format(self.parameters_['output_format']))
            prnt.error("must be one of: {}".format(', '.join(formats)))
            prnt.error("using default: {}".format(formats_default))
            self.outformat = formats_default
            
        # set the output file type if writing output files
        if self.parameters_['write_grids']:
            types = ['asc', 'bil', 'grd', 'mxe']
            types_default = 'bil'
            if outtype.lower() not in types:
                prnt.error("incorrect output data type specified: {}".format(outtype))
                prnt.error("must be one of: {}".format(', '.join(types)))
                prnt.error("using default: {}".format(types_default))
                self.outtype = types_default
            
        # then update with the flag - should be true if no problems arose
        self.initialized_ = flag
    
    def get_layers(self):
        """
        """
    
    def set_layers(self, directory, layers=None):
        """
        """
        
    def get_species(self):
        """
        """
        
    def set_species(self, species):
        """
        """
        
    def build_string(self):
        """
        """
        # first, check whether the options have been parsed through the initializer
        if not self.initialized_:
            self.initialize()
            
        # then get ready for just a stupid number of if statements
        s = []
        join = ' '
        sp_join = '_'
        
        # set the strings for running maxent command
        s.append(self.parameters_['path_java'])
        s.append('-mx{}m'.format(self.parameters_['mem']))
        s.append('-jar')
        s.append(self.parameters_['path_maxent'])

        # set it to autorun
        s.append('-a')
        
        # set the environmental layers
        s.append('-e')
        s.append(self.parameters_['env_layers'])
        
        # call out which layers will not be used, if set
        if not self.parameters_['all_layers']:
            diff = self.parameters_['layers_original'] - self.parameters_['layers_list']
            for layer in diff:
                s.append('-N')
                s.append(layer)
        
        # set the samples CSV
        s.append('-s')
        s.append(self.parameters_['samples'])
        
        # call out which species will be mapped if not all
        if not self.parameters_['all_species']:
            for sp in self.parameters_['species_list']:
                s.append('-E')
                split = sp.split()
                s.append(sp_join.join(split))
                
        # set the output directory
        s.append('-o')
        s.append(self.parameters_['model_dir'])
        
        # set the optional bias/test/prediction data
        if self.parameters_['bias_file'] is not None:
            s.append('biasfile={}'.format(self.parameters_['bias_file']))
        if self.parameters_['predict_layers'] is not None:
            s.append('-j')
            s.append(self.parameters_['predict_layers'])
        
        # set how test data are handled
        if self.parameters_['test_samples'] is not None:
            s.append('-T')
            s.append(self.parameters_['test_samples'])
        elif self.parameters_['pct_test_points'] is not 0:
            s.append('-X')
            s.append(self.parameters_['pct_test_points'])
            
        # set background and replicate data
        s.append('-MB')
        s.append(self.parameters_['n_background'])
        s.append('replicates={}'.format(self.parameters_['n_replicates']))
        s.append('replicatetype={}'.format(self.parameters_['replicate_type']))
        
        # set the features to calculate
        for feature in self.parameters_['features']:
            s.append(feature)
            
        # set options for writing grid data
        if self.parameters_['write_grids']:
            s.append('outputfiletype={}'.format(self.parameters_['output_type']))
        else:
            s.append('-x')
        s.append('outputformat={}'.format(self.parameters_['output_format']))
        
    def fit(self):
        """
        """
        # check that the object has been initialized to check on
        if not self.initialized:
            init = self.initalize()
            if not init:
                prnt.error("unable to run maxent. update your parameters then re-run {}.initialize()".format(self.__name__))
                return False
            
        # fist build the string to create the 