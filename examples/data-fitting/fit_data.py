from cassis_lte_python.LTEmodel import ModelSpectrum
from cassis_lte_python.sim.parameters import parameter_infos
import os


data_dir = './'
data_filename = '41505_28502_147_220GHz.txt'
data_file = os.path.join(data_dir, data_filename)

output_dir = './fit_results'  # output directory (will be created if does not exist)
source = 'mySource'
tag_list = ['41505',  # CH3CN
            '28502'  # H2CN
            ]
molecule = "_".join(tag_list)
myName = source + "_" + molecule
outputFile = myName + ".pdf"
outputConfig = myName + "_config.txt"

thresholds = {
    tag_list[0]: {'eup_max': 150},
    tag_list[1]: {'eup_max': 150, 'aij_min': 1.e-4}
}
# thresholds = None  # if no thresholds (find all lines)

continuum = 0.  # can be a float or a path to a 2-column file (freq_mhz, continuum_value)
input_dir = './inputs'  # directory containing the files rms_cal.txt and velocityRanges.txt

config = {
    'output_dir': output_dir,
    'data_file': data_file,
    'bandwidth': 30.0,
    'thresholds': thresholds,
    'tuning_info': {  # telescope associated to each frequency range present in the data file
        # 'iram': []  # leave empty if telescope applies to entire data range
        'iram': [100000., 150000.],
        'jcmt': [200000., 300000.]
    },
    'v_range': os.path.join(input_dir, "velocityRanges.txt"),
    'chi2_info': os.path.join(input_dir, "rms_cal.txt"),
    'oversampling': 5,
    'tc': continuum,
    'tcmb': 2.73,
    'minimize': True,
    'max_iter': 500000,
    'fit_kws': {'method': 'leastsq', 'xtol': 1.e-8, 'ftol': 1.e-8, 'gtol': 1.e-8},
    # 'fit_kws': {'method': 'basinhopping'},
    'exec_time': True,  # print execution time for minimization and file saving
    'components': {
        'c1': {
            'size': parameter_infos(min=1., max=50., value=30.0, vary=True),
            'tex': parameter_infos(min=10., max=300., value=100., vary=True),
            'vlsr': parameter_infos(min=0., max=10.0, value=4.0, vary=True),
            'interacting': True,
            'species': [
               {'tag': tag_list[0],
                'ntot': parameter_infos(min=0.001, max=1000., value=2.e16, factor=True),
                'fwhm': parameter_infos(min=1.0, max=15.0, value=3., vary=True)},
               {'tag': tag_list[1],
                'ntot': parameter_infos(min=0.001, max=1000., value=2.e15, factor=True),
                'fwhm': parameter_infos(expr=f'c1_fwhm_{tag_list[0]}')},  # to have the same fwhm as the 1st species
               ]
        },
    },

    'save_res_configs': True,  # to save the results of the fit and this configuration
    'name_config': outputConfig,
    'name_lam': myName,  # to save a CASSIS .lam configuration file
    # # CASSIS config file will be called 'name_lam' + '_' + 'tel_name' + '.lam' and saved in output_dir
    'plot_gui': True,
    'plot_file': False,
    'plot_kws': {
        'gui+file': {
            'model_err': True,
            'component_err': True,
            'display_all': True,  # if False, plot only windows with data used for chi2 calculation
            # 'windows': {43511: '1, 3-5'}  # dictionary with windows to plot per species ; overrides display_all
        },
        'gui_only': {  # keywords for GUI plot ; override keywords already in plot_kws
            'display_all': False
        },
        'file_only': {
            'filename': outputFile,
            'display_all': True,
            'ncols': 3, 'nrows': 6
        },
    },
}

reload = False  # 1st time run
# reload = True  # reload the config and change plotting parameters if

if not reload:
    res = ModelSpectrum(config)
else:
    new_kws = {
        'minimize': False, 'modeling': True,
        # 'bandwidth': 125.,
        'plot_gui': False,
        'plot_file': True,
        'plot_kws': {
            'gui+file': {
                'model_err': True,
                'component_err': True,
                'display_all': True,  # if False, plot only windows with data used for chi2 calculation
                # 'windows': {tag_list[0]: '1, 3-5'}  # dictionary with windows to plot per species ; overrides display_all
            },
            'gui_only': {  # keywords for GUI plot ; override keywords already in plot_kws
                'display_all': False
            },
            'file_only': {
                'filename': outputFile,
                'display_all': True,
                'model_err': False,
                'component_err': False,
                'ncols': 2, 'nrows': 6
            }
        }
    }
    ModelSpectrum(
        os.path.join(output_dir, outputConfig),
        **new_kws
    )


