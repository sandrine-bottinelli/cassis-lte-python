# Example script to create a single model with one component and two species.
# Output files are :
# - the model spectrum in fits and plain text formats, as well as a png image
# - a CASSIS configuration file (.ltm)
# A second component can be added (no output provided).
from cassis_lte_python.LTEmodel import ModelSpectrum


tag = '41505'  # CH3CN
tag2 = '28502'  # H2CN
fileout = f'{tag}_{tag2}_147_220GHz'  # name without extension for output files

thresholds = {
    tag: {'eup_max': 150},
    tag2: {'eup_max': 150, 'aij_min': 1.e-4}
}
component_desc = {
    'c1': {'size': 20.,
           'tex': 30.,
           'vlsr': 3.0,
           'interacting': True,
           'species': [{'tag': tag, 'ntot': 1.e14, 'fwhm': 2.0},
                       {'tag': tag2, 'ntot': 1.e15, 'fwhm': 2.0}]
           }
}

# Uncomment and edit lines below to add another component :
# component_desc['c2'] = {
#     'size': 50.,
#     'tex': 20.,
#     'vlsr': 0.0,
#     'interacting': True,
#     'species': [{'tag': tag, 'ntot': 5.e14, 'fwhm': 1.0}]
# }

config = {
    'output_dir': './ltm_example_results',  # directory will be created if does not exist but not overwritten
    'franges_ghz': [[147.090, 147.190], [220.655, 220.755]],
    'df_mhz': 0.1,
    'noise': [0.1, 0.2],  # in K, for each frequency range ; use a single float instead of a list if identical
#
    'tuning_info': {  # telescope associated to each frequency range
        'iram': [100000., 150000.],
        'jcmt': [200000., 300000.]
    },
#
    'tc': 0.,  # continuum : can be a float or a two-column ascii file with frequencies in Mhz and continuum values
    'tcmb': 2.73,
#
    'thresholds': thresholds,
    'components': component_desc,
    'modeling': True,
    'exec_time': True,  # print execution time for file saving
    'plot_gui': False,
    'plot_file': True,
    'plot_kws': {
        'file_only': {'filename': fileout + '.pdf',
                      'nrows': 2, 'ncols': 1}
    },
    'save_spec': True,  # save the spectrum in ascii or fits, depending on name below
    'file_spec': fileout + '.txt',  # filename for the spectrum
}

# Execute the configuration
ModelSpectrum(config)

# model.write_ltm(fileout)  # saves .ltm configuration that can be uploaded in CASSIS
