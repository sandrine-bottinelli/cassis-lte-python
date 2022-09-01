# Example script to create a single model with one component and two species.
# Output files are :
# - the model spectrum in fits and plain text formats, as well as a png image
# - a CASSIS configuration file (.ltm)
# A second component can be added (no output provided).
from cassis_lte_python.LTEmodel import ModelSpectrum, Species
from time import process_time


tag = '41505'
tag2 = '28502'
fileout = f'{tag}_{tag2}_220GHz'  # name without extension for output files

component_desc = {
    'c1': {'size': 20.,
           'tex': 30.,
           # 'vlsr': 0.0,
           # 'interacting': False,
           'species': [Species(tag, ntot=1.e15, fwhm=2.0),
                       Species(tag2, ntot=1.e15, fwhm=2.0)]
           }
}

# Uncomment and edit lines below to add another component :
# component_desc['c2'] = {
#     'size': 50.,
#     'tex': 20.,
#     'vlsr': 0.0,
#     'interacting': True,
#     'species': [Species(tag, ntot=1.e15, fwhm=1.0)]
# }

config = {
    'output_dir': './ltm_example_results',  # directory will be created if does not exist but not overwritten
    'fmin_ghz': 220.655,
    'fmax_ghz': 220.755,
    'df_mhz'  : 0.1,
    'noise_mk': 0.05,
#
    'tuning_info': {  # telescope associated to each frequency range
                   'jcmt': [200000., 300000.]
                  },
#
    'tc': 0.,
    'tcmb': 2.73,
#
    'components': component_desc
}

t1_start = process_time()

model = ModelSpectrum(config)
model.generate_lte_model()

model.save_spectrum(fileout, ext='fits')
model.save_spectrum(fileout, ext='txt')
model.make_plot(filename=fileout, gui=False,  # saves .png ; NB: takes about one second
                # basic=True,  # do not plot line positions (takes time in some cases)
                verbose=False  # do not print file location in terminal
                )
model.write_ltm(fileout)  # saves .ltm configuration that can be uploaded in CASSIS

print("Execution time : {:.2f} seconds".format(process_time() - t1_start))
