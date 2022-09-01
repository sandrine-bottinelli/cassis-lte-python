# Example script to compute models with one component (having 3 possible temperatures)
# and two species (each having 2 possible column densities).
# For each model, output files are :
# - the model spectrum in fits and plain text formats, as well as a png image
# - a CASSIS configuration file (.ltm)
from cassis_lte_python.LTEmodel import ModelSpectrum, Species, print_settings
import numpy as np
from itertools import product
from time import process_time


print_settings()  # to check user settings

all_models = True  # if False, only generates the 1st model out of all possible combinations

fwhm1 = 5.  # km/s
parameter_space = {
    'tex': np.linspace(100, 200, num=3),
    '49503': [8.0e14, 8.0e15],  # C4H
    '38002': [3.0e13, 3.0e14]  # c-C3H2
}

# Build a list of dictionary combining the values given by parameter_space
keys = parameter_space.keys()
values = (parameter_space[key] for key in keys)
model_parameters = [dict(zip(keys, combination)) for combination in product(*values)]

if not all_models:
    model_parameters = [model_parameters[0]]

print(f'Computing {len(model_parameters)} models\n...')
t1_start = process_time()

config = {
    'output_dir': './ltm_example_grid1_results',  # directory will be created if does not exist but not overwritten
    'fmin_ghz': 85.6,
    'fmax_ghz': 85.7,
    'df_mhz'  : 0.1,
    'noise_mk': 0.,
#
    'tuning_info': {  # telescope associated to each frequency range in MHz
        'iram': [85000., 86000.]
    },
#
    'tc': 0.,  # continuum value in K
    'tcmb': 2.73,  # background ('CMB') temperature in K
}

step = 20
for im, model_params in enumerate(model_parameters):
    iter_nb = im + 1
    if iter_nb % step == 0 and iter_nb != 0:
        print(f"Execution time after {iter_nb} iterations : {(process_time() - t1_start):.2f} seconds\n...")

    tex = model_params['tex']
    tags = list(model_params.keys())
    tags.remove('tex')
    species_list = []
    for tag in tags:
        species_list.append(Species(tag, ntot=model_params[tag], fwhm=fwhm1))
    config['components'] = {
        'c1': {'tex': tex,
               # 'size': 1.e5,  # if not given, the value in LTEmodel_user_settings.py will be used
               # 'vlsr': 0.,  # if not given, the value in LTEmodel_user_settings.py will be used
               # 'interacting': False,  # default is False
               'species': species_list
               }
    }

    if im == 0:
        model = ModelSpectrum(config)
        model.generate_lte_model()
    else:
        model.update_configuration(config)

    # save in output_dir with filename made of tex, tags and column density values, e.g. c1_tex_100_49503_8.0e+15...
    fileout = f'tex_{int(tex)}'
    for tag in tags:
        fileout += f'_{tag}_{model_params[tag]:.1e}'
    model.save_spectrum(fileout, ext='fits')
    model.save_spectrum(fileout, ext='txt')
    model.make_plot(filename=fileout, gui=False,   # saves .png ; NB: takes about one second
                    # basic=True,  # do not plot line positions (takes time in some cases)
                    verbose=False  # do not print file location in terminal
                    )
    model.write_ltm(fileout)  # saves .ltm configuration that can be uploaded in CASSIS

print("Execution time : {:.2f} seconds".format(process_time() - t1_start))
