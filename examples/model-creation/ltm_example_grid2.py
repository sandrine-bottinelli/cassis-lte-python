# Example script to compute models with two components (each having 3 possible temperatures)
# and one species (each having 2 possible column densities) in each component.
# For each model, output files are :
# - the model spectrum in fits and plain text formats, as well as a png image
# - a CASSIS configuration file (.ltm)
from cassis_lte_python.LTEmodel import ModelSpectrum, Species, print_settings
import numpy as np
from itertools import product
from time import process_time


print_settings()  # to check user settings

all_models = True  # if False, only generates the 1st model out of all possible combinations

cpt_names = ['c1', 'c2']
fwhm_vals = {  # assume same FWHM for all species in each component
    cpt_names[0]: 5.,  # km/s
    cpt_names[1]: 10.  # km/s
}
parameter_space = {
    # 1st component
    cpt_names[0]+'_tex': np.linspace(100, 200, num=3),
    cpt_names[0]+'_49503': [8.0e14, 8.0e15],  # C4H
    # 2nd component
    cpt_names[1]+'_tex': np.linspace(100, 200, num=3),
    cpt_names[1]+'_38002': [3.0e13, 3.0e14]  # c-C3H2
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
    'output_dir': './ltm_example_grid2_results',  # directory will be created if does not exist but not overwritten
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

    components = {}
    for cname in cpt_names:
        tex = model_params[cname+'_tex']
        ctags = [p for p in model_params.keys() if cname in p]
        ctags.remove(cname+'_tex')
        tags = [ctag.split('_')[1] for ctag in ctags]
        species_list = []
        for tag, ctag in zip(tags, ctags):
            species_list.append(Species(tag, ntot=model_params[ctag], fwhm=fwhm_vals[cname]))
        components[cname] = {
            'tex': tex,  #
            # 'size': 1.e5,  # if not given, the value in LTEmodel_user_settings.py will be used
            # 'vlsr': 0.,  # if not given, the value in LTEmodel_user_settings.py will be used
            # 'interacting': False,  # default is False
            'species': species_list
        }

    config['components'] = components

    if im == 0:
        model = ModelSpectrum(config)
        model.generate_lte_model()
    else:
        model.update_configuration(config)

    # save in output_dir with filename made of tex, tags and column density values, e.g. c1_tex_100_49503_8.0e+15...
    fileout = ''
    for key, val in model_params.items():
        if 'tex' in key:
            skey = key
            sval = f'{int(val)}'
        else:  # column densities
            skey = key.split('_')[1]  # only keep the tag
            sval = f'{val:.1e}'
        fileout += f'{skey}_{sval}_'
    fileout = fileout[:-1]  # remove trailing underscore
    model.save_spectrum(fileout, ext='fits')
    model.save_spectrum(fileout, ext='txt')
    model.make_plot(filename=fileout, gui=False,   # saves .png ; NB: takes about one second if basic=True
                    basic=True,  # do not plot line positions nor individual components (takes time)
                    verbose=False  # do not print file location in terminal
                    )
    model.write_ltm(fileout)  # saves .ltm configuration that can be uploaded in CASSIS

print("Execution time : {:.2f} seconds".format(process_time() - t1_start))
