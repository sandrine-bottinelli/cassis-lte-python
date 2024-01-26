from cassis_lte_python.LTEmodel import ModelSpectrum
import os


data_dir = './'
data_filename = '41505_28502_147_220GHz.txt'
data_file = os.path.join(data_dir, data_filename)

tag_list = [
    '41505',  # CH3CN
    '28502'  # H2CN
]
thresholds = {
    tag_list[0]: {'eup_max': 150},
    tag_list[1]: {'eup_max': 150, 'aij_min': 1.e-4}
}
# thresholds = None  # if no thresholds (find all lines)

config = {
    'data_file': data_file,
    'bandwidth': 30.0,
    'thresholds': thresholds,
    'inspect': tag_list,
    'plot_gui': True
}

show_vrange = False  # first run with False and copy/edit the file inputs/velocityRanges.txt
# show_vrange = True  # to check the velocity ranges that will be fitted

if show_vrange:
    config['v_range'] = os.path.join("inputs", "velocityRanges.txt")

ModelSpectrum(config)
