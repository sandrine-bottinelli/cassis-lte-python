import astropy.constants.codata2014 as const  # h,k_B,c # SI units
import numpy as np
from matplotlib.pyplot import get_cmap
"""
A few useful constants
"""


C_LIGHT = const.c.value  # speed of light [m/s]
K_B = const.k_B.value  # Boltzman's constant [J/K]
H = const.h.value

FWHM2SIGMA = 1. / (2. * np.sqrt(2. * np.log(2.)))  # to convert the FWHM of a Gaussian to sigma

TEL_DIAM = {'iram': 30.,
            'apex': 12.,
            'jcmt': 15.,
            'gbt': 100.,
            'alma_400m': 400.,
            'alma_170m': 170.}

# CPT_COLORS = ['blue', 'green', 'mediumorchid']
# CPT_COLORS = [
#     'blue', 'dodgerblue', 'deepskyblue',
#     'orange',
#     'gold',
#     # 'yellow',
#     'green',
#     'purple', 'mediumorchid', 'pink']
TAB20 = get_cmap('tab20')(np.linspace(0, 1, 20))
PLOT_COLORS = np.concatenate([TAB20[:][::2], TAB20[:][1::2]])
# PLOT_LINESTYLES = ['-', '--']
PLOT_LINESTYLES = ['-', ':']
