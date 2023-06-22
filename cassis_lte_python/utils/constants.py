import astropy.constants.codata2014 as const  # h,k_B,c # SI units
import numpy as np
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
