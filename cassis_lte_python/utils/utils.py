from __future__ import annotations

__all__ = [
    "File",
    "DataFile",
    "beam_function",
    "compute_jypb2k",
    "compute_tau0",
    "compute_weight",
    "concat_dict",
    "delta_v_to_delta_f",
    "dilution_factor",
    "dilution_factors",
    "expand_dict",
    "expand_string",
    "find_nearest",
    "find_nearest_id",
    "find_nearest_trans",
    "flatten_dic",
    "format_float",
    "format_time",
    "format_variable",
    "freq2velo",
    "frequency_to_velocity",
    "fwhm_to_sigma",
    "get_beam",
    "get_beam_size",
    "get_cubes",
    "get_df_row_from_freq_range",
    "get_extended_limits",
    "get_mask",
    "get_partition_function",
    "get_single_mask",
    "get_telescope",
    "get_tmb2ta_factor",
    "get_valid_pixels",
    "is_in_range",
    "jnu",
    "load_json",
    "make_map_image",
    "nearest_interp",
    "open_continuum_file",
    # "open_data_file",  # deprecated
    "pixels_snake_loop",
    "read_crtf",
    "read_noise_info",
    "read_species_info",
    "read_telescope_file",
    "reduce_wcs_dim",
    "remove_comments",
    "remove_trailing_comments",
    "retrieve_unit",
    "save_all_map_images",
    "save_all_map_images_one_file",
    "search_telescope_file",
    "select_from_ranges",
    "unflatten_dic",
    "velo2freq",
    "velocity_to_frequency",
    "write_continuum_file",
]

from typing import Literal, List

from cassis_lte_python.utils.logger import CassisLogger
from cassis_lte_python.utils.constants import C_LIGHT, K_B, H, UNITS
from cassis_lte_python.utils.settings import SETTINGS
from cassis_lte_python.database.species import get_partition_function

import numpy as np
import os
from io import StringIO
import astropy.io.fits as fits
from astropy import units as u
import pandas as pd
from spectral_cube import SpectralCube
from regions import Regions, PixCoord, EllipseSkyRegion
from astropy.wcs import WCS
from scipy.interpolate import interp1d
from datetime import timedelta
import warnings
import json
import matplotlib.colors as colors
import matplotlib.pyplot as plt
plt.ioff()


TELESCOPE_DIR = SETTINGS.TELESCOPE_DIR
NB_DECIMALS = SETTINGS.NB_DECIMALS

LOGGER = CassisLogger.create('utils')

FONT_DEF = 'Times'
fontsize = 12
plt.rc('font', size=fontsize)
plt.rc('axes', labelsize=fontsize)  # fontsize of the x and y labels
plt.rc(('xtick', 'ytick'), labelsize=fontsize)
# plt.rcParams.update({"text.usetex": True})
# plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = FONT_DEF

labels = {
    'Ntot': "$N_{tot}$",
    'Tex': "$T_{ex}$",
    'Vlsr': "$V_{LSR}$",
    'FWHM': "FWHM",
    'size': "Source size"
}
units = {
    'Ntot': "[cm$^{-2}$]",
    'Tex': "[K]",
    'Vlsr': "[km s$^{-1}$]",
    'FWHM': "[km s$^{-1}$]",
    'size': "[$''$]"
}
color_maps = {
    'Ntot': 'inferno',
    'Tex': 'inferno',
    'Vlsr': 'RdYlBu_r', #'coolwarm',
    'FWHM': 'viridis',
    'size': 'viridis'
}

fig_size_single = (5, 4.5)


class CustomJSONizer(json.JSONEncoder):
    def default(self, obj):
        return super().encode(bool(obj)) \
            if isinstance(obj, np.bool_) \
            else super().default(obj)


class File:
    def __init__(self, filepath, default_xunit='MHz', default_yunit='K'):
        self.filepath = filepath
        self._ext = os.path.splitext(filepath)[-1]
        self._raw_header = None
        self.header = {}
        self._xdata_mhz = None
        self._xdata = None
        self._ydata = None

        self.get_header()
        self.get_data()

        # self._xunit = self.header.get('xUnit', None)
        # self._yunit = self.header.get('yUnit', None)

        if self.xunit is None:
            # xunit = input(f"X-axis unit not found, please provide it : ")
            message = [f"X-axis unit not found in {self.filepath}, setting it to {default_xunit}.",
                       "To choose a different unit, e.g., GHz, "
                       "please add the following line at the beginning of your file:",
                       "    // unitx: GHz"
                       ]
            LOGGER.warning(message)
            self._xunit = 'MHz'
        if self.yunit is None:
            # yunit = input(f"Y-axis unit not found, please provide it : ")
            message = [f"Y-axis unit not found in {self.filepath}, setting it to {default_yunit}.",
                       "To choose a different unit, e.g., Jy/beam, "
                       "please add the following line at the beginning of your file:",
                       "    // unity: Jy/beam"
                       ]
            LOGGER.warning(message)
            self._yunit = 'K'

        if self.xunit != 'MHz':
            self._xdata_mhz = (self.xdata * u.Unit(self.xunit)).to('MHz').value
        else:
            self._xdata_mhz = self.xdata

    def get_header(self, comments=('#', '//')):
        if self._ext == '.fits':
            with fits.open(self.filepath) as hdu:
                self.header = hdu[1].header

        else:
            with open(self.filepath) as f:
                raw_header = []
                header = {}
                line = f.readline()
                try:
                    iter(comments)
                except TypeError:
                    comments = [comments]

                while any([line.startswith(comment) for comment in comments]) or len(line.strip()) == 0:
                    raw_header.append(line)
                    comment = [c for c in comments if line.startswith(c)][0]
                    if ":" in line and 'Point' not in line:
                        info = line.lstrip(comment).split(":", maxsplit=1)
                        header[info[0].strip()] = info[1].strip()
                    line = f.readline()

            self._raw_header = raw_header
            self.header = header

    def get_data(self):
        if self._ext == '.fits':
            self.read_fits()

        elif self._ext in ['.fus', '.lis']:
            self.read_cassis()

        else:
            # print('Unknown extension.')
            # return None
            data = np.genfromtxt(self.filepath, skip_header=len(self._raw_header), usecols=[0, 1], unpack=True)
            data = data[~np.isnan(data).any(axis=1)]  # TODO : check this line is working correctly

            self._xdata = data[0]
            self._ydata = data[1]
            self._xunit = retrieve_unit(self._raw_header)
            self._yunit = retrieve_unit(self._raw_header, unit_type='yaxis')

        inds = self.xdata.argsort()
        self._ydata = self.ydata[inds]
        self._xdata = self.xdata[inds]

    def read_fits(self):
        # TODO : distinguish cube fits from single spectrum fits
        with fits.open(self.filepath) as hdu:
            self.header = hdu[1].header
            data = hdu[1].data

            if isinstance(hdu[1], fits.BinTableHDU):
                self._xunit = hdu[1].columns.columns[0].unit
                self._yunit = hdu[1].columns.columns[1].unit
                # do the following to be able to use in a DataFrame
                # self._xdata = data[hdu[1].columns.columns[0].name].byteswap().newbyteorder()
                # self._ydata = data[hdu[1].columns.columns[1].name].byteswap().newbyteorder()
                dat = []
                for i in range(2):
                    A = data[hdu[1].columns.columns[i].name]
                    dat.append(A.view(A.dtype.newbyteorder()).byteswap())
                self._xdata, self._ydata = dat
            else:
                raise TypeError("Cannot open this fits file.")

    def read_cassis(self):
        data = np.genfromtxt(self.filepath, skip_header=len(self._raw_header), names=True)
        self._xdata = data['FreqLsb']
        self._xunit = 'MHz'
        self._yunit = 'K'
        try:
            self._ydata = data['Intensity']
        except ValueError:
            self._ydata = data['Int']

    def write(self, fileout, sep='\t'):
        if self._ext == '.txt':
            hdr = ''.join([f'{key}: {val}\n' for key, val in self.header])
            np.savetxt(fileout, np.array([self.xdata, self.ydata]).T, delimiter=sep, header=hdr)
        elif self._ext == '.fits':
            tab_hdu_out = fits.BinTableHDU.from_columns(
                [fits.Column(name='wave', format='D', array=self.xdata, unit=self.xunit),
                 fits.Column(name='flux', format='D', array=self.ydata, unit=self.yunit)],
                 header=self.header
            )
            with fits.open(self.filepath) as hdu:
                hdu[1] = tab_hdu_out
                hdu.writeto(fileout, overwrite=True)
        else:
            raise TypeError("Unknown extension.")

    @property
    def xdata(self):
        return self._xdata

    @property
    def xdata_mhz(self):
        return self._xdata_mhz

    @property
    def ydata(self):
        return self._ydata

    @property
    def xunit(self):
        return self._xunit

    @xunit.setter
    def xunit(self, value):
        if value != self._xunit:
            # convert x-axis
            self._xdata = self._xdata * u.Unit(self._xunit).to(value).value
            # update header
            if self.header is not None:
                if isinstance(self.header, dict):
                    self.header['xLabel'].replace(self._xunit, value)
                else:  # assume header from fits
                    if 'CUNIT1' in self.header:
                        self.header['CUNIT1'] = value
                    else:
                        self.header['TUNIT1'] = value
            try:
                self.header['WAVE'] = (self.header['WAVE'], f'[{value}]')
            except KeyError:
                pass
            # set new unit
            self._xunit = value

    @property
    def yunit(self):
        return self._yunit


class DataFile(File):
    def __init__(self, filepath, telescope=None):
        super().__init__(filepath)  # read header, x and y data, and set the units
        self.vlsr = 0.
        self.bmaj = None
        self.bmin = None
        self._telescope = telescope

        self.get_vlsr()

        if self.yunit in UNITS['flux']:
            self.read_beam()

    def get_vlsr(self):
        for k, v in self.header.items():
            if 'vlsr' in k or 'VLSR' in k:
                self.vlsr = float(v)
                return
        LOGGER.warning(f"Vlsr not found, using {self.vlsr} km/s")

    def read_beam(self):
        try:
            beam_info = self.header['beam size']
            elts = beam_info.split()
            beam = []
            for e in elts:
                try:
                    beam.append(float(e))
                except ValueError:
                    continue
            self.bmaj = max(beam)
            self.bmin = min(beam)
        except KeyError:
            try:
                bmaj_unit, bmin_unit = 'deg', 'deg'
                if self.header['NAXIS'] > 2:  # assume cube with AXIS1 = RA, AXIS2 = Dec
                    if 'CUNIT1' in self.header:
                        bmaj_unit = self.header['CUNIT1']
                        bmin_unit = self.header['CUNIT2']
                    else:
                        bmaj_unit = self.header['TUNIT1']
                        bmin_unit = self.header['TUNIT2']
                self.bmaj = (self.header['BMAJ'] * u.Unit(bmaj_unit)).to('arcsec').value
                self.bmin = (self.header['BMIN'] * u.Unit(bmin_unit)).to('arcsec').value
            except KeyError:
                warnings.warn(f"Warning : data in {self.yunit} but beam not found in data file, "
                              f"make sure it is present in telescope file.")

    def beam(self):
        return self.bmaj, self.bmin

    @property
    def yunit(self):
        return self._yunit

    @yunit.setter
    def yunit(self, value: str):
        if value != self._yunit:
            if ((self._yunit in UNITS['flux'] and value == 'K')
                    or (self._yunit == 'K' and value in UNITS['flux'])):
                if self._telescope is not None:
                    locations = [self._telescope, os.path.join(TELESCOPE_DIR, self._telescope)]
                    tel_info = None
                    for loc in locations:
                        try:
                            tel_info = read_telescope_file(loc)
                        except FileNotFoundError:
                            continue
                    if tel_info is None:
                        raise FileNotFoundError(f"None of the following telescope files were found : "
                                                f"{', '.join([loc for loc in locations])}")
                    jypb2k = compute_jypb2k(self._xdata_mhz, [get_beam(f_i, tel_info) for f_i in self.xdata_mhz])
                elif self.bmaj is not None and self.bmin is not None:
                    jypb2k = compute_jypb2k(self._xdata_mhz, np.sqrt(self.bmaj * self.bmin))
                else:
                    raise ValueError("No beam information found - cannot perform conversion.")
                # convert y-axis
                if value == 'K':  # from K to jy/beam
                    self._ydata = self.ydata * jypb2k
                else:  # from jy/beam to K
                    self._ydata = self.ydata / jypb2k
                # update header
                if self.header is not None:
                    if isinstance(self.header, dict):
                        self.header['yLabel'].replace(self._yunit, value)
                    else:  # assume header from fits
                        if 'CUNIT2' in self.header:
                            self.header['CUNIT2'] = value
                        else:
                            self.header['TUNIT2'] = value
                        try:
                            self.header['FLUX'] = (self.header['FLUX'], f'[{value}]')
                        except KeyError:
                            pass
                # set new unit
                self._yunit = value
            else:
                LOGGER.warning(f"Cannot convert from {self._yunit} to {value}, nothing done.")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)  # , cls=type(self))
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found : {path}")


def open_continuum_file(filepath):
    data = File(filepath)
    return data.xdata_mhz, data.ydata


def write_continuum_file(filepath, cont_data, xunit='MHz', yunit='K'):
    with open(filepath, 'w') as fileout:
        fileout.write(f'// xUnit: {xunit}\n')
        fileout.write(f'// yUnit: {yunit}\n')
        row = cont_data.iloc[0]
        fileout.write('{:.4f}\t{:.4f}\n'.format(round(row['fmhz_range'][0] - 10.0), row['continuum']))

        for i, row in cont_data.iterrows():
            fileout.write('{:.4f}\t{:.4f}\n'.format(row['fmhz_range'][0], row['continuum']))
            fileout.write('{:.4f}\t{:.4f}\n'.format(row['fmhz_range'][1], row['continuum']))

        row = cont_data.iloc[-1]
        fileout.write('{:.4f}\t{:.4f}\n'.format(round(row['fmhz_range'][1] + 10.0), row['continuum']))


def open_data_file(filepath, continuum=False):
    """Deprecated - for backward compatibility"""
    data = DataFile(filepath)
    return data.xdata_mhz, data.ydata, data.vlsr


def retrieve_unit(infos: str | list, unit_type='xaxis') -> str:
    """

    :param infos: string or list of strings containing a unit
    :param unit_type: choose between xaxis or yaxis
    :return: the unit
    """
    try:
        units = UNITS[unit_type]
    except KeyError:
        raise KeyError("unit_type can only be xaxis or yaxis")

    if not isinstance(infos, list):
        infos = [infos]

    for unit in units:
        for s in infos:
            if unit.lower() in s.lower():
                return unit


def format_float(value, fmt=None, nb_digits=6, nb_signif_digits=NB_DECIMALS, min_exponent=-2):
    """
    Convert a float value to a desired formatted string.

    :param value: the value to format
    :param fmt: the format to use, e.g., ".2f" or "{:.3e}"
    :param nb_digits: number of digits above which exponent format will be used
    :param nb_signif_digits: max number of decimal digits
    :param min_exponent: power of 10 below which exponent format will be used
    :return: the formatted value
    """
    if fmt:
        if fmt.startswith("{"):
            return fmt.format(value)
        else:
            fmt = '{:' + fmt + '}'
            return fmt.format(value)

    power = np.log10(abs(value)) if value != 0 else 0.
    rpst = "e" if (power < min_exponent or power > nb_digits) else "f"
    if nb_signif_digits is None:
        if rpst == "e":
            return np.format_float_scientific(value)
        else:
            return np.format_float_positional(value)
    else:
        f = "{:." + str(nb_signif_digits) + rpst + "}"
        return f.format(value)

def format_variable(var):
    if isinstance(var, str):
        return var
    elif isinstance(var, int):
        return str(var)
    else:
        return format_float(var)


def format_time(t_sec):
    """
    Formats a time in seconds into the appropriate shape depending on its value.

    :param t_sec: time in seconds
    :return:
    """
    delta = timedelta(seconds=t_sec)
    if delta.days == 0:
        if t_sec < 60:  # less than a minute, return seconds
            return f"{t_sec:.2f} seconds"
        else:
            res = str(delta).split(sep=":")
            ms = f"{int(res[1])} minutes {float(res[2]):.1f} seconds"
            if t_sec < 3600:  # less than an hour, return minutes seconds
                return ms
            else:  # hours but less than a day
                return f"{int(res[0])} hours {ms}"
    else:
        return str(delta)


def remove_trailing_comments(str_or_list_str: str | List[str]):
    if isinstance(str_or_list_str, str):
        str_or_list_str = [str_or_list_str]
    result = []
    for line in str_or_list_str:
        elts = line.split("#", maxsplit=1)
        if len(elts) > 0 and len(elts[0]) > 0:
            result.append(elts[0])
        else:
            result.append(line)
    return result if len(result) > 1 else result[0]


def remove_comments(list_str: List[str]) -> List[str]:
    list_str = [line for line in list_str if not line.startswith("#")]
    return remove_trailing_comments(list_str)


def select_from_ranges(x_values, ranges, y_values=None, oversampling=None):
    if type(ranges[0]) is not list:
        ranges = [ranges]

    x_new = []
    y_new = []

    for x_range in ranges:
        imin = find_nearest_id(x_values, min(x_range))
        imax = find_nearest_id(x_values, max(x_range))
        x_sub = x_values[imin:imax+1]
        # x_sub = x_values[(x_values >= min(x_range)) & (x_values <= max(x_range))]
        if len(x_sub) == 0:
            continue
        xmin, xmax = min(x_sub), max(x_sub)
        if oversampling is not None:
            # xmin, xmax = min(x_sub), max(x_sub)
            x_sub = np.linspace(xmin, xmax, num=len(x_sub)*oversampling, endpoint=True)
        x_new = np.append(x_new, x_sub)
        if y_values is not None:
            y_sub = y_values[imin:imax+1]
            # y_sub = y_values[(x_values >= xmin) & (x_values <= xmax)]
            y_new = np.append(y_new, y_sub)

    return x_new, y_new if y_values is not None else x_new


def nearest_interp(xi: int | float | list | np.ndarray,
                   x: list | np.ndarray,
                   y: list | np.ndarray | tuple):
    """
    Find y values corresponding to the x value closest to xi.

    :param xi: new x values ; must be within x values
    :param x: reference values
    :param y: values to be interpolated ; if want to interpolate several arrays, must be given as a tuple
    :return: interpolated values
    """
    # Shift x points to centers
    spacing = np.diff(x) / 2
    x = x + np.hstack([spacing, spacing[-1]])

    if isinstance(y, tuple):
        y_out = []
        for yarr in y:
            # Append the last point in y twice for ease of use
            yarr2 = np.hstack([yarr, yarr[-1]])
            y_out.append(yarr2[np.searchsorted(x, xi)])
        return tuple(y_out)

    # Append the last point in y twice for ease of use
    y = np.hstack([y, y[-1]])
    return y[np.searchsorted(x, xi)]


def find_nearest(arr, value):
    """
    Find the value in "arr" that is closest to "value".

    :param arr:
    :param value:
    :return:
    """
    idx = find_nearest_id(arr, value)
    return arr[idx]


def find_nearest_id(arr: np.ndarray | list, value):
    """
    Find the index of the value in "arr" that is closest to "value".

    :param arr:
    :param value:
    :return:
    """
    if isinstance(arr, list):
        arr = np.array(arr)

    return (abs(arr - value)).argmin()  # NB: could improve memory usage by using dichotomy


def find_nearest_trans(trans_list, value):
    f_trans_list = [tr.f_trans_mhz for tr in trans_list]
    idx = (abs(np.array(f_trans_list) - value)).argmin()
    return trans_list[idx]


def is_in_range(fmhz, list_ranges):
    res = False
    for rg in list_ranges:
        if rg[0] <= fmhz <= rg[1]:
            res = True
    return res


def concat_dict(d1: dict, d2: dict) -> dict:
    """
    Concatenate two dictionaries ; if a keyword is present in both, value from d2 is kept.

    :param d1: a dictionary
    :param d2: another dictionary whose keywords/values will be added to or will override those in d1
    :return: the concatenated dictionary
    """
    d1_2 = d1
    for k, v in d2.items():
        d1_2[k] = v

    return d1_2


def expand_string(s: str) -> list:
    """
    Expand a string of the form '1, 3-5' into a list of integer [1, 3, 4, 5]

    :param s:
    :return:
    """
    li = []
    for elt in s.split(','):
        if '-' in elt:
            nmin, nmax = elt.split('-')
            for i in range(int(nmin), int(nmax) + 1):
                li.append(i)
        else:
            li.append(int(elt))

    return li


def expand_dict(dic: dict, n_items=None, expand_vals=False):
    if '*' in dic:
        if n_items is not None:
            new_dic = {i + 1: dic['*'] for i in range(n_items)}
        else:
            # print("Missing number of transitions.")
            return dic
    else:
        new_dic = {}
        for k, v in dic.items():
            if expand_vals:
                new_dic[k] = expand_string(v)
            else:
                for nb in expand_string(k):
                    new_dic[nb] = v

    return new_dic


def flatten_dic(d, sep=";"):
    if all([not isinstance(v, dict) for v in d.values()]):
        return d
    else:
        dflat = {}
        for k, v in d.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    dflat[f'{k}{sep}{k2}'] = v2
            else:
                dflat[k] = v
        return flatten_dic(dflat)


def unflatten_dic(dic, sep=";"):
    d_unflat = dict()
    for k, v in dic.items():
        keys = k.split(sep)
        d = d_unflat
        for key in keys[:-1]:
            if key not in d:
                d[key] = dict()
            d = d[key]

        d[keys[-1]] = v

    return d_unflat


def get_extended_limits(values, padding=0.05):
    dx = max(values) - min(values)
    return [min(values) - padding * dx, max(values) + padding * dx]


def get_df_row_from_freq_range(df: pd.DataFrame, freq: float) -> tuple:
    try:
        row = df[(freq > df['fmin']) & (freq < df['fmax'])]
        return row
    except IndexError:
        raise IndexError("The dataframe must have 'fmin' and 'fmax' columns.")


def read_noise_info(noise_file):
    noise_info = {}
    with open(noise_file) as f:
        all_lines = f.readlines()
        for line in all_lines:
            if line.startswith('#') or len(line.strip()) == 0:
                continue
            elts = line.split()
            if len(elts) == 1:  # line only has one element -> tag
                tag = elts[0]
                noise_info[tag] = {}
            else:
                # noise info : vmin vmax line numbers or rms cal line numbers
                noise_info[tag][''.join(elts[2:])] = [float(elts[0]), float(elts[1])]

    return noise_info


def read_species_info(sp_file: str | StringIO, header='infer'):
    if isinstance(sp_file, str):
        with open(sp_file) as f:
            lines = f.readlines()
    elif isinstance(sp_file, StringIO):
        lines = sp_file.readlines()
    else:
        lines = sp_file
    lines = [line.rstrip() for line in lines]
    lines = remove_trailing_comments(lines)
    fh = StringIO("\n".join(lines))
    df = pd.read_csv(fh, delimiter='\t', comment='#', index_col=0, dtype=str, header=header)
    # perform check on number of columns for components
    # ncols_cpt = [col for col in df.columns if col.startswith('c')]
    # if len(ncols_cpt) % 4 != 0:  # ncols_cpt must be a multiple of 4
    #     raise ValueError(f"Number of columns for components in {sp_file} "
    #                      f"is not a multiple of 4.")
    if df.index.has_duplicates:
        dup = df.index[df.index.duplicated()]
        raise ValueError('Duplicate species infos detected for tags :',
                         ", ".join([str(val) for val in dup.values]))
    sp_infos = df.apply(pd.to_numeric, errors='coerce')
    sp_infos = sp_infos.fillna('*')
    # sp_infos['tag'] = sp_infos['tag'].astype("string")
    return sp_infos


def compute_tau0(tran, ntot, fwhm, tex, qtex=None):
    if qtex is None:
        qtex = get_partition_function(tran.tag, temp=tex)
    # nup = ntot * tran.gup / qtex / np.exp(tran.eup_J / const.k_B.value / tex)  # [cm-2]
    nup = ntot * tran.gup / qtex / np.exp(tran.eup / tex)  # [cm-2]
    tau0 = C_LIGHT ** 3 * tran.aij * nup * 1.e4 * (np.exp(H * tran.f_trans_mhz * 1.e6 / K_B / tex) - 1.) \
           / (4. * np.pi * (tran.f_trans_mhz * 1.e6) ** 3 * fwhm * 1.e3 * np.sqrt(np.pi / np.log(2.)))
    return tau0


def compute_weight(intensity, rms, cal):
    """
    Returns the weight as 1./sqrt(rms**2 + cal_uncertainty**2) where cal_uncertainty is the calibration uncertainty
    in percent * the intensity at the given frequency.

    :param intensity: intensity
    :param rms: noise for the given intensity, in the same units
    :param cal: calibration uncertainty in percent
    :return: 1. / sqrt(rms**2 + cal_uncertainty**2)
    """
    for arg in [intensity, rms, cal]:
        if isinstance(arg, list):  # convert to numpy array if list
            arg = np.array(arg)
    cal = cal / 100.
    return 1. / np.sqrt(rms**2 + (cal * intensity)**2)


def fwhm_to_sigma(value, reverse=False):
    if reverse:
        return value * (2. * np.sqrt(2. * np.log(2.)))
    else:
        return value / (2. * np.sqrt(2. * np.log(2.)))


def delta_v_to_delta_f(value, fref_mhz, reverse=False):
    if reverse:
        return (value / fref_mhz) * C_LIGHT * 1.e-3
    else:
        return value * 1.e3 * fref_mhz / C_LIGHT


def velocity_to_frequency(value, fref_mhz, vref_kms=0., reverse=False):
    """
    Convert a velocity to a frequency.

    :param value: velocity to be converted, in the same unit as vref_kms (generally km/s)
    :param fref_mhz:
    :param vref_kms:
    :param reverse: deprecated (kept for compatibility) ; use frequency_to_velocity
    :return:
    """
    if reverse:
        return frequency_to_velocity(value, fref_mhz, vref_kms=vref_kms)
    return fref_mhz * (1. - (value - vref_kms) * 1.e3 / C_LIGHT)


def frequency_to_velocity(value, fref_mhz, vref_kms=0.):
    return C_LIGHT * (1. - value / fref_mhz) * 1.e-3 + vref_kms


def velo2freq(f_ref, vref_kms):
    return lambda v: velocity_to_frequency(v, f_ref, vref_kms=vref_kms)


def freq2velo(f_ref, vref_kms):
    return lambda v: frequency_to_velocity(v, f_ref, vref_kms=vref_kms)


def Teq(fmhz):
    return H * fmhz * 1.e6 / K_B


def jnu(fmhz, temp: float):
    fmhz_arr = np.array(fmhz) if type(fmhz) == list else fmhz
    res = (H * fmhz_arr * 1.e6 / K_B) / \
          (np.exp(H * fmhz_arr * 1.e6 / (K_B * temp)) - 1.)
    return list(res) if type(fmhz) == list else res


def search_telescope_file(tel):
    if os.path.isfile(os.path.join(TELESCOPE_DIR, tel)):
        return os.path.join(TELESCOPE_DIR, tel)
    elif os.path.isfile(tel):
        return tel
    else:
        raise FileNotFoundError(f"Telescope file {tel} not found.")


def read_telescope_file(telescope_file):
    """
    Read telescope file into dataframe

    :param telescope_file: CASSIS telescope file
    :return: dataframe
    """
    with open(telescope_file, 'r') as f:
        col_names = ['Frequency (MHz)', 'Beff/Feff']
        tel_data = f.readlines()
        tel_diam = float(tel_data[1])
        if int(tel_diam) == 0:
            col_names += ['Bmaj (arcsec)', 'Bmin (arcsec)']
        # get column names:
        line = tel_data[2]
        col_names = line.replace('.', ',').lstrip('//').split(',')
        col_names = [c.strip() for c in col_names]

        tel_info = pd.read_csv(telescope_file, sep='\t', skiprows=3,
                               names=col_names, usecols=list(range(len(col_names))))

        if not tel_info['Frequency (MHz)'].is_monotonic_increasing:
            LOGGER.warning(f"Frequencies in telescope file {telescope_file} are not in ascending order.")

        tel_info['Diameter (m)'] = [tel_diam for _ in range(len(tel_info))]

        return tel_info


def get_telescope(fmhz, tuning_info: pd.DataFrame):
    if isinstance(fmhz, float):
        fmhz = np.array(fmhz)

    if len(tuning_info) == 1:
        return [tuning_info['telescope'][0]] * len(fmhz)

    tel_list = np.empty(len(fmhz), dtype=object)
    for i, row in tuning_info.iterrows():
        tel_list[np.where((fmhz >= row['fmhz_min']) & (fmhz <= row['fmhz_max']))] = row['telescope']

    if None in tel_list:
        raise ValueError(f"Telescope not defined for at least one frequency: {fmhz[np.equal(tel_list, None)]}")

    return tel_list


def get_tmb2ta_factor(freq_mhz: float | int, tel_data: pd.DataFrame) -> float:
    """
    Retrieves the beam efficiency to convert from main-beam temperature to antenna temperature

    :param freq_mhz: frequency in MHz ; float
    :param tel_data: telescope dataframe containing "tuning" information
    :return:
    """
    f_tel = tel_data[tel_data.columns[0]]
    beff_tel = tel_data[tel_data.columns[1]]
    return np.interp(freq_mhz, f_tel, beff_tel)


def beam_function(tel_data: pd.DataFrame):
    bmin, bmaj = None, None
    if 'Bmin (arcsec)' in tel_data.columns:
        bmin = interp1d(tel_data['Frequency (MHz)'], tel_data['Bmin (arcsec)'])
    if 'Bmaj (arcsec)' in tel_data.columns:
        bmaj = interp1d(tel_data['Frequency (MHz)'], tel_data['Bmaj (arcsec)'])

    if bmin is not None and bmaj is not None:  # explicit beam major and minor axes
        beam = lambda f: np.sqrt(bmin(f) * bmaj(f))
    else:  # beam size from telescope diameter
        beam = lambda f: get_beam_size(f, np.interp(f, tel_data['Frequency (MHz)'].values,
                                                    tel_data['Diameter (m)'].values))
    return beam


def get_beam(freq_mhz: float | int, tel_data: pd.DataFrame):
    """
    Determine the beam at a given frequency

    :param freq_mhz:
    :param tel_data:
    :return:
    """
    tel_data_f = tel_data.iloc[(tel_data['Frequency (MHz)'] - freq_mhz).abs().argmin()]  # index of the closest freq
    if tel_data_f['Diameter (m)'] == 0:
        return tel_data_f['Bmaj (arcsec)'], tel_data_f['Bmin (arcsec)']
    else:
        bs = get_beam_size(freq_mhz, tel_data_f['Diameter (m)'])
        return bs, bs


def get_beam_size(freq_mhz: float | int | np.ndarray, tel_diam: float):
    """
    Computes the beam size at the given frequency for the given telescope diameter

    :param freq_mhz: frequency in MHz ; float or numpy array
    :param tel_diam: telescope diameter in meters
    :return: the beam size in arcsec
    """

    return (1.22 * C_LIGHT / (freq_mhz * 1.e6)) / tel_diam * 3600. * 180. / np.pi


def dilution_factors(source_size: float | int,
                     beam: tuple | list | np.ndarray,
                     geometry: Literal['gaussian', 'disc'] = 'gaussian') -> float | np.ndarray:
    """
    Compute the dilution factors for a given source size and a list or array of beam sizes.

    :param source_size:
    :param beam:
    :param geometry:
    :return:
    """
    if isinstance(beam, tuple):
        return dilution_factor(source_size, beam, geometry)
    else:
        return np.array([dilution_factor(source_size, b, geometry) for b in beam])


def dilution_factor(source_size: float | int, beam: int | float | tuple,
                    geometry: Literal['gaussian', 'disc'] = 'gaussian') -> float:
    """
    Compute the dilution factor for the given source and beam sizes, depending on the geometry

    :param source_size: in arcsec
    :param beam: in arcsec
    :param geometry: gaussian (default) or disc
    :return: the dilution factor
    """
    # dilution_factor = tr.mol_size ** 2 / (tr.mol_size**2 + get_beam_size(model.telescope,freq)**2)
    # dilution_factor = (1. - np.cos(cpt.size/3600./180.*np.pi)) / ( (1. - np.cos(cpt.size/3600./180.*np.pi))
    #                    + (1. - np.cos(get_beam_size(self.telescope,self.frequencies)/3600./180.*np.pi)) )
    geometries = ['gaussian', 'disc']

    if isinstance(beam, (int, float)):
        beam_size_sq = beam ** 2
    else:
        beam_size_sq = beam[0] * beam[1]

    if geometry == 'disc':
        return 1. - np.exp(-np.log(2.) * (source_size ** 2 / beam_size_sq))

    if geometry == 'gaussian':
        return source_size ** 2 / (source_size ** 2 + beam_size_sq)

    raise TypeError(f"Unsupported geometry, can only be {', '.join(geometries[:-1])} or {geometries[-1]}.")


def compute_jypb2k(freq_mhz: float | int | list | np.ndarray,
                   beam_arcsec: float | int | list | np.ndarray) -> float | np.ndarray:
    """
    Compute the conversion factor from Jansky per beam to Kelvin.
    T = (conv_fact / nu^2) * I , with :
    conv_fact = c^2 / (2*k_B*omega) * 1.e-26 (to convert Jy to mks)
    omega = pi*bmaj*bmin/(4*ln2)

    :param freq_mhz: frequency in MHz ; float, integer, list or numpy array
    :param beam_arcsec: 1D beam size in arcsec ; float, integer, list or numpy array
    :return: the conversion factor from Jy/beam to Kelvin
    """
    input_type = type(freq_mhz)

    if not isinstance(freq_mhz, np.ndarray):
        freq_mhz = np.array(freq_mhz)

    if not isinstance(beam_arcsec, np.ndarray):
        beam_arcsec = np.array(beam_arcsec)

    if len(freq_mhz) != len(beam_arcsec):
        raise IndexError("Lengths of frequency and beam arcsec do not match.")

    omega = (beam_arcsec * u.Unit('arcsec')).to('rad').value ** 2
    # beam_rad = beam_arcsec / 3600 / 180 * np.pi
    # omega = beam_rad ** 2
    omega *= np.pi / (4 * np.log(2))
    conv_fact = C_LIGHT ** 2 / (2 * K_B * omega) * 1.e-26
    conv_fact /= (freq_mhz * 1.e6) ** 2
    if input_type is list:
        conv_fact = list(conv_fact)
    if input_type is int or input_type is float:
        conv_fact = conv_fact[0]
    return conv_fact


def get_cubes(file_list, check_spatial_shape=None):
    """
    Retrieve data as SpectralCube objects. Checks whether the cubes in file_list have the same spatial shape and extent.

    :param file_list: the list of files (absolute paths)
    :param check_spatial_shape: a two-element tuple representing the spatial shape of some other cube.
    :return: a list of SpectralCube objects
    """
    hduls = [fits.open(f) for f in file_list]

    # check that cubes have the same number of pixels in RA and Dec:
    nx_list = set([hdul[0].shape[-2] for hdul in hduls])
    ny_list = set([hdul[0].shape[-1] for hdul in hduls])
    if len(nx_list) > 1 or len(ny_list) > 1:
        raise ValueError("The cubes do not have the same dimension(s) in RA and/or Dec.")

    cubes = [SpectralCube.read(h) for h in hduls]
    # check that cubes have the same spatial extent:
    # lat_list = set([cube.latitude_extrema for cube in cubes])
    # lon_list = set([cube.longitude_extrema for cube in cubes])
    # if len(lat_list) > 1 or len(lon_list) > 1:
    #     raise ValueError("The cubes do not have the same extent in RA and/or Dec.")
    diff_lat = [cubes[0].latitude_extrema != c.latitude_extrema for c in cubes[1:]]
    diff_lat = [val.all() for val in diff_lat]
    diff_lon = [cubes[0].longitude_extrema != c.longitude_extrema for c in cubes[1:]]
    diff_lon = [val.all() for val in diff_lon]
    if any(diff_lat + diff_lon):
        files_diff_lat = np.array(file_list[1:])[np.array(diff_lat)]
        files_diff_lon = np.array(file_list[1:])[np.array(diff_lon)]
        # files_diff_lon = file_list[diff_lon]
        files_diff = list(set(files_diff_lat + files_diff_lon))
        message = (f"The following files have RA and/or Dec extent different from {file_list[0]}:\n"
                   f"    {', '.join(files_diff)}")
        raise ValueError(message)

    if check_spatial_shape is not None:
        if (next(iter(nx_list)), next(iter(ny_list))) != check_spatial_shape:
            message = ["RA/Dec dimensions are not the same: "]
            for hdul, file in zip(hduls, file_list):
                message.append(f"{file}: ({hdul[0].shape[-2]}, {hdul[0].shape[-1]})")
            message.append(f"compared to {check_spatial_shape}")
            LOGGER.error("\n    ".join(message))
            raise ValueError("The cubes do not have the same dimension(s) in RA and/or Dec.")

    return cubes


def reduce_wcs_dim(wcs):
    dim = wcs.naxis  # len(wcs.array_shape)
    while dim > 2:
        wcs = wcs.dropaxis(-1)
        dim = wcs.naxis  # len(wcs.array_shape)
    return wcs


def read_crtf(file, use_region=None):
    """
    Read a CRTF region file.

    :param file: a CRTF file
    :param use_region: if more than one region in the file, specify which region to use (starting at 0)
    :return: the region
    """
    try:
        with open(file) as f:
            regions_str = "".join(f.readlines())
            regions_str = regions_str.replace("polyline", "poly")
        regs = Regions.parse(regions_str, format='crtf')
        # regs = Regions.read(file, format='crtf')
        if use_region is None:
            use_region = len(regs) - 1
            if len(regs) > 1:
                LOGGER.warning(f'More than one regions were found in {file}: using the last one by default.\n    '
                      'If this is not what you want, please edit your CRTF file.')
        return regs[use_region]
    except TypeError:
        raise TypeError(f'{file} is not a CRTF file.')


def get_single_mask(wcs: WCS, file, exclude=False):
    wcs_image = reduce_wcs_dim(wcs)
    nx = wcs_image.array_shape[0]
    ny = wcs_image.array_shape[1]
    mask = np.full((nx, ny), True)
    try:
        reg = read_crtf(file)
        reg_pix = reg.to_pixel(wcs_image)
        reg_mask = reg_pix.to_mask(mode='center')
        im_mask = reg_mask.to_image((nx, ny))
        mask = im_mask.astype(bool)
        if exclude:
            mask = np.logical_not(mask)
    except TypeError:
        LOGGER.warning(f'Invalid region or region file for {file}. No masking.')
    return mask


def get_mask(wcs: WCS, file, exclude=False):
    if not isinstance(file, list):
        return get_single_mask(wcs, file, exclude=exclude)

    if len(file) == 1:
        return get_single_mask(wcs, file[0], exclude=exclude)

    mask_out = get_single_mask(wcs, file[0], exclude=False)
    mask_in = get_single_mask(wcs, file[1], exclude=True)
    return mask_out & mask_in
    # mask = get_single_mask(wcs, file[0], exclude=False)
    # for f in file[1:]:
    #     mask &= get_single_mask(wcs, f, exclude=exclude)
    # return mask


def get_valid_pixels(wcs: WCS, file, file2=None, masked=False, snr=5., mask_operation='or'):
    """
    Obtain a list of valid or of masked pixels

    :param wcs: the WCS of the data (RA, Dec only)
    :param file: can be :
        an image of signal-to-noise ratios : valid pixels are pixel with signal-to-noise ratio greater than snr
        a CRTF file
    :param file2: a CRTF file
    :param masked: set to True if want list of masked pixels
    :param snr:
    :param mask_operation: if two CRTF files, can either be 'or'
        (e.g., to keep pixels within an annulus made from two ellipses) or 'and'
    :return: list of valid or of masked pixels
    """
    wcs_image = reduce_wcs_dim(wcs)
    nx = wcs_image.array_shape[0]
    ny = wcs_image.array_shape[1]

    extension = os.path.splitext(file)[1]
    with open(file) as f:
        line = f.readline()
    if line.startswith('#CRTF'):
        mask_region = Regions.read(file, format='crtf')[0]
        if file2 is not None:
            region2 = Regions.read(file2, format='crtf')[0]
            if mask_operation == 'or':
                mask_region = mask_region ^ region2
            elif mask_operation == 'and':
                mask_region = mask_region & region2
            else:
                raise ValueError("Invalid mask_operation.")

        mask_pix = mask_region.to_pixel(wcs_image)

        mask = np.full(wcs_image.array_shape, True)
        if masked:
            return [(i, j) for i in range(nx) for j in range(ny) if PixCoord(j, i) not in mask_pix]
        else:
            return [(i, j) for i in range(nx) for j in range(ny) if PixCoord(j, i) in mask_pix]

    elif extension == '.fits':
        hdu = fits.open(file)[0]
        if masked:
            return [(i, j) for i in range(nx) for j in range(ny) if hdu.data[j, i] < snr]
        else:
            return [(i, j) for i in range(nx) for j in range(ny) if hdu.data[j, i] >= snr]
    else:
        raise ValueError("Unknown extension.")


def pixels_snake_loop(xref, yref, xmax, ymax, xmin=0, ymin=0, step=1):
    """
    Compute the list of pixels that scans the array in the following way :
     - start on (xref, yref) then go right, up, left, up, right... -> computes top half of the map
     - go back to (xref, yref) then go left, down, right, down, left... -> compute bottom half of the map

     NB : the starting position is included twice, at the beginning and when changing direction (up/down)
     for easier detection of this change in direction.

    :param xref: starting x-position
    :param yref: starting y-position
    :param xmax: the upper limit in x-direction (included)
    :param ymax: the upper limit in y-direction (included)
    :param xmin: the lower limit in x-direction
    :param ymin: the lower limit in y-direction
    :param step: the step size in pixels
    :return: list of (x,y)
    """
    def snake(pix_list, direction):
        full_line = list(range(xref, xmin - 1, -step))
        full_line = full_line[1:] + list(range(xref, xmax + 1, step))
        full_line.sort()
        if direction == 'up':
            increment = +1 * step
            right2left = True  # start by going right to left
        else:
            increment = -1 * step
            right2left = False  # start by going left to right
        y = yref + increment
        while (yref < y <= ymax) or (yref > y >= ymin):
            full_line.sort(reverse=right2left)
            pix_list.extend([(x, y) for x in full_line])
            right2left = not right2left  # reverse direction
            y = y + increment
        return pix_list

    # xmax, ymax = xmax+1, ymax+1
    pix_list = [(x, yref) for x in range(xref, xmax + 1, step)]  # complete the line at yref, going right
    pix_list = snake(pix_list, direction='up')
    pix_list.extend([(x, yref) for x in range(xref, xmin - 1, -1*step)])  # complete the line at yref, going left
    pix_list = snake(pix_list, direction='down')

    return pix_list


def make_map_image(hdu, ntot_scaling='sqrt'):
    """
    Make an image from the given hdu(s).

    :param hdu: a single hdu or a list of hdus or a dictionary of hdus
    :param ntot_scaling:
    :param fig_ax: a (fig, ax) tuple containing matplotlib figure axes
    :return:
    """

    if not isinstance(hdu, (dict, list)):
        hdu = [hdu]

    if isinstance(hdu, list):
        if isinstance(hdu[0], list):  # we have a list of lists -> multi rows
            hdu_dic = {i: hd for i, hd in enumerate(hdu)}
        else:  # simple list -> one row
            hdu_dic = {0: hdu}
    elif isinstance(hdu, dict):
        hdu_dic = hdu
    else:
        raise TypeError('hdu should be a single hdu, a dictionary or a list of hdus')

    nrows = len(hdu_dic)
    ncols = max([len(row) for row in hdu_dic.values()])

    fig_size = (fig_size_single[0] * ncols, fig_size_single[1] * nrows)

    wcs = WCS(next(iter(hdu_dic.values()))[0].header)
    fig, axes = plt.subplots(
        figsize=fig_size,
        nrows=nrows, ncols=ncols,
        subplot_kw=dict(projection=wcs)
    )

    i = 0
    # NB: loop over dictionary to make sure we deal with the axes correctly, depending on its shape
    for row, hdu_list in hdu_dic.items():
        for j in range(ncols):
            if ncols == 1:
                if nrows == 1:
                    ax = axes
                else:
                    ax = axes[i]
            elif nrows == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            if j >= len(hdu_list):
                ax.set_axis_off()
                continue

            map_hdu = hdu_list[j]
            param = map_hdu.header['TITLE']

            try:
                comp, param = param.split(" - ")
            except ValueError:
                comp = ""
                LOGGER.warning("No component info found for {}".format(param))

            if 'Ntot' in param:
                param, sp = param.split(maxsplit=1)
                sp = list(sp)
                char_groups = [sp[0]]
                is_alpha = sp[0].isalpha()
                for c, char in enumerate(sp[1:]):
                    types = [char.isalpha(), is_alpha]
                    if all(types) or not any(types):
                        char_groups[-1] += char
                    else:
                        char_groups.append(char)
                    is_alpha = char.isalpha()
                sp_latex = ""
                for elt in char_groups:
                    if elt.isalpha():
                        sp_latex += elt
                    else:
                        if int(elt) <= 6:
                            sp_latex += f'$_{elt}$'
                        else:
                            sp_latex += f'$^{{{elt}}}$'
                title = "$N_{tot}$" + f"({sp_latex})" + " [cm$^{-2}$]"
                scaling = colors.PowerNorm(gamma=0.5)
                if ntot_scaling == 'sqrt':
                    pass
                elif ntot_scaling == 'log':
                    scaling = colors.LogNorm()
                else:
                    LOGGER.warning("Unknown ntot_scaling option, assuming sqrt.")
            else:
                param = param.split()[-1]  # update param so that "Source size" becomes "size"
                title = labels[param] + " " + units[param]
                scaling = colors.Normalize()

            title = f"{comp} - " + title

            pcm = ax.imshow(map_hdu.data, origin='lower', cmap=color_maps[param], norm=scaling)
            cax = ax.inset_axes([1, 0, 0.05, 1])
            fig.colorbar(pcm, cax=cax)

            ax.set_title(title)

            if i == nrows - 1:
                ax.set_xlabel("Right Ascension")
            else:
                ax.set_xlabel("")
                ax.tick_params(axis='x', labelbottom=False)

            if j == 0:
                ax.set_ylabel("Declination")
                # ax.text(-0.3, 0.5, f'Component {cpt[-1]}', transform=ax.transAxes,
                #         horizontalalignment='center',
                #         verticalalignment='center',
                #         rotation=90, fontsize=1.5*fontsize
                #         )
            else:
                ax.set_ylabel("")
                ax.tick_params(axis='y', labelleft=False)

        i += 1

    bb = [0, 0, 1.0, 1.0]
    if ncols == 1:
        bb = [0, 0, 0.95, 1.0]
    plt.tight_layout(rect=bb)

    return fig, axes


def save_all_map_images(map_dir, ntot_scaling='sqrt'):
    """
    For all files ending in .fits but not in _err.fits in the given directory, save the corresponding png image.

    :param map_dir: directory where the fits files are located
    :param ntot_scaling: type of scaling for the column density (sqrt or log)
    :return:
    """
    plt.close('all')
    map_files = [f for f in os.listdir(map_dir) if f.endswith('fits') and 'err' not in f]
    for map_file in map_files:
        fig, ax = make_map_image(os.path.join(map_dir, map_file), ntot_scaling=ntot_scaling)
        plt.tight_layout()
        fig.savefig(os.path.join(map_dir, f'{os.path.splitext(map_file)[0]}.png'))


def save_all_map_images_one_file(map_dir, ntot_scaling='sqrt'):
    plt.close('all')
    map_dir = os.path.abspath(map_dir)
    map_files = [f for f in os.listdir(map_dir) if f.endswith('fits') and 'err' not in f]
    output_dir = map_dir

    components = [os.path.split(map_file)[-1].split('_')[0] for map_file in map_files]
    components = list(set(components))
    components.sort()
    map_hdus_by_cpt = {}
    for cpt in components:
        hdu_list = []
        par_list = ['ntot', 'tex', 'fwhm', 'size','vlsr']
        for par in par_list:
            sub_list = [map_file for map_file in map_files if cpt in map_file and par in map_file]
            sub_list.sort()
            hdu_list.extend([fits.open(os.path.join(map_dir, fits_file))[0] for fits_file in sub_list])
        map_hdus_by_cpt[cpt] = hdu_list

    fig, axes = make_map_image(map_hdus_by_cpt, ntot_scaling=ntot_scaling)

    fig.savefig(os.path.join(output_dir, f'c_maps_{os.path.normpath(output_dir).split(os.path.sep)[-1]}.png'))
    # fig.savefig(os.path.join(output_dir, f'c_maps_{"_".join(species)}.png'))
