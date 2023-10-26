from cassis_lte_python.utils.constants import C_LIGHT, K_B, H, UNITS
from cassis_lte_python.utils.settings import TELESCOPE_DIR
from cassis_lte_python.database.species import get_partition_function

import numpy as np
import os
import astropy.io.fits as fits
from astropy import units as u
import pandas as pd
from regions import Regions, PixCoord, EllipseSkyRegion
from astropy.wcs import WCS


class DataFile:
    def __init__(self, filepath, telescope=None):
        self.filepath = filepath
        self._ext = os.path.splitext(filepath)[-1]
        self._raw_header = None
        self.header = None
        self._xdata_mhz = None
        self._xdata = None
        self._ydata = None
        self._xunit = None
        self._yunit = None
        self.vlsr = 0.
        self.bmaj = None
        self.bmin = None
        self._telescope = telescope

        self.open_data_file()  # read header, x and y data, and set the units
        self.get_vlsr()
        if self.xunit is None:
            print(f"X-axis unit not found, setting to MHz.")
            self._xunit = 'MHz'

        if self.xunit != 'MHz':
            self._xdata_mhz = (self.xdata * u.Unit(self.xunit)).to('MHz').value
        else:
            self._xdata_mhz = self.xdata

        if self.yunit in UNITS['flux']:
            self.read_beam()

    def open_data_file(self):
        if self._ext == '.fits':
            self.read_fits()

        elif self._ext in ['.fus', '.lis']:
            self.read_cassis()

        else:
            # print('Unknown extension.')
            # return None
            self.get_header()
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
                self._xdata = data[hdu[1].columns.columns[0].name].byteswap().newbyteorder()
                self._ydata = data[hdu[1].columns.columns[1].name].byteswap().newbyteorder()
            else:
                raise TypeError("Cannot open this fits file.")

    def read_cassis(self):
        self.get_header(comment='//')

        data = np.genfromtxt(self.filepath, skip_header=len(self._raw_header), names=True)
        self._xdata = data['FreqLsb']
        try:
            self._ydata = data['Intensity']
        except ValueError:
            self._ydata = data['Int']

    def write(self, fileout, sep='/t'):
        if self._ext == '.txt':
            pass

    def get_header(self, comment='#'):
        with open(self.filepath) as f:
            raw_header = []
            header = {}
            line = f.readline()
            while line.startswith(comment) or len(line.strip()) == 0:
                raw_header.append(line)
                if ":" in line and 'Point' not in line:
                    info = line.lstrip(comment).split(":", maxsplit=1)
                    header[info[0].strip()] = info[1].strip()
                line = f.readline()

        self._raw_header = raw_header
        self.header = header

    def get_vlsr(self):
        try:
            self.vlsr = float(self.header.get('vlsr'))
        except TypeError:
            try:
                self.vlsr = self.header['VLSR']  # km/s
            except KeyError:
                print(f"Vlsr not found, using {self.vlsr} km/s")

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
                bmaj_unit = self.header['CUNIT1']
                bmin_unit = self.header['CUNIT2']
                self.bmaj = (self.header['BMAJ'] * u.Unit(bmaj_unit)).to('arcsec').value
                self.bmin = (self.header['BMIN'] * u.Unit(bmin_unit)).to('arcsec').value
            except KeyError:
                pass

    def beam(self):
        return self.bmaj, self.bmin

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
            # set new unit
            self._xunit = value

    @property
    def yunit(self):
        return self._yunit

    @yunit.setter
    def yunit(self, value: str):
        if value != self._yunit:
            if value in UNITS['flux']:
                bmaj, bmin = None, None
                if self._telescope is not None:
                    try:
                        read_telescope_file(os.path.join(TELESCOPE_DIR, self._telescope))
                    except FileNotFoundError:
                        try:
                            read_telescope_file(self._telescope)
                        except FileNotFoundError:
                            raise FileNotFoundError(f"None of the following telescope files were found : "
                                                    f"./{self._telescope}, {os.path.join(TELESCOPE_DIR, self._telescope)}")

                elif self.bmaj is not None and self.bmin is not None:
                    pass
                else:
                    raise ValueError("No beam information found - cannot perform conversion.")
                # convert y-axis from jy/beam to K
                self._ydata = self.ydata * compute_jypb2k(self._xdata_mhz, (bmaj, bmin))
                # set new unit
                self._yunit = value
            else:
                print(f"Cannot convert to {value}, nothing done.")


def open_data_file(filepath):
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
            if f'({unit})' in s or f'[{unit}]' in s:
                return unit


def format_float(value, fmt=None, nb_digits=6, nb_signif_digits=3):
    """

    :param value:
    :param fmt: the format to use, e.g., "{:.3e}"
    :param nb_digits:
    :param nb_signif_digits:
    :return:
    """
    if fmt:
        return fmt.format(value)

    power = np.log10(abs(value)) if value != 0 else 0.
    rpst = "e" if (power < -2 or power > nb_digits) else "f"
    f = "{:." + str(nb_signif_digits) + rpst + "}"
    return f.format(value)


def select_from_ranges(x_values, ranges, y_values=None, oversampling=None):
    if type(ranges[0]) is not list:
        ranges = [ranges]

    x_new = []
    y_new = []

    for x_range in ranges:
        imin = find_nearest_id(x_values, min(x_range))
        imax = find_nearest_id(x_values, max(x_range))
        x_sub = x_values[imin:imax+1]
        if len(x_sub) == 0:
            continue
        if oversampling is not None:
            xmin, xmax = min(x_sub), max(x_sub)
            x_sub = np.linspace(xmin, xmax, num=len(x_sub)*oversampling, endpoint=True)
        x_new = np.append(x_new, x_sub)
        if y_values is not None:
            y_sub = y_values[imin:imax+1]
            y_new = np.append(y_new, y_sub)

    return x_new, y_new if y_values is not None else x_new


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


def expand_dict(dic: dict, n_items=None):
    if '*' in dic:
        if n_items is not None:
            new_dic = {i + 1: dic['*'] for i in range(n_items)}
        else:
            # print("Missing number of transitions.")
            return dic
    else:
        new_dic = {}
        for k, v in dic.items():
            for e in k.split(','):
                if '-' in e:
                    nmin, nmax = e.split('-')
                    for i in range(int(nmin), int(nmax) + 1):
                        new_dic[i] = v
                else:
                    new_dic[int(e)] = v

    return new_dic


def get_extended_limits(values, padding=0.05):
    dx = max(values) - min(values)
    return [min(values) - padding * dx, max(values) + padding * dx]


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


def read_telescope_file(telescope_file):
    with open(telescope_file, 'r') as f:
        col_names = ['Frequency (MHz)', 'Beff/Feff']
        tel_data = f.readlines()
        tel_diam = int(tel_data[1])
        if tel_diam == 0:
            col_names += ['Bmaj (arcsec)', 'Bmin (arcsec)']
        # get column names:
        line = tel_data[2]
        col_names = line.replace('.', ',').lstrip('//').split(',')
        col_names = [c.strip() for c in col_names]

        tel_info = pd.read_csv(telescope_file, sep='\t', skiprows=3,
                               names=col_names, usecols=list(range(len(col_names))))
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
                     geometry: str = 'gaussian') -> float | np.ndarray:
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


def dilution_factor(source_size: float | int, beam: tuple, geometry: str = 'gaussian') -> float:
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

    beam_size_sq = beam[0] ** 2 + beam[1] ** 2

    if geometry == 'disc':
        return 1. - np.exp(-np.log(2.) * (source_size ** 2 / beam_size_sq))

    if geometry == 'gaussian':
        return source_size ** 2 / (source_size ** 2 + beam_size_sq)

    raise TypeError(f"Unsupported geometry, can only be {', '.join(geometries[:-1])} or {geometries[-1]}.")


def compute_jypb2k(freq_mhz: float | int | list | np.ndarray,
                   beam_arcsec: tuple | list | np.ndarray) -> float | np.ndarray:
    """
    Compute the conversion factor from Jansky per beam to Kelvin.
    T = (conv_fact / nu^2) * I , with :
    conv_fact = c^2 / (2*k_B*omega) * 1.e-26 (to convert Jy to mks)
    omega = pi*bmaj*bmin/(4*ln2)

    :param freq_mhz:
    :param beam_arcsec:
    :return:
    """
    if isinstance(freq_mhz, list):
        freq_mhz = np.array(freq_mhz)

    if isinstance(beam_arcsec, list):
        beam_arcsec = np.array(beam_arcsec)

    try:
        bmaj_arcsec, bmin_arcsec = beam_arcsec[:, 0], beam_arcsec[:, 1]
    except (IndexError, TypeError):
        bmaj_arcsec, bmin_arcsec = beam_arcsec[0], beam_arcsec[1]
    omega = (bmaj_arcsec * u.Unit('arcsec')).to('rad').value * (bmin_arcsec * u.Unit('arcsec')).to('rad').value
    omega *= np.pi / (4 * np.log(2))
    conv_fact = C_LIGHT ** 2 / (2 * K_B * omega) * 1.e-26
    return conv_fact / (freq_mhz * 1.e6)**2


def reduce_wcs_dim(wcs):
    dim = len(wcs.array_shape)
    while dim > 2:
        wcs = wcs.dropaxis(2)
        dim = len(wcs.array_shape)
    return wcs


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
    if extension == '.crtf':
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
