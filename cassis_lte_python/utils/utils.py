from cassis_lte_python.utils.constants import C_LIGHT, K_B, H

from numpy import exp, genfromtxt, array, log, log10, abs, linspace, append, interp, sqrt, pi, empty, where, ndarray
import os
import astropy.io.fits as fits
from astropy import units as u
import pandas as pd
from regions import Regions, PixCoord, EllipseSkyRegion
from astropy.wcs import WCS


def open_data_file(filepath):
    vlsr = 0.
    ext = os.path.splitext(filepath)[-1]
    if ext == '.fits':
        with fits.open(filepath) as hdu:
            data = hdu[1].data
            try:
                vlsr = hdu[1].header['VLSR']  # km/s
            except KeyError:
                vlsr = 0.
            if isinstance(hdu[1], fits.BinTableHDU):
                xunit = hdu[1].columns.columns[0].unit
                freq = data[hdu[1].columns.columns[0].name].byteswap().newbyteorder()  # to be able to use in a DataFrame
                flux = data[hdu[1].columns.columns[1].name].byteswap().newbyteorder()
                freq = (freq * u.Unit(xunit)).to('MHz').value
            else:
                raise TypeError("Cannot open this fits file.")
    elif ext in ['.fus', '.lis']:
        with open(filepath) as f:
            nskip = 0
            line = f.readline()
            while '//' in line:
                nskip += 1
                if 'vlsr' in line:
                    vlsr = float(line.split()[-1])
                line = f.readline()

        data = genfromtxt(filepath, skip_header=nskip, names=True)
        freq = data['FreqLsb']
        try:
            flux = data['Intensity']
        except ValueError:
            flux = data['Int']

    else:
        # print('Unknown extension.')
        # return None
        freq, flux = [], []
        with open(filepath) as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip()) == 0:
                    continue
                if line[0] == '#':
                    continue
                elts = line.split()
                if elts[1] not in ['NaN', 'nan']:
                    freq.append(float(elts[0]))
                    flux.append(float(elts[1]))
        freq = array(freq) * 1000.
        flux = array(flux)

    inds = freq.argsort()
    flux = flux[inds]
    freq = freq[inds]

    return freq, flux, vlsr


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

    power = log10(abs(value)) if value != 0 else 0.
    rpst = "e" if (power < -2 or power > nb_digits) else "f"
    f = "{:." + str(nb_signif_digits) + rpst + "}"
    return f.format(value)


def select_from_ranges(x_values, ranges, y_values=None, oversampling=None, extend=False):
    if type(ranges[0]) is not list:
        ranges = [ranges]

    x_new = []
    y_new = []

    for x_range in ranges:
        imin = find_nearest_id(x_values, min(x_range))
        imax = find_nearest_id(x_values, max(x_range))
        if extend:
            if x_values[imin] > min(x_range) and imin > 0:
                imin = imin - 1
            if x_values[imax] < max(x_range) and imax < len(x_values):
                imax = imax + 1
        x_sub = x_values[imin:imax+1]
        if len(x_sub) == 0:
            continue
        xmin, xmax = min(x_sub), max(x_sub)
        if oversampling is not None:
            x_sub = linspace(xmin, xmax, num=len(x_sub)*oversampling, endpoint=True)
        x_new = append(x_new, x_sub)
        if y_values is not None:
            y_sub = y_values[(x_values >= xmin) & (x_values <= xmax)]
            y_new = append(y_new, y_sub)

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


def find_nearest_id(arr: ndarray | list, value):
    """
    Find the index of the value in "arr" that is closest to "value".
    :param arr:
    :param value:
    :return:
    """
    if isinstance(arr, list):
        arr = array(arr)

    return (abs(arr - value)).argmin()


def find_nearest_trans(trans_list, value):
    f_trans_list = [tr.f_trans_mhz for tr in trans_list]
    idx = (abs(array(f_trans_list) - value)).argmin()
    return trans_list[idx]


def is_in_range(fmhz, list_ranges):
    res = False
    for rg in list_ranges:
        if rg[0] <= fmhz <= rg[1]:
            res = True
    return res


def expand_dict(dic: dict, n_items):
    if '*' in dic:
        new_dic = {i + 1: dic['*'] for i in range(n_items)}
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
            arg = array(arg)
    cal = cal / 100.
    return 1. / sqrt(rms**2 + (cal * intensity)**2)


def fwhm_to_sigma(value, reverse=False):
    if reverse:
        return value * (2. * sqrt(2. * log(2.)))
    else:
        return value / (2. * sqrt(2. * log(2.)))


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
    fmhz_arr = array(fmhz) if type(fmhz) == list else fmhz
    res = (H * fmhz_arr * 1.e6 / K_B) / \
          (exp(H * fmhz_arr * 1.e6 / (K_B * temp)) - 1.)
    return list(res) if type(fmhz) == list else res


def get_telescope(fmhz, tuning_info: pd.DataFrame):
    if isinstance(fmhz, float):
        fmhz = array(fmhz)

    if len(tuning_info) == 1:
        return [tuning_info['telescope'][0]] * len(fmhz)

    tel_list = empty(len(fmhz), dtype=object)
    for i, row in tuning_info.iterrows():
        tel_list[where((fmhz >= row['fmhz_min']) & (fmhz <= row['fmhz_max']))] = row['telescope']

    return tel_list


def get_tmb2ta_factor(fmhz, tel_data):
    if isinstance(tel_data, pd.DataFrame):
        f_tel = tel_data[tel_data.columns[0]]
        beff_tel = tel_data[tel_data.columns[1]]
        return interp(fmhz, f_tel, beff_tel)
    else:
        raise TypeError("Not implemented yet.")


def get_beam_size(freq_mhz, tel_diam):
    """
    Computes the beam size at the given frequency
    :param freq_mhz: frequency in MHz ; float or numpy array
    :param tel_info: telescope name or dataframe containing "tuning" information
    :return: the beam size in arcsec
    """

    return (1.22 * C_LIGHT / (freq_mhz * 1.e6)) / tel_diam * 3600. * 180. / pi


def dilution_factor(source_size, beam_size, geometry='gaussian'):
    # dilution_factor = tr.mol_size ** 2 / (tr.mol_size**2 + get_beam_size(model.telescope,freq)**2)
    # dilution_factor = (1. - np.cos(cpt.size/3600./180.*np.pi)) / ( (1. - np.cos(cpt.size/3600./180.*np.pi))
    #                    + (1. - np.cos(get_beam_size(self.telescope,self.frequencies)/3600./180.*np.pi)) )
    if geometry == 'disc':
        return 1. - exp(-log(2.) * (source_size / beam_size) ** 2)
    else:
        return source_size ** 2 / (source_size ** 2 + beam_size ** 2)


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
