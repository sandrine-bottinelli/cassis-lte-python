import numpy as np
import sqlite3
import astropy.constants.codata2014 as const  # h,k_B,c # SI units
# from astropy.modeling import Fittable1DModel, Parameter
# from astropy import units as u
from lmfit import Model, Parameters, Parameter
from scipy import stats, signal
import matplotlib.pyplot as plt
from matplotlib import ticker
import astropy.io.fits as fits
import os
import pandas as pd
import datetime
import json
import configparser
import tkinter
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('agg')

module_dir = os.path.dirname(__file__)
user_config = os.path.join(module_dir, 'config.ini')
default_config = os.path.join(module_dir, 'config_defaults.ini')
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation(),
                                   inline_comment_prefixes=('#',))
if os.path.isfile(user_config):
    config.read(user_config)
else:
    if os.path.isfile(default_config):
        print(f'{user_config} not found, using {default_config}\n')
        config.read(default_config)
    else:
        raise FileNotFoundError('No configuration file found.')

CASSIS_DIR = config.get('GENERAL', 'CASSIS_DIR')
SQLITE_FILE = config.get('DATABASE', 'SQLITE_FILE')
PARTITION_FUNCTION_DIR = config.get('DATABASE', 'PARTITION_FUNCTION_DIR')
TELESCOPE_DIR = config.get('MODEL', 'TELESCOPE_DIR')
SIZE_DEF = config.getfloat('MODEL', 'SIZE')
VLSR_DEF = config.getfloat('MODEL', 'VLSR')
FWHM_DEF = config.getfloat('MODEL', 'FWHM')
DPI_DEF = config.getint('PLOT', 'DPI')

if not os.path.isfile(SQLITE_FILE):
    parent_dir = os.path.dirname(SQLITE_FILE)
    # try to find a cassisYYYYMMDD.db file in the parent directory :
    if os.path.isdir(parent_dir):
        try:
            db_list = [f for f in os.listdir(parent_dir)
                       if (f.endswith('.db') and f.startswith('cassis'))]
            db_list.sort()
            SQLITE_FILE_NEW = os.path.join(parent_dir, db_list[-1])
            if 'YYYYMMDD' in SQLITE_FILE:
                print(f'Using sqlite file {SQLITE_FILE_NEW}.\n')
            else:
                print(f'{SQLITE_FILE} not found, using {SQLITE_FILE_NEW} instead.\n')
            SQLITE_FILE = SQLITE_FILE_NEW
            # update config :
            config['DATABASE']['SQLITE_FILE'] = SQLITE_FILE
        except IndexError:
            raise FileNotFoundError(f'No cassisYYYYMMDD.db found in {parent_dir}.')
    else:
        raise FileNotFoundError(f'{SQLITE_FILE} not found.')

if os.path.isfile(SQLITE_FILE):
    conn = sqlite3.connect(SQLITE_FILE)
else:
    raise FileNotFoundError(f'{SQLITE_FILE} not found.')

db = conn.cursor()

CPT_COLORS = ['blue', 'green', 'mediumorchid']
PLOT_COLORS = plt.get_cmap('tab20')(np.linspace(0, 1, 20))
PLOT_LINESTYLES = ['-', '--', ':']

EUP_MIN_DEF = 0.
EUP_MAX_DEF = None  # 150.
AIJ_MIN_DEF = 0.
AIJ_MAX_DEF = None
ERR_MAX_DEF = None
THRESHOLDS_DEF = {'eup_min': EUP_MIN_DEF,
                  'eup_max': EUP_MAX_DEF,
                  'aij_min': AIJ_MIN_DEF,
                  'aij_max': AIJ_MAX_DEF,
                  'err_max': ERR_MAX_DEF}

TEL_DIAM = {'iram': 30.,
            'apex': 12.,
            'jcmt': 15.,
            'gbt': 100.,
            'alma_400m': 400.,
            'alma_170m': 170.}

# Matplotlib global parameters
matplotlib.rcParams['xtick.direction'] = 'in'  # Ticks inside
matplotlib.rcParams['ytick.direction'] = 'in'  # Ticks inside
matplotlib.rcParams['ytick.right'] = True  # draw ticks on the right side
# axes.formatter.limits: -5, 6  # use scientific notation if log10
                               # of the axis range is smaller than the
                               # first or larger than the second
# axes.formatter.use_mathtext: False  # When True, use mathtext for scientific notation.
# axes.formatter.min_exponent: 0  # minimum exponent to format in scientific notation
matplotlib.rcParams['axes.formatter.useoffset'] = False  # No offset for tick labels
# axes.formatter.useoffset: True  # If True, the tick label formatter
                                 # will default to labeling ticks relative
                                 # to an offset when the data range is
                                 # small compared to the minimum absolute
                                 # value of the data.
# axes.formatter.offset_threshold: 4  # When useoffset is True, the offset
                                     # will be used when it can remove
                                     # at least this number of significant
                                     # digits from tick labels.


def print_settings():
    print("Settings are :")
    for section in config.sections():
        for key, val in dict(config.items(section)).items():
            unit = ""
            if "size" in key:
                unit = "arcsec"
            elif "vlsr" in key or "fwhm" in key:
                unit = "km/s"
            print(f"    {key.upper()} = {val} {unit}")
    print("")


def get_species_info(database, species):
    tag = species.tag if isinstance(species, Species) else int(species)
    # retrieve infos from catdir :
    res_catdir = database.execute("SELECT * FROM catdir WHERE speciesid = {}".format(tag))
    cols_sp_info = [t[0] for t in res_catdir.description]
    all_rows = database.fetchall()
    if len(all_rows) == 0:
        return None
    sp_info = all_rows[0]
    sp_info_dic = dict(zip(cols_sp_info, sp_info))

    # retrieve database name :
    database.execute("SELECT name FROM cassis_databases WHERE id = {}".format(sp_info_dic["id_database"]))
    sp_info_dic["database_name"] = database.fetchall()[0][0]

    return sp_info_dic


def get_transition_list(database, species, fmhz_ranges, return_type='dict', **thresholds):
    species_list = [species] if type(species) is not list else species  # make sure it is a list
    tag_list = [sp.tag for sp in species_list] if isinstance(species_list[0], Species) else species_list
    transition_dict = {}
    for tag in tag_list:
        if thresholds:
            eup_min = thresholds[str(tag)].get('eup_min', EUP_MIN_DEF)
            eup_max = thresholds[str(tag)].get('eup_max', EUP_MAX_DEF)
            aij_min = thresholds[str(tag)].get('aij_min', AIJ_MIN_DEF)
            aij_max = thresholds[str(tag)].get('aij_max', AIJ_MAX_DEF)
            err_max = thresholds[str(tag)].get('err_max', ERR_MAX_DEF)
        else:
            eup_min = 0.
            eup_max = None
            aij_min = 0.
            aij_max = None
            err_max = None

        transition_list = []
        # retrieve catdir_id :
        sp_dic = get_species_info(database, tag)
        if sp_dic is None:
            continue
        sp_id = sp_dic['id']
        sp_name = sp_dic['name']
        for frange in fmhz_ranges:
            fmhz_min, fmhz_max = frange
            cmd = "SELECT * FROM transitions WHERE catdir_id = {} and fMhz < {} and fMhz > {}" \
                  " and eup > {} and aint > {}".format(sp_id, fmhz_max, fmhz_min, eup_min, aij_min)
            if eup_max is not None:
                cmd += " and eup < {}".format(eup_max)
            if aij_max is not None:
                cmd += " and aint < {}".format(aij_max)
            if err_max is not None:
                cmd += " and err < {}".format(err_max)
            res = database.execute(cmd)
            col_names = [t[0] for t in res.description]
            all_rows = database.fetchall()
            if len(all_rows) == 0:
                continue
            for row in all_rows:
                row_dic = dict(zip(col_names, row))
                trans = Transition(tag, row_dic['fMHz'], row_dic['aint'], row_dic['elow'], row_dic['igu'],
                                   f_err_mhz=row_dic['err'], name=sp_name, db_id=row_dic['id_transitions'],
                                   qn=row_dic['qn'])
                transition_list.append(trans)
        if len(transition_list) > 0:
            transition_list.sort(key=lambda x: x.f_trans_mhz)
            transition_dict[str(tag)] = transition_list

    if len(transition_dict) == 0:
        raise IndexError('No transitions found. Please check your thresholds.')

    if return_type == 'dict':  # return dictionary {tag: transition list}
        return transition_dict
    elif return_type == 'df':  # return dataframe
        tmp = {'transition': [], 'tag': [], 'db_id': []}
        for tag, tran_list in transition_dict.items():
            for tr in tran_list:
                tmp['transition'].append(tr)
                tmp['tag'].append(tr.tag)
                tmp['db_id'].append(tr.db_id)
        return pd.DataFrame(tmp)
    else:  # return list
        tr_list = [item for sublist in list(transition_dict.values()) for item in sublist]
        tr_list.sort(key=lambda x: x.f_trans_mhz)
        return tr_list


def select_transitions(tran_df_or_dict, thresholds=None, xrange=None, return_type=None):
    def is_selected(tran, sp_thresholds):
        constraints = []
        for key, val in sp_thresholds.items():
            attr = key.rsplit('_', maxsplit=1)[0]
            if 'min' in key:
                constraints.append(val <= getattr(tran, attr))
            if 'max' in key and val is not None:
                constraints.append(val >= getattr(tran, attr if key != 'err_max' else 'f_err_mhz'))
        return True if all(constraints) else False

    if thresholds is None:
        thresholds = {}

    tag_list = []
    for tag in tran_df_or_dict.tag:
        if tag not in tag_list:
            tag_list.append(tag)

    if xrange is not None:
        for tag in tag_list:
            if tag in thresholds:
                thresholds[str(tag)]['f_trans_mhz_min'] = min(xrange)
                thresholds[str(tag)]['f_trans_mhz_max'] = max(xrange)
            else:
                thresholds[str(tag)] = {'f_trans_mhz_min': min(xrange),
                                        'f_trans_mhz_max': max(xrange)}

    if isinstance(tran_df_or_dict, pd.DataFrame):
        if len(thresholds) == 0:
            selected = tran_df_or_dict
        else:
            indices = []
            for tag, thres in thresholds.items():
                for row in tran_df_or_dict[tran_df_or_dict.tag == int(tag)].iterrows():
                    tr = row[1].transition
                    if is_selected(tr, thres):
                        indices.append(row[0])
            selected = tran_df_or_dict.loc[indices]

        if return_type is None or return_type == 'df':
            return selected
        elif return_type == 'dict':
            return {str(tag): list(selected[selected.tag == int(tag)].transition) for tag in tag_list}
    # elif isinstance(tran_df_or_dict, dict):
    #     pass
    else:
        raise TypeError("First argument must be a DataFrame.")
        # raise TypeError("First argument must be a DataFrame or a dictionary.")


def generate_lte_model_func(config):

    def lte_model_func(fmhz, **params):
        norm_factors = config.get('norm_factors', {key: 1. for key in params.keys()})
        tc = config['tc']
        tcmb = config['tcmb']
        vlsr_file = config['vlsr_file']
        fmhz_mod = fmhz  # config['x_mod']
        beam_sizes = config['beam_sizes']
        line_list = config['line_list']
        tau_max = config['tau_max']
        file_rejected = config['file_rejected']
        intensity_before = tc + jnu(fmhz_mod, tcmb)
        intensity = 0.
        for icpt, cpt in enumerate(config['cpt_list']):
            tex = params['{}_tex'.format(cpt.name)] * norm_factors['{}_tex'.format(cpt.name)]
            vlsr = params['{}_vlsr'.format(cpt.name)] * norm_factors['{}_vlsr'.format(cpt.name)]
            size = params['{}_size'.format(cpt.name)] * norm_factors['{}_size'.format(cpt.name)]

            sum_tau = 0
            for isp, tag in enumerate(cpt.tag_list):
                if isinstance(line_list, list):
                    tran_list = line_list
                else:  # assume it is a DataFrame
                    tran_list = list(line_list.loc[line_list['tag'] == tag].transition)
                ntot = params['{}_ntot_{}'.format(cpt.name, tag)] * norm_factors['{}_ntot_{}'.format(cpt.name, tag)]
                fwhm = params['{}_fwhm_{}'.format(cpt.name, tag)] * norm_factors['{}_fwhm_{}'.format(cpt.name, tag)]
                qtex = cpt.species_list[isp].get_partition_function(tex)
                for tran in tran_list:
                    # nup = ntot * tran.gup / qtex / np.exp(tran.eup_J / const.k_B.value / tex)  # [cm-2]
                    nup = ntot * tran.gup / qtex / np.exp(tran.eup / tex)  # [cm-2]
                    tau0 = const.c.value ** 3 * tran.aij * nup * 1.e4 \
                           * (np.exp(const.h.value * tran.f_trans_mhz * 1.e6 / const.k_B.value / tex) - 1.) \
                           / (4. * np.pi * (tran.f_trans_mhz * 1.e6) ** 3 * fwhm * 1.e3 * np.sqrt(np.pi / np.log(2.)))
                    if isinstance(tau_max, (float, int)) and tau0 >= tau_max:
                        with open(file_rejected, 'a') as f:
                            f.write('\n')
                            f.write('\t'.join([str(tag), format_float(ntot), format_float(tex), format_float(fwhm),
                                               format_float(tran.f_trans_mhz),
                                               format_float(tran.eup),
                                               format_float(tran.aij),
                                               str(tran.gup),
                                               format_float(tau0)]))
                        continue

                    num = fmhz_mod - velocity_to_frequency(vlsr, tran.f_trans_mhz, vref_kms=vlsr_file)
                    den = fwhm_to_sigma(delta_v_to_delta_f(fwhm, tran.f_trans_mhz))
                    if config['line_center_only']:
                        offset = find_nearest_id(fmhz_mod,
                                                 velocity_to_frequency(vlsr, tran.f_trans_mhz, vref_kms=vlsr_file))
                        pulse = signal.unit_impulse(len(fmhz_mod), offset)
                        sum_tau += tau0 * pulse
                    else:
                        sum_tau += tau0 * np.exp(-0.5 * (num / den) ** 2)

            ff = np.array([dilution_factor(size, bs) for bs in beam_sizes])
            if not cpt.isInteracting:
                intensity_cpt = jnu(fmhz_mod, tex) * (1. - np.exp(-sum_tau)) - \
                                intensity_before * (1. - np.exp(-sum_tau))
                intensity += ff * intensity_cpt
            else:
                intensity_before += intensity
                intensity_cpt = jnu(fmhz_mod, tex) * (1. - np.exp(-sum_tau)) - \
                                intensity_before * (1. - np.exp(-sum_tau))
                intensity = ff * intensity_cpt

        intensity = intensity + intensity_before - jnu(fmhz_mod, tcmb)
        intensity += np.random.normal(0., config['noise'], len(intensity))  # add gaussian noise

        # return np.interp(fmhz, fmhz_mod, intensity)
        return intensity

    return lte_model_func


class SimpleSpectrum:
    def __init__(self, xarray, yarray, xunit='mhz', yunit='K'):
        self.xval = xarray
        self.yval = yarray
        self.xunit = xunit
        self.yunit = yunit


class ModelSpectrum:
    def __init__(self, configuration, verbose=True, check_tel_range=False):
        config = configuration
        if not isinstance(configuration, dict):
            try:  # assume config is a path to a file
                config = self.load_config(configuration)
            except TypeError:
                print("Configuration must be a dictionary or a path to a configuration file.")

        self.output_dir = config.get('output_dir', os.path.curdir)
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        self.data_file = config.get('data_file', None)
        self.x_file = config.get('x_obs', None)
        self.y_file = config.get('y_obs', None)
        self.vlsr_file = config.get('vlsr_obs', 0.)
        if self.data_file is not None:
            self.x_file, self.y_file, self.vlsr_file = open_data_file(self.data_file)

        self.jypb = None
        if 'beam_info' in config:
            f_hz_beam = np.concatenate(config['beam_info']['f_mhz']) * 1.e6
            omega = np.concatenate(config['beam_info']['beam_omega'])
            self.jypb = 1.e-26 * const.c.value ** 2 / (f_hz_beam * f_hz_beam) / (2. * const.k_B.value * omega)

        self.cont_info = config.get('tc', 0.)

        if isinstance(self.cont_info, (float, int)):
            self.tc = self.cont_info
        elif os.path.isfile(self.cont_info):  # cont_info is a CASSIS continuum file : MHz [tab] K
            f_cont, t_cont = np.loadtxt(self.cont_info, delimiter='\t', unpack=True)
            self.tc = np.array([np.interp(x, f_cont, t_cont) for x in self.x_file])
        elif isinstance(self.cont_info, dict):  # to compute continuum over ranges given by the user
            tc = []
            for freqs, fluxes, franges in zip(self.x_file, self.y_file, self.cont_info.values()):
                cont_data = []
                for frange in franges:
                    cont_data.append(fluxes[(freqs >= min(frange)) & (freqs <= max(frange))])
                tc.append([np.mean(cont_data) for _ in freqs])
            self.tc = np.concatenate(tc)
        else:
            raise TypeError("Continuum must be a float, an integer or a dictionary.")
        # else:  # HDU
        #     pass
            # tc = cont_info[datafile].data.squeeze()[j, i]

        if self.x_file is not None and isinstance(self.x_file[0], np.ndarray):
            self.x_file = np.concatenate(self.x_file)
        if self.y_file is not None and isinstance(self.y_file[0], np.ndarray):
            self.y_file = np.concatenate(self.y_file)

        if self.jypb is not None:
            self.y_file *= self.jypb
            self.tc *= self.jypb
        if not isinstance(self.tc, (float, int)):
            self.tc = np.array([(f, t) for f, t in zip(self.x_file, self.tc)],
                               dtype=[('f_mhz', np.float32), ('tc', np.float32)])

        fmin_ghz = 115.
        fmax_ghz = 116.
        dfmhz = 0.1
        if 'fmin_ghz' in config and config['fmin_ghz'] is not None:
            fmin_ghz = config['fmin_ghz']
        if 'fmax_ghz' in config and config['fmax_ghz'] is not None:
            fmax_ghz = config['fmax_ghz']
        self.fmin_mhz = fmin_ghz * 1.e3
        self.fmax_mhz = fmax_ghz * 1.e3
        if 'df_mhz' in config and config['df_mhz'] is not None:
            dfmhz = config['df_mhz']
        self.dfmhz = dfmhz
        self.noise = 1.e-3 * config.get('noise_mk', 0.)
        if self.data_file is None:
            self.x_mod = np.linspace(self.fmin_mhz, self.fmax_mhz,
                                     num=int((self.fmax_mhz - self.fmin_mhz) / self.dfmhz) + 1)
        else:
            self.x_mod = None

        self._tuning_info_user = config.get('tuning_info', None)
        self.tuning_info = None
        if 'tuning_info' in config:
            # if telescope is not in TEL_DIAM, try to find it in TELESCOPE_DIR
            for tel in config['tuning_info'].keys():
                if tel not in TEL_DIAM.keys():
                    with open(os.path.join(TELESCOPE_DIR, tel), 'r') as f:
                        for line in f.readlines():
                            if 'Diameter' in line:
                                continue
                            TEL_DIAM[tel] = float(line)
                            break
            if check_tel_range :  # check telescope ranges cover all data / all model values:
                x_vals = self.x_file if self.x_file is not None else [min(self.x_mod), max(self.x_mod)]
                extended = False
                for x in x_vals:
                    is_in_range = [val[0] <= x <= val[1] for val in config['tuning_info'].values()]
                    limits = list(config['tuning_info'].values())
                    limits = [item for sublist in limits for item in sublist]
                    if not any(is_in_range):
                        # raise LookupError("Telescope ranges do not cover some of the data, e.g. at {} MHz.". format(x))
                        extended = True
                        nearest = find_nearest(np.array(limits), x)
                        for key, val in config['tuning_info'].items():
                            # new_lo = 5. * np.floor(x / 5.) if val[0] == nearest else val[0]
                            # new_hi = 5. * np.ceil(x / 5.) if val[1] == nearest else val[1]
                            new_lo = np.floor(x) if val[0] == nearest else val[0]
                            new_hi = np.ceil(x) if val[1] == nearest else val[1]
                            config['tuning_info'][key] = [new_lo, new_hi]
                if extended:
                    print("Some telescope ranges did not cover some of the data ; ranges were extended to :")
                    print(config['tuning_info'])

            tuning_info = {'fmhz_range': [], 'telescope': []}
            for key, val in config['tuning_info'].items():
                arr = np.array(val)
                if len(arr.shape) == 1:
                    arr = np.array([val])
                for r in arr:
                    r_min, r_max = min(r), max(r)
                    if self.x_file is not None:
                        x_sub = self.x_file[(self.x_file >= r_min) & (self.x_file <= r_max)]
                        r_min = max(r_min, min(x_sub))
                        r_max = min(r_max, max(x_sub))
                        if len(self.x_file[(self.x_file >= r_min) & (self.x_file <= r_max)]) == 0:
                            continue
                    # if self.fmax_mhz is not None and self.fmin_mhz is not None:
                    else:
                        r_min = max(r_min, self.fmin_mhz)
                        r_max = min(r_max, self.fmax_mhz)
                    tuning_info['fmhz_range'].append([r_min, r_max])
                    tuning_info['telescope'].append(key)
            self.tuning_info = pd.DataFrame(tuning_info)

        self.bandwidth = config.get('bandwidth', None)  # km/s ; None for 1 window with entire spectrum

        self.oversampling = int(config.get('oversampling', 3))

        self.tcmb = config.get('tcmb', 2.73)

        self.tau_max = config.get('tau_max', None)
        self.file_rejected = None
        if self.tau_max is not None:
            self.file_rejected = config.get('file_rejected', 'rejected_lines.txt')
        if self.file_rejected is not None:
            self.file_rejected = os.path.join(self.output_dir, self.file_rejected)
            with open(self.file_rejected, 'w') as f:
                f.writelines(['# Rejected lines with tau >= {}\n'.format(self.tau_max),
                              '\t'.join(['# Tag ', 'Ntot', 'Tex', 'FWHM', 'f_mhz', 'Eup', 'Aij', 'gup', 'tau'])
                              ])

        self.tag_list = []
        self.cpt_list = []
        for key, cpt_dic in config.get('components').items():
            sp_list = cpt_dic['species']
            cpt = Component(key, sp_list, isInteracting=cpt_dic.get('interacting', False),
                            vlsr=cpt_dic.get('vlsr'), tex=cpt_dic.get('tex'), size=cpt_dic.get('size'))
            self.cpt_list.append(cpt)
            for sp in cpt.species_list:
                if sp.tag not in self.tag_list:
                    self.tag_list.append(sp.tag)

        self.thresholds = {}
        if 'thresholds' in config:
            for tag in self.tag_list:
                user_dict = config['thresholds'].get(str(tag), THRESHOLDS_DEF)
                self.thresholds[str(tag)] = {label: user_dict.get(label, value)
                                             for label, value in THRESHOLDS_DEF.items()}
        else:
            for tag in self.tag_list:
                self.thresholds[str(tag)] = THRESHOLDS_DEF

        self.line_list_all = get_transition_list(db, self.tag_list, self.tuning_info['fmhz_range'],
                                                 return_type='df')
        tr_list_by_tag = select_transitions(self.line_list_all, self.thresholds, return_type='dict')
        if all([len(t_list) == 0 for t_list in tr_list_by_tag.values()]):
            raise LookupError("No transition found within the thresholds.")

        self._v_range_user = config.get('v_range', None)
        self._rms_cal_user = config.get('chi2_info', None)
        if len(self.tag_list) == 1 and self._v_range_user is not None and self._rms_cal_user is not None:
            if str(self.tag_list[0]) not in self._v_range_user:
                self._v_range_user = {str(self.tag_list[0]): self._v_range_user}
            if str(self.tag_list[0]) not in self._rms_cal_user:
                self._rms_cal_user = {str(self.tag_list[0]): self._rms_cal_user}

        if self.bandwidth is None:
            self.win_list = [Window(tr_list_by_tag[str(self.tag_list[0])][0], 1)]
        else:
            self.win_list = []
            for tag, tr_list in tr_list_by_tag.items():
                win_list_tag = []  # first find all windows with enough data
                for i, tr in enumerate(tr_list):
                    f_range_plot = [velocity_to_frequency(v, tr.f_trans_mhz, vref_kms=self.vlsr_file)
                                    for v in [-1.1 * self.bandwidth / 2 + self.vlsr_file,
                                              1.1 * self.bandwidth / 2 + self.vlsr_file]]
                    f_range_plot.sort()
                    if self.x_file is not None:
                        x_win, y_win = select_from_ranges(self.x_file, f_range_plot, y_values=self.y_file)
                        if len(x_win) == 0 or len(set(y_win)) == 1:
                            continue
                    win = Window(tr, len(win_list_tag) + 1)
                    win_list_tag.append(win)

                nt = len(win_list_tag)
                if verbose:
                    print('{} : {}/{} transitions found with enough data within thresholds'.format(tag, nt,
                                                                                                   len(tr_list)))

                if self._v_range_user is not None and self._rms_cal_user is not None:
                    v_range = expand_dict(self._v_range_user[tag], nt)
                    rms_cal = expand_dict(self._rms_cal_user[tag], nt)
                    for win in win_list_tag:
                        win_num = win.plot_nb
                        if win_num in v_range and win_num in rms_cal:
                            win.v_range_fit = v_range[win_num]
                            win.rms_mk = rms_cal[win_num][0]
                            if self.jypb is not None:
                                win.rms_mk *= self.jypb[find_nearest_id(self.x_file, win.transition.f_trans_mhz)]
                            win.cal = rms_cal[win_num][1]
                            f_range = [velocity_to_frequency(v, win.transition.f_trans_mhz, vref_kms=self.vlsr_file)
                                       for v in v_range[win_num]]
                            f_range.sort()
                            win.f_range_fit = f_range
                            if self.x_file is not None:
                                win.x_fit, win.y_fit = select_from_ranges(self.x_file, f_range, y_values=self.y_file)

                        self.win_list.append(win)
                else:
                    self.win_list.extend(win_list_tag)

        self.win_list_fit = [w for w in self.win_list if w.in_fit]
        self.win_list_plot = []

        if self.x_file is not None:
            self.x_fit = np.concatenate([w.x_fit for w in self.win_list_fit], axis=None)
            self.y_fit = np.concatenate([w.y_fit for w in self.win_list_fit], axis=None)
            # x_mod = select_from_ranges(self.x_file, self.line_list['fmhz_range_plot'], oversampling=self.oversampling)
            # self.x_mod = x_mod[0]  # TODO: why do I have to do this?
        else:
            self.x_fit = None
        self.y_mod = None  # np.zeros_like(self.frequencies)

        self.params2fit = None
        self.norm_factors = None
        self.model = None
        if self.x_file is not None:
            self.model = self.generate_lte_model()
        self.best_params = None
        self.model_fit = None
        self.normalize = False
        self.figure = None

    def load_config(self, path):
        with open(path) as f:
            return json.load(f)  # , cls=type(self))

    def save_config(self, filename, dirname=None):
        config_save = {
            'data_file': os.path.abspath(self.data_file) if self.data_file is not None else None,
            'output_dir': os.path.abspath(self.output_dir) if self.output_dir is not None else None,
            'tc': self.cont_info,
            'tcmb': self.tcmb,
            'tuning_info': self._tuning_info_user,
            'v_range': self._v_range_user,
            'chi2_info': self._rms_cal_user,
            'bandwidth': self.bandwidth,
            'oversampling': self.oversampling,
            'fghz_min': self.fmin_mhz / 1.e3,
            'fghz_max': self.fmax_mhz / 1.e3,
            'df_mhz': self.dfmhz,
            'noise_mk': self.noise * 1.e3,
            'tau_max': self.tau_max,
            'thresholds': self.thresholds,
            'components': {cpt.name: cpt.as_json() for cpt in self.cpt_list}
        }
        json_dump = json.dumps(config_save, indent=4)  # separators=(', \n', ': '))
        if dirname is not None:
            if not os.path.isdir(os.path.abspath(dirname)):
                os.makedirs(os.path.abspath(dirname))
        else:
            dirname = self.output_dir
        path = os.path.join(dirname, filename)
        with open(path, 'w') as f:
            f.write(json_dump)

    def update_configuration(self, config):
        # update components only
        new_cpts = config["components"]
        for cpt, cpt_info in new_cpts.items():
            cname = cpt
            for par, pval in cpt_info.items():
                if par != 'species':
                    self.params2fit[cname+'_'+par].value = pval
                else:
                    for sp in pval:
                        self.params2fit[f"{cname}_ntot_{sp.tag}"].value = sp.ntot
                        self.params2fit[f"{cname}_fwhm_{sp.tag}"].value = sp.fwhm

        self.y_mod = self.model(self.x_mod, **self.params2fit)

    def update_parameters(self, params=None):
        if params is None:
            params = self.best_pars()
        for cpt in self.cpt_list:
            pars = {par: value for par, value in params.items() if cpt.name in par}
            cpt.update_parameters(pars)

    def model_info(self, x_mod, line_list=None, cpt_list=None, line_center_only=False):
        return {
            'tc': self.get_tc(x_mod),
            'tcmb': self.tcmb,
            'vlsr_file': self.vlsr_file,
            'norm_factors': self.norm_factors,
            'beam_sizes': [get_beam_size(f_i, get_telescope(f_i, self.tuning_info)) for f_i in x_mod],
            'line_list': self.line_list_all if line_list is None else line_list,
            'cpt_list': self.cpt_list if cpt_list is None else cpt_list,
            'noise': self.noise,
            'tau_max': self.tau_max,
            'file_rejected': self.file_rejected,
            'line_center_only': line_center_only
        }

    def get_tc(self, x_mod):
        if isinstance(self.tc, (int, float)):
            tc = self.tc
        else:  # assume it is a structured array with fields frequency, continuum
            # if all([x1 == x2 for x1, x2 in zip(x_mod, self.tc[0])])
            names = self.tc.dtype.names
            tc = []
            for x in x_mod:
                id = find_nearest_id(self.tc[names[0]], x)
                tc.append(self.tc[names[1]][id])
            tc = np.array(tc)
        return tc

    def get_rms(self, fmhz):
        if type(fmhz) == float:
            fmhz = list(fmhz)

        rms = []
        for freq in fmhz:
            for win in self.win_list_fit:
                if min(win.f_range_fit) <= freq <= max(win.f_range_fit):
                    rms.append(win.rms_mk / 1000.)

        return rms if len(rms) > 1 else rms[0]

    def get_weight(self, fmhz, intensity=0.):
        """
        Returns the weight as 1./sqrt(rms**2 + cal_uncertainty**2) where cal_uncertainty is the calibration uncertainty
        in percent * the intensity at the given frequency.
        :param fmhz: frequency in MHz
        :param intensity: intensity at fmhz
        :return: 1. / sqrt(rms**2 + cal_uncertainty**2)
        """
        for win in self.win_list_fit:
            if min(win.f_range_fit) <= fmhz <= max(win.f_range_fit):
                rms = win.rms_mk / 1000.
                cal = win.cal / 100.
                return 1. / np.sqrt(rms**2 + (cal * intensity)**2)

    def generate_lte_model(self, normalize=False):
        params2fit = Parameters()

        for icpt, cpt in enumerate(self.cpt_list):
            for par in cpt.parameters:
                if 'size' in par.name:
                    par.set(min=0. if not isinstance(par.min, (float, int)) else par.min)
                if 'tex' in par.name:
                    par.set(min=2.73 if not isinstance(par.min, (float, int)) else par.min,
                            max=cpt.tmax if not isinstance(par.min, (float, int)) else min(par.max, cpt.tmax))
                params2fit[par.name] = par

            for isp, sp in enumerate(cpt.species_list):
                for par in sp.parameters:
                    par.set(min=0. if not isinstance(par.min, (float, int)) else par.min)
                    params2fit[par.name] = par

        self.params2fit = params2fit

        norm_factors = {}
        for parname in self.params2fit:
            param = self.params2fit[parname]
            nf = np.abs(param.value) if (normalize and param.value != 0.) else 1.
            norm_factors[param.name] = nf
            if normalize and param.expr is None:
                param.set(min=param.min/nf, max=param.max/nf, value=param.value/nf)
        self.norm_factors = norm_factors

        if self.x_fit is not None:
            lte_model_func = generate_lte_model_func(self.model_info(self.x_fit))
        else:
            lte_model_func = generate_lte_model_func(self.model_info(self.x_mod))
            self.y_mod = lte_model_func(self.x_mod, **self.params2fit)
        res = Model(lte_model_func)
        self.model = res

        return res

    def fit_model(self, normalize=False, max_nfev=None, fit_kws=None, print_report=True, report_kws=None,
                  method='leastsq'):
        """
        Computes weights and perform the fit.
        :param normalize: whether to normalize the parameters (default=False)
        :param max_nfev: maximum number of iterations (default value depends on the algorithm)
        :param fit_kws: keywords for the fit function
        :param print_report: whether to print the statistics and best values (default=True)
        :param report_kws: keywords for the fit_report function
        :param method: name of the fitting method
        :return:
        """
        self.normalize = normalize

        if self.model is None:
            self.generate_lte_model(normalize=normalize)

        wt = np.array([self.get_weight(x, intensity=y)
                       for x, y in zip(self.x_fit, self.y_fit - self.get_tc(self.x_fit))])
        # wt = None
        self.model_fit = self.model.fit(self.y_fit, params=self.params2fit, fmhz=self.x_fit,
                                        weights=wt,
                                        method=method,
                                        max_nfev=max_nfev, fit_kws=fit_kws)

        self.best_pars()
        self.update_parameters()

        if print_report:
            print(self.fit_report(report_kws=report_kws))

    def fit_report(self, report_kws=None):
        if report_kws is None:
            report_kws = {}

        if self.normalize:
            report_kws['modelpars'] = self.best_pars()

        if 'show_correl' not in report_kws:
            report_kws['show_correl'] = False

        report = self.model_fit.fit_report(**report_kws)
        pvalue = '    p-value            = {}'.format(format_float(stats.chi2.sf(self.model_fit.chisqr,
                                                                                 self.model_fit.nfree),
                                                                   nb_signif_digits=6))
        lines = report.split(sep='\n')

        return '\n'.join(lines[:9] + [pvalue] + lines[9:])

    def compute_model(self, params=None, x_values=None, line_list=None, line_center_only=False):
        if self.model is None:
            self.generate_lte_model()
        if params is None:
            params = self.best_params if self.best_params is not None else self.params2fit
        if x_values is None:
            x_values = self.x_mod
        elif type(x_values) is list:
            x_values = np.array(x_values)
        lte_func = generate_lte_model_func(self.model_info(x_values, line_list=line_list,
                                                           line_center_only=line_center_only))

        return lte_func(x_values, **params)

    # def get_best_pars(self):
    #     for parname in self.params2fit:
    #         self.best_params[parname] = self.params2fit[parname] * self.config['norm_factors'][parname]
    #     return self.best_params

    def best_pars(self):
        if self.model_fit is not None and self.best_params is None:
            params = self.model_fit.params
            for par in params:
                p = params[par]
                nf = self.norm_factors[p.name]
                p.set(min=p.min * nf, max=p.max * nf, value=p.value * nf)
                if p.stderr is not None:
                    p.stderr *= nf
            self.best_params = params
            # reset norm factors
            self.norm_factors = {key: 1 for key in self.norm_factors.keys()}

        return self.best_params
        # parnames = list(self.model_fit.params.keys())
        # namelen = max(len(n) for n in parnames)
        # output = []
        # for name in parnames:
        #     par = self.model_fit.params[name]
        #     space = ' ' * (namelen - len(name))
        #     norm = self.config['norm_factors'][name]
        #     output.append(f"    {name}:{space} {par.value} x {norm} = {par.value*norm: .5g}")
        # return '\n'.join(output)

    def plot_pars(self):
        return self.best_params if self.best_params is not None else self.params2fit

    def integrated_intensities(self):
        best_pars = self.best_params if self.best_params is not None else self.params2fit
        res = {}
        for cpt in self.cpt_list:
            fluxes = []
            for win in self.win_list:
                f_ref = win.transition.f_trans_mhz
                fwhm = best_pars['{}_fwhm_{}'.format(cpt.name, win.transition.tag)].value
                vlsr = best_pars['{}_vlsr'.format(cpt.name)].value
                fmin_mod, fmax_mod = [velocity_to_frequency(vlsr + v, f_ref, vref_kms=self.vlsr_file)
                                      for v in [fwhm, -fwhm]]
                # x_file_win = self.x_file[(fmin_mod <= self.x_file) & (self.x_file <= fmax_mod)]
                # x_mod = np.linspace(min(x_file_win), max(x_file_win),
                #                     num=self.oversampling * len(x_file_win))
                npts = 100
                x_mod = np.linspace(fmin_mod, fmax_mod, num=npts)
                y_mod = self.compute_model(params=best_pars, x_values=x_mod, line_list=[win.transition])

                dv = 2. * fwhm / (npts - 1)
                K_kms = 0.
                for i in range(len(y_mod) - 1):
                    K_kms += np.mean([y_mod[i], y_mod[i+1]]) * dv

                fluxes.append([win.transition.tag, win.plot_nb, f_ref, K_kms])

            res[cpt.name] = pd.DataFrame(fluxes, columns=['tag', 'line number', 'f_mhz', 'K.km/s'])

        return res

    def plot_window(self, win, list_other_species=None, thresholds_other=None,
                    other_species_selection=None, ax=None, fig=None, basic=False, dpi=None):
        if fig is None:
            fig = Figure(figsize=(5, 4), dpi=dpi)
        if ax is None:
            ax = fig.add_subplot()
        if dpi is None:
            dpi = DPI_DEF

        vlsr = self.cpt_list[0].vlsr if self.vlsr_file == 0. else self.vlsr_file
        best_pars = self.best_params if self.best_params is not None else self.params2fit
        fwhm = max([best_pars[par].value for par in best_pars if 'fwhm' in par])

        tr = win.transition
        f_ref = tr.f_trans_mhz

        ax2 = ax.twiny()  # instantiate a second axes that shares the same y-axis
        padding = 0.05

        if self.bandwidth is not None:  # velocity at bottom (1), freq at top (2)
            vmin, vmax = -self.bandwidth / 2 + vlsr, self.bandwidth / 2 + vlsr
            fmin, fmax = [velocity_to_frequency(v, f_ref, vref_kms=self.vlsr_file)
                          for v in [vmax, vmin]]
            xmin1, xmax1 = vmin, vmax
            xmin2, xmax2 = fmin, fmax
            dx2 = xmax2 - xmin2
            dx1 = xmax1 - xmin1
            xlim1 = (xmin1 - padding * dx1, xmax1 + padding * dx1)
            xlim2 = (xmax2 + padding * dx2, xmin2 - padding * dx2)

            # Number of ticks
            ax2.xaxis.set_major_locator(plt.MaxNLocator(4))

        else:  # frequency at top and bottom
            fmin, fmax = self.fmin_mhz, self.fmax_mhz
            xmin1, xmax1 = fmin, fmax
            xmin2, xmax2 = fmin, fmax
            dx2 = xmax2 - xmin2
            dx1 = xmax1 - xmin1
            xlim1 = (xmin1 - padding * dx1, xmax1 + padding * dx1)
            xlim2 = xlim1

            # Number of ticks
            ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))

        ax2.set_xlim(xlim2)
        ax.set_xlim(xlim1)

        # Minor ticks
        ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        if self.x_file is not None:
            fmin_mod = fmin
            fmax_mod = fmax
            x_file_win = self.x_file[(fmin_mod <= self.x_file) & (self.x_file <= fmax_mod)]
            x_mod = np.linspace(min(x_file_win), max(x_file_win),
                                num=self.oversampling * len(x_file_win))
        else:
            x_mod = self.x_mod[(self.x_mod <= fmax) & (self.x_mod >= fmin)]
            y_mod = self.y_mod[(self.x_mod <= fmax) & (self.x_mod >= fmin)]

        # compute model for all transitions (no thresholds)
        all_lines = select_transitions(self.line_list_all, xrange=[fmin - 2 * fwhm, fmax + 2 * fwhm])  # for model calculation
        if self.x_file is not None:
            y_mod = self.compute_model(params=best_pars, x_values=x_mod, line_list=all_lines)

        all_lines_display = select_transitions(all_lines, thresholds=self.thresholds)  # for position display
        other_lines_display = pd.concat([all_lines, all_lines_display]).drop_duplicates(subset='db_id', keep=False)

        if list_other_species is not None:
            try:
                other_lines_thresholds = get_transition_list(db, list_other_species, [[fmin, fmax]],
                                                             **thresholds_other, return_type='df')
                tmp = pd.concat([all_lines_display, other_lines_display,
                                 other_lines_thresholds]).drop_duplicates(subset='db_id', keep=False)
                other_lines_display = pd.concat([other_lines_display, tmp])
            except IndexError:
                pass  # do nothing

        if self.bandwidth is None:
            other_lines_display = None

        ymin = min(y_mod)
        ymax = max(y_mod)
        if self.x_file is not None:
            x_file = self.x_file[(fmin <= self.x_file) & (self.x_file <= fmax)]
            y_file = self.y_file[(fmin <= self.x_file) & (self.x_file <= fmax)]
            ax2.plot(x_file, y_file, 'k-', drawstyle='steps-mid')
            ymin = min(ymin, min(y_file))
            ymax = max(ymax, max(y_file))
        ymin = ymin - 0.05 * (ymax - ymin)
        ymax = ymax + 0.1 * (ymax - ymin)
        if ymin == ymax:
            ymin -= 0.001  # arbitrary
            ymax += 0.001  # arbitrary
        ax.set_ylim(ymin, ymax)

        # plot overall model
        ax2.plot(x_mod, y_mod, drawstyle='steps-mid', color='r', linewidth=2)

        # write transition number (center, bottom)
        if len(self.win_list_plot) > 1:
            ax2.text(0.5, 0.05, "{}".format(win.plot_nb),
                     transform=ax2.transAxes, horizontalalignment='center',
                     fontsize='large', color=PLOT_COLORS[0])

        # plot range used for chi2 calculation
        v_range = win.v_range_fit
        if v_range is not None:
            ax.axvspan(v_range[0], v_range[1], facecolor='purple', alpha=0.1)

        if not basic :  # plot line position(s) and plot components if more than one
            for icpt, cpt in enumerate(self.cpt_list):
                par_vlsr = best_pars['{}_vlsr'.format(cpt.name)].value
                y_pos = ymax - (ymax - ymin) * np.array([0., 0.075])

                # plot line positions w/i user's constraints
                self.plot_line_position(ax2, tr, par_vlsr, y_pos,
                                        color=PLOT_COLORS[icpt], label=str(tr.tag))
                tag_colors = {tr.tag: PLOT_COLORS[icpt]}

                line_list = pd.concat([all_lines_display, other_lines_display])
                for t in line_list['tag']:
                    if t not in tag_colors:
                        tag_colors[t] = PLOT_COLORS[(icpt + 2 * len(tag_colors)) % len(PLOT_COLORS)]

                for row in all_lines_display.iterrows():
                    tran = row[1].transition
                    lbl = str(tran.tag)
                    lw = 1.5
                    if tran != tr:
                        self.plot_line_position(ax2, tran, par_vlsr, y_pos,
                                                color=tag_colors[tran.tag], label=lbl, linewidth=lw)

                if other_lines_display is not None:
                    for row in other_lines_display.iterrows():
                        tran = row[1].transition
                        lbl = "s{}".format(tran.tag)
                        lw = 0.75
                        if tran.tag == other_species_selection:  # plot at the bottom
                            ypos_other = ymin + (ymax - ymin) * np.array([0., 0.075])
                        else:  # plot at the top
                            ypos_other = y_pos - 0.025 * (ymax - ymin)
                        self.plot_line_position(ax2, tran, par_vlsr, ypos_other,
                                                color=tag_colors[tran.tag], label=lbl, linewidth=lw)

                if len(self.cpt_list) > 1:
                    c_lte_func = generate_lte_model_func(self.model_info(x_mod,
                                                                         line_list=all_lines,
                                                                         cpt_list=[self.cpt_list[icpt]]))
                    c_best_pars = {}
                    for pname, par in best_pars.items():
                        if cpt.name in pname:
                            c_best_pars[pname] = par.value
                    c_y_mod = c_lte_func(x_mod, **c_best_pars)

                    ax2.plot(x_mod, c_y_mod, drawstyle='steps-mid',
                             color=CPT_COLORS[icpt % len(CPT_COLORS)], linewidth=0.75)

            handles, labels = ax2.get_legend_handles_labels()
            newLabels, newHandles = [], []  # for main lines
            satLabels, satHandles = [], []  # for satellites lines
            for handle, label in zip(handles, labels):
                if label not in newLabels and label[0] != 's':
                    newLabels.append(label)
                    newHandles.append(handle)
                elif label[1:] not in satLabels and label[0] == 's':
                    satLabels.append(label[1:])
                    satHandles.append(handle)
            leg = ax.legend(newHandles, newLabels, labelcolor='linecolor', frameon=False,
                             bbox_to_anchor=(xmin1 - padding * dx1, y_pos[1] - 0.02 * (ymax - ymin)),
                             bbox_transform=ax.transData, loc='upper left',
                             fontsize='small',
                             handlelength=0, handletextpad=0, fancybox=True)

            sat_leg = ax.legend(satHandles, satLabels, frameon=False, labelcolor='linecolor',
                                 bbox_to_anchor=(xmax1 + padding * dx1, y_pos[1] - 0.02 * (ymax - ymin)),
                                 bbox_transform=ax.transData, loc='upper right',
                                 fontsize='small',
                                 handlelength=0, handletextpad=0, fancybox=True)
            for text in sat_leg.get_texts():
                text.set_fontstyle("italic")

            # Manually add the first legend back
            ax.add_artist(leg)

        return fig

    def plot_line_position(self, freq_axis, transition, vel, y_range, err_color=None, **kwargs):
        x_pos = velocity_to_frequency(vel, transition.f_trans_mhz, vref_kms=self.vlsr_file)
        freq_axis.plot([x_pos, x_pos], y_range, **kwargs)
        # plot error on line frequency
        if transition.f_err_mhz is not None:
            if err_color is None:
                err_color = kwargs['color']
            freq_axis.plot([x_pos - transition.f_err_mhz, x_pos + transition.f_err_mhz], 2 * [np.average(y_range)],
                           color=err_color, linewidth=0.75)

    def make_plot(self, tag=None, filename=None, dirname=None, gui=False, verbose=True, basic=False,
                  other_species=None, display_all=True, other_species_selection=None, dpi=None):
        """
        Produces a plot of the fit results.
        :param tag: specify a tag to plot ; if None, all tags are plotted.
        :param filename: name of output file.
        :param dirname: path to an output directory.
        :param gui: if True, plot on screen, one window at a time.
        :param verbose: if True, prints some information in the terminal, such as png file location
        :param basic: if True, does not plot line position and individual components (time consuming)
        :param other_species: dictionary with key = tag, value = thresholds for other species
        for which to plot line position
        :param display_all: if False, display only lines with velocity selection.
        :param other_species_selection: (int) select only windows with other lines from this tag.
        :param dpi: the dpi value
        :return: None
        """

        plt.close()
        plt.ticklabel_format(style='plain')

        if dpi is None:
            dpi = DPI_DEF

        if tag is not None:
            win_list_plot = [w for w in self.win_list if w.transition.tag == tag]
        else:
            win_list_plot = self.win_list

        if not display_all:
            win_list_plot = [w for w in win_list_plot if w.in_fit]

        list_other_species, thresholds_other = get_other_species(other_species)

        best_pars = self.best_params if self.best_params is not None else self.params2fit
        self.update_parameters(params=best_pars)

        if other_species_selection is None:
            self.win_list_plot = win_list_plot
        else:
            fwhm_all_cpt = [best_pars[par].value for par in best_pars if 'fwhm' in par]
            fwhm = max(fwhm_all_cpt)
            for win in win_list_plot:
                f_ref = win.transition.f_trans_mhz
                # t_ref = win.transition.tag
                # fwhm_all_cpt = [best_pars[par].value for par in best_pars if 'fwhm_{}'.format(t_ref) in par]
                # fwhm = max(fwhm_all_cpt)
                delta_f = 3. * delta_v_to_delta_f(fwhm, fref_mhz=f_ref)
                try:
                    res = get_transition_list(db, int(other_species_selection),
                                              [[f_ref - delta_f, f_ref + delta_f]],
                                              **thresholds_other, return_type='df')
                    win.other_species_selection = res
                    self.win_list_plot.append(win)
                except IndexError:
                    pass

        nplots = len(self.win_list_plot)
        if nplots == 0:
            raise LookupError("No lines found for the plot. Please check your tag selection.")

        # if verbose:
        #     print("Tag {}, plot number {} :".format(tr.tag, line_list_plot.plot_nb.iloc[i]))
        #     print("Main transitions :")
        #     for t in all_lines_display['transition']:
        #         print(t)
        #     print("Satellite transitions :")
        #     for t in other_lines_display['transition']:
        #         print(t)
        #     print(" ")

        if gui:
            root = tkinter.Tk()
            root.wm_title("LTEmodel - Results")
            root.geometry("700x500")
            # root.columnconfigure(0, weight=1)
            # root.columnconfigure(1, weight=3)
            # root.rowconfigure(0, weight=3)
            # root.rowconfigure(1, weight=1)

            fig = self.plot_window(self.win_list_plot[0], list_other_species, thresholds_other,
                                   other_species_selection)
            canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
            canvas.draw()

            # pack_toolbar=False will make it easier to use a layout manager later on.
            toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
            # navigation toolbar
            # toolbarFrame = tkinter.Frame(master=root)
            # toolbarFrame.grid(row=1, column=1)
            # toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)

            toolbar.update()
            # toolbar.grid(row=1, column=1, sticky='ew')

            # canvas.mpl_connect(
            #     "key_press_event", lambda event: print(f"you pressed {event.key}"))
            # canvas.mpl_connect("key_press_event", key_press_handler)

            # Create a frame for the listbox+scrollbar, attached to the root window
            win_frame = tkinter.Frame(root)
            win_names = [win.name for win in self.win_list_plot] if nplots > 1 else [self.win_list_plot[0].name[:-4]]
            # Create a Listbox and attaching it to its frame
            win_list = tkinter.Listbox(win_frame, width=10, selectmode='single', activestyle='none',
                                       listvariable=tkinter.StringVar(value=win_names))
            win_list.select_set(0)
            win_list.activate(0)
            win_list.focus_set()
            # Insert elements into the listbox
            # for values in range(100):
            #     win_list.insert(tkinter.END, values)

            # handle event
            def win_selected(event):
                """
                Handle item selected event for the windows' listbox
                """
                # get selected indices
                iwin = event.widget.curselection()[0]
                # update fig
                fig.clear()
                self.plot_window(self.win_list_plot[iwin], list_other_species, thresholds_other,
                                 other_species_selection, fig=fig)
                canvas.draw_idle()
                # canvas.flush_events()
                toolbar.update()

            def OnEntryUpDown(event):
                selection = event.widget.curselection()[0]

                if event.keysym == 'Up':
                    selection = selection - 1 if selection > 0 else (event.widget.size() - 1)

                if event.keysym == 'Down':
                    selection = selection + 1 if selection < (event.widget.size() - 1) else 0

                event.widget.selection_clear(0, tkinter.END)
                event.widget.select_set(selection)
                event.widget.activate(selection)
                event.widget.selection_anchor(selection)
                event.widget.see(selection)
                win_selected(event)

            win_list.bind('<<ListboxSelect>>', win_selected)
            win_list.bind("<Down>", OnEntryUpDown)
            win_list.bind("<Up>", OnEntryUpDown)

            # Create a Scrollbar attached to the listbox's frame
            win_scroll = tkinter.Scrollbar(win_frame, orient='vertical')
            # setting scrollbar command parameter to have a vertical view
            win_scroll.config(command=win_list.yview)
            # Attaching Listbox to Scrollbar
            # Since we need to have a vertical scroll we use yscrollcommand
            win_list.config(yscrollcommand=win_scroll.set)

            # button_quit = tkinter.Button(master=root, text="Quit", command=root.quit)
            # Packing order is important. Widgets are processed sequentially and if there
            # is no space left, because the window is too small, they are not displayed.
            # The canvas is rather flexible in its size, so we pack it last which makes
            # sure the UI controls are displayed as long as possible.

            # Add Listbox to the left side of its frame
            win_list.pack(side=tkinter.LEFT, fill=tkinter.BOTH)
            # Add Scrollbar to the right side
            win_scroll.pack(side=tkinter.RIGHT, fill='y')
            # Add list frame to root
            win_frame.pack(side="left", fill='y')
            # win_frame.grid(rowspan=2, column=0, sticky='ns')

            # button_quit.pack(side=tkinter.BOTTOM)
            toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
            canvas.get_tk_widget().pack(side=tkinter.RIGHT, fill=tkinter.BOTH, expand=1)
            # canvas.get_tk_widget().grid(row=0, column=1)

            tkinter.mainloop()

        if filename is not None:  # save to file
            file_path = self.set_filepath(filename, dirname=dirname, ext='png')

            if verbose:
                print("\nSaving plot to {} \n...".format(file_path))

            nx = int(np.ceil(np.sqrt(nplots)))
            ny = int(np.ceil(nplots / nx))
            scale = 4
            fig, axes = plt.subplots(nx, ny, figsize=(nx * scale, ny * scale))
            for i, ax in enumerate(fig.axes):
                if i >= nplots:
                    ax.set_frame_on(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                self.plot_window(self.win_list_plot[i], list_other_species, thresholds_other, other_species_selection,
                                 ax=ax, fig=fig, basic=basic, dpi=dpi)

            fig.savefig(file_path, bbox_inches='tight', dpi=dpi)

            if verbose:
                print("Done\n")

    def set_filepath(self, filename, dirname=None, ext=None):
        sub_dir = self.output_dir
        if '/' in filename:  # filename contains directory
            dirs = os.path.split(filename)
            sub_dir = os.path.join(*dirs[:-1])
            filename = dirs[-1]

        if dirname is not None:
            if sub_dir is not None:
                dirname = os.path.join(dirname, sub_dir)
        else:
            dirname = sub_dir
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        if ext is not None:
            if ext[0] != '.':
                ext = '.' + ext
            if filename[-len(ext):] != ext:
                filename += ext

        return os.path.join(dirname, filename)

    def save_spectrum(self, filename, dirname=None, ext='txt', spec=None, continuum=False):
        file_path = self.set_filepath(filename, dirname=dirname, ext=ext)

        spectrum_type = ''
        if spec is not None:
            x_values, y_values = spec
        elif continuum:
            spectrum_type = 'continuum'
            ext = 'txt'  # force txt extension
            file_path = self.set_filepath(filename + '_cont', dirname=dirname, ext=ext)
            x_values = self.tc['f_mhz']
            y_values = self.tc['tc']
        elif self.data_file is None and self.x_file is not None:
            self.data_file = file_path
            x_values, y_values = self.x_file, self.y_file
        else:
            x_values, y_values = self.x_mod, self.y_mod
            spectrum_type = 'synthetic'

        if ext == 'fits':
            col1 = fits.Column(name='wave', format='D', unit='MHz', array=x_values)
            col2 = fits.Column(name='flux', format='D', unit='K', array=y_values)
            hdu = fits.BinTableHDU.from_columns([col1, col2])

            hdu.header['DATE-HDU'] = (datetime.datetime.now().strftime("%c"), 'Date of HDU creation')
            if spectrum_type == 'synthetic':
                hdu.header['DATABASE'] = ('SQLITE ({})'.format(os.path.split(SQLITE_FILE)[-1]), 'Database used')
                hdu.header['MODEL'] = ('Full LTE', 'Model used to compute this spectrum')
                hdu.header['NOISE'] = (self.noise * 1000., '[mK]Noise added to the spectrum')

            params = self.params2fit
            if self.best_params is None:
                params = self.best_pars()
            try:
                vlsr = params['{}_vlsr'.format(self.cpt_list[0].name)].value
            except TypeError:
                vlsr = self.cpt_list[0].vlsr
            hdu.header['VLSR'] = (vlsr, '[km/s]')

            hdu.writeto(file_path, overwrite=True)

        if ext == 'txt':
            with open(file_path, 'w') as f:
                if spectrum_type != 'continuum':
                    f.writelines(['#title: Spectral profile\n',
                                  '#date: {}\n'.format(datetime.datetime.now().strftime("%c")),
                                  '#coordinate: world\n',
                                  '#xLabel: frequency [MHz]\n',
                                  '#yLabel: [Kelvin] Mean\n'])
                for x, y in zip(x_values, y_values):
                    f.write('{}\t{}\n'.format(format_float(x), format_float(y)))

        return os.path.abspath(file_path)

    def save_fit_results(self, filename, dirname=None):
        with open(self.set_filepath(filename, dirname=dirname, ext='txt'), 'w') as f:
            f.writelines(self.fit_report(report_kws={'show_correl': True}))

    def write_cassis_file(self, filename, dirname=None):
        def lam_item(name, value):
            if isinstance(value, float):
                fmt = '{:.3e}' if (value >= 1.e4 or value <= 1.e-2) else '{:.2f}'
                return '{}={}\n'.format(name, fmt.format(value))
            return '{}={}\n'.format(name, value)

        ext = 'ltm' if self.x_file is None else 'lam'

        filebase = self.set_filepath(filename, dirname)

        tels = self.tuning_info['telescope'].values
        if len(tels) > 1:
            filepaths = [filebase + '_' + tel + '.' + ext for tel in tels]
        else:
            filepaths = [filebase + '.' + ext]

        if ext == 'ltm':
            tuning = {
                'tuningMode': 'RANGE',
                'minValue': min(self.x_mod) / 1000.,
                'maxValue': max(self.x_mod) / 1000.,
                'valUnit': 'GHZ',
                'lineValue': '115.5',
                'dsbSelected': 'false',
                'dsb': 'LSB',
                'loFreq': '121.5',
                'telescope': '',  # TBD when writing the ltm file
                'bandwidth': (max(self.x_mod) - min(self.x_mod)) / 1000.,
                'bandUnit': 'GHZ',
                'resolution': self.dfmhz,
                'resolutionUnit': 'MHZ'
            }
        else:
            tuning = {
                'nameData': os.path.abspath(self.data_file) if self.data_file is not None else '',
                'telescopeData': '',  # TBD when writing the lam file
                'typeFrequency': 'SKY' if self.vlsr_file == 0. else 'REST',
                'minValue': min(self.x_file) / 1000.,
                'maxValue': max(self.x_file) / 1000.,
                'valUnit': 'GHZ',
                'bandValue': self.bandwidth,
                'bandUnit': 'KM_SEC_MOINS_1'
            }

        eup_max = 1.e304
        eup_max_vals = [val['eup_max'] for val in self.thresholds.values() if val['eup_max'] is not None]
        if len(eup_max_vals) > 0:
            eup_max = max(eup_max_vals)
        aij_max = 1.e304
        aij_max_vals = [val['aij_max'] for val in self.thresholds.values() if val['aij_max'] is not None]
        if len(aij_max_vals) > 0:
            aij_max = max(aij_max_vals)
        thresholds = {'jupMin': '*',
                      'jupMax': '*',
                      'jLowMin': '*',
                      'jLowMax': '*',
                      'kupMin': '*',
                      'kupMax': '*',
                      'kLowMin': '*',
                      'kLowMax': '*',
                      'lupMin': '*',
                      'lupMax': '*',
                      'lLowMin': '*',
                      'lLowMax': '*',
                      'mupMin': '*',
                      'mupMax': '*',
                      'mLowMin': '*',
                      'mLowMax': '*',
                      'thresEupMin': min([val['eup_min'] for val in self.thresholds.values()]),
                      'thresEupMax': eup_max,
                      'thresAijMin': min([val['aij_min'] for val in self.thresholds.values()]),
                      'thresAijMax': aij_max
                      }

        species = {'template': 'All Species',
                   'nbMoleculesSelected': len(self.tag_list)
                   }
        for it, tag in enumerate(self.tag_list):
            species['moleculeSelectedNum{}'.format(it)] = tag

        lte_radex = {
            'telescope': '',  # TBD when writing the file
            'tmbBox': 'false',
            'observing': 'PSDBS',
            'tbg': self.tcmb,
            'tbgUnit': 'K',
            'noise': self.noise * 1000.,
            'noiseUnit': 'mK',
            'frequency': 'SKY' if self.cpt_list[0].vlsr != 0. else 'REST'
        }
        if ext == 'lam':
            lte_radex = {'lteRadexSelected': 'true', **lte_radex, 'overSampling': self.oversampling,
                         'frequency': 'SKY' if self.vlsr_file == 0. else 'REST'}

        # Define continuum
        if isinstance(self.tc, (float, int)):
            tc = self.tc
        else:  # assume cont defined as array => need to write it to a file
            tc = self.save_spectrum(filename, dirname=dirname, continuum=True)

        components = {'# Component parameters 1':
                          {'Comp1Name': 'Continuum',
                           'Comp1Enabled': 'true',
                           'Comp1Interacting': 'false',
                           'Comp1ContinuumSelected': 'CONSTANT',
                           'Comp1Continuum': 'Continuum 0 [K]',
                           'Comp1ContinuumSize': tc
                           }
                      }

        # Define other components
        params = self.params2fit
        if self.best_params is None and ext == 'lam':
            params = self.best_pars()

        for ic, cpt in enumerate(self.cpt_list):
            c_id = ic + 1
            c_num = ic + 2
            basename = 'Comp{}'.format(c_num)
            cdic = {'Name': 'Component {}'.format(c_id),
                    'Enabled': 'true',
                    'Interacting': 'true' if cpt.isInteracting else 'false',
                    'Algo': 'FULL_LTE_ALGO',
                    'Density': 7.5e22,
                    'Geometry': 'SPHERE_MODE',
                    'Vlsr': params['{}_vlsr'.format(cpt.name)].value,
                    'VlsrUnit': 'km/s',
                    'NbMol': len(cpt.tag_list)
                    }
            for isp, sp in enumerate(cpt.species_list):
                molname = 'Mol{}'.format(isp + 1)
                mol_dic = {
                    'Tag': sp.tag,
                    'Species': sp.name,
                    'Database': sp.database,
                    'Collision': '-no -',
                    'Compute': 'true',
                    'NSp': params['{}_ntot_{}'.format(cpt.name, sp.tag)].value,
                    'Abundance': params['{}_ntot_{}'.format(cpt.name, sp.tag)].value / cdic['Density'],
                    'Tex': params['{}_tex'.format(cpt.name)].value,
                    'TKin': '10.0',
                    'FWHM': params['{}_fwhm_{}'.format(cpt.name, sp.tag)].value,
                    'Size': params['{}_size'.format(cpt.name)].value
                    }
                cdic[molname] = {basename + molname + key: val for key, val in mol_dic.items()}

            components['# Component parameters {}'.format(c_num)] = {basename + key: val for key, val in cdic.items()}

        all_lines = ['# {}\n'.format(datetime.datetime.now().strftime("%c"))]
        if ext == 'lam':
            all_lines.append('# Generals Parameters\n\n')

        for filepath, tel in zip(filepaths, tels):
            with open(filepath, 'w') as f:
                if 'telescopeData'in tuning:
                    tuning['telescopeData'] = tel
                if 'telescope'in tuning:
                    tuning['telescope'] = tel
                lte_radex['telescope'] = tel
                f.writelines(all_lines)
                if ext == 'ltm':
                    items = [tuning, thresholds, lte_radex]
                else:
                    items = [tuning, thresholds, species, lte_radex]
                for dic in items:
                    for k, v in dic.items():
                        f.write(lam_item(k, v))
                    f.write('\n')
                for c_title, c_dic in components.items():
                    f.write(c_title + '\n\n')
                    for k, v in c_dic.items():
                        if isinstance(v, dict):
                            f.write('\n')
                            for kk, vv in v.items():
                                f.write(lam_item(kk, vv))
                        else:
                            f.write(lam_item(k, v))
                    f.write('\n')

    def write_ltm(self, filename, dirname=None):
        self.write_cassis_file(filename, dirname=dirname)

    def write_lam(self, filename, dirname=None, save_fit=True):
        if save_fit:
            self.save_fit_results(filename + '_fit_res', dirname=dirname)
        self.write_cassis_file(filename, dirname=dirname)


class Window:
    def __init__(self, transition, plot_nb, v_range_fit=None, f_range_fit=None, rms_mk=None, cal=None):
        self.transition = transition
        self.plot_nb = plot_nb
        self._name = "{} - {}".format(transition.tag, plot_nb)
        self._v_range_fit = v_range_fit
        self._f_range_fit = f_range_fit
        self._rms_mk = rms_mk
        self._cal = cal
        self._x_fit = None
        self._y_fit = None
        self._in_fit = False
        self.other_species_selection = None

    @property
    def name(self):
        return self._name

    @property
    def v_range_fit(self):
        return self._v_range_fit

    @v_range_fit.setter
    def v_range_fit(self, value):
        self._v_range_fit = value

    @property
    def f_range_fit(self):
        return self._f_range_fit

    @f_range_fit.setter
    def f_range_fit(self, value):
        self._f_range_fit = value

    @property
    def rms_mk(self):
        return self._rms_mk

    @rms_mk.setter
    def rms_mk(self, value):
        self._rms_mk = value

    @property
    def cal(self):
        return self._cal

    @cal.setter
    def cal(self, value):
        self._cal = value

    @property
    def x_fit(self):
        return self._x_fit

    @x_fit.setter
    def x_fit(self, value):
        self._x_fit = value

    @property
    def y_fit(self):
        return self._y_fit

    @y_fit.setter
    def y_fit(self, value):
        self._y_fit = value
        if len(self._y_fit) > 3:
            self._in_fit = True

    @property
    def in_fit(self):
        return self._in_fit


class Component:
    def __init__(self, name, species_list, isInteracting=False, vlsr=None, size=None, tex=100., config=None):
        # super().__init__()
        self.name = name
        self.species_list = []
        for sp in species_list:
            if isinstance(sp, Species):
                sp2add = sp
            elif isinstance(sp, dict):
                sp2add = Species(sp['tag'], ntot=sp['ntot'], fwhm=sp['fwhm'], component=self)
            elif isinstance(sp, int):
                sp2add = Species(sp)
            else:
                try:
                    sp2add = Species(int(sp))
                except TypeError:
                    print("Elements of species_list must be a Species, a dictionary, "
                          "or the tag as an integer or a string")
            if sp2add._component is None:
                sp2add.set_component(self.name)
            self.species_list.append(sp2add)

        self.tag_list = [sp.tag for sp in self.species_list]
        self.isInteracting = isInteracting
        if vlsr is None:
            vlsr = VLSR_DEF  # TODO: find out why VLSR_DEF inside __init__ does not work
        self._vlsr = create_parameter('{}_vlsr'.format(self.name), vlsr)  # km/s
        if size is None:
            size = SIZE_DEF
        self._size = create_parameter('{}_size'.format(self.name), size)  # arcsec
        self._tex = create_parameter('{}_tex'.format(self.name), tex)  # K
        for sp in self.species_list:
            sp.tex = self.tex
        self._tmax = min([max(sp.pf[0]) for sp in self.species_list])
        self.transition_list = None
        self.parameters = [self._vlsr, self._size, self._tex]

    def as_json(self):
        return {
            'vlsr': round(self.vlsr, 3),
            'size': round(self.size, 3),
            'tex': round(self.tex, 3),
            'isInteracting': self.isInteracting,
            'species': [sp.as_json() for sp in self.species_list]
        }

    def update_parameters(self, new_pars):
        self.vlsr = new_pars['{}_vlsr'.format(self.name)].value
        self.size = new_pars['{}_size'.format(self.name)].value
        self.tex = new_pars['{}_tex'.format(self.name)].value
        for sp in self.species_list:
            sp.ntot = new_pars['{}_ntot_{}'.format(self.name, sp.tag)].value
            sp.fwhm = new_pars['{}_fwhm_{}'.format(self.name, sp.tag)].value
            # sp.tex = new_pars['{}_tex'.format(self.name, sp.tag)].value

    @property
    def vlsr(self):
        return self._vlsr.value

    @vlsr.setter
    def vlsr(self, value):
        if self._vlsr.value != value:
            self._vlsr.value = value

    @property
    def size(self):
        return self._size.value

    @size.setter
    def size(self, value):
        if self._size.value != value:
            self._size.value = value

    @property
    def tex(self):
        return self._tex.value

    @tex.setter
    def tex(self, value):
        if self._tex.value != value:
            self._tex.value = value
            for sp in self.species_list:
                sp.tex = value

    @property
    def tmax(self):
        self._tmax = min([max(sp.pf[0]) for sp in self.species_list])
        return self._tmax

    def get_transitions(self, fmhz_ranges, **thresholds):
        self.transition_list = get_transition_list(db, self.species_list, fmhz_ranges, **thresholds)
        return self.transition_list

    def get_fwhm(self, transition):
        tag = transition.tag
        return next(sp for sp in self.species_list if sp.tag == tag).fwhm

    def get_tex(self, transition):
        tag = transition.tag
        return next(sp for sp in self.species_list if sp.tag == tag).tex

    def get_ntot(self, transition):
        tag = transition.tag
        return next(sp for sp in self.species_list if sp.tag == tag).ntot

    def assign_spectrum(self, spec: SimpleSpectrum):
        self.model_spec = spec


class Species:
    def __init__(self, tag, ntot=7.0e14, tex=100., fwhm=FWHM_DEF, component=None):
        # super().__init__(self)
        self._tag = int(tag)  # make sure tag is stored as an integer
        if isinstance(ntot, (float, dict)):
            self._ntot = create_parameter('ntot_{}'.format(tag), ntot)  # total column density [cm-2]
        elif isinstance(ntot, Parameter):
            self._ntot = ntot
        else:
            raise TypeError("ntot must be a float, a dictionary or a Parameter")
        if isinstance(fwhm, (float, dict)):
            self._fwhm = create_parameter('fwhm_{}'.format(tag), fwhm)  # line width [km/s]
        elif isinstance(fwhm, Parameter):
            self._fwhm = fwhm
        else:
            raise TypeError("fwhm must be a float, a dictionary or a Parameter")

        self._tex = tex  # excitation temperature [K]
        self._component = component
        if component is not None:
            self.set_component(component.name)

        sp_dic = get_species_info(db, tag)
        if sp_dic is None:
            raise IndexError("Tag {} not found in the database.".format(tag))
        self._id = sp_dic['id']
        self._name = sp_dic['name']
        self._database = sp_dic['database_name']

        self.pf = get_partition_function(db, self._tag)  # (tref, qlog)

    def as_json(self):
        return {
            'tag': self.tag,
            'ntot': round(self.ntot, 3),
            'fwhm': round(self.fwhm, 3)
        }

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def database(self):
        return self._database

    @property
    def tag(self):
        return self._tag

    @property
    def ntot(self):
        return self._ntot.value

    @ntot.setter
    def ntot(self, value):
        if self._ntot.value != value:
            self._ntot.value = value

    @property
    def tex(self):
        return self._tex

    @tex.setter
    def tex(self, value):
        if self._tex != value:
            self._tex = value

    @property
    def fwhm(self):
        return self._fwhm.value

    @fwhm.setter
    def fwhm(self, value):
        if self._fwhm.value != value:
            self._fwhm.value = value

    @property
    def parameters(self):
        return [self._ntot, self._fwhm]

    def set_component(self, comp_name):
        self._component = comp_name
        self._ntot.name = '{}_ntot_{}'.format(comp_name, self.tag)
        self._fwhm.name = '{}_fwhm_{}'.format(comp_name, self.tag)

    def get_partition_function(self, tex):
        tmp = np.interp(np.log10(tex), np.log10(self.pf[0]), self.pf[1])
        # tmp = find_nearest_id(self.pf[0], tex)
        # tmp = self.pf[1][tmp]
        qex = np.power(10., tmp)
        return get_partition_function_tex(self.pf[0], self.pf[1], tex)


class Transition:
    def __init__(self, tag, f_trans_mhz, aij, elo_cm, gup, name='', f_err_mhz=None, db_id=None, qn=None):
        self.f_trans_mhz = f_trans_mhz
        self.f_err_mhz = f_err_mhz
        self.aij = aij
        self.elo_cm = elo_cm
        self.elo_J = self.elo_cm * const.h.value * const.c.value * 100
        self.eup_J = self.elo_J + self.f_trans_mhz * 1.e6 * const.h.value
        self.gup = gup
        # self.eup = (elo_cm + self.f_trans_mhz * 1.e6 / (const.c.value * 100)) * 1.4389  # [K]
        self.eup = self.eup_J / const.k_B.value  # k_B in J/K
        self.tag = tag
        self.name = name
        self.db_id = db_id
        self.qn = qn

    def __str__(self):
        infos = ["{} ({})".format(self.name, self.qn),
                 "Tag = {}".format(self.tag),
                 "f = {} MHz (+/-{})".format(self.f_trans_mhz, self.f_err_mhz),
                 "Eup = {:.2f} K".format(self.eup),
                 "Aij = {:.2e} s-1".format(self.aij),
                 "gup = {}".format(self.gup)]
        return " ; ".join(infos)

    def __eq__(self, other):
        return True if self.db_id == other.db_id else False


def parameter_infos(value=None, min=None, max=None, expr=None, vary=True,
                    factor=False, difference=False):
    if factor and difference:
        raise KeyError("Can only have factor=True OR difference=True")
    if factor and value is not None and min is not None:
        min *= value
        max *= value
    if difference and value is not None and max is not None:
        min += value
        max += value
    return {'value': value, 'min': min, 'max': max, 'expr': expr, 'vary': vary}


def create_parameter(name, param):
    if isinstance(param, (float, int)):
        return Parameter(name, value=param)

    elif isinstance(param, dict):
        return Parameter(name, **param)
                         # value=param['value'], min=param.get('min', None), max=param.get('max', None),
                         # expr=param.get('expr', None))

    else:
        raise TypeError("Param must be a float or a dictionary.")


def select_from_ranges(x_values, ranges, y_values=None, oversampling=None):
    if type(ranges[0]) is not list:
        ranges = [ranges]
    x_new = []
    if y_values is not None:
        y_new = []
        df = pd.DataFrame({"x": x_values, "y": y_values})
    for x_range in ranges:
        x_sub = x_values[(x_values >= min(x_range)) & (x_values <= max(x_range))]
        if len(x_sub) == 0:
            continue
        xmin, xmax = min(x_sub), max(x_sub)
        if oversampling is not None:
            x_sub = np.linspace(xmin, xmax, num=len(x_sub)*oversampling, endpoint=True)
        x_new = np.append(x_new, x_sub)
        if y_values is not None:
            y_sub = y_values[(x_values >= xmin) & (x_values <= xmax)]
            y_new = np.append(y_new, y_sub)

    return x_new, y_new if y_values is not None else x_new


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


def get_other_species(other_species):
    thresholds_other = {}
    if other_species is None:
        return None, thresholds_other

    list_other_species = []
    if other_species is type(list):
        for other_sp in other_species:
            if isinstance(other_sp, list):
                new_sp = other_sp[0]
                new_th = {
                    'eup_max': other_sp[1] if other_sp[1] != '*' else EUP_MAX_DEF,
                    'aij_min': other_sp[2] if other_sp[2] != '*' else AIJ_MIN_DEF,
                    'err_max': other_sp[3] if other_sp[3] != '*' else ERR_MAX_DEF
                }
            else:
                new_sp = other_sp
                new_th = THRESHOLDS_DEF
            list_other_species.append(new_sp)
            thresholds_other[str(new_sp)] = new_th

    elif os.path.isfile(other_species):
        df = pd.read_csv(other_species, delimiter='\t', comment='#')
        col_names = df.columns[1:]
        list_other_species = list(df.tag)
        for index, row in df.iterrows():
            thresholds_other[str(int(row.tag))] = {c: row[c] for c in col_names if '*' not in str(row[c])}

    else:
        raise TypeError("other_species should be a list or a path to a file.")

    return list_other_species, thresholds_other


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_nearest_id(array, value):
    if isinstance(array, list):
        array = np.array(array)
    return (np.abs(array - value)).argmin()


def find_nearest_trans(trans_list, value):
    f_trans_list = []
    for tr in trans_list:
        f_trans_list.append(tr.f_trans_mhz)
    idx = (np.abs(np.array(f_trans_list) - value)).argmin()
    return trans_list[idx]


def get_partition_function(db, tag, temp=None):
    tref = []
    qlog = []
    pf_file = os.path.join(PARTITION_FUNCTION_DIR, '{}.txt'.format(tag))
    if os.path.isfile(pf_file):
        tref, qlog = np.genfromtxt(pf_file, comments='//', unpack=True)
    else:
        # retrieve catdir_id :
        db.execute("SELECT id FROM catdir WHERE speciesid = {}".format(tag))
        sp_id = db.fetchall()[0][0]
        for row in db.execute("SELECT * FROM cassis_parti_funct WHERE catdir_id = " + str(sp_id)):
            tref.append(row[1])
            qlog.append(row[2])

    # qlin = [np.power(10., q) for q in qlog]
    # qlin.sort()
    tref, qlog = zip(*sorted(zip(tref, qlog)))
    if temp is None:
        return tref, qlog
    else:
        return get_partition_function_tex(tref, qlog, temp)


def get_partition_function_tex(tref, qlog, temp):
    if temp <= tref[0]:
        print(f'{temp} is below the lowest temperature of the partition function ({tref[0]}) : '
              f'setting Q({temp}K)=Q({tref[0]}K)')
        return np.power(10., qlog[0])
    if temp >= tref[-1]:
        print(f'{temp} is above the highest temperature of the partition function ({tref[-1]}) : '
              f'setting Q({temp}K)=Q({tref[-1]}K)')
        return np.power(10., qlog[-1])
    for i, t in enumerate(tref[:-1]):
        if tref[i+1] >= temp >= t:
            tmp = np.interp(np.log10(temp), np.log10(tref[i:i+2]), qlog[i:i+2])
            qex = np.power(10., tmp)
            # qex = np.interp(temp, tref, qlin)
            return qex
            # return np.power(10., qlog[find_nearest_id(np.array(tref),temp)])


def fwhm_to_sigma(value, reverse=False):
    if reverse:
        return value * (2. * np.sqrt(2. * np.log(2.)))
    else:
        return value / (2. * np.sqrt(2. * np.log(2.)))


def delta_v_to_delta_f(value, fref_mhz, reverse=False):
    if reverse:
        return (value / fref_mhz) * const.c.value * 1.e-3
    else:
        return value * 1.e3 * fref_mhz / const.c.value


def velocity_to_frequency(value, fref_mhz, reverse=False, vref_kms=0.):
    if reverse:
        return const.c.value * (1. - value / fref_mhz) * 1.e-3 + vref_kms
    else:
        return fref_mhz * (1. - (value - vref_kms) * 1.e3 / const.c.value)


def Teq(fmhz):
    return const.h.value * fmhz * 1.e6 / const.k_B.value


def jnu(fmhz, temp: float):
    fmhz_arr = np.array(fmhz) if type(fmhz) == list else fmhz
    res = (const.h.value * fmhz_arr * 1.e6 / const.k_B.value) / \
          (np.exp(const.h.value * fmhz_arr * 1.e6 / (const.k_B.value * temp)) - 1.)
    return list(res) if type(fmhz) == list else res


def get_telescope(fmhz, tuning_info: pd.DataFrame):
    tel = None
    for row in tuning_info.itertuples(index=False):
        if row.fmhz_range[0] <= fmhz <= row.fmhz_range[1]:
            return row.telescope
    if tel is None:
        raise LookupError("No telescope found at {} Mhz.".format(fmhz))


def get_beam_size(freq_mhz, tel_info):
    """
    Computes the beam size at the given frequency
    :param freq_mhz: frequency in MHz
    :param tel_info: telescope name or dataframe containing "tuning" information
    :return: the beam size in arcsec
    """
    tel = tel_info
    if isinstance(tel_info, pd.DataFrame):
        tel = get_telescope(freq_mhz, tel_info)
    return (1.22 * const.c.value / (freq_mhz * 1.e6)) / TEL_DIAM[tel] * 3600. * 180. / np.pi


def dilution_factor(source_size, beam_size, geometry='gaussian'):
    # dilution_factor = tr.mol_size ** 2 / (tr.mol_size**2 + get_beam_size(model.telescope,freq)**2)
    # dilution_factor = (1. - np.cos(cpt.size/3600./180.*np.pi)) / ( (1. - np.cos(cpt.size/3600./180.*np.pi))
    #                    + (1. - np.cos(get_beam_size(self.telescope,self.frequencies)/3600./180.*np.pi)) )
    if geometry == 'disc':
        return 1. - np.exp(-np.log(2.) * (source_size / beam_size) ** 2)
    else:
        return source_size ** 2 / (source_size ** 2 + beam_size ** 2)

    # hdr = cube.hdu.header
    # if hdr['BUNIT'] == 'Jy/beam':  # calculate conversion factor Jy/beam to K
    #     try:
    #         bmaj = hdr['BMAJ'] * np.pi / 180.  # major axis in radians, assuming unit = degrees
    #         bmin = hdr['BMIN'] * np.pi / 180.  # major axis in radians, assuming unit = degrees
    #     except KeyError:
    #         for hdu in cube.hdulist[1:]:
    #             try:
    #                 unit = hdu.columns['BMAJ'].unit  # assume bmaj and bmin have the same unit
    #                 fact = u.Quantity("1. {}".format(unit)).to(u.rad).value
    #                 bmaj = np.mean(hdu.data['BMAJ']) * fact
    #                 bmin = np.mean(hdu.data['BMIN']) * fact
    #             except KeyError:
    #                 raise KeyError("Beam information not found in file {}.".format(file_list[h]))
    #     omega = np.pi * bmaj * bmin / (4. * np.log(2.))

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

    power = np.log10(np.abs(value)) if value != 0 else 0.
    rpst = "e" if (power < -2 or power > nb_digits) else "f"
    f = "{:." + str(nb_signif_digits) + rpst + "}"
    return f.format(value)


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
            freq = data['wave'].byteswap().newbyteorder()  # to be able to use in a DataFrame
            flux = data['flux'].byteswap().newbyteorder()
    elif ext in ['.fus', '.lis']:
        with open(filepath) as f:
            nskip = 0
            line = f.readline()
            while '//' in line:
                nskip += 1
                if 'vlsr' in line:
                    vlsr = float(line.split()[-1])
                line = f.readline()

        data = np.genfromtxt(filepath, skip_header=nskip, names=True)
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
        freq = np.array(freq) * 1000.
        flux = np.array(flux)

    inds = freq.argsort()
    flux = flux[inds]
    freq = freq[inds]

    return freq, flux, vlsr
