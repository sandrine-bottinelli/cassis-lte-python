from cassis_lte_python.utils.utils import delta_v_to_delta_f, velocity_to_frequency,\
    open_data_file, is_in_range, select_from_ranges, find_nearest, expand_dict, read_noise_info
from cassis_lte_python.utils.constants import C_LIGHT, K_B, H, TEL_DIAM
from lmfit import Parameters, Parameter
import os
import configparser
import sqlite3
import pandas as pd
from numpy import concatenate, loadtxt, array, interp, power, log10, genfromtxt, ndarray, mean, floor, ceil, float32


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
    print(f"Using database : {SQLITE_FILE}")
    conn = sqlite3.connect(SQLITE_FILE)
else:
    raise FileNotFoundError(f'{SQLITE_FILE} not found.')

DATABASE_SQL = conn.cursor()

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


class ModelConfiguration:
    def __init__(self, configuration, verbose=True, check_tel_range=False):
        self._configuration_dict = configuration
        self.tag_list = []
        self.cpt_list = []
        for key, cpt_dic in configuration.get('components').items():
            sp_list = cpt_dic['species']
            cpt = Component(key, sp_list, isInteracting=cpt_dic.get('interacting', False),
                            vlsr=cpt_dic.get('vlsr'), tex=cpt_dic.get('tex'), size=cpt_dic.get('size'))
            self.cpt_list.append(cpt)
            for sp in cpt.species_list:
                if sp.tag not in self.tag_list:
                    self.tag_list.append(sp.tag)

        self.output_dir = configuration.get('output_dir', os.path.curdir)
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        self.base_name = configuration.get('base_name', 'lte_model')

        self.data_file = None
        self.x_file = None
        self.y_file = None
        self.vlsr_file = None
        self.vlsr_plot = None
        self.cont_info = None
        self.tc = None
        self.jypb = None
        self._v_range_user = configuration.get('v_range', None)
        self._rms_cal_user = configuration.get('chi2_info', None)
        self.win_list = []
        self.win_list_fit = None
        self.win_list_plot = []

        self.x_fit = None
        self.y_fit = None
        self.x_mod = None
        self.y_mod = None

        fmin_ghz = 115.
        fmax_ghz = 116.
        dfmhz = 0.1
        if 'fmin_ghz' in configuration and configuration['fmin_ghz'] is not None:
            fmin_ghz = configuration['fmin_ghz']
        if 'fmax_ghz' in configuration and configuration['fmax_ghz'] is not None:
            fmax_ghz = configuration['fmax_ghz']
        self.fmin_mhz = fmin_ghz * 1.e3
        self.fmax_mhz = fmax_ghz * 1.e3
        if 'df_mhz' in configuration and configuration['df_mhz'] is not None:
            dfmhz = configuration['df_mhz']
        self.dfmhz = dfmhz
        self.noise = 1.e-3 * configuration.get('noise_mk', 0.)

        self.t_a_star = configuration.get('t_a*', False)
        self._tuning_info_user = configuration.get('tuning_info', None)
        self.tuning_info = None

        if 'thresholds' in configuration:
            self.thresholds = get_species_thresholds(configuration['thresholds'],
                                                     select_species=self.tag_list,
                                                     return_list_sp=False)
        else:
            self.thresholds = {}
            for tag in self.tag_list:
                self.thresholds[str(tag)] = THRESHOLDS_DEF

        self._telescope_data = {}
        self.line_list_all = None
        self.tr_list_by_tag = None

        self.bandwidth = configuration.get('bandwidth', None)  # km/s ; None for 1 window with entire spectrum

        self.oversampling = int(configuration.get('oversampling', 3))

        self.tcmb = configuration.get('tcmb', 2.73)

        self.tau_max = configuration.get('tau_max', None)
        self.file_rejected = None
        if self.tau_max is not None:
            self.file_rejected = configuration.get('file_rejected', 'rejected_lines.txt')
        if self.file_rejected is not None:
            self.file_rejected = os.path.join(self.output_dir, self.file_rejected)
            with open(self.file_rejected, 'w') as f:
                f.writelines(['# Rejected lines with tau >= {}\n'.format(self.tau_max),
                              '\t'.join(['# Tag ', 'Ntot', 'Tex', 'FWHM', 'f_mhz', 'Eup', 'Aij', 'gup', 'tau'])
                              ])

    def get_data(self, config=None):
        if config is None:
            config = self._configuration_dict

        self.data_file = config.get('data_file', None)
        self.x_file = config.get('x_obs', None)
        self.y_file = config.get('y_obs', None)
        self.vlsr_file = config.get('vlsr_obs', 0.)
        if self.data_file is not None and self.x_file is None:
            self.x_file, self.y_file, self.vlsr_file = open_data_file(self.data_file)
        self.vlsr_plot = self.vlsr_file
        if self.vlsr_plot == 0.:
            self.vlsr_plot = self.cpt_list[0].vlsr

        if self.x_file is not None and isinstance(self.x_file[0], ndarray):
            self.x_file = concatenate(self.x_file)
        if self.y_file is not None and isinstance(self.y_file[0], ndarray):
            self.y_file = concatenate(self.y_file)

    def get_jypb(self, config=None):
        if config is None:
            config = self._configuration_dict
        if 'beam_info' in config:
            f_hz_beam = concatenate(config['beam_info']['f_mhz']) * 1.e6
            omega = concatenate(config['beam_info']['beam_omega'])
            self.jypb = 1.e-26 * C_LIGHT ** 2 / (f_hz_beam * f_hz_beam) / (2. * K_B * omega)

    def get_continuum(self, config=None):
        if config is None:
            config = self._configuration_dict

        if self.x_file is None:
            self.get_data()

        self.cont_info = config.get('tc', 0.)

        if isinstance(self.cont_info, (float, int)):
            self.tc = self.cont_info
        elif isinstance(self.cont_info, str) and os.path.isfile(self.cont_info):  # cont_info is a CASSIS continuum file : MHz [tab] K
            f_cont, t_cont = loadtxt(self.cont_info, delimiter='\t', unpack=True)
            self.tc = array([interp(x, f_cont, t_cont) for x in self.x_file])
        elif isinstance(self.cont_info, dict):  # to compute continuum over ranges given by the user
            tc = []
            for freqs, fluxes, franges in zip(self.x_file, self.y_file, self.cont_info.values()):
                cont_data = []
                for frange in franges:
                    cont_data.append(fluxes[(freqs >= min(frange)) & (freqs <= max(frange))])
                tc.append([mean(cont_data) for _ in freqs])
            self.tc = concatenate(tc)
        else:
            raise TypeError("Continuum must be a float, an integer or a 2-column tab-separated file (MHz K).")
        # else:  # HDU
        #     pass
            # tc = cont_info[datafile].data.squeeze()[j, i]

        if not isinstance(self.tc, (float, int)):
            self.tc = array([(f, t) for f, t in zip(self.x_file, self.tc)],
                               dtype=[('f_mhz', float32), ('tc', float32)])

    def get_linelist(self, config=None, check_tel_range=False):
        if config is None:
            config = self._configuration_dict

        if 'tuning_info' in config:
            # if telescope is not in TEL_DIAM, try to find it in TELESCOPE_DIR
            for tel in config['tuning_info'].keys():
                # if tel not in TEL_DIAM.keys():
                with open(os.path.join(TELESCOPE_DIR, tel), 'r') as f:
                    col_names = ['Frequency (MHz)', 'Beff/Feff']
                    tel_data = f.readlines()
                    for line in tel_data:
                        if 'Diameter' in line:
                            continue
                        TEL_DIAM[tel] = float(line)
                        break
                    for line in tel_data:
                        if 'Frequency' in line:
                            col_names = line.replace('.', ',').lstrip('//').split(',')
                            col_names = [c.strip() for c in col_names]
                            break
                self._telescope_data[tel] = pd.read_csv(os.path.join(TELESCOPE_DIR, tel), sep='\t', skiprows=3,
                                                        names=col_names, usecols=list(range(len(col_names))))

            if check_tel_range:  # check telescope ranges cover all data / all model values:
                x_vals = self.x_file if self.x_file is not None else [min(self.x_mod), max(self.x_mod)]
                extended = False
                for x in x_vals:
                    # is_in_range = [val[0] <= x <= val[1] for val in config['tuning_info'].values()]
                    limits = list(config['tuning_info'].values())
                    limits = [item for sublist in limits for item in sublist]
                    if not is_in_range(x, config['tuning_info'].values):
                        # raise LookupError("Telescope ranges do not cover some of the data, e.g. at {} MHz.". format(x))
                        extended = True
                        nearest = find_nearest(array(limits), x)
                        for key, val in config['tuning_info'].items():
                            # new_lo = 5. * np.floor(x / 5.) if val[0] == nearest else val[0]
                            # new_hi = 5. * np.ceil(x / 5.) if val[1] == nearest else val[1]
                            new_lo = floor(x) if val[0] == nearest else val[0]
                            new_hi = ceil(x) if val[1] == nearest else val[1]
                            config['tuning_info'][key] = [new_lo, new_hi]
                if extended:
                    print("Some telescope ranges did not cover some of the data ; ranges were extended to :")
                    print(config['tuning_info'])

            tuning_info = {'fmhz_range': [], 'telescope': [], 'fmhz_min': [], 'fmhz_max': []}
            for key, val in config['tuning_info'].items():
                arr = array(val)
                if len(arr.shape) == 1:
                    arr = array([val])
                for r in arr:
                    r_min, r_max = min(r), max(r)
                    # if self.x_file is not None:
                    #     x_sub = self.x_file[(self.x_file >= r_min) & (self.x_file <= r_max)]
                    #     r_min = max(r_min, min(x_sub))
                    #     r_max = min(r_max, max(x_sub))
                    #     if len(self.x_file[(self.x_file >= r_min) & (self.x_file <= r_max)]) == 0:
                    #         continue
                    # # if self.fmax_mhz is not None and self.fmin_mhz is not None:
                    # else:
                    #     r_min = max(r_min, self.fmin_mhz)
                    #     r_max = min(r_max, self.fmax_mhz)
                    tuning_info['fmhz_range'].append([r_min, r_max])
                    tuning_info['telescope'].append(key)
                    tuning_info['fmhz_min'].append(r_min)
                    tuning_info['fmhz_max'].append(r_max)
            self.tuning_info = pd.DataFrame(tuning_info)

        self.line_list_all = get_transition_list(DATABASE_SQL, self.tag_list, self.tuning_info['fmhz_range'],
                                                 return_type='df')
        tr_list_tresh = select_transitions(self.line_list_all, self.thresholds)
        self.tr_list_by_tag = {tag: list(tr_list_tresh[tr_list_tresh.tag == tag].transition) for tag in self.tag_list}
        if all([len(t_list) == 0 for t_list in self.tr_list_by_tag.values()]):
            raise LookupError("No transition found within the thresholds.")

    def get_v_range_info(self):
        # extract v_range info
        if self._v_range_user is not None:
            if isinstance(self._v_range_user, dict):
                if '*' in self._v_range_user:  # same velocity range for all species/lines
                    self._v_range_user = {str(tag): {'*': self._v_range_user['*']} for tag in self.tag_list}

                if len(self.tag_list) == 1 and str(self.tag_list[0]) not in self._v_range_user:
                    # only one species and tag not given => "reformat" dictionary to contain tag
                    self._v_range_user = {str(self.tag_list[0]): self._v_range_user}
                    if str(self.tag_list[0]) not in self._rms_cal_user:
                        self._rms_cal_user = {str(self.tag_list[0]): self._rms_cal_user}

            elif isinstance(self._v_range_user, str):
                self._v_range_user = read_noise_info(self._v_range_user)

            else:
                raise TypeError("v_range must be a dictionary or a path to an appropriate file.")

    def get_rms_cal_info(self):
        # extract rms/cal info
        if self._rms_cal_user is not None:
            if isinstance(self._rms_cal_user, dict):
                if '*' in self._rms_cal_user:
                    self._rms_cal_user = {str(tag): {'*': self._rms_cal_user['*']} for tag in self.tag_list}

                if len(self.tag_list) == 1 and str(self.tag_list[0]) not in self._rms_cal_user:
                    # only one species and tag not given => "reformat" dictionary to contain tag
                    self._rms_cal_user = {str(self.tag_list[0]): self._rms_cal_user}

            elif isinstance(self._rms_cal_user, str):
                self._rms_cal_user = read_noise_info(self._rms_cal_user)

            else:
                raise TypeError("chi2_info must be a dictionary or a path to an appropriate file.")

    def get_windows(self, verbose=True):
        if self.bandwidth is None:
            self.win_list = [Window(self.tr_list_by_tag[str(self.tag_list[0])][0], 1)]
        else:
            self.get_v_range_info()
            self.get_rms_cal_info()
            for tag, tr_list in self.tr_list_by_tag.items():
                win_list_tag = []  # first find all windows with enough data
                for i, tr in enumerate(tr_list):
                    f_range_plot = [velocity_to_frequency(v, tr.f_trans_mhz, vref_kms=self.vlsr_file)
                                    for v in [-1. * self.bandwidth / 2 + self.vlsr_plot,
                                              1. * self.bandwidth / 2 + self.vlsr_plot]]
                    f_range_plot.sort()
                    x_win, y_win = None, None
                    if self.x_file is not None:
                        x_win, y_win = select_from_ranges(self.x_file, f_range_plot, y_values=self.y_file)
                        if len(x_win) == 0 or len(set(y_win)) == 1:
                            continue
                    win = Window(tr, len(win_list_tag) + 1)
                    win.x_file, win.y_file = x_win, y_win
                    win_list_tag.append(win)

                nt = len(win_list_tag)
                if verbose or verbose == 2:
                    print('{} : {}/{} transitions found with enough data within thresholds'.format(tag, nt,
                                                                                                   len(tr_list)))
                if verbose == 2:
                    for iw, w in enumerate(win_list_tag):
                        print('  {}. {}'.format(iw + 1, w.transition))

                # if self._v_range_user is not None and self._rms_cal_user is not None:
                if (all([self._v_range_user is not None, self._rms_cal_user is not None])
                        and (tag in self._v_range_user or '*' in self._v_range_user)
                        and (tag in self._rms_cal_user or '*' in self._rms_cal_user)):
                    v_range = expand_dict(self._v_range_user[tag], nt)
                    rms_cal = expand_dict(self._rms_cal_user[tag], nt)
                    for win in win_list_tag:
                        win_num = win.plot_nb
                        if win_num in v_range and win_num in rms_cal:
                            win.v_range_fit = v_range[win_num]
                            win.rms = rms_cal[win_num][0]
                            # if self.jypb is not None:
                            #     win.rms_mk *= self.jypb[find_nearest_id(self.x_file, win.transition.f_trans_mhz)]
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
            self.x_fit = concatenate([w.x_fit for w in self.win_list_fit], axis=None)
            self.y_fit = concatenate([w.y_fit for w in self.win_list_fit], axis=None)
            # x_mod = select_from_ranges(self.x_file, self.line_list['fmhz_range_plot'], oversampling=self.oversampling)
            # self.x_mod = x_mod[0]  # TODO: why do I have to do this?
        else:
            self.x_fit = None

        return self.win_list


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
        self.transition_list = get_transition_list(DATABASE_SQL, self.species_list, fmhz_ranges, **thresholds)
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

    # def assign_spectrum(self, spec: SimpleSpectrum):
    #     self.model_spec = spec


class Species:
    def __init__(self, tag, ntot=7.0e14, tex=100., fwhm=FWHM_DEF, component=None):
        # super().__init__(self)
        self._tag = str(tag)  # make sure tag is stored as a string
        self._ntot = create_parameter('ntot_{}'.format(tag), ntot)  # total column density [cm-2]
        self._fwhm = create_parameter('fwhm_{}'.format(tag), fwhm)  # line width [km/s]

        self._tex = tex  # excitation temperature [K]
        self._component = component
        if component is not None:
            self.set_component(component.name)

        sp_dic = get_species_info(DATABASE_SQL, self._tag)
        if sp_dic is None:
            raise IndexError("Tag {} not found in the database.".format(self._tag))
        self._id = sp_dic['id']
        self._name = sp_dic['name']
        self._database = sp_dic['database_name']

        self.pf = get_partition_function(DATABASE_SQL, self._tag)  # (tref, qlog)

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
        tmp = interp(log10(tex), log10(self.pf[0]), self.pf[1])
        # tmp = find_nearest_id(self.pf[0], tex)
        # tmp = self.pf[1][tmp]
        qex = power(10., tmp)
        return get_partition_function_tex(self.pf[0], self.pf[1], tex)


class Transition:
    def __init__(self, tag, f_trans_mhz, aij, elo_cm, gup, name='', f_err_mhz=None, db_id=None, qn=None):
        self.f_trans_mhz = f_trans_mhz
        self.f_err_mhz = f_err_mhz
        self.aij = aij
        self.elo_cm = elo_cm
        self.elo_J = self.elo_cm * H * C_LIGHT * 100
        self.eup_J = self.elo_J + self.f_trans_mhz * 1.e6 * H
        self.gup = gup
        # self.eup = (elo_cm + self.f_trans_mhz * 1.e6 / (const.c.value * 100)) * 1.4389  # [K]
        self.eup = self.eup_J / K_B  # k_B in J/K
        self.tag = str(tag)  # make sure tag is stored as a string
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


class Window:
    def __init__(self, transition, plot_nb, v_range_fit=None, f_range_fit=None, rms=None, cal=None):
        self.transition = transition
        self.plot_nb = plot_nb
        self._name = "{} - {}".format(transition.tag, plot_nb)
        self._v_range_fit = v_range_fit
        self._f_range_fit = f_range_fit
        self._rms = rms
        self._cal = cal
        self._x_fit = None
        self._y_fit = None
        self._in_fit = False
        self._x_mod = None
        self._y_mod = None
        self._x_file = None
        self._y_file = None
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
    def rms(self):
        return self._rms

    @rms.setter
    def rms(self, value):
        self._rms = value

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

    @property
    def x_mod(self):
        return self._x_mod

    @x_mod.setter
    def x_mod(self, value):
        self._x_mod = value

    @property
    def y_mod(self):
        return self._y_mod

    @y_mod.setter
    def y_mod(self, value):
        self._y_mod = value

    @property
    def x_file(self):
        return self._x_file

    @x_file.setter
    def x_file(self, value):
        self._x_file = value

    @property
    def y_file(self):
        return self._y_file

    @y_file.setter
    def y_file(self, value):
        self._y_file = value


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
        return Parameter(name, **parameter_infos(**param))

    elif isinstance(param, Parameter):
        return param

    else:
        raise TypeError(f"{name} must be a float, an integer, a dictionary or an instance of the Parameter class.")


def get_species_info(database, species):
    tag = species.tag if isinstance(species, Species) else str(species)
    # retrieve infos from catdir :
    res_catdir = database.execute("SELECT * FROM catdir WHERE speciesid = {}".format(tag))
    cols_sp_info = [t[0] for t in res_catdir.description]
    all_rows = database.fetchall()
    if len(all_rows) == 0:
        print(f"{tag} not found in the database.")
        return None
    sp_info = all_rows[-1]
    sp_info_dic = dict(zip(cols_sp_info, sp_info))

    # retrieve database name :
    database.execute("SELECT name FROM cassis_databases WHERE id = {}".format(sp_info_dic["id_database"]))
    sp_info_dic["database_name"] = database.fetchall()[0][0]

    return sp_info_dic


def get_transition_list(database, species, fmhz_ranges, return_type='dict', **thresholds):
    species_list = [species] if type(species) is not list else species  # make sure it is a list
    if isinstance(species_list[0], Species):
        tag_list = [sp.tag for sp in species_list]
    else:
        tag_list = [str(sp) for sp in species_list]

    transition_dict = {}
    for tag in tag_list:  # NB: tag should be a string
        if thresholds:
            eup_min = thresholds[tag].get('eup_min', EUP_MIN_DEF)
            eup_max = thresholds[tag].get('eup_max', EUP_MAX_DEF)
            aij_min = thresholds[tag].get('aij_min', AIJ_MIN_DEF)
            aij_max = thresholds[tag].get('aij_max', AIJ_MAX_DEF)
            err_max = thresholds[tag].get('err_max', ERR_MAX_DEF)
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
            transition_dict[tag] = transition_list

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


def select_transitions(tran_df, thresholds=None, xrange=None, return_type=None, bright_lines_only=False, vlsr=None):
    def is_selected(tran, sp_thresholds):
        constraints = []
        for key, val in sp_thresholds.items():
            attr = key.rsplit('_', maxsplit=1)[0]
            if bright_lines_only and key == 'eup_min':
                val = 0
            if bright_lines_only and key == 'aij_max':
                val = None
            if 'min' in key:
                constraints.append(val <= getattr(tran, attr))
            if 'max' in key and val is not None:
                constraints.append(val >= getattr(tran, attr if key != 'err_max' else 'f_err_mhz'))
        return True if all(constraints) else False

    if not isinstance(tran_df, pd.DataFrame):
        raise TypeError("First argument must be a DataFrame.")

    if vlsr is not None:
        if xrange is not None:
            xrange = [x + delta_v_to_delta_f(vlsr, sum(xrange)/len(xrange)) for x in xrange]
        else:
            print("INFO - No frequency range specified, ignoring the vlsr keyword.")

    if thresholds is None:
        thresholds = {}

    if len(thresholds) == 0:
        selected = tran_df
    else:
        indices = []
        for tag, thres in thresholds.items():
            if xrange is not None:
                thres['f_trans_mhz_min'] = min(xrange)
                thres['f_trans_mhz_max'] = max(xrange)

            for row in tran_df[tran_df.tag == tag].iterrows():
                tr = row[1].transition
                if is_selected(tr, thres):
                    indices.append(row[0])
        selected = tran_df.loc[indices]

    if return_type is not None and return_type != 'df':
        print("Not implemented, returning dataframe.")
        # return {str(tag): list(selected[selected.tag == tag].transition) for tag in tag_list}
    return selected


def get_species_thresholds(sp_threshold_infos, select_species=None, return_list_sp=True):
    sp_thresholds = {}
    list_species = []

    if sp_threshold_infos is type(list):
        for other_sp in sp_threshold_infos:
            if isinstance(other_sp, list):  # if list, assume it is : [tag, eup_max, aij_min, err_max]
                new_sp = str(other_sp[0])  # make sure the tag is a string
                new_th = {
                    'eup_max': other_sp[1] if other_sp[1] != '*' else EUP_MAX_DEF,
                    'aij_min': other_sp[2] if other_sp[2] != '*' else AIJ_MIN_DEF,
                    'err_max': other_sp[3] if other_sp[3] != '*' else ERR_MAX_DEF
                }
            else:
                new_sp = other_sp
                new_th = THRESHOLDS_DEF
            list_species.append(new_sp)
            sp_thresholds[str(new_sp)] = new_th

    elif isinstance(sp_threshold_infos, dict):
        sp_thresholds = {str(key): val for key, val in sp_threshold_infos.items()}  # make sure tag is a string
        list_species = list(sp_threshold_infos.keys())

    elif os.path.isfile(sp_threshold_infos):
        df = pd.read_csv(sp_threshold_infos, delimiter='\t', comment='#', index_col=False, dtype=str)
        col_names = df.columns[1:]
        list_species = [t.strip() for t in df.tag]
        for index, row in df.iterrows():
            sp_thresholds[str(int(row.tag))] = {c: float(row[c]) for c in col_names if '*' not in row[c]}

    else:
        raise TypeError("other_species should be a list, a dictionary or a path to a file.")

    if len(sp_thresholds) > 0:  # if not empty, make sure add default thresholds if necessary
        for sp in sp_thresholds.keys():
            sp_thresholds[sp] = {label: sp_thresholds[sp].get(label, value)
                                 for label, value in THRESHOLDS_DEF.items()}

    if select_species is not None:
        if not isinstance(select_species, list):
            select_species = list(select_species)
        sp_thresholds = {str(sp): sp_thresholds[str(sp)] for sp in select_species}
        list_species = select_species

    if return_list_sp:
        return list_species, sp_thresholds
    else:
        return sp_thresholds

def get_partition_function(db, tag, temp=None):
    tref = []
    qlog = []
    pf_file = os.path.join(PARTITION_FUNCTION_DIR, '{}.txt'.format(tag))
    if os.path.isfile(pf_file):
        tref, qlog = genfromtxt(pf_file, comments='//', unpack=True)
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
        return power(10., qlog[0])
    if temp >= tref[-1]:
        print(f'{temp} is above the highest temperature of the partition function ({tref[-1]}) : '
              f'setting Q({temp}K)=Q({tref[-1]}K)')
        return power(10., qlog[-1])
    for i, t in enumerate(tref[:-1]):
        if tref[i+1] >= temp >= t:
            tmp = interp(log10(temp), log10(tref[i:i+2]), qlog[i:i+2])
            qex = power(10., tmp)
            # qex = np.interp(temp, tref, qlin)
            return qex
            # return np.power(10., qlog[find_nearest_id(np.array(tref),temp)])
