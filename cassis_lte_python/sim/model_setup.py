from cassis_lte_python.utils.utils import velocity_to_frequency,\
    open_data_file, is_in_range, select_from_ranges, find_nearest, expand_dict, read_noise_info
from cassis_lte_python.utils.constants import C_LIGHT, K_B, TEL_DIAM
from cassis_lte_python.utils.observer import Observable
from cassis_lte_python.database.constantsdb import THRESHOLDS_DEF
from cassis_lte_python.database.species import Species, get_species_thresholds
from cassis_lte_python.database.transitions import get_transition_df, select_transitions
from cassis_lte_python.sim.parameters import create_parameter
from cassis_lte_python.utils.settings import TELESCOPE_DIR, VLSR_DEF, SIZE_DEF
import os
import pandas as pd
from numpy import concatenate, loadtxt, array, interp, ndarray, mean, floor, ceil, float32, linspace, inf


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
        self._rms_cal = None
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
        self.get_tuning_info()

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

        if self.x_file is not None:
            x_mod = []
            for i, row in self.tuning_info.iterrows():
                x_sub = self.x_file[(self.x_file >= row['fmhz_min']) & (self.x_file <= row['fmhz_max'])]
                x_mod.extend(linspace(min(x_sub), max(x_sub), num=self.oversampling * len(x_sub)))
            self.x_mod = array(x_mod)

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
        elif isinstance(self.cont_info, str) and os.path.isfile(self.cont_info):
            # cont_info is a CASSIS continuum file : MHz [tab] K
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

    def get_tuning_info(self, config=None, check_tel_range=False):
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

    def get_linelist(self, config=None, check_tel_range=False):
        # if config is None:
        #     config = self._configuration_dict

        self.line_list_all = get_transition_df(self.tag_list, self.tuning_info['fmhz_range'])
        tr_list_tresh = select_transitions(self.line_list_all, thresholds=self.thresholds)
        # tr_list_tresh = get_transition_df(self.tag_list, self.tuning_info['fmhz_range'], **self.thresholds)
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
        if self.tr_list_by_tag is None:
            raise ValueError("Missing transition list.")

        # extract rms/cal info
        if self._rms_cal_user is not None:
            if isinstance(self._rms_cal_user, dict):
                if next(iter(self._rms_cal_user))[0] == '[':  # info by frequency range
                    frange, fmin, fmax, rms, cal = [], [], [], [], []
                    for k, v in self._rms_cal_user.items():
                        k = k.strip('[').strip(']').strip()  # remove brackets and spaces ; could be improved
                        k = k.split(',')
                        frange.append([float(elt) for elt in k])
                        fmin.append(float(k[0]))
                        fmax.append(float(k[1]))
                        rms.append(float(v[0]))
                        cal.append(float(v[1]))
                    self._rms_cal = pd.DataFrame({'freq_range': frange,
                                                  'fmin': fmin,
                                                  'fmax': fmax,
                                                  'rms': rms,
                                                  'cal': cal})

                else:
                    if '*' in self._rms_cal_user:
                        self._rms_cal_user = {str(tag): {'*': self._rms_cal_user['*']} for tag in self.tag_list}

                    if len(self.tag_list) == 1 and str(self.tag_list[0]) not in self._rms_cal_user:
                        # only one species and tag not given => "reformat" dictionary to contain tag
                        self._rms_cal_user = {str(self.tag_list[0]): self._rms_cal_user}

                    tup, rms, cal = [], [], []
                    for tag, chi2_info in self._rms_cal_user.items():
                        for k, v in expand_dict(chi2_info).items():
                            tup.append((tag, k))
                            rms.append(float(v[0]))
                            cal.append(float(v[1]))
                    self._rms_cal = pd.DataFrame({'win_id': tup,
                                                  'rms': rms,
                                                  'cal': cal})

            elif isinstance(self._rms_cal_user, str):
                # TODO: TBC
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
                        if len(x_win) < 5 or len(set(y_win)) == 1:
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
                        and (tag in self._v_range_user or '*' in self._v_range_user)):
                        # and (tag in self._rms_cal_user or '*' in self._rms_cal_user)):
                    v_range = expand_dict(self._v_range_user[tag], nt)
                    # rms_cal = expand_dict(self._rms_cal_user[tag], nt)
                    for win in win_list_tag:
                        win_num = win.plot_nb
                        if 'freq_range' in self._rms_cal.columns:
                            rms_cal = self._rms_cal[(win.transition.f_trans_mhz > self._rms_cal['fmin'])
                                                    & (win.transition.f_trans_mhz < self._rms_cal['fmax'])]
                        else:
                            rms_cal = self._rms_cal[self._rms_cal['win_id'] == (tag, win_num)]
                            if len(rms_cal) == 0:
                                rms_cal = self._rms_cal[self._rms_cal['win_id'] == (tag, '*')]
                        if len(rms_cal) == 0:
                            raise IndexError(f"rms/cal info not found for {win.transition}.")

                        if win_num in v_range:
                            win.v_range_fit = v_range[win_num]
                            win.rms = rms_cal['rms'].values[0]
                            # if self.jypb is not None:
                            #     win.rms_mk *= self.jypb[find_nearest_id(self.x_file, win.transition.f_trans_mhz)]
                            win.cal = rms_cal['cal'].values[0]
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
        self.transition_list = get_transition_df(self.species_list, fmhz_ranges, **thresholds)
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


class Window:
    def __init__(self, transition, plot_nb, v_range_fit=None, f_range_fit=None, rms=None, cal=None):
        self.transition = transition
        self.plot_nb = plot_nb
        self._name = "{} - {}".format(transition.tag, plot_nb)
        self._v_range_fit = v_range_fit
        self._f_range_fit = f_range_fit
        self._v_range_plot = None
        self._f_range_plot = None
        self._rms = rms
        self._cal = cal
        self._x_fit = None
        self._y_fit = None
        self._in_fit = False
        self._x_mod = None
        self._y_mod = None
        self.y_mod_cpt = []
        self.bottom_unit = 'MHz'
        self.top_unit = 'MHz'
        self.bottom_lim = None
        self.top_lim = None
        self.x_mod_plot = None
        self.x_file_plot = None
        self._x_file = None
        self._y_file = None
        self._y_res = None
        self.other_species_selection = None
        self.main_lines_display = {}
        self.other_lines_display = {}
        self.other_species_display = pd.DataFrame()
        self._y_min = inf
        self._y_max = -inf

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
    def v_range_plot(self):
        return self._v_range_plot

    @v_range_plot.setter
    def v_range_plot(self, value):
        self._v_range_plot = value

    @property
    def f_range_plot(self):
        return self._f_range_plot

    @f_range_plot.setter
    def f_range_plot(self, value):
        self._f_range_plot = value

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
        self.y_min = min([self._y_min, min(self._y_mod)])
        self.y_max = max([self._y_max, max(self._y_mod)])

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
        self.y_min = min([self._y_min, min(self._y_file)])
        self.y_max = max([self._y_max, max(self._y_file)])

    @property
    def y_res(self):
        return self._y_res

    @y_res.setter
    def y_res(self, value):
        self._y_res = value

    @property
    def y_min(self):
        return self._y_min

    @y_min.setter
    def y_min(self, value):
        self._y_min = value

    @property
    def y_max(self):
        return self._y_max

    @y_max.setter
    def y_max(self, value):
        self._y_max = value
