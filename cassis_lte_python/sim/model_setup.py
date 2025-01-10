from __future__ import annotations

import io

from cassis_lte_python.utils import utils
from cassis_lte_python.utils.observer import Observable
from cassis_lte_python.database.constantsdb import THRESHOLDS_DEF
from cassis_lte_python.database.species import Species, get_species_thresholds
from cassis_lte_python.database.transitions import get_transition_df, select_transitions
from cassis_lte_python.sim.parameters import create_parameter, parameter_infos
from cassis_lte_python.utils.settings import VLSR_DEF, SIZE_DEF, NROWS_DEF, NCOLS_DEF
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import warnings


class ModelConfiguration:
    def __init__(self, configuration, verbose=True, check_tel_range=False):
        self._configuration_dict = configuration

        self.modeling = configuration.get('modeling', False)
        self.minimize = configuration.get('minimize', False)
        self.line_analysis = configuration.get('line_analysis', 'inspect' in configuration)

        self.jparams = configuration.get('params', None)
        self.jmodel_fit = configuration.get('model_fit', None)

        self.species_infos = None
        if 'species_infos' in configuration:
            self.species_infos = utils.read_species_info(configuration['species_infos'], header=0)

        self.species_dict = configuration.get('species_dict', {})

        self.fwhm_max = 0.
        self.tag_list = configuration.get('inspect', [])
        self.cpt_list = []
        self.comp_config_file = None
        if 'components' in configuration:
            self.comp_config_file = configuration['components'].get('config', None)
            self.get_components(configuration['components'])

        self.output_dir = configuration.get('output_dir', os.path.curdir)
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        self.data_file = configuration.get('data_file', None)
        self.data_file_obj = None
        self.xunit = 'MHz'
        yunit = configuration.get('yunit', 'K')
        self.yunit = yunit.strip()
        self._x_file = []
        self._y_file = []
        self.x_obs = configuration.get('x_obs', None)
        self.y_obs = configuration.get('y_obs', None)
        self.vlsr_file = 0.
        self.vlsr_plot = configuration.get('vlsr_plot', 0.)
        self.line_shift_kms = configuration.get('line_shift_kms', 0.)
        self._cont_info = configuration.get('tc', 0.)
        self._tc = None
        self.cont_free = configuration.get('continuum_free', False)
        self.bl_corr = configuration.get('baseline_corr', False)
        if self.bl_corr and not self.cont_free:
            raise ValueError("'baseline_corr' can only be True if 'continuum_free' is also True.")
        self._telescope_data = {}
        self.t_a_star = configuration.get('t_a*', False)
        self.tmb2ta = None
        if 'beam_info' in configuration:
            self.get_jypb(configuration)
        self.jypb = None
        self.bmaj = None
        self.bmin = None
        self.beam = {'bmaj': 0, 'bmin': 0}

        self.fit_full_range = False
        self.fit_freq_except = []
        self._fit_freq_except_user = configuration.get('fit_freq_except', None)
        if self._fit_freq_except_user is not None:
            self.fit_full_range = True
            if isinstance(self._fit_freq_except_user, str):
                fmin, fmax = np.loadtxt(self._fit_freq_except_user, unpack=True, comments='#')
                if len(fmin) > 0 and len(fmax) > 0:
                    self.fit_freq_except = [[f1, f2] for f1, f2 in zip(fmin, fmax)]
            elif isinstance(self._fit_freq_except_user, list):
                if len(self._fit_freq_except_user) == 2 and not isinstance(self._fit_freq_except_user[0], list):
                    # "simple" list -> make it list of lists
                    self.fit_freq_except = [self._fit_freq_except_user]
                else:
                    self.fit_freq_except = self._fit_freq_except_user
            else:
                raise TypeError("fit_freq_except must be a list of ranges or a path to an appropriate file.")

        self._v_range_user = configuration.get('v_range', None)
        self._rms_cal_user = configuration.get('rms_cal', None)
        if 'chi2_info' in configuration:
            warnings.warn("The chi2_info keyword is deprecated, please use the keyword rms_cal instead.")
            self._rms_cal_user = configuration.get('chi2_info', None)
        self._rms_cal = None
        self.win_list = []
        self.win_list_fit = []
        self.win_list_plot = []
        self.win_list_gui = []
        self.win_list_file = []

        self.x_fit = None
        self.y_fit = None
        self.x_mod = configuration.get('x_mod', None)
        self.y_mod = None

        self.dfmhz = configuration.get('df_mhz', 0.1)

        if self.line_analysis and 'franges_ghz' not in configuration and 'tuning_info' not in configuration:
            raise KeyError("You must provide a frequency range with the 'franges_ghz' keyword.")
        if any([key in configuration for key in ['fmin_ghz', 'fmax_ghz']]):
            print("The keywords 'fmin_ghz' and 'fmax_ghz' are deprecated.")
            if self.minimize or self.modeling:
                print("The model will be computed over the ranges in 'tuning_info'.")
            else:
                print("Data selection performed with 'tuning_info' or 'franges_ghz', whichever is provided.")

        self.franges_mhz = []
        if 'tuning_info' in configuration:  # mandatory if model or fit
            self._tuning_info_user = configuration.get('tuning_info')
            self.tuning_info = []
            self.get_tuning_info()
        else:
            if self.modeling or self.minimize:
                raise KeyError("'tuning_info' is mandatory when modeling or minimizing.")

        franges_ghz = configuration.get('franges_ghz', [])
        if len(franges_ghz) > 0:
            # make sure it is a list of lists
            if not isinstance(franges_ghz[0], list):
                franges_ghz = [franges_ghz]

            if 'tuning_info' in configuration:
                print("INFO - 'franges_ghz' supersedes the ranges in 'tuning_info' for model computation")
                print("       make sure the ranges in tuning_info are wider than those in 'franges_ghz'")
            self.franges_mhz = [[min(r) * 1000, max(r) * 1000] for r in franges_ghz]

        self.fmin_mhz = min(self.franges_mhz[0])
        self.fmax_mhz = max(self.franges_mhz[-1])

        self.mask = []

        if self.x_mod is None and self.data_file is None:
            # self.x_mod = np.arange(self.fmin_mhz, self.fmax_mhz + self.dfmhz, self.dfmhz)
            self.x_mod = np.array([])
            for r in franges_ghz:
                self.franges_mhz.append([min(r) * 1000, max(r) * 1000])
                self.x_mod = np.concatenate([self.x_mod,
                                             np.arange(min(r) * 1000, max(r) * 1000 + self.dfmhz / 2, self.dfmhz)],
                                            dtype=np.float32)
            self.fmin_mhz = min(self.x_mod)
            self.fmax_mhz = max(self.x_mod)

        noise = 0.
        if 'noise' in configuration:
            noise = configuration.get('noise', 0.)
        if 'noise_mk' in configuration:
            noise = 1.e-3 * configuration.get('noise_mk', 0.)
        if isinstance(noise, (float, int)):
            self.noise = lambda x: noise
        elif isinstance(noise, list):  # TODO : check if this is used
            yvals = np.array([[n, n] for n in noise])
            self.noise = interp1d(np.concatenate(self.franges_mhz), np.concatenate(yvals), kind='nearest')
        else:
            print('Noise format not supported. Should be a integer, a float or a list.')

        self.f_err_mhz_max = configuration.get('f_err_mhz_max', None)

        if 'thresholds' in configuration and configuration['thresholds'] is not None:
            self.thresholds = get_species_thresholds(configuration['thresholds'],
                                                     select_species=self.tag_list,
                                                     return_list_sp=False)
        elif self.species_infos is not None:
            cols_thresholds = self.species_infos[[col for col in self.species_infos.columns if not col.startswith('c')]]
            sp_thresholds = {}
            for index, row in self.species_infos.iterrows():
                sp_thresholds[str(index)] = {c: row[c] for c in cols_thresholds if row[c] != '*'}
            self.thresholds = get_species_thresholds(sp_thresholds,
                                                     select_species=self.tag_list,
                                                     return_list_sp=False)
        else:
            self.thresholds = {}
            for tag in self.tag_list:
                self.thresholds[str(tag)] = THRESHOLDS_DEF

        self.sort = configuration.get('sort', 'frequency')
        sort_parameters = ['frequency', 'eup', 'aij']
        if self.sort not in sort_parameters:
            print("Sort should be one of the following :", ", ".join(sort_parameters))
        self.line_list_all = None
        self.tr_list_by_tag = None  # w/i thresholds

        self.snr_threshold = configuration.get('snr_threshold', None)

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

        self.constraints = configuration.get('constraints', None)
        if self.constraints is not None:
            constraints_dict_user = self.constraints
            constraints_dict = {}
            symbols = ['+', '*', '/', '-']
            for nc in range(1, len(self.cpt_list) + 1):
                for key, val in constraints_dict_user.items():
                    if key.startswith('c'):  # key already has component info, keep and go to the next constraint
                        constraints_dict[key] = val
                        continue
                    constraints_dict[f'c{nc}_{key}'] = f'c{nc}_{val}'  # assume no mathematical expression
                    for symb in symbols:
                        if symb in val:  # we have a mathematical expression
                            elts = val.split(symb)
                            for i, elt in enumerate(elts):
                                if not elt[0].isdigit() and not elt[0].startswith('c'):
                                    elts[i] = f'c{nc}_{elt}'
                            val = symb.join(elts)
                        constraints_dict[f'c{nc}_{key}'] = val

            self.constraints = constraints_dict

        self.ref_pixel_info = None
        self.latest_valid_params = None
        self.tau_lim = configuration.get('tau_lim', np.inf)
        self.max_iter = configuration.get('max_iter', None)
        self.fit_kws = configuration.get('fit_kws', None)
        self.print_report = configuration.get('print_report', True)

        # outputs other than plots
        self.base_name = configuration.get('base_name', configuration.get('name_lam', 'lte_model'))
        self.save_configs = configuration.get('save_configs', True) or configuration.get('save_res_configs', True)
        self._name_lam = configuration.get('name_lam', self.base_name)  # do not add extension here
        self._name_config = configuration.get('name_config', self.base_name + '_config.txt')
        self.save_results = configuration.get('save_results', True) or configuration.get('save_res_configs', True)
        self.save_infos_components = configuration.get('save_infos_components', True)
        save_model = configuration.get('save_res_configs', True)
        if 'save_spec' in configuration:
            warnings.warn("'save_spec' is deprecated, use 'save_model' instead.")
            save_model = configuration['save_spec']
        self.save_model_spec = configuration.get('save_model', save_model)
        self.save_obs_spec = configuration.get('save_obs', False)
        self._file_spec = configuration.get('file_spec', 'synthetic_spectrum.txt')
        output_files_def = {}
        if self.save_configs:
            output_files_def['lam'] = self._name_lam
            output_files_def['config'] = self._name_config
        if self.save_results:
            output_files_def['results'] = self.base_name + '_fit_res.txt'
        if self.save_obs_spec:
            output_files_def['obs'] = self.base_name + '_obs.txt'
        if self.save_model_spec:
            output_files_def['model'] = self.base_name + '_model.txt'

        self._output_files = configuration.get('output_files', output_files_def)

        # Default plot keywords :
        self.plot_kws = {
            'tag': None,
            'display_all': True,
            'windows': {},
            'verbose': True,
            'basic': False,
            'other_species': None,
            'other_species_plot': 'all',
            'other_species_win_selection': None,
            'model_err': False,
            'component_err': False
        }
        self.user_plot_kws = configuration.get('plot_kws', {})
        new_plot_kws = {}
        if 'gui+file' in self.user_plot_kws:
            self.plot_kws.update(self.user_plot_kws.get('gui+file', {}))
        else:
            self.plot_kws.update(self.user_plot_kws)  # old config
            new_plot_kws = {'gui+file': self.user_plot_kws.copy()}
        # Make sure 'tag' is a list of strings :
        tag = self.plot_kws['tag']
        if tag is not None:
            if not isinstance(tag, list):
                tag = [tag]
            self.plot_kws['tag'] = [str(t) for t in tag]

        kws_plot_only = ['other_species']

        self.plot_gui = configuration.get('plot_gui', True)  # do gui plot by default
        self.gui_kws = self.plot_kws.copy()  # Default gui keywords = plot_kws
        gui_kws = self.user_plot_kws.get('gui_only', {})
        if 'gui_kws' in configuration:
            gui_kws = configuration.get('gui_kws')
            new_plot_kws['gui_only'] = gui_kws
        for k in kws_plot_only:
            if k in gui_kws.keys():
                # print(f'N.B. : {k} in gui keywords is not used.')
                gui_kws.pop(k)
        self.gui_kws.update(gui_kws)

        self.plot_file = configuration.get('plot_file', False)
        self.file_kws = self.plot_kws.copy()  # Default file keywords = plot_kws
        self.file_kws.update({'nrows': NROWS_DEF, 'ncols': NCOLS_DEF})
        file_kws = self.user_plot_kws.get('file_only', {})
        if 'file_kws' in configuration:
            file_kws = configuration.get('file_kws')
            new_plot_kws['file_only'] = file_kws
        if 'plot_filename' in configuration:
            file_kws['filename'] = configuration['plot_filename']
        # if self.plot_file and 'filename' not in file_kws:
        #     raise NameError("Please provide a name for the output pdf file.")
        for k in kws_plot_only:
            if k in file_kws.keys():
                # print(f'N.B. : {k} in file keywords is not used.')
                file_kws.pop(k)
        self.file_kws.update(file_kws)

        if len(new_plot_kws) > 0:
            self.user_plot_kws = new_plot_kws

        self.exec_time = configuration.get('exec_time', True)

        if 'data_file' in self._configuration_dict or 'x_obs' in self._configuration_dict:
            self.get_data()

        if self.vlsr_plot == 0. and 'components' in configuration:
            self.vlsr_plot = self.cpt_list[0].vlsr

        if self.vlsr_file == 0 and self.line_shift_kms == 0:
            print("Your data seem to be in sky frequency : "
                  "you should provide the 'line_shift_kms' to ensure adequate search of transitions.")
            shift = max([cpt.vlsr for cpt in self.cpt_list])
            ans = input(f"Do you want to use the largest Vlsr found in the components' starting values "
                        f"({shift})? [Y/n] : ")
            if ans.strip() in ['y', 'Y', '']:
                self.line_shift_kms = shift

        if 'tc' in self._configuration_dict:
            self.get_continuum()

        if len(self.x_file) > 0:
            self.get_linelist()
            self.get_windows()
        if self._v_range_user is not None:
            self.win_nb_fit = {}
            self.get_velocity_ranges()
        if self.minimize or self.modeling:
            self.get_data_to_fit()

    def get_data(self, config=None):
        if config is None:
            config = self._configuration_dict

        self.data_file = config.get('data_file', None)
        if 'x_obs' in config:
            self.x_file = config['x_obs']
        if 'y_obs' in config:
            self.y_file = config['y_obs']
        self.vlsr_file = config.get('vlsr_obs', 0.)
        if self.data_file is not None and len(self.x_file) == 0:
            self.data_file_obj = utils.DataFile(self.data_file)
            self.x_file, self.y_file = self.data_file_obj.xdata_mhz, self.data_file_obj.ydata
            self.vlsr_file = self.data_file_obj.vlsr
            self.yunit = self.data_file_obj.yunit
        if len(self.y_file) == 0 and len(self.x_file) > 0:
            self.y_file = np.random.rand(len(self.x_file))  # random y-values just to be able to set up windows.

        self.vlsr_plot = self.vlsr_file

        if len(self.x_file) > 0 and isinstance(self.x_file[0], np.ndarray):
            self.x_file = np.concatenate(self.x_file)
        if len(self.y_file) > 0 and isinstance(self.y_file[0], np.ndarray):
            self.y_file = np.concatenate(self.y_file)

        if len(self.x_file) > 0:
            idx = self.x_file.argsort()
            self.x_file = self.x_file[idx]
            self.y_file = self.y_file[idx]

        if len(self.x_file) > 0:
            # first check if some ranges are below/above the data's min/max :
            frange_mhz = []
            for frange in self.franges_mhz:
                if frange[1] < min(self.x_file):
                    print(f"No data in {frange} (data start at {min(self.x_file)}) - skipping this range")
                elif frange[0] > max(self.x_file):
                    print(f"No data in {frange} (data end at {max(self.x_file)}) - skipping this range")
                else:
                    frange_mhz.append(frange)
            if len(frange_mhz) == 0:
                raise ValueError("No frequency ranges selected ; check your values.")
            self.franges_mhz = frange_mhz
            # modify min of first range to be the greatest value between itself and min(x_file)
            self.franges_mhz[0][0] = max(min(self.x_file), self.franges_mhz[0][0])
            # modify max of last range to be the small value between itself and max(x_file)
            self.franges_mhz[-1][-1] = min(max(self.x_file), self.franges_mhz[-1][-1])
            # if necessary, cut out data below/above min/max frange_mhz
            self.y_file = self.y_file[(self.x_file >= self.franges_mhz[0][0]) & (self.x_file <= self.franges_mhz[-1][-1])]
            self.x_file = self.x_file[(self.x_file >= self.franges_mhz[0][0]) & (self.x_file <= self.franges_mhz[-1][-1])]
            mask = np.full(len(self.x_file), False)
            for r in self.franges_mhz:
                mask[(min(r) <= self.x_file) & (self.x_file <= max(r))] = True

            # if self.fit_freq_except is not None:
            for r in self.fit_freq_except:
                mask[(min(r) <= self.x_file) & (self.x_file <= max(r))] = False

            self.mask = mask

            # x_sub, y_sub = utils.select_from_ranges(self.x_file, self.franges_mhz,
            #                                         y_values=self.y_file)
            # self.x_file, self.y_file = x_sub, y_sub

            if self.oversampling == 1:
                self.x_mod = self.x_file[self.mask]
            else:
                x_mod = []
                for rg in self.franges_mhz:
                    x_sub = self.x_file[(self.x_file >= min(rg)) & (self.x_file <= max(rg))]
                    if len(x_sub) == 0:
                        continue
                    x_mod.extend(np.linspace(min(x_sub), max(x_sub), num=self.oversampling * len(x_sub)))
                self.x_mod = np.array(x_mod)

    def get_jypb(self, config=None):
        if config is None:
            config = self._configuration_dict
        if 'beam_info' in config:
            f_beam = config['beam_info']['f_mhz']
            jypb2k = utils.compute_jypb2k(f_beam, config['beam_info']['beam_omega'])
            self.jypb = interp1d(f_beam, jypb2k, kind='nearest')

    def get_continuum(self, config=None):
        if isinstance(self.cont_info, (float, int)):
            self._tc = lambda x: self.cont_info
        elif isinstance(self.cont_info, str):
            try:
                # cont_info is a CASSIS continuum file : MHz [tab] K
                # f_cont, t_cont = np.loadtxt(self.cont_info, delimiter='\t', comments=['#', '//'], unpack=True)
                f_cont, t_cont = utils.open_continuum_file(self.cont_info)
                self._tc = interp1d(f_cont, t_cont, kind='linear')  # nearest??
            except FileNotFoundError:
                raise FileNotFoundError(f"{os.path.isfile(self.cont_info)} not found.")
        elif isinstance(self.cont_info, dict):
            # to compute continuum over ranges given by the user : { '[fmin, fmax]': value, ...}
            f_cont, t_cont = [], []
            for frange, val in self.cont_info.items():
                frange = frange.replace('[', ' ').replace(']', ' ')
                frange = [float(f) for f in frange.split(sep=',')]
                f_cont.extend(frange)
                t_cont.extend([val, val])
            self._tc = interp1d(f_cont, t_cont, kind='nearest')
        else:
            raise TypeError("Continuum must be a float, an integer or a 2-column tab-separated file (MHz K).")

    def get_tuning_info(self, config=None, check_tel_range=False):
        if config is None:
            config = self._configuration_dict

        # if 'tuning_info' in config:
        if check_tel_range:  # check telescope ranges cover all data / all model values:
            x_vals = self.x_file if len(self.x_file) > 0 else [min(self.x_mod), max(self.x_mod)]
            extended = False
            for x in x_vals:
                # is_in_range = [val[0] <= x <= val[1] for val in config['tuning_info'].values()]
                limits = list(config['tuning_info'].values())
                limits = [item for sublist in limits for item in sublist]
                if not utils.is_in_range(x, config['tuning_info'].values):
                    # raise LookupError("Telescope ranges do not cover some of the data, e.g. at {} MHz.".format(x))
                    extended = True
                    nearest = utils.find_nearest(np.array(limits), x)
                    for key, val in config['tuning_info'].items():
                        # new_lo = 5. * np.floor(x / 5.) if val[0] == nearest else val[0]
                        # new_hi = 5. * np.ceil(x / 5.) if val[1] == nearest else val[1]
                        new_lo = np.floor(x) if val[0] == nearest else val[0]
                        new_hi = np.ceil(x) if val[1] == nearest else val[1]
                        config['tuning_info'][key] = [new_lo, new_hi]
            if extended:
                print("Some telescope ranges did not cover some of the data ; ranges were extended to :")
                print(config['tuning_info'])

        tuning_info = {'fmhz_range': [], 'telescope': [], 'fmhz_min': [], 'fmhz_max': []}
        for key, val in config['tuning_info'].items():
            # make sure we have a list of frequency ranges
            if len(val) == 0:  # empty list, use entire data range
                arr = [[min(self.x_file), max(self.x_file)]]
            elif isinstance(val[0], list):
                arr = val
            else:
                arr = [val]
            # arr = np.array(val)  # convert list to array
            # if len(arr.shape) == 1:
            #     arr = np.array([val])
            for r in arr:
                r_min, r_max = min(r), max(r)
                # if len(self.x_file) > 0:
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
        self.franges_mhz = tuning_info['fmhz_range']

        for tel, freq_user in self._configuration_dict['tuning_info'].items():
            if isinstance(freq_user[0], list):
                freq_user = [el for li in freq_user for el in li]
            tel_info = utils.read_telescope_file(utils.search_telescope_file(tel),
                                                 fmin_mhz=min(freq_user),
                                                 fmax_mhz=max(freq_user))
            self._telescope_data[tel] = tel_info

        # tuning_data = pd.DataFrame()
        #
        # for i, row in self.tuning_info.iterrows():
        #     tel_data = self._telescope_data[row['telescope']]
        #     index_min = tel_data['Frequency (MHz)'].searchsorted(row['fmhz_min']) - 1
        #     # NB: searchsorted returns the index where value could be inserted and maintain order
        #     index_max = tel_data['Frequency (MHz)'].searchsorted(row['fmhz_max'])
        #     tel_data_sub = tel_data.loc[index_min: index_max]
        #     if tuning_data.empty:
        #         tuning_data = tel_data_sub
        #     else:
        #         tuning_data = pd.concat([tuning_data, tel_data_sub])

        tuning_data = pd.concat(list(self._telescope_data.values()))
        tuning_data = tuning_data.sort_values(by=['Frequency (MHz)'])

        if self.t_a_star:
            self.tmb2ta = interp1d(tuning_data['Frequency (MHz)'], tuning_data['Beff/Feff'])
        else:
            self.tmb2ta = lambda x: 1.

        self.beam = utils.beam_function(tuning_data)

        if self.jypb is None:
            self.jypb = lambda f: utils.compute_jypb2k(f, self.beam(f))

    def get_linelist(self, verbose=True):
        """
        Retrieving lists of transitions.
        :param verbose: if True, print number of transitions w/i thresholds
                        if 2, print
        :return: None

        self.line_list_all (dataframe): the list of all transitions (no thresholds)
        self.tr_list_by_tag (dictionary): the list of transitions (w/i thresholds if applies), for each tag
        If computing a model only or fitting by velocity, apply thresholds.
        """

        # Search for all transition between min/max of data or model
        if self.fit_full_range:
            line_list_all = get_transition_df(self.tag_list,
                                              # fmhz_ranges=[[self.fmin_mhz, self.fmax_mhz]],
                                              fmhz_ranges=self.franges_mhz,
                                              shift_kms=self.line_shift_kms)
        else:
            line_list_all = get_transition_df(self.tag_list,
                                              fmhz_ranges=[[self.fmin_mhz, self.fmax_mhz]],
                                              # fmhz_ranges=self.franges_mhz,
                                              shift_kms=self.line_shift_kms)
            print(f"INFO - {len(line_list_all)} transitions found (no thresholds) "
                  f"within {'data' if len(self.x_file) > 0 else 'model'}'s min/max.")
        line_list_all_by_tag = {tag: list(line_list_all[line_list_all.tag == tag].transition)
                                for tag in self.tag_list}

        if self.f_err_mhz_max is not None:
            line_list_all = line_list_all[line_list_all.f_err_mhz <= self.f_err_mhz_max]
            print(f"INFO - {len(line_list_all)} transitions found with f_err_mhz <= {self.f_err_mhz_max}.")

        self.line_list_all = select_transitions(line_list_all, xrange=self.franges_mhz)
        print(f"INFO - {len(self.line_list_all)} transitions found (no thresholds) "
              f"within tuning frequencies : {self.tuning_info['fmhz_range'].tolist()}.")
        self.tr_list_by_tag = {tag: list(self.line_list_all[self.line_list_all.tag == tag].transition)
                               for tag in self.tag_list}

        # get linelist w/i thresholds and tuning frequencies
        tr_list_tresh = select_transitions(line_list_all, thresholds=self.thresholds)
        if len(self.tuning_info) > 1:
            # more than one telescope range => search only in data within telescope ranges
            # NB : this assumes that if only one range, it encompasses the data's min/max
            tr_list_tresh = select_transitions(tr_list_tresh, xrange=self.franges_mhz)
            print(f"INFO - {len(tr_list_tresh)} transitions within thresholds and within tuning frequencies : "
                  f"{self.tuning_info['fmhz_range'].tolist()}")

        if self.sort == 'frequency':
            tr_list_tresh.sort_values('fMHz', inplace=True)
        else:
            tr_list_tresh.sort_values(self.sort, inplace=True)
        # for comparison with CASSIS look for number of transitions w/i min/max of data :
        # if len(self.x_file) > 0:
        #     print(f"{len(tr_list_tresh)} transitions within thresholds",
        #           f"and within data's min/max : [{min(self.x_file)}, {max(self.x_file)}].")
        # else:
        #     self.line_list_all = tr_list_tresh

        self.tr_list_by_tag = {tag: list(tr_list_tresh[tr_list_tresh.tag == tag].transition)
                               for tag in self.tag_list}
        if all([len(t_list) == 0 for t_list in self.tr_list_by_tag.values()]):
            raise LookupError("No transition found within the thresholds.")

        ltag = max([len(t) for t in self.tag_list])  # max length for tags
        lntr = max([len(l) for l in self.tr_list_by_tag.values()])  # max number of lines w/i thresholds
        lntr = len(str(lntr))  # max length of number of lines
        lntr_all = max([len(l) for l in line_list_all_by_tag.values()])  # max number of lines w/o thresholds
        lntr_all = len(str(lntr_all))  # max length of number of lines
        tags_no_tran = []
        for tag, tr_list in self.tr_list_by_tag.items():
            if len(tr_list) == 0:
                tags_no_tran.append(tag)
            thr_info = "within thresholds"
            if self.fit_full_range:
                thr_info = ""
                if self.f_err_mhz_max is not None:
                    thr_info = f"with f_err_mhz <= {self.f_err_mhz_max}"
            if verbose or verbose == 2:
                print(f'{tag:>{ltag}s} : {len(tr_list):{lntr}d}'
                      f' /{len(line_list_all_by_tag[tag]):{lntr_all}d} transitions found {thr_info}')
            if verbose == 2:
                for it, tr in enumerate(tr_list):
                    print('  {}. {}'.format(it + 1, tr))

        if len(tags_no_tran) > 0:
            print(f"WARNING - No transitions found for the following species : {', '.join(tags_no_tran)}"
                  f" ; removing it/them from the analysis.")
            for tag in tags_no_tran:
                self.tag_list.remove(tag)
                for cpt in self.cpt_list:
                    try:
                        cpt.tag_list.remove(tag)
                        cpt.species_list = [sp for sp in cpt.species_list if sp.tag != tag]
                    except ValueError:  # tag already not in component -> do nothing
                        pass
            cpt_no_sp = [cpt.name for _, cpt in enumerate(self.cpt_list) if len(cpt.tag_list) == 0]
            if len(cpt_no_sp) > 0:
                if len(cpt_no_sp) == 1:
                    message = (f"Component {cpt_no_sp[0]} does not have any species,"
                               f"please check your thresholds.")
                else:
                    message = (f"Components {', '.join(cpt_no_sp)} do not have any species, "
                               f"please check your thresholds.")
                raise LookupError(message)

            # recompute the linelist
            self.line_list_all = get_transition_df(
                self.tag_list, fmhz_ranges=self.franges_mhz,
                shift_kms=self.line_shift_kms
            )
            tmp_list = select_transitions(self.line_list_all, thresholds=self.thresholds)
            self.tr_list_by_tag = {tag: list(tmp_list[tmp_list.tag == tag].transition)
                                   for tag in self.tag_list}

        # for cpt in self.cpt_list:  # currently not used
        #     # cpt.transition_list = self.line_list_all[self.line_list_all['tag'].isin(cpt.tag_list)]
        #     cpt.transition_list = {key: val for key, val in self.tr_list_by_tag.items() if key in cpt.tag_list}

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
                self._v_range_user = utils.read_noise_info(self._v_range_user)

            else:
                raise TypeError("v_range must be a dictionary or a path to an appropriate file.")

    def get_rms_cal_info(self):
        if self.tr_list_by_tag is None:
            raise ValueError("Missing transition list.")

        # extract rms/cal info
        if self._rms_cal_user is not None:
            if isinstance(self._rms_cal_user, dict):
                # make sure we have floats:
                for key, val in self._rms_cal_user.items():
                    self._rms_cal_user[key] = [float(elt) for elt in val]
                if '*' in self._rms_cal_user:
                    self._rms_cal = pd.DataFrame({'freq_range': [[min(self.x_file), max(self.x_file)]],
                                                  'fmin': [min(self.x_file)],
                                                  'fmax': [max(self.x_file)],
                                                  'rms': [self._rms_cal_user['*'][0]],
                                                  'cal': [self._rms_cal_user['*'][1]]})

                elif next(iter(self._rms_cal_user))[0] == '[':  # info by frequency range
                    frange, fmin, fmax, rms, cal = [], [], [], [], []
                    for k, v in self._rms_cal_user.items():
                        range = k.strip('[').strip(']').strip()  # remove brackets and spaces ; could be improved
                        range = [float(elt) for elt in range.split(',')]  # convert comma-separated values to float
                        range.sort()  # make sure it is order by increasing values
                        frange.append(range)
                        fmin.append(range[0])
                        fmax.append(range[1])
                        rms_val = float(v[0])
                        if rms_val == 0.:
                            rms_val = np.nan
                        rms.append(rms_val)
                        cal.append(float(v[1]))
                    self._rms_cal = pd.DataFrame({'freq_range': frange,
                                                  'fmin': fmin,
                                                  'fmax': fmax,
                                                  'rms': rms,
                                                  'cal': cal})

                else:  # TODO : check the following
                    if len(self.tag_list) == 1 and str(self.tag_list[0]) not in self._rms_cal_user:
                        # only one species and tag not given => "reformat" dictionary to contain tag
                        self._rms_cal_user = {str(self.tag_list[0]): self._rms_cal_user}

                    tup, rms, cal = [], [], []
                    for tag, chi2_info in self._rms_cal_user.items():
                        for k, v in utils.expand_dict(chi2_info).items():
                            tup.append((tag, k))
                            rms.append(float(v[0]))
                            cal.append(float(v[1]))
                    self._rms_cal = pd.DataFrame({'win_id': tup,
                                                  'rms': rms,
                                                  'cal': cal})

            elif isinstance(self._rms_cal_user, str):
                # TODO: TBC
                # self._rms_cal_user = utils.read_noise_info(self._rms_cal_user)
                fmin, fmax, rms, cal = np.loadtxt(self._rms_cal_user, delimiter='\t', unpack=True)
                if isinstance(fmin, float):
                    frange = [fmin, fmax]
                else:
                    frange = [[fmini, fmaxi] for fmini, fmaxi in zip(fmin, fmax)]
                self._rms_cal = pd.DataFrame({'freq_range': frange,
                                              'fmin': fmin,
                                              'fmax': fmax,
                                              'rms': rms,
                                              'cal': cal})

            else:
                raise TypeError("rms_cal must be a dictionary or a path to an appropriate file.")

    def get_tag_list_from_sp_dict(self, labels):
        if not isinstance(labels, list):
            labels = [labels]
        tags = []
        for label in labels:
            try:
                # check if the label can be converted to an integer
                # if so, we directly have a tag, append it as a string
                int(label)
                tags.append(str(label))
            except ValueError:
                # the label is not numeric, search for the corresponding tag(s) in the user's dictionary
                try:
                    tags_temp = self.species_dict[label]
                    if not isinstance(tags_temp, list):
                        tags_temp = [tags_temp]
                    tags.extend(tags_temp)
                except KeyError:
                    raise IndexError("No tags found for species '{}'".format(label))
        return tags

    def read_comp_infos(self, lines_from_file, cpt_dict=None):
        if cpt_dict is None:
            cpt_dict = {}
        # get species list per component and whether component is interacting
        interacting = [line for line in lines_from_file if '_interacting' in line]
        species = [line for line in lines_from_file if '_species' in line]
        for line in interacting:
            cname = line.split('_')[0]
            if cname not in cpt_dict:
                cpt_dict[cname] = {'interacting': line.split()[1].strip() == "True"}
            else:
                cpt_dict[cname]['interacting'] = line.split()[1].strip() == "True"
        for line in species:
            cname = line.split('_')[0]
            labels = line.split('\t')[1]
            labels = [elt.strip() for elt in labels.split(',')]
            tags = self.get_tag_list_from_sp_dict(labels)

            if cname not in cpt_dict:
                cpt_dict[cname] = {'species': tags}
            else:
                cpt_dict[cname]['species'] = tags

        return cpt_dict

    def read_comp_params(self, lines_from_file, cpt_dict=None):
        if cpt_dict is None:
            cpt_dict = {}

        # Component parameters (size, tex, vlsr, fwhm)
        n_start_comps = 0
        for j, line in enumerate(lines_from_file):
            if line.startswith('name'):
                n_start_comps = j
                break
        n_end_comps = n_start_comps + 1
        for j, line in enumerate(lines_from_file[n_start_comps + 1:]):
            if line.startswith('c'):
                n_end_comps += 1
                continue
            break
        rows_comps = lines_from_file[n_start_comps:n_end_comps]
        rows_comps = [line.rstrip() for line in rows_comps]
        fcomp = io.StringIO("\n".join(rows_comps))
        cpt_df = pd.read_csv(fcomp, sep='\t', comment='#')
        cpt_df = cpt_df.rename(columns=lambda x: x.strip())
        cpt_names = [name.split('_')[0] for name in cpt_df['name']]
        cpt_names = list(set(cpt_names))
        cpt_names.sort()
        # cpt_info = {cname: {'interacting': True} for cname in cpt_names}
        # for cname in cpt_names:
        #     if cname not in cpt_dict:  # ignore
        #         # cpt_dict[cname] = {}
        #         continue
            # if 'interacting' not in cpt_dict[cname]:
            #     cpt_dict[cname]['interacting'] = True
        valid_par_names = ['size', 'tex', 'vlsr', 'fwhm']
        for i, row in cpt_df.iterrows():
            cpt_name, par_name = row['name'].rsplit('_', maxsplit=1)
            # check validity of labels
            if par_name not in valid_par_names:
                raise Exception(f"Invalid parameter name {cpt_name}_{par_name}.")
            if cpt_name in cpt_dict:
                cpt_dict[cpt_name][par_name] = parameter_infos(min=row['min'], max=row['max'], value=row['value'],
                                                               vary=row['vary'])

        return cpt_dict

    def read_sp_table(self, cpt_dict=None, **kwargs):
        # species infos are given as a table, stored in self.species_infos :
        # tag   eup_min ...     c1_ntot     c1_ntot_min ...
        if cpt_dict is None:
            cpt_dict = {}
        ntot_min_fact = kwargs.get('ntot_min_fact', 1e-3)
        ntot_max_fact = kwargs.get('ntot_max_fact', 1e3)
        # self.species_infos[[col for col in self.species_infos.columns if not col.startswith('c')]]
        cpt_names = [key for key in cpt_dict.keys() if key[1].isdigit()]
        for cname in cpt_names:
            cname_dic = cpt_dict[cname]
            sp_list = cname_dic.get('species', None)
            if sp_list is None and self.species_infos is None:
                raise Exception("Missing species info.")
            if not isinstance(sp_list, list):  # make sure it is a list
                sp_list = [sp_list]
            species_list = []
            for sp in sp_list:
                ntot = self.species_infos.at[int(sp), f'{cname}_ntot']
                ntot_min = ntot_min_fact
                ntot_max = ntot_max_fact
                is_factor = True

                if f'{cname}_ntot_min_fact' in self.species_infos.keys():
                    ntot_min = self.species_infos.at[int(sp), f'{cname}_ntot_min_fact']
                    if ntot_min == 0. or ntot_min == '*':
                        ntot_min = ntot_min_fact
                if f'{cname}_ntot_max_fact' in self.species_infos.keys():
                    ntot_max = self.species_infos.at[int(sp), f'{cname}_ntot_max_fact']
                    if ntot_max == 0. or ntot_max == '*':
                        ntot_max = ntot_max_fact

                if f'{cname}_ntot_min' in self.species_infos.keys() or f'{cname}_ntot_max' in self.species_infos.keys():
                    is_factor = False
                    if f'{cname}_ntot_min' in self.species_infos.keys():
                        ntot_min = self.species_infos.at[int(sp), f'{cname}_ntot_min']
                        if ntot_min == 0. or ntot_min == '*':
                            ntot_min = ntot_min_fact * ntot
                    if f'{cname}_ntot_max' in self.species_infos.keys():
                        ntot_max = self.species_infos.at[int(sp), f'{cname}_ntot_max']
                        if ntot_max == 0. or ntot_max == '*':
                            ntot_max = ntot_max_fact * ntot

                sp_dict = {
                    'tag': str(sp),
                    'ntot': parameter_infos(value=ntot, min=ntot_min, max=ntot_max, factor=is_factor)
                }
                if cpt_dict[cname]['fwhm'] is None:
                    if f'{cname}_fwhm_min_d' in self.species_infos.keys():
                        diff = True
                        min_f = self.species_infos.at[int(sp), f'{cname}_fwhm_min_d']
                        max_f = self.species_infos.at[int(sp), f'{cname}_fwhm_max_d']
                    elif f'{cname}_fwhm_min' in self.species_infos.keys():
                        diff = False
                        min_f = self.species_infos.at[int(sp), f'{cname}_fwhm_min']
                        max_f = self.species_infos.at[int(sp), f'{cname}_fwhm_max']
                    else:
                        raise Exception("Missing fwhm info.")
                    sp_dict['fwhm'] = parameter_infos(value=self.species_infos.at[int(sp), f'{cname}_fwhm'],
                                                      min=min_f, max=max_f, difference=diff)
                species_list.append(sp_dict)

            cname_dic['species'] = species_list

        return cpt_dict

    def read_sp_block(self, blocks_from_file, cpt_dict=None):
        # species infos are given as one "block" per species :
        # tag   eup_min     ...
        # name   min    value   max vary
        # c1_ntot   ...
        if cpt_dict is None:
            cpt_dict = {}
        cpt_names = [key for key in cpt_dict.keys() if key[1].isdigit()]
        sp_list_by_cpt = {}
        for c_name in cpt_names:
            if 'species' in cpt_dict[c_name] and isinstance(cpt_dict[c_name]['species'][0], (int, str)):
                species_labels = cpt_dict[c_name]['species']
                sp_list_by_cpt[c_name] = self.get_tag_list_from_sp_dict(species_labels)
                cpt_dict[c_name]['species'] = []

        sp_dfs = []
        sp_names = []
        pars = ['ntot', 'fwhm']
        for block in blocks_from_file:
            sp_name = block[1].split()[0]
            sp_names.append(sp_name)
            sp_df = pd.read_csv(io.StringIO("\n".join(block[2:])), sep='\t', comment='#', index_col=0)
            sp_df = sp_df.rename(columns=lambda x: x.strip())
            # check validity of labels
            for lbl in sp_df.index.tolist():
                if lbl.split('_')[-1] not in pars:
                    raise Exception(f"Invalid parameter name {lbl} for tag {sp_name}.")

            sp_dfs.append(sp_df)
        df = pd.concat(sp_dfs, axis=1, keys=sp_names)
        # print(df)
        for cname, sp_list in sp_list_by_cpt.items():
            for tag in sp_list:
                if tag not in sp_names:
                    raise KeyError(f'Tag {tag} not found in the component configuration.')
                sp_dict = {'tag': tag}
                for par in pars:
                    if f'{cname}_{par}' in df.index:
                        pmin, val, pmax, var = df.loc[f'{cname}_{par}'][tag].values
                        if np.isnan(float(val)):
                            raise ValueError(f"Missing {par} information for tag {tag}.")
                        factor = False
                        if par == 'ntot' and df.loc[f'{cname}_{par}'][tag]['max'] < 1.e6:
                            factor = True
                        sp_dict[par] = parameter_infos(value=val,
                                                       min=pmin,
                                                       max=pmax,
                                                       vary=var,
                                                       factor=factor)
                cpt_dict[cname]['species'].append(sp_dict)

        return cpt_dict

    def get_components(self, cpt_info):
        """
        Creates the components.
        :param cpt_info: dictionary with the components' information ; can contain an item providing a config file
        :return: None
        Possible formats for the config file :
        1. (componentConfig.txt)
           Whether a component is interacting and infos (min, value, max, vary) on size, tex, vlsr, fwhm
           e.g.:
            c1_interacting	True
            [...]
            # Add at least 3 lines per component (size, tex, vlsr) ; can add a 4th line with fwhm info
            name	min	value	max	vary # do not change this line
            c1_size	1	2	3	True
            c1_tex	100	200	300	True
            c1_vlsr	0	3	5	True
            c1_fwhm	2.0	5.0	12.0	True

        2. (infosComponentSpecies.txt)
           List of species (or key) per component
           Whether component is interacting
           Infos (min, value, max, vary) on size, tex, vlsr, fwhm
           Table with thresholds and column density info
           e.g.:
            c1_species	CH3OHs
            [...]
            c1_interacting	True
            [...]
            name	min	value	max	vary # do not change this line
            c1_size	1	1	3	False
            c1_tex	100	350	1000	True
            c1_vlsr	5	6.7	8	True
            c1_fwhm	1.5	3.5	6.5	False
            c2_size	1	1	50	False
            c2_tex	5	160	450	True
            c2_vlsr	5	7	8	True
            c2_fwhm	0.9	1.8	4.4	False
            [...]
            tag	    eup_min	eup_max	aij_min	aij_max	err_max	c1_ntot	c1_ntot_min_fact	c1_ntot_max_fact
            28503	0.0	    300.0	1.0e-6	*	    3	    6.0e17	1e-3                1e3

        3. (infosComponentSpecies.txt)
           Infos (min, value, max, vary) on size, tex, vlsr, fwhm plus
           one block per species with thresholds and column density
           (species and interacting are in the user's script)
           e.g.:
            name	min	value	max	vary # do not change this line
            c1_size	1	6	10	False
            c1_tex	50	100	150	False
            c1_vlsr	2	4	6	True
            c1_fwhm	2.0	4.5	12.0	False
            [...]
            tag	eup_min	eup_max	aij_min	aij_max	err_max
            44501	0.0	600.0	1.0e-6	*	3
            name	min	    value	max	vary # do not change this line
            c1_ntot	1e-3	1.0e15	1e3	True
            c2_ntot	1e-3	1.0e14	1e3	True
            c3_ntot	1e-3	1.0e14	1e3	True
        """
        config_key = [key for key in cpt_info.keys() if 'config' in key]  # look for key indicating a config file
        if len(config_key) > 0:
            cpt_config_file = cpt_info[config_key[0]]
            cpt_info.pop(config_key[0])  # remove the config item
            ntot_min_fact = None
            ntot_max_fact = None
            try:
                with open(cpt_config_file) as f:
                    all_lines = f.readlines()
                    line_sp_infos = 0
                    sp_table = True  # assume 2nd format with species infos (thresholds, column density) as table
                    for i, line in enumerate(all_lines):
                        if 'SPECIES INFOS' in line:  # 3rd format
                            sp_table = False
                        if 'SPECIES INFOS' in line or line.startswith('tag'):
                            # fmt 2 or 3 : identify where the species information starts
                            line_sp_infos = i
                            break

                    if line_sp_infos == 0:  # only component info, no thresholds per species
                        lines = all_lines
                    else:  # both component and species infos -> separate them
                        lines = [line.rstrip() for line in all_lines[:line_sp_infos]]  # comp info, dealt with later
                        # here, deal with species infos
                        lines_sp = all_lines[line_sp_infos:]
                        for _ in range(len(lines_sp)):
                            if lines_sp[0].startswith('#'):
                                lines_sp.pop(0)  # remove the comment lines at the beginning of the block
                            else:
                                break

                        if sp_table:
                            self.species_infos = utils.read_species_info(io.StringIO("".join(lines_sp)))
                        else:
                            # species infos by block : first need to separate each block
                            infos_by_sp = []  # list of list containing species infos
                            infos_sp = []

                            # First remove all comment lines
                            # (otherwise there is an issue if the last line is a comment line)
                            lines_sp = [line for line in lines_sp if not line.startswith('#')]

                            for i, line in enumerate(lines_sp):
                                if line.strip() != '':  # info line (not blank line)
                                    infos_sp.append(line.rstrip() + "\n")  # append info to the species list
                                if line.strip() == '' or i == len(lines_sp) - 1:
                                    # blank line or end of list : end of a species block
                                    infos_by_sp.append(infos_sp)
                                    infos_sp = []
                                    continue

                            self.read_sp_block(infos_by_sp, cpt_dict=cpt_info)  # add column density info to cpt_info

                            thresholds = [infos_by_sp[0][0]]  # column names
                            for infos_sp in infos_by_sp:
                                thresholds.append(infos_sp[1])  # for each species block, thresholds are at index 1
                                # sp = infos_sp[1].split()[0]
                                # sp_df =
                            self.species_infos = utils.read_species_info(io.StringIO("".join(thresholds)))

                    cpt_info = self.read_comp_infos(lines, cpt_dict=cpt_info)
                    cpt_info = self.read_comp_params(lines, cpt_dict=cpt_info)

                # Default ntot factors
                ntot_factors = [line for line in lines if "ntot_m" in line]
                for elt in ntot_factors:
                    if "min" in elt:
                        ntot_min_fact = float(elt.split("=")[-1])
                    if "max" in elt:
                        ntot_max_fact = float(elt.split("=")[-1])

                if sp_table:
                    cpt_info = self.read_sp_table(cpt_dict=cpt_info, **{'ntot_min_fact': ntot_min_fact,
                                                                        'ntot_max_fact': ntot_max_fact})

            except FileNotFoundError:
                raise FileNotFoundError(f"{cpt_config_file} was not found.")

        for cname in cpt_info.keys():
            if 'fwhm' not in cpt_info[cname]:
                cpt_info[cname]['fwhm'] = None

        for cname, cpt_dic in cpt_info.items():
            sp_list = cpt_dic.get('species', None)
            if sp_list is None:
                    raise Exception("Missing species info.")
            if not isinstance(sp_list, list):  # make sure it is a list
                sp_list = [sp_list]
            if isinstance(sp_list[0], (int, str)):
                cpt_info = self.read_sp_table(cpt_dict=cpt_info)
                sp_list = cpt_dic['species']
            if 'set_fwhm' in cpt_dic and cpt_dic['set_fwhm'] is not None:
                tag_ref = str(cpt_dic['set_fwhm'])
                expr = f'{cname}_fwhm_{tag_ref}'
                sp_list_ord = []
                for sp in sp_list:
                    if sp['tag'] != tag_ref:
                        sp['fwhm'].update({'expr': expr})
                        sp_list_ord.append(sp)
                    else:
                        sp_list_ord = [sp] + sp_list_ord  # make sure the reference species is first
                sp_list = sp_list_ord
            cpt = Component(cname, sp_list,
                            isInteracting=cpt_dic.get('interacting', False) or cpt_dic.get('isInteracting', False),
                            vlsr=cpt_dic.get('vlsr'), tex=cpt_dic.get('tex'), size=cpt_dic.get('size'),
                            fwhm=cpt_dic.get('fwhm'))
            self.cpt_list.append(cpt)

            if cpt_dic.get('fwhm') is not None:
                if cpt_dic.get('fwhm')['max'] is not None and cpt_dic.get('fwhm')['max'] != np.inf:
                    self.fwhm_max = max(self.fwhm_max, cpt_dic.get('fwhm')['max'])
                else:
                    self.fwhm_max = max(self.fwhm_max, cpt_dic.get('fwhm')['value'])

            for sp in cpt.species_list:
                try:
                    if sp.parameters[1].max is not None and sp.parameters[1].max != np.inf:
                        self.fwhm_max = max(self.fwhm_max, sp.parameters[1].max)
                    else:
                        self.fwhm_max = max(self.fwhm_max, sp.parameters[1].value)
                except IndexError:
                    pass  # do nothing

                if sp.tag not in self.tag_list:
                    self.tag_list.append(sp.tag)

        self.tag_list.sort(key=int)

    def get_windows(self, verbose=True):

        if self.bandwidth is None:
            if len(self.franges_mhz) > 1:
                self.win_list = []
                for r in self.franges_mhz:
                    ndigit = 0
                    rmin = np.round(min(r), ndigit)
                    rmax = np.round(max(r), ndigit)
                    while rmin == rmax:
                        ndigit += 1
                        rmin = np.round(min(r), ndigit)
                        rmax = np.round(max(r), ndigit)
                    if ndigit == 0:
                        rmin = int(rmin)
                        rmax = int(rmax)
                    self.win_list.append(Window(name=f'{rmin}-{rmax} MHz',
                                                x_mod=self.x_mod[(self.x_mod >= min(r)) & (self.x_mod <= max(r))]))
            else:
                self.win_list = [Window(name='Full spectrum')]
            return

        self.get_rms_cal_info()

        if self.fit_full_range:  # fitting entire frequency range except a few windows
            # NB: windows already excluded in data selection
            f_fit = self.x_file[self.mask]
            y_fit = self.y_file[self.mask]
            # if len(self.fit_freq_except) > 0:
            #     if not isinstance(self.fit_freq_except[0], list):
            #         self.fit_freq_except = [self.fit_freq_except]
            #     fmin = min(self.x_file)
            #     f2fit = []
            #     for f_range in self.fit_freq_except:
            #         fmax = min(f_range)
            #         f2fit.append([fmin, fmax])
            #         fmin = max(f_range)
            #     f2fit.append([fmin, max(self.x_file)])
            #     f_fit, y_fit = utils.select_from_ranges(self.x_file, f2fit, y_values=self.y_file)

            win = Window(name='Full spectrum')
            win.x_file = self.x_file
            win.y_file = self.y_file
            win.x_mod = self.x_mod
            # if win.x_file is not None:
            #     x_mod = []
            #     for i, row in self.tuning_info.iterrows():
            #         x_sub = win.x_file[(win.x_file >= row['fmhz_min']) & (win.x_file <= row['fmhz_max'])]
            #         if len(x_sub) == 0:
            #             continue
            #         x_mod.extend(np.linspace(min(x_sub), max(x_sub), num=self.oversampling * len(x_sub)))
            #     win.x_mod = np.array(x_mod)
            # else:
            #     pass
            win.f_ranges_nofit = self.fit_freq_except

            rms_cal_array = []
            for i, row in self._rms_cal.iterrows():
                rms_cal_array.append([row['fmin'], row['rms'], row['cal']])
                rms_cal_array.append([row['fmax'], row['rms'], row['cal']])
            rms_cal_array = pd.DataFrame(np.array(rms_cal_array), columns=['freq', 'rms', 'cal'])
            if min(f_fit) < rms_cal_array['freq'].min():
                raise ValueError(f"At least one value ({min(f_fit)}) is below "
                                 f"the minimum value of the interpolation range ({rms_cal_array['freq'].min()})")
            if max(f_fit) > rms_cal_array['freq'].max():
                raise ValueError(f"At least one value ({max(f_fit)}) is above "
                                 f"the maximum value of the interpolation range ({rms_cal_array['freq'].max()})")
            rms = interp1d(rms_cal_array['freq'], rms_cal_array['rms'])(f_fit)
            cal = interp1d(rms_cal_array['freq'], rms_cal_array['cal'])(f_fit)
            # rms = np.empty(len(f_fit), dtype=float)
            # cal = np.empty(len(f_fit), dtype=float)
            # if 'freq_range' in self._rms_cal.columns:
            #     for i, row in self._rms_cal.iterrows():
            #         indices = np.where((f_fit > row['fmin']) & (f_fit < row['fmax']))
            #         rms[indices] = row['rms']
            #         cal[indices] = row['cal']
            #
            #     if None in rms:
            #         raise ValueError("rms not defined for at least one frequency.")
            #     if None in cal:
            #         raise ValueError("calibration error not defined for at least one frequency.")
            # else:
            #     raise TypeError("rms and calibration information must be given in frequency ranges.")

            win.x_fit = f_fit
            win.y_fit = y_fit
            win.in_fit = True
            win.rms = rms
            win.cal = cal
            self.win_list = [win]
            self.win_list_fit = [w for w in self.win_list if w.in_fit]
            # self.line_list_all = get_transition_df(self.tag_list, fmhz_ranges=[min(self.x_file), max(self.x_file)])

        else:  # fit_freq_except is None : fitting by velocity range
            self.get_v_range_info()

            win_list_limits = []
            for tag, tr_list in self.tr_list_by_tag.items():
                win_list_tag = []  # first find all windows with enough data
                for i, tr in enumerate(tr_list):
                    f_range_plot = [utils.velocity_to_frequency(v, tr.f_trans_mhz, vref_kms=self.vlsr_file)
                                    for v in [-1. * self.bandwidth / 2 + self.vlsr_plot,
                                              1. * self.bandwidth / 2 + self.vlsr_plot]]
                    f_range_plot.sort()
                    x_win, y_win = None, None
                    if len(self.x_file) > 0:
                        x_win, y_win = utils.select_from_ranges(self.x_file, f_range_plot, y_values=self.y_file)
                        if len(x_win) <= 5 or len(set(y_win)) == 1:
                            continue
                    win = Window(tr, len(win_list_tag) + 1, bl_corr=self.bl_corr)
                    win.x_file, win.y_file = x_win, y_win
                    win_list_tag.append(win)
                    fwhm_mhz = utils.delta_v_to_delta_f(self.fwhm_max, tr.f_trans_mhz)
                    win_list_limits.append([min(f_range_plot) - 0.5 * fwhm_mhz, max(f_range_plot + 0.5 * fwhm_mhz)])

                nt = len(win_list_tag)
                if (verbose or verbose == 2) and nt != len(tr_list):
                    print('{} : {}/{} transitions found with enough data within thresholds'.format(tag, nt,
                                                                                                   len(tr_list)))
                if verbose == 2:
                    for iw, w in enumerate(win_list_tag):
                        print('  {}. {}'.format(iw + 1, w.transition))

                self.win_list.extend(win_list_tag)

            # Find transitions in win_list:
            self.line_list_all = get_transition_df(self.tag_list, fmhz_ranges=win_list_limits,
                                                   shift_kms=self.line_shift_kms)
            if self.f_err_mhz_max is not None:
                self.line_list_all = self.line_list_all[self.line_list_all.f_err_mhz <= self.f_err_mhz_max]
            self.line_list_all = self.line_list_all.drop_duplicates(subset='db_id', keep='first')

        if self.sort == 'frequency':
            self.line_list_all.sort_values('fMHz', inplace=True)
        else:
            self.line_list_all.sort_values(self.sort, inplace=True)

        return

    def get_velocity_ranges(self):
        for tag in self.tag_list:
            win_list_tag = [win for win in self.win_list if tag in win.name]
            nt = len(win_list_tag)
            if self._v_range_user is not None and (tag in self._v_range_user or '*' in self._v_range_user):
                v_range = utils.expand_dict(self._v_range_user[tag], nt)
                self.win_nb_fit[tag] = list(v_range.keys())
                for win in win_list_tag:
                    win_num = win.plot_nb

                    if win_num in v_range:  # window has range to be fitted
                        win.v_range_fit = v_range[win_num]
                        f_range = [utils.velocity_to_frequency(v, win.transition.f_trans_mhz,
                                                               vref_kms=self.vlsr_file)
                                   for v in v_range[win_num]]
                        f_range.sort()
                        win.f_range_fit = f_range

                        # get rms and cal for windows to be fitted
                        if self._rms_cal_user is not None:
                            try:
                                win.set_rms_cal(self.rms_cal)
                            except KeyError:
                                raise KeyError(f"rms/cal info not found.")
                        else:
                            pass

    def get_data_to_fit(self, update=False):
        # find windows with data to be fitted
        if not update:
            for win in self.win_list:
                if win.f_range_fit is not None:
                    win.x_fit, win.y_fit = utils.select_from_ranges(self.x_file, win.f_range_fit,
                                                                    y_values=self.y_file)
                    # with open('freq2fit.txt', 'a') as f:
                    #     f.write(win.name + f' - {win.f_range_fit}\n')
                    #     f.writelines('\n'.join([str(freq) for freq in win.x_fit]))
                    #     f.write('\n\n')

        self.win_list_fit = [w for w in self.win_list if w.in_fit]

        if len(self.win_list_fit) > 0:
            self.x_fit = np.concatenate([w.x_fit for w in self.win_list_fit], axis=None)
            self.y_fit = np.concatenate([w.y_fit for w in self.win_list_fit], axis=None)
            print(f"\nNumber of points used for the minimization : {len(self.x_fit)}/{len(self.x_file)}")
            if len(self.franges_mhz) > 1:
                for frange in self.franges_mhz:
                    print(f"    {frange}: {len(self.x_fit[(self.x_fit >= min(frange)) & (self.x_fit <= max(frange))])}"
                          f" / {len(self.x_file[(self.x_file >= min(frange)) & (self.x_file <= max(frange))])}"
                          f" points used")
                print("\n")

        else:
            self.x_fit, self.y_fit = None, None

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
        if len(value) > 0:
            self._y_file = value
            if self.x_obs is not None and not np.array_equal(self.x_obs, self.x_file):
                self._y_file = value[self.x_obs.argsort()]
            # update windows :
            if self.ref_pixel_info is not None:
                win_list = self.ref_pixel_info['windows']
            else:
                win_list = self.win_list
            new_win_list = []
            if len(win_list) > 0 and self.x_file is not None:
                for win in win_list:
                    x_win, y_win = utils.select_from_ranges(self.x_file, [min(win.x_file), max(win.x_file)],
                                                            y_values=self._y_file)
                    if len(x_win) <= 5 or len(set(y_win)) == 1:
                        continue
                    win.y_file = y_win
                    new_win_list.append(win)
                self.win_list = new_win_list
                self.get_data_to_fit()

    @property
    def tc(self):
        return self._tc

    @property
    def cont_info(self):
        return self._cont_info

    @cont_info.setter
    def cont_info(self, value):
        # self._configuration_dict['tc'] = value
        self._cont_info = value
        self.get_continuum()

    @property
    def rms_cal(self):
        return self._rms_cal

    @rms_cal.setter
    def rms_cal(self, value):
        self._rms_cal_user = value
        self.get_rms_cal_info()
        for win in self.win_list:
            row = utils.get_df_row_from_freq_range(self.rms_cal, win.transition.f_trans_mhz)
            idx = row.index
            win.rms = self.rms_cal.loc[idx, 'rms'].values[0]
            win.cal = self.rms_cal.loc[idx, 'cal'].values[0]

    @property
    def file_spec(self):
        warnings.warn("This property is deprecated, please use `model_spec` instead.")
        return self._file_spec

    @file_spec.setter
    def file_spec(self, value):
        warnings.warn("This property is deprecated, please use `model_spec` instead.")
        self._file_spec = value

    @property
    def output_files(self):
        return self._output_files

    @output_files.setter
    def output_files(self, dic):
        self._output_files = dic

    @property
    def name_config(self):
        return self.output_files['config']

    @property
    def name_lam(self):
        return self.output_files['lam']


class Component:
    def __init__(self, name, species_list, isInteracting=False, vlsr=None, size=None, tex=100., fwhm=None, config=None):
        # super().__init__()
        self.name = name

        self.isInteracting = isInteracting

        if vlsr is None:
            vlsr = VLSR_DEF  # TODO: find out why VLSR_DEF inside __init__ does not work
        self._vlsr = create_parameter('{}_vlsr'.format(self.name), vlsr)  # km/s

        if size is None:
            size = SIZE_DEF
        self._size = create_parameter('{}_size'.format(self.name), size)  # arcsec

        self._tex = create_parameter('{}_tex'.format(self.name), tex)  # K

        if fwhm is not None:
            self._fwhm = create_parameter('{}_fwhm'.format(self.name), fwhm)
        else:
            self._fwhm = None

        cpt_species_list = []
        if not isinstance(species_list, list):
            species_list = [species_list]
        for sp in species_list:
            if isinstance(sp, Species):
                sp2add = sp
            elif isinstance(sp, dict):
                sp2add = Species(sp['tag'], ntot=sp['ntot'],
                                 fwhm=None if self._fwhm is not None else sp['fwhm'],
                                 component=self)
            elif isinstance(sp, (int, str)):
                sp2add = Species(sp)
            else:
                try:
                    sp2add = Species(int(sp))
                except TypeError:
                    print("Elements of species_list must be a Species, a dictionary, "
                          "or the tag as an integer or a string")
            if sp2add._component is None:
                sp2add.set_component(self.name)
            cpt_species_list.append(sp2add)

        self._species_list = cpt_species_list
        self._tag_list = [sp.tag for sp in self.species_list]
        self._parameters = []

        for sp in self.species_list:
            sp.tex = self.tex
        # highest temp for the component should be the lowest value among the max values of the partition functions
        self._tmax = min([max(sp.pf[0]) for sp in self.species_list])
        # lowest temp for the component should be the highest value among the min values of the partition functions
        self._tmin = max([min(sp.pf[0]) for sp in self.species_list])

        self.transition_list = None

    def __repr__(self):
        """Return printable representation of a Component object."""
        s = []
        return f"<Component '{self.name}', {', '.join(s)}>"

    def as_json(self):
        return {
            'vlsr': round(self.vlsr, 3),
            'size': round(self.size, 3),
            'tex': round(self.tex, 3),
            'fwhm': round(self.fwhm, 3) if self._fwhm is not None else None,
            'isInteracting': self.isInteracting,
            'species': [sp.as_json() for sp in self.species_list]
        }

    def update_parameters(self, new_pars):
        self._vlsr = new_pars['{}_vlsr'.format(self.name)]
        self._size = new_pars['{}_size'.format(self.name)]
        self._tex = new_pars['{}_tex'.format(self.name)]
        if '{}_fwhm'.format(self.name) in new_pars:
            self._fwhm = new_pars['{}_fwhm'.format(self.name)]
        for sp in self.species_list:
            sp._ntot = new_pars['{}_ntot_{}'.format(self.name, sp.tag)]
            if self._fwhm is None:
                sp._fwhm = new_pars['{}_fwhm_{}'.format(self.name, sp.tag)]

        # self.vlsr.set(value=new_pars['{}_vlsr'.format(self.name)].value, is_init_value=False)
        # self.size.set(value=new_pars['{}_size'.format(self.name)].value, is_init_value=False)
        # self.tex.set(value=new_pars['{}_tex'.format(self.name)].value, is_init_value=False)
        # for sp in self.species_list:
        #     sp.ntot.set(value=new_pars['{}_ntot_{}'.format(self.name, sp.tag)].value, is_init_value=False)
        #     sp.fwhm.set(value=new_pars['{}_fwhm_{}'.format(self.name, sp.tag)].value, is_init_value=False)
        #     # sp.tex = new_pars['{}_tex'.format(self.name, sp.tag)].value

    @property
    def vlsr(self):
        return self._vlsr.value

    @property
    def size(self):
        return self._size.value

    @property
    def tex(self):
        return self._tex.value

    @property
    def fwhm(self):
        return self._fwhm.value if self._fwhm is not None else None

    @property
    def tmax(self):
        self._tmax = min([max(sp.pf[0]) for sp in self.species_list])
        return self._tmax

    @property
    def tmin(self):
        self._tmin = max([min(sp.pf[0]) for sp in self.species_list])
        return self._tmin

    @property
    def tag_list(self):
        return self._tag_list

    @tag_list.setter
    def tag_list(self, value):
        self._tag_list = value

    @property
    def species_list(self):
        return self._species_list

    @species_list.setter
    def species_list(self, value):
        self._species_list = value

    @property
    def parameters(self):
        pars = [self._vlsr, self._size, self._tex]
        if self._fwhm is not None:
            pars.append(self._fwhm)
        for sp in self.species_list:
            pars.append(sp._ntot)
            if self._fwhm is None:
                pars.append(sp._fwhm)
        return pars

    # def get_transitions(self, fmhz_ranges, **thresholds):  # not used -> keep??
    #     self.transition_list = get_transition_df(self.species_list, fmhz_ranges, **thresholds)
    #     return self.transition_list

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
    def __init__(self, transition=None, plot_nb=0, name="", v_range_fit=None, f_range_fit=None, rms=None, cal=None,
                 x_mod=None, bl_corr=False):
        self.transition = transition
        self.plot_nb = plot_nb
        self._name = name
        if self.transition is not None:
            self._name = "{} - {}".format(transition.tag, plot_nb)
        self._v_range_fit = v_range_fit
        self._f_range_fit = f_range_fit
        self.f_ranges_nofit = []
        self._v_range_plot = None
        self._f_range_plot = None
        self._rms = rms
        self._cal = cal
        self._x_fit = None
        self._y_fit = None
        self._in_fit = False
        self._x_mod = x_mod
        self._y_mod = None
        self._y_mod_err = None
        self.y_mod_cpt = {}
        self.y_mod_err_cpt = {}
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
        self.tag_colors = {}
        self._y_min = np.inf
        self._y_max = -np.inf
        self._bl_corr = bl_corr

    def __repr__(self):
        """Return printable representation of a Window object."""
        return f"<Window '{self.name}'>"

    def set_rms_cal(self, rms_cal_df):
        if 'freq_range' in rms_cal_df.columns:
            rms_cal = rms_cal_df[(self.transition.f_trans_mhz > rms_cal_df['fmin'])
                                 & (self.transition.f_trans_mhz < rms_cal_df['fmax'])]
        else:
            rms_cal = rms_cal_df[rms_cal_df['win_id'] == (self.transition.tag, self.plot_nb)]
            if len(rms_cal) == 0:
                rms_cal = rms_cal_df[rms_cal_df['win_id'] == (self.transition.tag, '*')]
        if len(rms_cal) == 0:
            raise IndexError(f"rms/cal info not found for {self.transition}.")
        self._rms = rms_cal['rms'].values[0]
        # if self.jypb is not None:
        #     win.rms_mk *= self.jypb[find_nearest_id(self.x_file, win.transition.f_trans_mhz)]
        self._cal = rms_cal['cal'].values[0]

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
        if self._bl_corr and min(value) < 0:
            value -= min(value)
        self._y_fit = value
        if len(self._y_fit) > 3:
            self._in_fit = True

    @property
    def in_fit(self):
        return self._in_fit

    @in_fit.setter
    def in_fit(self, value):
        if isinstance(value, bool) and value != self._in_fit:
            self._in_fit = value

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
    def y_mod_err(self):
        return self._y_mod_err

    @y_mod_err.setter
    def y_mod_err(self, value):
        self._y_mod_err = value

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
        if self._bl_corr and min(value) < 0:
            value -= min(value)
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
