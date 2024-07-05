from __future__ import annotations

from cassis_lte_python.utils import utils
from cassis_lte_python.utils.observer import Observable
from cassis_lte_python.database.constantsdb import THRESHOLDS_DEF
from cassis_lte_python.database.species import Species, get_species_thresholds
from cassis_lte_python.database.transitions import get_transition_df, select_transitions
from cassis_lte_python.sim.parameters import create_parameter, parameter_infos
from cassis_lte_python.utils.settings import TELESCOPE_DIR, VLSR_DEF, SIZE_DEF, NROWS_DEF, NCOLS_DEF
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import warnings


class ModelConfiguration:
    def __init__(self, configuration, verbose=True, check_tel_range=False):
        self._configuration_dict = configuration

        self.jparams = configuration.get('params', None)
        self.jmodel_fit = configuration.get('model_fit', None)

        self.species_infos = None
        if 'species_infos' in configuration:
            # sp_infos = {}
            df = pd.read_csv(configuration['species_infos'], delimiter='\t', comment='#', index_col=0, dtype=str)
            # perform check on number of columns for components
            ncols_cpt = [col for col in df.columns if col.startswith('c')]
            if len(ncols_cpt) % 4 != 0:  # ncols_cpt must be a multiple of 4
                raise ValueError(f"Number of columns for components in {configuration['species_infos']} "
                                 f"is not a multiple of 4.")
            if df.index.has_duplicates:
                dup = df.index[df.index.duplicated()]
                raise ValueError('Duplicate species infos detected for tags :',
                                 ", ".join([str(val) for val in dup.values]))
            sp_infos = df.apply(pd.to_numeric, errors='coerce')
            sp_infos = sp_infos.fillna('*')
            # sp_infos['tag'] = sp_infos['tag'].astype("string")
            self.species_infos = sp_infos

        self.fwhm_max = 0.
        self.tag_list = configuration.get('inspect', [])
        self.cpt_list = []
        if 'components' in configuration:
            for key, cpt_dic in configuration.get('components').items():
                sp_list = cpt_dic['species']
                if not isinstance(sp_list, list):  # make sure it is a list
                    sp_list = [sp_list]
                if isinstance(sp_list[0], (int, str)) and self.species_infos is not None:
                    # we have a list of tags and infos from file
                    species_list = []
                    for sp in sp_list:
                        if f'{key}_fwhm_min_d' in self.species_infos.keys():
                            diff = True
                            min_f = self.species_infos.at[int(sp), f'{key}_fwhm_min_d']
                            max_f = self.species_infos.at[int(sp), f'{key}_fwhm_max_d']
                        else:
                            diff = False
                            min_f = self.species_infos.at[int(sp), f'{key}_fwhm_min']
                            max_f = self.species_infos.at[int(sp), f'{key}_fwhm_max']
                        species_list.append(
                            {'tag': str(sp),
                             'ntot': parameter_infos(
                                 value=self.species_infos.at[int(sp), f'{key}_ntot'],
                                 min=1e-2, max=1e2, factor=True
                             ),
                             'fwhm': parameter_infos(
                                 value=self.species_infos.at[int(sp), f'{key}_fwhm'],
                                 min=min_f, max=max_f, difference=diff
                             )
                             }
                        )
                    sp_list = species_list
                if 'set_fwhm' in cpt_dic and cpt_dic['set_fwhm'] is not None:
                    tag_ref = str(cpt_dic['set_fwhm'])
                    expr = f'{key}_fwhm_{tag_ref}'
                    sp_list_ord = []
                    for sp in sp_list:
                        if sp['tag'] != tag_ref:
                            sp['fwhm'].update({'expr': expr})
                            sp_list_ord.append(sp)
                        else:
                            sp_list_ord = [sp] + sp_list_ord  # make sure the reference species is first
                    sp_list = sp_list_ord
                cpt = Component(key, sp_list,
                                isInteracting=cpt_dic.get('interacting', False) or cpt_dic.get('isInteracting', False),
                                vlsr=cpt_dic.get('vlsr'), tex=cpt_dic.get('tex'), size=cpt_dic.get('size'))
                self.cpt_list.append(cpt)
                for sp in cpt.species_list:
                    if sp.parameters[1].max is not None and sp.parameters[1].max != np.inf:
                        self.fwhm_max = max(self.fwhm_max, sp.parameters[1].max)
                    if sp.tag not in self.tag_list:
                        self.tag_list.append(sp.tag)

        self.output_dir = configuration.get('output_dir', os.path.curdir)
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        self.base_name = configuration.get('base_name', 'lte_model')

        self.data_file = configuration.get('data_file', None)
        self.data_file_obj = None
        self.xunit = 'MHz'
        yunit = configuration.get('yunit', 'K')
        self.yunit = yunit.strip()
        self._x_file = None
        self._y_file = None
        self.x_obs = configuration.get('x_obs', None)
        self.y_obs = configuration.get('y_obs', None)
        self.vlsr_file = 0.
        self.vlsr_plot = configuration.get('vlsr_plot', 0.)
        self._cont_info = configuration.get('tc', 0.)
        self._tc = None
        self._telescope_data = {}
        self.tmb2ta = None
        if 'beam_info' in configuration:
            self.get_jypb(configuration)
        self.jypb = None
        self.bmaj = None
        self.bmin = None
        self.beam = {'bmaj': 0, 'bmin': 0}
        self._fit_freq_except_user = configuration.get('fit_freq_except', None)
        if self._fit_freq_except_user is not None:
            if isinstance(self._fit_freq_except_user, str):
                fmin, fmax = np.loadtxt(self._fit_freq_except_user, unpack=True)
                self.fit_freq_except = [[f1, f2] for f1, f2 in zip(fmin, fmax)]
            elif isinstance(self._fit_freq_except_user, list):
                if not isinstance(self._fit_freq_except_user[0], list):
                    self.fit_freq_except = [self._fit_freq_except_user]
            else:
                raise TypeError("fit_freq_except must be a list or a path to an appropriate file.")
        else:
            self.fit_freq_except = None

        self._v_range_user = configuration.get('v_range', None)
        self._rms_cal_user = configuration.get('rms_cal', None)
        if 'chi2_info' in configuration:
            warnings.warn("The chi2_info keyword is deprecated, please use the keyword rms_cal instead.")
            self._rms_cal_user = configuration.get('chi2_info', None)
        self._rms_cal = None
        self.win_list = []
        self.win_list_fit = None
        self.win_list_plot = []
        self.win_list_gui = []
        self.win_list_file = []

        self.x_fit = None
        self.y_fit = None
        self.x_mod = configuration.get('x_mod', None)
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

        franges_ghz = configuration.get('franges_ghz', [[fmin_ghz, fmax_ghz]])
        # make sure it is a list of lists
        if not isinstance(franges_ghz[0], list):
            franges_ghz = [franges_ghz]
        self.franges_mhz = []

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
        elif isinstance(noise, list):
            yvals = np.array([[n, n] for n in noise])
            self.noise = interp1d(np.concatenate(self.franges_mhz), np.concatenate(yvals), kind='nearest')
        else:
            print('Noise format not supported. Should be a integer, a float or a list.')

        self.t_a_star = configuration.get('t_a*', False)
        self._tuning_info_user = configuration.get('tuning_info', None)
        self.tuning_info = []

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

        self.modeling = configuration.get('modeling', False)

        self.constraints = configuration.get('constraints', None)
        self.ref_pixel_info = None
        self.minimize = configuration.get('minimize', False)
        self.tau_lim = configuration.get('tau_lim', np.inf)
        self.max_iter = configuration.get('max_iter', None)
        self.fit_kws = configuration.get('fit_kws', None)
        self.save_configs = configuration.get('save_configs', True) or configuration.get('save_res_configs', True)
        self.save_results = configuration.get('save_results', True) or configuration.get('save_res_configs', True)
        self.name_lam = configuration.get('name_lam', None)
        self.name_config = configuration.get('name_config', None)
        self.save_spec = configuration.get('save_spec', False)
        self.file_spec = configuration.get('file_spec', 'synthetic_spectrum.txt')

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
        if self.plot_file and 'filename' not in file_kws:
            raise NameError("Please provide a name for the output pdf file.")
        for k in kws_plot_only:
            if k in file_kws.keys():
                # print(f'N.B. : {k} in file keywords is not used.')
                file_kws.pop(k)
        self.file_kws.update(file_kws)

        if len(new_plot_kws) > 0:
            self.user_plot_kws = new_plot_kws

        self.exec_time = configuration.get('exec_time', True)

        self.get_tuning_info()

        if 'data_file' in self._configuration_dict or 'x_obs' in self._configuration_dict:
            self.get_data()

        if self.vlsr_plot == 0. and 'components' in configuration:
            self.vlsr_plot = self.cpt_list[0].vlsr

        if 'tc' in self._configuration_dict:
            self.get_continuum()

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
        self.x_file = config.get('x_obs', None)
        self.y_file = config.get('y_obs', None)
        self.vlsr_file = config.get('vlsr_obs', 0.)
        if self.data_file is not None and self.x_file is None:
            self.data_file_obj = utils.DataFile(self.data_file)
            self.x_file, self.y_file = self.data_file_obj.xdata_mhz, self.data_file_obj.ydata
            self.vlsr_file = self.data_file_obj.vlsr
            self.yunit = self.data_file_obj.yunit

        self.vlsr_plot = self.vlsr_file

        if self.x_file is not None and isinstance(self.x_file[0], np.ndarray):
            self.x_file = np.concatenate(self.x_file)
        if self.y_file is not None and isinstance(self.y_file[0], np.ndarray):
            self.y_file = np.concatenate(self.y_file)

        if self.x_file is not None:
            idx = self.x_file.argsort()
            self.x_file = self.x_file[idx]
            self.y_file = self.y_file[idx]

        if self.x_file is not None and len(self.tuning_info) > 0:
            # select data within telescope range
            x_sub, y_sub = utils.select_from_ranges(self.x_file, self.tuning_info['fmhz_range'].array,
                                                    y_values=self.y_file)
            self.x_file, self.y_file = x_sub, y_sub

            if self.oversampling == 1:
                self.x_mod = self.x_file
            else:
                x_mod = []
                for i, row in self.tuning_info.iterrows():
                    x_sub = self.x_file[(self.x_file >= row['fmhz_min']) & (self.x_file <= row['fmhz_max'])]
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
                f_cont, t_cont = np.loadtxt(self.cont_info, delimiter='\t', unpack=True)
                self._tc = interp1d(f_cont, t_cont, kind='nearest')
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

        if 'tuning_info' in config:
            if check_tel_range:  # check telescope ranges cover all data / all model values:
                x_vals = self.x_file if self.x_file is not None else [min(self.x_mod), max(self.x_mod)]
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

            for tel in config['tuning_info'].keys():
                freq_user = config['tuning_info'][tel]
                if isinstance(freq_user[0], list):
                    freq_user = [el for li in freq_user for el in li]
                tel_info = utils.read_telescope_file(utils.search_telescope_file(tel),
                                                     fmin_mhz=min(freq_user),
                                                     fmax_mhz=max(freq_user))
                self._telescope_data[tel] = tel_info

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

    def get_linelist(self, config=None):
        # if config is None:
        #     config = self._configuration_dict

        if self.x_file is not None:
            x_vals = self.x_file
        else:
            x_vals = self.x_mod

        tr_list_tresh = get_transition_df(self.tag_list, [[min(x_vals), max(x_vals)]], **self.thresholds)
        if self.sort == 'frequency':
            tr_list_tresh.sort_values('fMHz', inplace=True)
        else:
            tr_list_tresh.sort_values(self.sort, inplace=True)
        # for comparison with CASSIS look for number of transitions w/i min/max of data :
        if self.x_file is not None:
            print(f"{len(tr_list_tresh)} transitions within thresholds ",
                  f"and within data's min/max : [{min(self.x_file)}, {max(self.x_file)}].")
        else:
            self.line_list_all = tr_list_tresh

        if len(self.tuning_info) > 1:
            # more than one telescope range => search only in data within telescope ranges
            # NB : this assumes that if only one range, it encompasses the data's min/max
            f_range_search = []
            for f_range in self.tuning_info['fmhz_range']:
                x_sub = x_vals[(x_vals >= min(f_range)) & (x_vals <= max(f_range))]
                f_range_search.append([min(x_sub), max(x_sub)])
            tr_list_tresh = select_transitions(tr_list_tresh, xrange=f_range_search)
            print(f"{len(tr_list_tresh)} transitions within thresholds and within tuning frequencies : "
                  f"{self.tuning_info['fmhz_range'].tolist()}")
            if self.sort == 'frequency':
                tr_list_tresh.sort_values('fMHz', inplace=True)
            else:
                tr_list_tresh.sort_values(self.sort, inplace=True)

        # tr_list_tresh = get_transition_df(self.tag_list, self.tuning_info['fmhz_range'], **self.thresholds)
        self.tr_list_by_tag = {tag: list(tr_list_tresh[tr_list_tresh.tag == tag].transition) for tag in self.tag_list}
        if all([len(t_list) == 0 for t_list in self.tr_list_by_tag.values()]):
            raise LookupError("No transition found within the thresholds.")

        for cpt in self.cpt_list:
            # cpt.transition_list = self.line_list_all[self.line_list_all['tag'].isin(cpt.tag_list)]
            cpt.transition_list = {key: val for key, val in self.tr_list_by_tag.items() if key in cpt.tag_list}

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
                if '*' in self._rms_cal_user:
                    self._rms_cal = pd.DataFrame({'freq_range': [[min(self.x_file), max(self.x_file)]],
                                                  'fmin': [min(self.x_file)],
                                                  'fmax': [max(self.x_file)],
                                                  'rms': [self._rms_cal_user['*'][0]],
                                                  'cal': [self._rms_cal_user['*'][1]]})

                elif next(iter(self._rms_cal_user))[0] == '[':  # info by frequency range
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

    def get_windows(self, verbose=True):
        if self.bandwidth is None or self.fit_freq_except is not None:
            for tag, tr_list in self.tr_list_by_tag.items():
                if verbose or verbose == 2:
                    print('{} : {} transitions found within thresholds'.format(tag, len(tr_list)))
                if verbose == 2:
                    for iw, w in enumerate(tr_list):
                        print('  {}. {}'.format(iw + 1, w.transition))

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

        if self.fit_freq_except is not None:
            f_fit = self.x_file
            y_fit = self.y_file
            if len(self.fit_freq_except) > 0:
                if not isinstance(self.fit_freq_except[0], list):
                    self.fit_freq_except = [self.fit_freq_except]
                fmin = min(self.x_file)
                f2fit = []
                for f_range in self.fit_freq_except:
                    fmax = min(f_range)
                    f2fit.append([fmin, fmax])
                    fmin = max(f_range)
                f2fit.append([fmin, max(self.x_file)])
                f_fit, y_fit = utils.select_from_ranges(self.x_file, f2fit, y_values=self.y_file)

            win = Window(name='Full spectrum')
            win.x_file = self.x_file
            win.y_file = self.y_file
            win.f_ranges_nofit = self.fit_freq_except
            rms = np.empty(len(f_fit), dtype=float)
            cal = np.empty(len(f_fit), dtype=float)
            if 'freq_range' in self._rms_cal.columns:
                for i, row in self._rms_cal.iterrows():
                    indices = np.where((f_fit > row['fmin']) & (f_fit < row['fmax']))
                    rms[indices] = row['rms']
                    cal[indices] = row['cal']

                if None in rms:
                    raise ValueError("rms not defined for at least one frequency.")
                if None in cal:
                    raise ValueError("calibration error not defined for at least one frequency.")
            else:
                raise TypeError("rms and calibration information must be given in frequency ranges.")

            win.x_fit = f_fit
            win.y_fit = y_fit
            win.rms = rms
            win.cal = cal
            self.win_list = [win]
            self.win_list_fit = [w for w in self.win_list if w.in_fit]
            self.line_list_all = get_transition_df(self.tag_list, fmhz_ranges=[min(self.x_file), max(self.x_file)])

        else:
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
                    if self.x_file is not None:
                        x_win, y_win = utils.select_from_ranges(self.x_file, f_range_plot, y_values=self.y_file)
                        if len(x_win) <= 5 or len(set(y_win)) == 1:
                            continue
                    win = Window(tr, len(win_list_tag) + 1)
                    win.x_file, win.y_file = x_win, y_win
                    win_list_tag.append(win)
                    fwhm_mhz = utils.delta_v_to_delta_f(self.fwhm_max, tr.f_trans_mhz)
                    win_list_limits.append([min(f_range_plot) - 0.5 * fwhm_mhz, max(f_range_plot + 0.5 * fwhm_mhz)])

                nt = len(win_list_tag)
                if verbose or verbose == 2:
                    print('{} : {}/{} transitions found with enough data within thresholds'.format(tag, nt,
                                                                                                   len(tr_list)))
                if verbose == 2:
                    for iw, w in enumerate(win_list_tag):
                        print('  {}. {}'.format(iw + 1, w.transition))

                self.win_list.extend(win_list_tag)

            # Find transitions in win_list:
            self.line_list_all = get_transition_df(self.tag_list, fmhz_ranges=win_list_limits)
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
        # update windows :
        if len(self.win_list) > 0 and self.x_file is not None:
            for win in self.win_list:
                x_win, y_win = utils.select_from_ranges(self.x_file, [min(win.x_file), max(win.x_file)],
                                                        y_values=value)
                win.y_file = y_win
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


class Component:
    def __init__(self, name, species_list, isInteracting=False, vlsr=None, size=None, tex=100., config=None):
        # super().__init__()
        self.name = name
        self.species_list = []
        if not isinstance(species_list, list):
            species_list = [species_list]
        for sp in species_list:
            if isinstance(sp, Species):
                sp2add = sp
            elif isinstance(sp, dict):
                sp2add = Species(sp['tag'], ntot=sp['ntot'], fwhm=sp['fwhm'], component=self)
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
            self.species_list.append(sp2add)

        self._tag_list = [sp.tag for sp in self.species_list]
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
        # highest temp for the component should be the lowest value among the max values of the partition functions
        self._tmax = min([max(sp.pf[0]) for sp in self.species_list])
        # lowest temp for the component should be the highest value among the min values of the partition functions
        self._tmin = max([min(sp.pf[0]) for sp in self.species_list])
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
        self._vlsr = new_pars['{}_vlsr'.format(self.name)]
        self._size = new_pars['{}_size'.format(self.name)]
        self._tex = new_pars['{}_tex'.format(self.name)]
        for sp in self.species_list:
            sp._ntot = new_pars['{}_ntot_{}'.format(self.name, sp.tag)]
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
                 x_mod=None):
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
