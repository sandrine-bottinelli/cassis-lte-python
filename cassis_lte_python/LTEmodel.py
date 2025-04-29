from __future__ import annotations

from cassis_lte_python.utils import utils
from cassis_lte_python.gui.plots import file_plot, gui_plot
from cassis_lte_python.sim.model_setup import ModelConfiguration, Component
from cassis_lte_python.utils.settings import NCOLS_DEF, NROWS_DEF, DPI_DEF, NB_DECIMALS
from cassis_lte_python.utils.constants import PLOT_COLORS, CPT_COLORS, UNITS
from cassis_lte_python.database.species import get_species_thresholds, Species
from cassis_lte_python.database.transitions import get_transition_df, select_transitions
from cassis_lte_python.utils.settings import SQLITE_FILE
from cassis_lte_python.utils.utils import get_df_row_from_freq_range
import numpy as np
from numpy.random import normal
from lmfit import Model, Parameters
from lmfit.model import ModelResult
from scipy import stats, signal
from scipy.special import erf
from scipy.stats import t
import astropy.io.fits as fits
import astropy.units as u
import os
import pandas as pd
import datetime
import json
from spectral_cube import SpectralCube
# from astropy.wcs import WCS
from time import process_time
from warnings import warn
from typing_extensions import Literal
import math
import copy


def generate_lte_model_func(config: dict):
    """
    Function to generate the model function using information provided in config.
    The generated function depends on the frequency in MHz and on a set of lmfit parameters.

    :param config: a dictionary containing :
        - the following frequency-dependent functions:
            - tc : continuum values ; no default
            - beam_sizes : 1-D equivalent beam size ; no default
            - tmb2ta : conversion factor from Tmb to Ta* scale ; default to 1
            - jypb2k : conversion factor from Jy/beam to K ; default to 1
            - noise : rms noise values ; default to 0
        - line_list : list of transitions to be modeled
        - cpt_list : list of components
    :return: The model function to be minimized.
    """
    line_list = config['line_list']
    try:
        tr_list_by_tag = {tag: list(line_list.loc[line_list['tag'] == tag].transition)
                          for tag in line_list['tag'].unique()}
    except KeyError:
        raise TypeError("Not implemented yet, line_list should be a DataFrame.")

    def lte_model_func(fmhz, log=False, cpt='from_config', line_center_only=False, return_tau=False, **params):
        norm_factors = config.get('norm_factors', {key: 1. for key in params.keys()})
        vlsr_file = config.get('vlsr_file', 0.)
        tc = config['tc'](fmhz)
        beam_sizes = config['beam_sizes'](fmhz)
        tmb2ta = config.get('tmb2ta', lambda x: 1.)(fmhz)
        jypb2k = config.get('jypb2k', lambda x: 1.)(fmhz)
        noise = config.get('noise', lambda x: 0.)(fmhz)
        tcmb = config.get('tcmb', 2.73)
        # line_list = config['line_list']
        cpt_list = config['cpt_list']
        if not isinstance(cpt_list, list):
            cpt_list = [cpt_list]
        if cpt != 'from_config':
            cpt_list = cpt if isinstance(cpt, list) else [cpt]

        tau_max = config.get('tau_max', None)
        file_rejected = config.get('file_rejected', None)
        tc = tc * jypb2k  # if jypb2k not 1, data, and so tc, are in Jy/beam -> convert to K
        intensity_before = tc + utils.jnu(fmhz, tcmb)
        intensity = 0.
        for icpt, cpt in enumerate(cpt_list):
            tex = params['{}_tex'.format(cpt.name)] * norm_factors['{}_tex'.format(cpt.name)]
            # if log:
            #     tex = round(10. ** tex, 6)
            vlsr = params['{}_vlsr'.format(cpt.name)] * norm_factors['{}_vlsr'.format(cpt.name)]
            size = params['{}_size'.format(cpt.name)] * norm_factors['{}_size'.format(cpt.name)]
            try:
                fwhm = params['{}_fwhm'.format(cpt.name)] * norm_factors['{}_fwhm'.format(cpt.name)]
            except KeyError:
                fwhm = None
                pass  # assume fwhm will be given by species, do nothing

            sum_tau = 0
            for isp, tag in enumerate(cpt.tag_list):
                # if isinstance(line_list, list):
                #     tran_list = line_list
                # else:  # assume it is a DataFrame
                #     tran_list = list(line_list.loc[line_list['tag'] == tag].transition)
                if tag not in tr_list_by_tag.keys():
                    continue
                tran_list = tr_list_by_tag[tag]
                ntot = params['{}_ntot_{}'.format(cpt.name, tag)] * norm_factors['{}_ntot_{}'.format(cpt.name, tag)]
                if log:
                    ntot = 10. ** ntot

                if fwhm is None:
                    fwhm = params['{}_fwhm_{}'.format(cpt.name, tag)] * norm_factors['{}_fwhm_{}'.format(cpt.name, tag)]

                qtex = cpt.species_list[isp].get_partition_function(tex)
                for tran in tran_list:
                    tau0 = utils.compute_tau0(tran, ntot, fwhm, tex, qtex=qtex)
                    if isinstance(tau_max, (float, int)) and tau0 >= tau_max:
                        with open(file_rejected, 'a') as f:
                            f.write('\n')
                            f.write('\t'.join([str(tag), utils.format_float(ntot), utils.format_float(tex),
                                               utils.format_float(fwhm), utils.format_float(tran.f_trans_mhz),
                                               utils.format_float(tran.eup), utils.format_float(tran.aij),
                                               str(tran.gup),
                                               utils.format_float(tau0)]))
                        continue

                    num = fmhz - utils.velocity_to_frequency(vlsr, tran.f_trans_mhz, vref_kms=vlsr_file)
                    den = utils.fwhm_to_sigma(utils.delta_v_to_delta_f(fwhm, tran.f_trans_mhz))
                    if line_center_only:
                        offset = utils.find_nearest_id(
                            fmhz, utils.velocity_to_frequency(vlsr, tran.f_trans_mhz, vref_kms=vlsr_file))
                        pulse = signal.unit_impulse(len(fmhz), offset)
                        sum_tau += tau0 * pulse
                    else:
                        sum_tau += tau0 * np.exp(-0.5 * (num / den) ** 2)

            ff = utils.dilution_factors(size, beam_sizes)
            if not cpt.isInteracting:
                intensity_cpt = utils.jnu(fmhz, tex) * (1. - np.exp(-sum_tau)) - \
                                intensity_before * (1. - np.exp(-sum_tau))
                intensity += ff * intensity_cpt
            else:
                intensity_before += intensity
                intensity_cpt = utils.jnu(fmhz, tex) * (1. - np.exp(-sum_tau)) - \
                                intensity_before * (1. - np.exp(-sum_tau))
                intensity = ff * intensity_cpt

        intensity = intensity + intensity_before - utils.jnu(fmhz, tcmb)
        if config.get('cont_free'):
            intensity -= tc
        intensity += normal(0., noise, len(intensity))  # add gaussian noise
        intensity *= tmb2ta  # convert to Ta
        intensity /= jypb2k  # convert to Jy/beam

        if return_tau:
            intensity = (intensity, sum_tau)
        return intensity

    return lte_model_func


class SimpleSpectrum:
    def __init__(self, xarray, yarray, xunit='mhz', yunit='K'):
        self.xval = xarray
        self.yval = yarray
        self.xunit = xunit
        self.yunit = yunit


class ModelSpectrum(object):
    def __init__(self, configuration: (dict, str, ModelConfiguration),
                 verbose=True, check_tel_range=False, **kwargs):
        if isinstance(configuration, (dict, str)):  # dictionary or string
            if isinstance(configuration, str):  # string : load the file
                config = self.load_config(configuration)
                # date = config.get('creation-date', datetime.datetime.now())
                # if isinstance(date, str):
                #     date = datetime.datetime.strptime(date.split(",")[0], "%Y-%m-%d")
                if 'creation-date' not in config:  # old format for plot keywords
                    config['plot_kws'] = {'gui+file': config.get('plot_kws', {}),
                                          'gui_only': config.get('gui_kws', {}),
                                          'file_only': config.get('file_kws', {})}
                    config.pop('gui_kws')
                    config.pop('file_kws')
                if ('gui_kws' in kwargs) or ('file_kws' in kwargs):
                    print('gui_kws and file_kws are deprecated keywords, please use the following instead:')
                    comment = "  # N.B. : can be also be removed/commented"
                    info_gui = "" if 'gui_kws' in kwargs else comment
                    info_file = "" if 'file_kws' in kwargs else comment
                    print("  'plot_kws': {")
                    print("       'gui_only':", kwargs.get('gui_kws', {}), info_gui, end=",\n")
                    print("       'file_only':", kwargs.get('file_kws', {}), info_file)
                    print("  }")

                    # update kwargs to new format
                    kwargs['plot_kws'] = {}
                    if 'gui_kws' in kwargs:
                        kwargs['plot_kws']['gui_only'] = kwargs['gui_kws']
                        kwargs.pop('gui_kws')
                    if 'file_kws' in kwargs:
                        kwargs['plot_kws']['file_only'] = kwargs['file_kws']
                        kwargs.pop('file_kws')

                # update configuration
                dflat = utils.flatten_dic(config)
                dflat.update(utils.flatten_dic(kwargs))
                config = utils.unflatten_dic(dflat)

            else:  # it is a dictionary
                config = configuration

            model_config = ModelConfiguration(config, verbose=verbose, check_tel_range=check_tel_range)

            # if 'data_file' in model_config._configuration_dict or 'x_obs' in model_config._configuration_dict:
            #     model_config.get_data()
            #
            # if 'tc' in model_config._configuration_dict:
            #     model_config.get_continuum()

            # model_config.get_linelist()
            # model_config.get_windows()

        elif isinstance(configuration, ModelConfiguration):
            model_config = configuration
            model_config.get_data_to_fit()
            model_config.get_continuum()
            # update tag list, parameters, and windows to fit (intensities, noise)
            tag_list = []
            for cpt in model_config.cpt_list:
                tag_list.extend([str(t) for t in cpt.tag_list if str(t) not in tag_list])
            # determine whether tag_list has changed :
            recompute_win = True
            if len(tag_list) == len(model_config.tag_list) and all([tag in model_config.tag_list for tag in tag_list]):
                recompute_win = False
            if recompute_win:
                for cpt in model_config.cpt_list:  # should contain new tag list
                    current_tags = {sp.tag: sp for sp in cpt.species_list}
                    new_sp_list = []
                    for tag in cpt.tag_list:
                        if tag in current_tags.keys():
                            new_sp_list.append(current_tags[tag])
                        else:
                            new_sp_list.append(model_config.ref_pixel_info[cpt.name][tag])
                    cpt.species_list = new_sp_list
                model_config.tag_list = tag_list
                for win in model_config.win_list:
                    win.set_rms_cal(model_config.rms_cal)
                    tag = win.transition.tag
                    if tag in model_config.tag_list and win.plot_nb in model_config.win_nb_fit[tag]:
                        win.in_fit = True
                    else:
                        win.in_fit = False
                model_config.win_list_fit = [win for win in model_config.win_list if win.in_fit]
                if len(model_config.win_list_fit) > 0:
                    model_config.x_fit = np.concatenate([w.x_fit for w in model_config.win_list_fit], axis=None)
                    model_config.y_fit = np.concatenate([w.y_fit for w in model_config.win_list_fit], axis=None)

        else:  # unknown
            raise TypeError("Configuration must be a dictionary or a path to a configuration file "
                            "or of type ModelConfiguration.")

        self.model_config = model_config

        self.params = None
        self.norm_factors = None
        self.log = False
        self.normalize = False
        # self.make_params()

        self.model = None
        self.model_fit = None
        self.model_fit_cpt = []
        self.figure = None

        self.tag_colors = None
        self.tag_other_sp_colors = None
        self.cpt_cols = None
        self.thresholds_other = None

        if self.model_config.fit_kws is not None:
            if 'log' in self.model_config.fit_kws:
                self.log = self.model_config.fit_kws['log']
                self.model_config.fit_kws.pop('log')

        # try:
        #     self.make_params(json_params=self.model_config.jparams)
        # except TypeError:
        #     pass

        if self.modeling:
            self.do_modeling()

        if self.minimize:
            self.do_minimization()

    def __getattr__(self, item):
        # method called for the times that __getattribute__ raised an AttributeError
        # in this case, assume the item is in model_config
        return self.model_config.__getattribute__(item)

    def load_config(self, path):
        try:
            with open(path) as f:
                return json.load(f)  # , cls=type(self))
        except FileNotFoundError:
            print(f"File not found : {path}")

    def save_config_dict(self):
        try:
            name_config = os.path.abspath(self.name_config)
        except TypeError:
            name_config = self.name_config

        cpt_config = {}
        if self.model_config.comp_config_file is not None:
            cpt_config['config'] = os.path.abspath(self.model_config.comp_config_file)
        cpt_config = {**cpt_config, **{cpt.name: cpt.as_json() for cpt in self.cpt_list}}

        config_save = {}

        for attr in ('data_file', 'output_dir', 'tcmb', 'fit_full_range', 'bandwidth', 'oversampling',
                     'tau_max', 'thresholds', 'fit_kws', 'name_lam', 'save_configs', 'save_results',
                     'plot_gui', 'plot_file', 'exec_time', 'constraints'):
            try:
                val = getattr(self, attr)
            except AttributeError:
                continue
            if isinstance(val, (str, os.PathLike)) and (os.path.isdir(val) or os.path.isfile(val)):
                val = os.path.abspath(val)
            config_save[attr] = val

        tuning_info = {os.path.abspath(k): v for k, v in self.model_config._tuning_info_user.items()}
        rms_cal = self.model_config._rms_cal_user
        rms_cal = os.path.abspath(rms_cal) if isinstance(rms_cal, str) else rms_cal
        plot_kws = self.user_plot_kws
        try:
            plot_kws['gui+file']['other_species'] = os.path.abspath(plot_kws['gui+file']['other_species'])
        except (KeyError, TypeError):
            pass
        try:
            plot_kws['file_only']['filename'] = os.path.abspath(plot_kws['file_only']['filename'])
        except (KeyError, TypeError):
            pass

        config_save_ctd = {
            'tc': os.path.abspath(self.cont_info) if isinstance(self.cont_info, (str, os.PathLike)) else self.cont_info,
            'baseline_corr': self.bl_corr,
            'continuum_free': self.cont_free,
            'tuning_info': tuning_info,
            # 'v_range': self.model_config._v_range_user,
            # 'v_range': None,  # update later
            # 'fit_full_range': self.model_config.fit_full_range,
            # 'fit_freq_except': self.model_config._fit_freq_except_user,
            'rms_cal': rms_cal,
            # 'fghz_min': self.fmin_mhz / 1.e3,
            # 'fghz_max': self.fmax_mhz / 1.e3,
            'df_mhz': self.dfmhz,
            'noise': np.mean(self.noise(self.x_mod)),  # TODO : write range
            'minimize': False,  # by default, do not (re-)minimize
            'modeling': self.modeling or self.minimize,  # if minimization was done, want modeling too by default
            'max_iter': self.max_iter,
            # 'fit_kws': self.fit_kws,
            # 'name_lam': self.name_lam,
            'name_config': name_config,
            # 'save_configs': self.save_configs,
            # 'save_results': self.save_results,
            # 'plot_gui': self.plot_gui,
            # # 'gui_kws': self.gui_kws,
            # 'plot_file': self.plot_file,
            # 'plot_file': os.path.abspath(self.plot_file) if os.path.isfile(self.plot_file) else self.plot_file,
            # 'file_kws': self.file_kws,
            'plot_kws': plot_kws,
            # 'exec_time': self.exec_time,
            'components': cpt_config,
            # 'constraints': self.model_config.constraints,
            # 'params': self.params.dumps(cls=utils.CustomJSONizer)
            # 'params': self.params.dumps()
        }

        config_save = {**config_save, **config_save_ctd}

        if not self.model_config.fit_full_range:
            config_save['v_range'] = {tag: {win.name.split()[-1]: win.v_range_fit
                                            for win in self.model_config.win_list_fit}
                                      for tag in self.model_config.tag_list}
        else:
            config_save['fit_freq_except'] = self.model_config._fit_freq_except_user

        # if self.model_fit is not None:
        #     config_save['model_fit'] = self.model_fit.dumps(cls=utils.CustomJSONizer)
        # NB: the above line yields an "Object of type bool is not JSON serializable" error
        #     when the dumps method of the lmfit ModelResult class calls the dumps method of the Parameters class ;
        #     however, I do not understand why the later yields an error, since individual parameters do not yield
        #     any error.

        config_save = {'creation-date': datetime.datetime.now().strftime("%Y-%m-%d,  %H:%M:%S"),
                       **dict(sorted(config_save.items())),
                       'params': self.params.dumps(cls=utils.CustomJSONizer)}
        return config_save

    def save_config(self, filename, dirname=None):
        try:
            json_dump = json.dumps(self.save_config_dict(), indent=4,
                                   cls=utils.CustomJSONizer)  # separators=(', \n', ': '))

            if dirname is not None:
                if not os.path.isdir(os.path.abspath(dirname)):
                    os.makedirs(os.path.abspath(dirname))
            else:
                dirname = self.output_dir
            path = os.path.join(dirname, filename)
            with open(path, 'w') as f:
                f.write(json_dump)
        except Exception as e:
            print("WARNING - Encountered the following error while writing the json config : ")
            print(e)
            print("-> skipping this step.")
            # pass

    def update_configuration(self, config):  # TODO : needs to be updated
        # update components only
        new_cpts = config["components"]
        for cpt, cpt_info in new_cpts.items():
            cname = cpt
            for par, pval in cpt_info.items():
                if par != 'species':
                    self.params[cname+'_'+par].value = pval
                else:
                    for sp in pval:
                        self.params[f"{cname}_ntot_{sp.tag}"].value = sp.ntot
                        self.params[f"{cname}_fwhm_{sp.tag}"].value = sp.fwhm

        # self.y_mod = self.compute_model_intensities(params=self.params, x_values=self.x_mod)

    def update_tag_list(self, new_tags):
        current_tags = self.model_config.tag_list
        current_tags.sort()
        new_tags.sort()
        if new_tags == current_tags:  # list has not changed : exit method
            return

        for cpt in self.model_config.cpt_list:
            # update tag list and species list
            new_sp_list = []
            for tag in new_tags:
                if tag in cpt.tag_list:  # species already in tag list, keep it
                    new_sp_list.append(list(filter(lambda sp: sp.tag == tag, cpt.species_list))[0])
                else:
                    if (self.model_config.ref_pixel_info is not None
                            and tag in self.model_config.ref_pixel_info[cpt.name]):
                        # we have info on a reference pixel and the species is in the original component, take it
                        new_sp_list.append(self.model_config.ref_pixel_info[cpt.name][tag])

            cpt.species_list = new_sp_list
            cpt.tag_list = [sp.tag for sp in new_sp_list]

        self.model_config.tag_list = new_tags

        # update parameters
        params = Parameters()
        params.add_many(*[par for cpt in self.model_config.cpt_list for par in cpt.parameters])
        self.params = params

        # update windows to fit
        for win in self.model_config.win_list:
            win.set_rms_cal(self.model_config.rms_cal)
            tag = win.transition.tag
            if tag in self.model_config.tag_list and win.plot_nb in self.model_config.win_nb_fit[tag]:
                win.in_fit = True
            else:
                win.in_fit = False
        self.model_config.win_list_fit = [win for win in self.model_config.win_list if win.in_fit]

        # update data to fit
        if len(self.model_config.win_list_fit) > 0:
            self.model_config.x_fit = np.concatenate([w.x_fit for w in self.model_config.win_list_fit], axis=None)
            self.model_config.y_fit = np.concatenate([w.y_fit for w in self.model_config.win_list_fit], axis=None)

    def model_info(self, cpt=None):

        mdl_info = {
            'cont_free': self.cont_free,
            'tc': self.tc,
            'tcmb': self.tcmb,
            'vlsr_file': self.vlsr_file,
            'norm_factors': self.norm_factors,
            'log': self.log,
            'beam_sizes': self.beam,
            'tmb2ta': self.tmb2ta,
            'line_list': self.line_list_all,
            'cpt_list': self.cpt_list if cpt is None else [cpt],
            'noise': self.noise,
            'tau_max': self.tau_max,
            'file_rejected': self.file_rejected
        }
        if self.yunit in UNITS['flux']:
            mdl_info['jypb2k'] = self.jypb
        return mdl_info

    def get_tc(self, x_mod):
        return self.tc(x_mod)

    def get_rms_cal(self, fmhz):
        row = get_df_row_from_freq_range(self._rms_cal, fmhz)
        try:
            return row['rms'].values[0], row['cal'].values[0]
        except IndexError:
            raise IndexError(f"rms/cal not found at {fmhz} MHz "
                             f"(min/max are {min(self._rms_cal.fmin), max(self._rms_cal.fmax)}).")

    def get_rms(self, fmhz):
        if type(fmhz) == float:
            fmhz = list(fmhz)

        rms = []
        for freq in fmhz:
            for win in self.win_list_fit:
                if min(win.f_range_fit) <= freq <= max(win.f_range_fit):
                    rms.append(win.rms)

        return rms if len(rms) > 1 else rms[0]

    def param_names(self):
        params = []
        for cpt in self.cpt_list:
            params.extend(p.name for p in cpt.parameters)
            # for sp in cpt.species_list:
            #     params.extend(p.name for p in sp.parameters)
        return params

    def make_params(self, json_params: str | None = None, normalize=False):
        params = Parameters()

        for icpt, cpt in enumerate(self.cpt_list):
            for par in cpt.parameters:
                if 'size' in par.name:
                    par.set(min=0. if not isinstance(par.min, (float, int)) else par.min)
                if 'tex' in par.name:
                    par.set(min=self.tcmb if not isinstance(par.min, (float, int)) else max(par.min, self.tcmb))
                params[par.name] = par

            for isp, sp in enumerate(cpt.species_list):
                for par in sp.parameters:
                    par.set(min=0. if not isinstance(par.min, (float, int)) else par.min)
                    params[par.name] = par

        # Update parameters if possible :
        if json_params is not None:
            params.loads(json_params)

        params = self.check_params_boundaries(params)
        # # Check min/max Tex:
        # for icpt, cpt in enumerate(self.cpt_list):
        #     par = params[f'{cpt.name}_tex']
        #     mini = cpt.tmin if not isinstance(par.min, (float, int)) else max(par.min, cpt.tmin)
        #     maxi = cpt.tmax if not isinstance(par.max, (float, int)) else min(par.max, cpt.tmax)
        #     if par.min < cpt.tmin and self.minimize:
        #         sp_tmin = [sp for sp in cpt.species_list if min(sp.pf[0]) == cpt.tmin]
        #         if len(sp_tmin) > 1:
        #             tags_tmin = ", ".join([sp.tag for sp in sp_tmin])
        #         else:
        #             tags_tmin = sp_tmin[0].tag
        #         message = [
        #             f'The requested lower boundary for Tex is {par.min} K, but the lowest temperature',
        #             f'for which the partition function (pf) is defined for all species is {cpt.tmin} K;',
        #             f'Species for which T_pf_min = {cpt.tmin} is/are: {tags_tmin}',
        #             f'-> limiting Tex search to temperatures > {cpt.tmin} \n'
        #         ]
        #         print(f'INFO - Component {cpt.name} :')
        #         for line in message:
        #             print(" " * 6, line)
        #         # print(f'Component {cpt.name} : limiting Tex search to temperatures > {cpt.tmin} '
        #         #       f'(smallest temperature for which the partition function is defined for all species).')
        #         val = (mini + maxi) / 2
        #     if par.max > cpt.tmax and self.minimize:
        #         sp_tmax = [sp for sp in cpt.species_list if max(sp.pf[0]) == cpt.tmax]
        #         if len(sp_tmax) > 1:
        #             tags_tmax = ", ".join([sp.tag for sp in sp_tmax])
        #         else:
        #             tags_tmax = sp_tmax[0].tag
        #         message = [
        #             f'The requested upper boundary for Tex is {par.max} K, but the highest temperature',
        #             f'for which the partition function (pf) is defined for all species is {cpt.tmax} K;',
        #             f'Species for which T_pf_max = {cpt.tmax} is/are: {tags_tmax}',
        #             f'-> limiting Tex search to temperatures < {cpt.tmax} \n'
        #         ]
        #         print(f'INFO - Component {cpt.name} :')
        #         for line in message:
        #             print(" " * 6, line)
        #         # print(f'Component {cpt.name} : limiting Tex search to temperatures < {cpt.tmax} '
        #         #       f'(highest temperature for which the partition function is defined for all species).')
        #         val = (mini + maxi) / 2
        #
        #     par.set(min=mini, max=maxi)

        # user constraints
        if self.model_config.constraints is not None and self.model_config.ref_pixel_info is None:
            for key, val in self.model_config.constraints.items():
                if 'set_fwhm' in key:
                    tag_ref = val.split('_')[-1]
                    cpt = val.split('_')[0]
                    for tag in self.model_config.tag_list:
                        if tag != tag_ref:
                            try:
                                params[f'{cpt}_fwhm_{tag}'].set(expr=val)
                            except KeyError:  # key does not exist, do nothing
                                pass
                else:
                    try:
                        params[key].set(expr=val)
                    except KeyError:
                        print(f"Constraints: {key.split('_')[-1]} is not in the tag list "
                              f"or has been removed due to lack of transitions.")
                        pass

        # reset bounds if a parameters contains an expression to make sure it does not interfere
        for par in params:
            if params[par].expr is not None:
                params[par].set(min=-np.inf, max=np.inf)

        self.params = params

        # if self.model_config.jparams is None:
        #     self.model_config.jparams = self.params.dumps()

        norm_factors = {self.params[parname].name: 1. for parname in self.params}
        if normalize:
            for parname in self.params:
                param = self.params[parname]
                nf = abs(param.value) if param.value != 0. else 1.
                norm_factors[param.name] = nf
                if param.expr is None:
                    param.set(min=param.min / nf, max=param.max / nf, value=param.value / nf)
        self.norm_factors = norm_factors

    def check_params_boundaries(self, params):
        # Check min for params that cannot have negative values:
        for parname in params:
            if any([p in parname for p in ['size', 'fwhm', 'ntot']]):
                if params[parname].min < 0:
                    params[parname].set(min=0)

        # Check min/max Tex:
        for icpt, cpt in enumerate(self.cpt_list):
            par = params[f'{cpt.name}_tex']
            mini = cpt.tmin if not isinstance(par.min, (float, int)) else max(par.min, cpt.tmin)
            maxi = cpt.tmax if not isinstance(par.max, (float, int)) else min(par.max, cpt.tmax)
            if par.min < cpt.tmin and self.minimize:
                sp_tmin = [sp for sp in cpt.species_list if min(sp.pf[0]) == cpt.tmin]
                if len(sp_tmin) > 1:
                    tags_tmin = ", ".join([sp.tag for sp in sp_tmin])
                else:
                    tags_tmin = sp_tmin[0].tag
                message = [
                    f'The requested lower boundary for Tex is {par.min} K, but the lowest temperature',
                    f'for which the partition function (pf) is defined for all species is {cpt.tmin} K;',
                    f'Species for which T_pf_min = {cpt.tmin} is/are: {tags_tmin}',
                    f'-> limiting Tex search to temperatures > {cpt.tmin} \n'
                ]
                print(f'INFO - Component {cpt.name} :')
                for line in message:
                    print(" " * 6, line)
                # print(f'Component {cpt.name} : limiting Tex search to temperatures > {cpt.tmin} '
                #       f'(smallest temperature for which the partition function is defined for all species).')
                val = (mini + maxi) / 2
            if par.max > cpt.tmax and self.minimize:
                sp_tmax = [sp for sp in cpt.species_list if max(sp.pf[0]) == cpt.tmax]
                if len(sp_tmax) > 1:
                    tags_tmax = ", ".join([sp.tag for sp in sp_tmax])
                else:
                    tags_tmax = sp_tmax[0].tag
                message = [
                    f'The requested upper boundary for Tex is {par.max} K, but the highest temperature',
                    f'for which the partition function (pf) is defined for all species is {cpt.tmax} K;',
                    f'Species for which T_pf_max = {cpt.tmax} is/are: {tags_tmax}',
                    f'-> limiting Tex search to temperatures < {cpt.tmax} \n'
                ]
                print(f'INFO - Component {cpt.name} :')
                for line in message:
                    print(" " * 6, line)
                # print(f'Component {cpt.name} : limiting Tex search to temperatures < {cpt.tmax} '
                #       f'(highest temperature for which the partition function is defined for all species).')
                val = (mini + maxi) / 2

            par.set(min=mini, max=maxi)

        return params

    def generate_lte_model(self, normalize=False):
        if self.params is None:
            self.make_params(normalize=normalize)

        self.model = Model(generate_lte_model_func(self.model_info()),
                           independent_vars=['fmhz', 'log', 'cpt', 'line_center_only', 'return_tau'
                                             # 'tc', 'beam_sizes', 'tmb2ta', 'jypb2k'
                                             ]
                           )

    def do_modeling(self):
        if self.model_config.line_list_all is None:
            self.model_config.get_linelist()
        if len(self.model_config.win_list) == 0:
            self.get_windows()
        if self.minimize and len(self.model_config.win_list_fit) == 0:
            self.model_config.get_data_to_fit()

        if self.params is None:
            self.make_params(json_params=self.model_config.jparams)
            self.generate_lte_model()

        if self.plot_gui or self.plot_file:
            self.do_plots()

    def do_minimization(self, print_report: 'short' | 'long' | None=None, report_kws=None):
        if self.x_fit is None:
            return None

        if print_report is None:
            print_report = 'long' if self.model_config.print_report else 'short'
        # if self.model_config.line_list_all is None:
        #     self.model_config.get_linelist()
        # if len(self.model_config.win_list) == 0:
        #     self.get_windows()
        # if len(self.model_config.win_list_fit) == 0:
        #     self.model_config.get_data_to_fit()
        if self.model_config.jmodel_fit is not None:
            self.model_fit = ModelResult(self.model, self.params)
            self.model_fit.components = self.model.components
            self.model_fit.loads(self.model_config.jmodel_fit,
                                 funcdefs={'lte_model_func': generate_lte_model_func(self.model_info())})

        if self.params is None:
            self.make_params()

        if self.model_config.jparams is not None:
            start_pars = Parameters().loads(self.model_config.jparams)
            for parname in self.params:
                self.params[parname] = start_pars[parname]

        # if self.model is None:
        #     self.generate_lte_model()
        self.generate_lte_model()

        t_start = process_time()
        # Perform the fit
        self.fit_model(max_nfev=self.max_iter, fit_kws=self.fit_kws)
        t_stop = process_time()
        if self.exec_time:
            print("Execution time for minimization : {}.".format(utils.format_time(t_stop - t_start)))

        # If finite tau_lim, check if some lines have tau0 > tau_lim, if so, drop them from win_list_fit
        # and re-do minimization
        if self.model_config.tau_lim < np.inf:
            print("Checking line-center opacities...")
            rerun = True
            while rerun:
                for win in self.model_config.win_list_fit:
                    for cpt in self.model_config.cpt_list:
                        par_names = [f'{cpt.name}_ntot_{win.transition.tag}',
                                     f'{cpt.name}_fwhm_{win.transition.tag}',
                                     f'{cpt.name}_tex']
                        par_vals = []
                        for pname in par_names:
                            par = self.model_fit.params[pname]
                            if par.user_data is not None and par.user_data.get('log', False):
                                val = 10 ** par.value if par.vary or par.expr is not None else par.user_data['value']
                            else:
                                val = par.value
                            par_vals.append(val)
                        tau0 = utils.compute_tau0(win.transition, par_vals[0], par_vals[1], par_vals[2])
                        if tau0 > self.model_config.tau_lim:
                            win.in_fit = False
                            break
                if len([w for w in self.model_config.win_list if w.in_fit]) < len(self.model_config.win_list_fit):
                    print("Some opacities are above the user-defined limit. Re-run the minimization.")
                    # update data to be fitted and min/max values if factor
                    self.model_config.get_data_to_fit(update=True)
                    for par in self.params:
                        p = self.params[par]
                        if p.user_data is not None and 'factor' in p.user_data:
                            p.set(min=p.value*p.user_data.get('min_fact', 1.),
                                  max=p.value*p.user_data.get('max_fact', 1.))
                    t_start = process_time()
                    # Perform the fit
                    self.fit_model(max_nfev=self.max_iter, fit_kws=self.fit_kws)
                    t_stop = process_time()
                    if self.exec_time:
                        print("Execution time for minimization : {}.".format(utils.format_time(t_stop - t_start)))
                else:
                    print("All line-center are below the user-defined limit.")
                    rerun = False

        # reset norm factors and log scale
        self.norm_factors = {key: 1 for key in self.norm_factors.keys()}
        self.log = False

        # save parameters for re-use
        # if self.model_config.jparams is None:
        #     self.model_config.jparams = self.params.dumps()
        if self.model_config.latest_valid_params is None:
            self.model_config.latest_valid_params = self.params.copy()
        else:
            REL_TOL = 1e-3
            for parname, par in self.params.items():
                tol = np.abs(par.max - par.min) * 0.1  # 10% of the range
                if par.stderr != 0 and all([not math.isclose(par.value, extrem, abs_tol=tol)
                                            for extrem in [par.min, par.max]]):
                # if par.stderr != 0 and (par.min + tol < par.value < par.max - tol):
                    self.model_config.latest_valid_params[parname] = self.params[parname]

        if print_report == 'long':
            # print(self.model_fit.fit_report())
            print(self.fit_report(report_kws=report_kws))
            print("")
        elif print_report == 'short':
            print("Best values:")
            for cpt in self.model_config.cpt_list:
                res = []
                for p in cpt.parameters:
                    res.append(f"{p.name} = {utils.format_float(p.value, nb_signif_digits=2)}")
                print(f"  " + " ; ".join(res))
            print("")
        else:
            raise KeyError("print_report can only be 'long' or 'short'.")

        self.do_savings()
        self.do_plots()

        return {
            'iterations': self.model_fit.nfev,
            'Exec time (s)': t_stop - t_start,
            'red_chi2': self.model_fit.redchi,
            'params': self.model_fit.params
        }

    def do_savings(self):
        if self.model_config.fit_full_range:
            self.save_line_list_cassis("linelist_cassis", snr_threshold=self.model_config.snr_threshold)

        if self.model_fit is not None and self.save_results:
            # filename = ''
            # if self.name_lam is not None:
            #     filename = self.name_lam + '_'
            # filename = filename + 'fit_res'
            self.save_fit_results(self.model_config.output_files['results'])

        if self.model_config.save_infos_components:  # save a new info components
            if self.model_config.comp_config_file is not None:
                comp_name, ext = os.path.splitext(self.model_config.comp_config_file)
                comp_file_new = comp_name + '_from_results' + ext
                # header1, header2 = None, None
                with open(self.model_config.comp_config_file) as fin:
                    all_lines = fin.readlines()
                    hdr_comp_params = [(i, line) for i, line in enumerate(all_lines) if line.startswith('name')]
                    hdr2_end, hdr_thr = [(i, line) for i, line in enumerate(all_lines) if line.startswith('tag')][0]
                    col_names_thr = hdr_thr.split()
                    header1 = all_lines[:hdr_comp_params[0][0]]
                    hdr2_start = [i for i, line in enumerate(all_lines) if 'SPECIES INFOS' in line][0]
                    header2 = all_lines[max(0, hdr2_start-1):hdr2_end]

                n_digits = 2
                with open(comp_file_new, 'w') as f:
                    f.writelines(header1)
                    f.write(hdr_comp_params[0][1])
                    for comp in self.model_config.cpt_list:
                        pars = [par for par in comp.parameters if 'ntot' not in par.name]
                        for par in pars:
                            pmin = par.min if abs(par.min) not in [np.inf, float("inf")] else par.value
                            pmax = par.max if abs(par.max) not in [np.inf, float("inf")] else par.value
                            line = "\t".join([par.name, utils.format_float(pmin, nb_signif_digits=n_digits),
                                              utils.format_float(par.value, nb_signif_digits=n_digits),
                                              utils.format_float(pmax, nb_signif_digits=n_digits), str(par.vary)])
                            f.write(line + "\n")

                    f.writelines(header2)

                    for tag in self.model_config.tag_list:
                        f.write(hdr_thr)
                        thr = self.model_config.thresholds[tag]
                        elts = [str(tag)]
                        for name in col_names_thr[1:]:
                            elt = thr[name]
                            elts.append('*' if elt is None else utils.format_float(elt, nb_signif_digits=1))
                        f.write("\t".join(elts) + "\n")
                        f.write(hdr_comp_params[0][1])
                        for comp in self.model_config.cpt_list:
                            try:
                                species = [sp for sp in comp.species_list if sp.tag == tag][0]
                                Ntot = species.parameters[0]
                                Nmin = utils.format_float(Ntot.min, nb_signif_digits=n_digits)
                                Nmax = utils.format_float(Ntot.max, nb_signif_digits=n_digits)
                                if Ntot.user_data is not None and 'factor' in Ntot.user_data:
                                    Nmin = utils.format_float(Ntot.user_data['min_fact'], fmt='{:.1e}')
                                    Nmax = utils.format_float(Ntot.user_data['max_fact'], fmt='{:.1e}')
                                line = "\t".join([f"{comp.name}_ntot", Nmin,
                                                  utils.format_float(Ntot.value, nb_signif_digits=n_digits), Nmax,
                                                  str(Ntot.vary)])
                                f.write(line + "\n")

                            except IndexError:
                                pass  # do nothing

                        f.write("\n")

        if self.model is not None and self.save_model_spec:
            filename, ext = os.path.splitext(self.model_config.output_files['model'])
            if len(ext) == 0:
                ext = 'txt'
            self.save_model(filename, ext=ext)

        if len(self.x_file) > 0 and self.save_obs_spec:
            filename, ext = os.path.splitext(self.model_config.output_files['obs'])
            if len(ext) == 0:
                ext = 'txt'
            self.save_spectrum(filename, ext=ext, spectrum_type='observed')
            if self.model_config.yunit in UNITS['flux']:
                self.save_spectrum(filename + '_K', ext=ext, spectrum_type='observed', yunit='K')

        if (self.modeling or self.minimize) and self.save_configs:
            # must be last to make sure we have appropriate data file in memory
            if 'lam' in self.model_config.output_files:
                # save both lam and ltm if no baseline correction (local correction)
                self.write_lam(self.model_config.output_files['lam'])
                if not self.model_config.bl_corr:
                    self.write_ltm(self.model_config.output_files['lam'])

            if 'config' in self.model_config.output_files:
                self.save_config(self.model_config.output_files['config'])

    def do_plots(self):
        if self.plot_gui or self.plot_file:
            print('Finding windows for gui and file plots.')
            self.setup_plot()
            if self.plot_gui and len(self.model_config.win_list_gui) > 0:
                t_start = process_time()
                print("Preparing windows for GUI plot...")
                if self.bandwidth is None or self.model_config.fit_full_range:
                    self.setup_plot_fus()
                else:
                    self.setup_plot_la(self.model_config.win_list_gui, **self.gui_kws)
                if self.exec_time:
                    print(f"Execution time for preparing GUI plot : {utils.format_time(process_time() - t_start)}.")

                self.make_plot('gui')

            if self.plot_file and len(self.model_config.win_list_file) > 0:
                if ((self.model_config.win_list_file != self.model_config.win_list_gui) or
                        (self.file_kws['model_err'] and not self.gui_kws['model_err']) or
                        (self.file_kws['component_err'] and not self.gui_kws['component_err'])):
                    t_start = process_time()
                    print("Preparing windows for file plot...")
                    # look for windows not already in gui
                    if self.model_config.plot_gui:
                        win_list = [w for w in self.model_config.win_list_file
                                    if w not in self.model_config.win_list_gui]
                    else:
                        win_list = self.model_config.win_list_file

                    if len(win_list) > 0:
                        if self.bandwidth is None or self.model_config.fit_full_range:
                            self.setup_plot_fus()
                        else:
                            self.setup_plot_la(win_list, **self.file_kws)
                    # Compute errors if necessary
                    if self.model_fit.covar is not None:
                        for win in self.model_config.win_list_file:
                            if win.y_mod_err is None and self.file_kws['model_err']:
                                win.y_mod_err = self.model_fit.eval_uncertainty(fmhz=win.x_mod)
                            if len(win.y_mod_err_cpt) == 0 and self.file_kws['component_err']:
                                win.y_mod_err_cpt = {f'c{nc+1}': y for nc, y in
                                                     enumerate(self.eval_uncertainties_components(fmhz=win.x_mod))}
                    else:
                        print("## Warning: could not compute model errors.")

                    if self.exec_time:
                        print(f"Execution time for preparing file plot : {utils.format_time(process_time() - t_start)}.")

                self.make_plot('file')

    def fit_model(self, max_nfev=None, fit_kws=None):
        """
        Computes weights and perform the fit.
        :param max_nfev: maximum number of iterations (default value depends on the algorithm)
        :param fit_kws: keywords for the fit function
        :return:
        """

        def fit_callback(pars, iter, resid, *args, **kws):
            # Function called after each iteration to print the iteration number every 100 iterations
            if iter % 100 == 0:
                print(f"    Iteration {int(iter // 100) * 100 + 1} : chi2 = {sum(resid**2)} ; "
                      f"reduced chi2 = {sum(resid**2) / (len(resid) - len([p for p in pars if pars[p].vary]))} ...")

        if self.log:  # take log10 for tex and ntot
            params = self.params
            for par in params:
                # if 'tex' in par or 'ntot' in par:
                if 'ntot' in par:
                    user_data = {'value': params[par].value,
                                 'min': params[par].min,
                                 'max': params[par].max,
                                 'log': True}
                    if params[par].user_data is None:
                        params[par].user_data = user_data
                    else:
                        params[par].user_data = {**params[par].user_data, **user_data}
                    if params[par].expr is None:
                        params[par].set(value=np.log10(params[par].value),
                                        min=np.log10(params[par].min),
                                        max=np.log10(params[par].max))

        wt = []
        for win in self.win_list_fit:
            if self.model_config.cont_free:
                tc = 0.
            else:
                tc = self.get_tc(win.x_fit)
            wt.extend(utils.compute_weight(win.y_fit - tc, win.rms, win.cal))
        wt = np.array(wt)

        # wt = None
        method = fit_kws.get('method', 'leastsq')
        # print(f'Performing minimization with the {method} method...')
        if 'method' in fit_kws:
            fit_kws.pop('method')

        cb = None
        if self.print_report and method != "emcee":
            cb = fit_callback

        """NB: nan_policy:
        'raise': Raise a ValueError (default)
        'propagate': Do not check for NaNs or missing values. The fit will try to ignore them.
        'omit': Remove NaNs or missing observations in data."""
        self.model_fit = self.model.fit(self.y_fit, params=self.params, fmhz=self.x_fit, log=self.log,
                                        # tc=self.tc(self.x_fit), beam_sizes=self.beam(self.x_fit),
                                        # tmb2ta=self.tmb2ta(self.x_fit), jypb2k=self.jypb(self.x_fit),
                                        nan_policy='omit',  #
                                        cpt='from_config',
                                        line_center_only=False, return_tau=False,
                                        weights=wt,
                                        method=method,
                                        max_nfev=max_nfev, fit_kws=fit_kws,
                                        iter_cb=cb)
        if not self.print_report:
            if self.model_fit.nfev == self.model_fit.max_nfev:
                message = f"Maximum number of iterations reached ({self.model_fit.max_nfev}) ; "
            else:
                message = f"Fit performed in {self.model_fit.nfev} iterations ; "
            print(message + f"reduced chi-square = {self.model_fit.redchi:.2f}.")
        # if len(self.cpt_list) > 1:
        #     for cpt in self.cpt_list:
        #         model_fit_cpt = copy.deepcopy(self.model_fit)
        #         c_par = Parameters()
        #         for par in model_fit_cpt.params:
        #             if cpt.name in par:
        #                 c_par[par] = self.model_fit.params[par]
        #         model_fit_cpt.params = c_par
        #         indices = [i for i, var_name in enumerate(model_fit_cpt.var_names) if cpt.name in var_name]
        #         nvarys = len(indices)
        #         model_fit_cpt.var_names = [model_fit_cpt.var_names[i] for i in indices]
        #         model_fit_cpt.best_values = {k: v for k, v in model_fit_cpt.best_values.items() if cpt.name in k}
        #         model_fit_cpt.nvarys = nvarys
        #         covar = np.zeros((nvarys, nvarys))
        #         for i in range(nvarys):
        #             for j in range(nvarys):
        #                 covar[i, j] = model_fit_cpt.covar[indices[i], indices[j]]
        #         model_fit_cpt.covar = covar
        #         model_fit_cpt.model.func = generate_lte_model_func(self.model_info(cpt=cpt))
        #         self.model_fit_cpt.append(model_fit_cpt)

        self.model = self.model_fit.model

        # update parameters
        self.params = self.model_fit.params.copy()
        for par in self.params:
            p = self.params[par]
            pfit = self.model_fit.params[par]

            p.correl = pfit.correl

            nf = self.norm_factors[p.name]
            if nf != 1.:
                p.set(min=pfit.min * nf, max=pfit.max * nf, value=pfit.value * nf, is_init_value=False, expr=pfit.expr)
                if pfit.stderr is not None:
                    p.stderr = nf * pfit.stderr

            if p.user_data is not None and p.user_data.get('log', False):
                p.init_value = p.user_data['value']
                if pfit.stderr is not None:
                    try:
                        p.stderr = (10 ** (pfit.value + pfit.stderr) - 10 ** (pfit.value - pfit.stderr)) / 2
                    except (OverflowError, RuntimeWarning):
                        p.stderr = np.inf
                val = round(10 ** pfit.value, 6) if p.vary or p.expr is not None else p.user_data['value']
                p.set(value=val, min=p.user_data['min'], max=p.user_data['max'], is_init_value=False, expr=pfit.expr)
                p.user_data['log'] = False

        # update components
        for cpt in self.model_config.cpt_list:
            cpt.update_parameters(self.params)

        # update vlr_plot
        if self.vlsr_file == 0:
            self.model_config.vlsr_plot = self.cpt_list[0].vlsr

    def fit_report(self, report_kws=None):
        fit_params = self.model_fit.params.copy()
        self.model_fit.params = self.params

        if report_kws is None:
            report_kws = {}

        if 'show_correl' not in report_kws:
            report_kws['show_correl'] = False

        report = self.model_fit.fit_report(**report_kws)
        pvalue = '    p-value            = {}'.format(
            utils.format_float(stats.chi2.sf(self.model_fit.chisqr, self.model_fit.nfree))
        )
        lines = report.split(sep='\n')
        new_lines = []
        lmax = 6 + NB_DECIMALS
        for line in lines:
            if '(init' in line:  # reformat initial value
                begin_line, end_line = line.rsplit(sep=" (", maxsplit=1)
                init_label, init_val = end_line.split(' = ')
                init_val = init_val.strip(')')
                end_line = f"({init_label} = {utils.format_float(float(init_val))})"

                err = ""
                if "+/-" in begin_line:  # redefine begin_line and reformat error
                    begin_line, rest = begin_line.split(sep="+/-")
                    rest = rest.strip()
                    err, pc = rest.split(' ', maxsplit=1)
                    err = f" +/- {utils.format_float(float(err)): <{lmax}} {pc}"

                begin_line = begin_line.rstrip()
                label, val = begin_line.rsplit(sep=' ', maxsplit=1)
                begin_line = f"{label} {utils.format_float(float(val)): <{lmax}}"

                new_lines.append(" ".join([begin_line, err, end_line]))

            elif "=" in line and ":" not in line and "#" not in line:  # reformat statistics if not integers
                elts = line.rsplit(sep="=", maxsplit=1)
                new_lines.append("= ".join([elts[0], utils.format_float(float(elts[1]))]))

            elif "fixed" in line:  # keep if expr is None, else ignore
                par = line.split(sep=":")[0].strip()
                if self.params[par].expr is None:
                    new_lines.append(line)

            else:  # keep
                new_lines.append(line)

        self.model_fit.params = fit_params

        return '\n'.join(new_lines[:9] + [pvalue] + new_lines[9:])

    def eval_uncertainties_components(self, fmhz, sigma=1):
        """From lmfit.model"""
        res = []

        if self.model_fit.covar is not None:
            for cpt in self.cpt_list:
                params = Parameters()
                for par in self.params:
                    if cpt.name in par:
                        params[par] = self.params[par]

                indices = [i for i, var_name in enumerate(self.model_fit.var_names) if cpt.name in var_name]
                var_names = [self.model_fit.var_names[i] for i in indices]
                nvarys = len(var_names)
                cpt_model_func = generate_lte_model_func(self.model_info(cpt=cpt))

                # ensure fjac and df2 are correct size if independent var updated by kwargs
                feval = cpt_model_func(fmhz=fmhz, log=False, **params)
                ndata = len(feval.view('float64'))        # allows feval to be complex
                covar = np.zeros((nvarys, nvarys))
                for i in range(nvarys):
                    for j in range(nvarys):
                        covar[i, j] = self.model_fit.covar[indices[i], indices[j]]
                if any(p.stderr is None for p in params.values()):
                    res.append(np.zeros(ndata))
                    continue

                fjac = np.zeros((nvarys, ndata), dtype='float64')
                df2 = np.zeros(ndata, dtype='float64')

                # find derivative by hand!
                pars = params.copy()
                for i in range(nvarys):
                    pname = var_names[i]
                    val0 = pars[pname].value
                    dval = pars[pname].stderr/3.0
                    pars[pname].value = val0 + dval
                    res1 = cpt_model_func(fmhz=fmhz, log=False, **pars)

                    pars[pname].value = val0 - dval
                    res2 = cpt_model_func(fmhz=fmhz, log=False, **pars)

                    pars[pname].value = val0
                    fjac[i] = (res1.view('float64') - res2.view('float64')) / (2*dval)

                for i in range(nvarys):
                    for j in range(nvarys):
                        df2 += fjac[i] * fjac[j] * covar[i, j]

                if sigma < 1.0:
                    prob = sigma
                else:
                    prob = erf(sigma/np.sqrt(2))

                scale = t.ppf((prob+1)/2.0, self.model_fit.ndata-nvarys)

                res.append(scale * np.sqrt(df2))

        return res

    def compute_model_intensities(self, params=None, x_values=None, line_list=None, line_center_only=False,
                                  cpt=None):
        if x_values is None:
            x_values = self.x_mod

        if type(x_values) is list:
            x_values = np.array(x_values)

        if self.model is None:
            self.generate_lte_model()

        if params is None:
            params = self.params

        if cpt is None:
            cpt = self.model_config.cpt_list

        return self.model.func(x_values, log=False, cpt=cpt, line_center_only=line_center_only, return_tau=False,
                               **params)
        # if cpt is not None:
        #     c_best_pars = {}
        #     for pname, par in params.items():
        #         if cpt.name in pname:
        #             c_best_pars[pname] = par.value
        #     params = c_best_pars
        #     return self.model_cpt[cpt.name].func(x_values, log=False, **params)
        # else:
        #     return self.model.func(x_values, log=False, **params)

        # if cpt is not None:
        #     lte_func = generate_lte_model_func(self.model_info(x_values, line_list=line_list,
        #                                                        cpt=[cpt]))
        # else:
        #     lte_func = generate_lte_model_func(self.model_info(x_values, line_list=line_list,
        #                                                        line_center_only=line_center_only))
        #
        # return lte_func(x_values, **params)

    def compute_model(self, params=None, x_values=None, line_list=None, line_center_only=False):
        """
        For backward compatibility.
        :param params:
        :param x_values:
        :param line_list:
        :param line_center_only:
        :return:
        """
        return self.compute_model_intensities(params=params, x_values=x_values, line_list=line_list,
                                              line_center_only=line_center_only)

    def integrated_intensities(self):
        pars = self.params
        res = {}
        for cpt in self.cpt_list:
            fluxes = []
            for win in self.win_list:
                f_ref = win.transition.f_trans_mhz
                fwhm = pars['{}_fwhm_{}'.format(cpt.name, win.transition.tag)].value
                vlsr = pars['{}_vlsr'.format(cpt.name)].value
                fmin_mod, fmax_mod = [utils.velocity_to_frequency(vlsr + v, f_ref, vref_kms=self.vlsr_file)
                                      for v in [fwhm, -fwhm]]
                # x_file_win = self.x_file[(fmin_mod <= self.x_file) & (self.x_file <= fmax_mod)]
                # x_mod = np.linspace(min(x_file_win), max(x_file_win),
                #                     num=self.oversampling * len(x_file_win))
                npts = 100
                x_mod = np.linspace(fmin_mod, fmax_mod, num=npts)
                y_mod = self.compute_model_intensities(params=pars, x_values=x_mod, line_list=[win.transition])

                dv = 2. * fwhm / (npts - 1)
                K_kms = 0.
                for i in range(len(y_mod) - 1):
                    K_kms += np.mean([y_mod[i], y_mod[i+1]]) * dv

                fluxes.append([win.transition.tag, win.plot_nb, f_ref, K_kms])

            res[cpt.name] = pd.DataFrame(fluxes, columns=['tag', 'line number', 'f_mhz', 'K.km/s'])

        return res

    def setup_plot_fus(self):  # TODO: rename/refactor, most instructions are for model only
        """
        Plot in full spectrum mode (self.bandwidth is None)
        :return:
        """
        # info to be saved in line list file
        # cols = ['tag', 'sp_name', 'fMHz', 'f_err_mhz', 'aij', 'elow', 'eup', 'igu', 'catdir_id', 'qn']
        cols = ['tag', 'sp_name', 'fMHz', 'f_err_mhz', 'aij', 'elow', 'eup', 'igu', 'qn']

        for win in self.win_list_plot:
            # win.x_mod = self.x_mod
            # win.f_range_plot = [min(win.x_file), max(win.x_file)]
            win.f_range_plot = [min(win.x_mod), max(win.x_mod)]
            win.bottom_unit = 'MHz'
            win.bottom_lim = utils.get_extended_limits(win.f_range_plot)
            win.top_lim = win.bottom_lim

            # compute the model :
            plot_pars = self.params
            # win.x_mod, win.y_mod = select_from_ranges(self.x_mod, win.f_range_plot, y_values=self.y_mod)
            # TODO: check if following is necessary
            # if win.x_file is not None:
            #     x_mod = []
            #     for i, row in self.tuning_info.iterrows():
            #         x_sub = win.x_file[(win.x_file >= row['fmhz_min']) & (win.x_file <= row['fmhz_max'])]
            #         if len(x_sub) == 0:
            #             continue
            #         x_mod.extend(np.linspace(min(x_sub), max(x_sub), num=self.oversampling * len(x_sub)))
            #     win.x_mod = np.array(x_mod)
            #
            # else:
            #     pass
            line_list = select_transitions(self.line_list_all, xrange=[min(win.x_mod), max(win.x_mod)])

            # Create file for saving the lines if model only
            if not self.model_config.minimize:
                with open(os.path.join(self.output_dir, 'linelist.txt'), "w") as f:
                    title = f"{win.name} : list of modelled lines"
                    # if self.model_config.minimize:
                    #     if self.model_config.f_err_mhz_max is not None:
                    #         title += f"with f_err_mhz <= {self.model_config.f_err_mhz_max}"
                    # else:
                    #     title += "within thresholds"
                    title += "within thresholds"
                    f.write(title + "\n\n")
                    # f.writelines(line_list[cols].to_string(index=False))
                    for itag, tag in enumerate(self.model_config.tag_list):
                        line_list_tag = line_list[line_list.tag == tag]
                        line_list_tag = line_list_tag.reset_index(drop=True)
                        line_list_tag.index += 1
                        f.writelines(line_list_tag[cols].to_string(index=True))
                        f.write("\n\n")

            win.y_mod = self.compute_model_intensities(params=plot_pars, x_values=win.x_mod,
                                                       line_list=self.line_list_all)
            if len(self.cpt_list) > 1:
                for cpt in self.cpt_list:
                    win.y_mod_cpt[cpt.name] = self.compute_model_intensities(params=plot_pars, x_values=win.x_mod,
                                                                             line_list=self.line_list_all,
                                                                             cpt=cpt)

            win.x_mod_plot = win.x_mod
            win.x_file_plot = win.x_file

            if win.x_file is not None:
                win.y_res = win.y_file - self.compute_model_intensities(params=plot_pars, x_values=win.x_file,
                                                                        line_list=self.line_list_all)
                win.y_res += self.get_tc(win.x_file)

    def setup_plot_la(self, win_list: list, verbose=True, other_species_dict: dict | None = None, **kwargs):
        """
        Prepare all data to do the plots in line analysis mode
        :param win_list: the list of windows
        :param verbose:
        :param other_species_dict: a dictionary of other species and their thresholds
        :return:
        """

        # Define some useful quantities
        plot_pars = self.params
        # vlsr = self.cpt_list[0].vlsr if self.vlsr_file == 0. else self.vlsr_file
        vlsr = self.vlsr_file
        if len(self.cpt_list) > 1 and self.vlsr_file == 0.:
            vlsr = self.cpt_list[0].vlsr
        try:
            fwhm = max([plot_pars[par].value for par in plot_pars if 'fwhm' in par])
        except TypeError:
            fwhm = 0.

        # self.update_parameters(params=plot_pars)

        if other_species_dict is not None:  # list of tags for which the user wants line positions
            thresholds_other = other_species_dict
        else:
            thresholds_other = self.thresholds_other if self.thresholds_other is not None else {}

        # for t in self.tag_list:
        #     if t in thresholds_other:
        #         thresholds_other.pop(t)
        list_other_species = list(thresholds_other.keys())

        # lines from other species : if many other species, more efficient to first find all transitions
        # across entire observed range, then filter in each window
        if len(list_other_species) > 0:
            other_species_lines = get_transition_df(list_other_species,
                                                    [[min(win.x_file), max(win.x_file)] for win in win_list],
                                                    **thresholds_other)
        # other_species_lines = get_transition_df(list_other_species, [[min(self.x_file), max(self.x_file)]],
        #                                         **thresholds_other)
        else:
            other_species_lines = pd.DataFrame()  # empty dataframe

        # info to be saved in line list file
        # cols = ['tag', 'sp_name', 'fMHz', 'f_err_mhz', 'aij', 'elow', 'eup', 'igu', 'catdir_id', 'qn']
        cols = ['tag', 'sp_name', 'x_pos', 'fMHz', 'f_err_mhz', 'aij', 'elow', 'eup', 'igu', 'qn']

        # Compute model overall model : takes longer than cycling through windows unless strong overlap of windows (TBC)
        # self.y_mod = self.compute_model_intensities(params=plot_pars, x_values=self.x_mod,
        #                                             line_list=self.line_list_all)

        # Compute model and line positions for each window
        t_win = datetime.datetime.now()
        print(f"Start preparing windows : {t_win.strftime('%H:%M:%S')}...")
        t_start = process_time()
        for iwin, win in enumerate(win_list):
            tr = win.transition
            f_ref = tr.f_trans_mhz
            win.v_range_plot = [-self.bandwidth / 2 + vlsr, self.bandwidth / 2 + vlsr]
            win.f_range_plot = [utils.velocity_to_frequency(v, f_ref, vref_kms=vlsr)
                                for v in win.v_range_plot]
            win.bottom_unit = 'km/s'
            win.bottom_lim = utils.get_extended_limits(win.v_range_plot)
            win.top_lim = [utils.velocity_to_frequency(v, f_ref, vref_kms=vlsr)
                           for v in win.bottom_lim]

            # all transitions in the window (no thresholds), to compute the model :
            fwhm_mhz = utils.delta_v_to_delta_f(fwhm, f_ref)
            # model_lines_win = get_transition_df(self.tag_list, [min(win.f_range_plot) - 0.5 * fwhm_mhz,
            #                                                     max(win.f_range_plot) + 0.5 * fwhm_mhz])
            model_lines_win = select_transitions(self.line_list_all, xrange=[min(win.f_range_plot) - 0.5 * fwhm_mhz,
                                                                             max(win.f_range_plot) + 0.5 * fwhm_mhz])

            # compute the model :
            # win.x_mod, win.y_mod = select_from_ranges(self.x_mod, win.f_range_plot, y_values=self.y_mod)
            if self.modeling or self.minimize:
                if self.oversampling == 1:
                    win.x_mod = win.x_file
                else:
                    win.x_mod = np.linspace(min(win.x_file), max(win.x_file), num=self.oversampling * len(win.x_file))
                if self.vlsr_file == 0.:
                    win.x_mod_plot = utils.frequency_to_velocity(win.x_mod, f_ref, vref_kms=self.vlsr_file)
                else:
                    win.x_mod_plot = utils.frequency_to_velocity(win.x_mod, f_ref, vref_kms=vlsr)

            if win.x_file is not None:
                if self.vlsr_file == 0.:
                    win.x_file_plot = utils.frequency_to_velocity(win.x_file, f_ref, vref_kms=self.vlsr_file)
                else:
                    win.x_file_plot = utils.frequency_to_velocity(win.x_file, f_ref, vref_kms=vlsr)

            if self.modeling or self.minimize:
                win.y_mod = self.model.eval(fmhz=win.x_mod, **plot_pars)
                if self.oversampling == 1:  # x_mod=x_file, so no need to recompute
                    win.y_res = win.y_file - win.y_mod
                else:
                    win.y_res = win.y_file - self.model.eval(fmhz=win.x_file, **plot_pars)
                if not self.model_config.cont_free:
                    win.y_res += self.get_tc(win.x_file)

            if self.model_fit is not None:
                if 'model_err' in kwargs and kwargs['model_err']:
                    # win.y_mod_err = self.model_fit.eval_uncertainty(fmhz=win.x_mod, cpt=self.cpt_list[0], params=c_par)
                    win.y_mod_err = self.model_fit.eval_uncertainty(fmhz=win.x_mod)
                if 'component_err' in kwargs and kwargs['component_err']:
                    # c_par = Parameters()
                    # for par in self.model_fit.params:
                    #     if self.cpt_list[0].name in par:
                    #         c_par[par] = self.model_fit.params[par]
                    # # self.model_fit.params = c_par
                    # self.model_fit.var_names = [par for par in c_par]
                    # self.model_fit.nvarys = len(c_par)
                    # # self.model_fit.params = plot_pars
                    # self.model_fit.var_names = [par for par in plot_pars]
                    # self.model_fit.nvarys = len(plot_pars)
                    # win.y_mod_err_cpt = [fit_cpt.eval_uncertainty(fmhz=win.x_mod, cpt=cpt)
                    #                      for fit_cpt, cpt in zip(self.model_fit_cpt, self.cpt_list)]
                    win.y_mod_err_cpt = {f'c{nc+1}': y
                                         for nc, y in enumerate(self.eval_uncertainties_components(fmhz=win.x_mod))}

            if (self.modeling or self.minimize) and (len(self.cpt_list) > 1):
                for cpt in self.cpt_list:
                    win.y_mod_cpt[cpt.name] = self.compute_model_intensities(params=plot_pars, x_values=win.x_mod,
                                                                             line_list=model_lines_win,
                                                                             cpt=cpt)

            # transitions from model species, w/i thresholds, for display :
            model_lines_user = select_transitions(model_lines_win,
                                                  thresholds=self.thresholds)
            # find "bright" lines (if aij_max not None and/or eup_min non-zero):
            # bright_lines = select_transitions(all_lines_win,  # xrange=[fmin, fmax],
            #                                   thresholds=self.thresholds,
            #                                   # bright_lines_only=True)

            # transitions from model species, outside thresholds :
            # model_lines_other = pd.concat([model_lines_user,
            #                                model_lines_win]).drop_duplicates(subset='db_id', keep=False)

            # transitions from other species :
            # other_species_win_all = get_transition_df(list_other_species,
            #                                           [[min(win.f_range_plot), max(win.f_range_plot)]],
            #                                           **thresholds_other)

            # if win.f_range_fit is not None:
            #     other_species_win_all = select_transitions(other_species_lines,
            #                                                xrange=[min(win.f_range_fit), max(win.f_range_fit)],
            #                                                vlsr=vlsr if self.vlsr_file == 0 else None)
            # else:
            #     other_species_win_all = select_transitions(other_species_lines,
            #                                                xrange=[min(win.f_range_plot), max(win.f_range_plot)],
            #                                                vlsr=vlsr if self.vlsr_file == 0 else None)
            other_species_win = select_transitions(other_species_lines,
                                                   xrange=[min(win.f_range_plot), max(win.f_range_plot)],
                                                   vlsr=vlsr if self.vlsr_file == 0 else None)

            # Drop transition if already in model_lines_user
            if not other_species_win.empty:
                for i, row in other_species_win.iterrows():
                    if row['db_id'] in list(model_lines_user['db_id']):
                        other_species_win.drop([i], inplace=True)
                # NB: cannot use the following because not all lines in model_lines_user are in other_species_win
                # other_species_win = pd.concat([model_lines_user,
                #                                other_species_win]).drop_duplicates(subset='db_id', keep=False)

            win_colors = {t: PLOT_COLORS[itag] for itag, t in enumerate(model_lines_user.tag.unique())}
            if len(self.cpt_list) > 0:
                for icpt, cpt in enumerate(self.cpt_list):
                    # build list of dataframes containing lines to be plotted for each component
                    win.main_lines_display[icpt] = self.get_lines_plot_params(
                        model_lines_user[model_lines_user['tag'].isin(cpt.tag_list)], cpt.vlsr, f_ref,
                        tag_colors=win_colors)
                    # win.other_lines_display[icpt] = self.get_lines_plot_params(
                    #     model_lines_other[model_lines_other['tag'].isin(cpt.tag_list)], cpt, f_ref, tag_colors=win_colors)
            else:
                win.main_lines_display[0] = self.get_lines_plot_params(
                    model_lines_user[model_lines_user['tag'].isin(self.tag_list)], self.model_config.vlsr_plot, f_ref,
                    tag_colors=win_colors)

            if len(other_species_win) > 0:
                # line plot parameters for other lines and other species (not component-dependent)
                plot_colors_sub = PLOT_COLORS[len(win_colors):]
                icol = 0
                for itag, t in enumerate(other_species_win.tag.unique()):
                    if t not in win_colors.keys():
                        win_colors[t] = plot_colors_sub[icol % len(plot_colors_sub)]
                        icol += 1

                win.other_species_display = self.get_lines_plot_params(other_species_win, self.cpt_list[0].vlsr, f_ref,
                                                                       tag_colors=win_colors)

            win.tag_colors = win_colors

            if iwin == 0:
                prep_time = (process_time() - t_start)
                print(f"    Time for one window : {prep_time:.2f} seconds")
                prep_time *= len(win_list)
                t_win += datetime.timedelta(seconds=prep_time)
                print(f"    Expected end time for {len(win_list)} windows: {t_win.strftime('%H:%M:%S')}")

            # Create file for saving the lines
            with open(os.path.join(self.output_dir, 'linelist.txt'), "w") as f:
                f.write("List of plotted lines\n\n")

                f.write(f"{win.name} : model species within thresholds (top lines)\n")
                f.writelines(win.main_lines_display[0][cols].to_string(index=False))
                f.write("\n\n")
                if len(win.other_species_display) > 0:
                    f.write(f"{win.name} : other species within thresholds (bottom lines)\n")
                    f.writelines(win.other_species_display[cols].to_string(index=False))
                    f.write("\n\n")

    def get_lines_plot_params(self, line_list: pd.DataFrame, vlsr: float, f_ref: float,
                              tag_colors: dict):

        colors = tag_colors
        lines_plot_params = line_list.copy()
        lines_plot_params['x_pos'] = [utils.frequency_to_velocity(row.fMHz, f_ref, vref_kms=vlsr)
                                      for i, row in lines_plot_params.iterrows()]
        lines_plot_params['x_pos_err'] = [utils.delta_v_to_delta_f(row.f_err_mhz, f_ref, reverse=True)
                                          for i, row in lines_plot_params.iterrows()]
        lines_plot_params['label'] = [row.tag for i, row in lines_plot_params.iterrows()]
        lines_plot_params['color'] = [colors[row.tag] for i, row in lines_plot_params.iterrows()]

        return lines_plot_params

    def select_windows(self, **kwargs):
        """
        Determine windows to plot
        :param tag: tag selection if do not want all the tags
        :param display_all: if False, only display windows with fitted data
        :param windows: a dictionary of the windows to be plotted (keys=tags, vals=window numbers)
        :return:
        """
        tag = kwargs.get('tag', None)
        if tag is not None:
            if not isinstance(tag, list):
                tag = [tag]
            tag = [str(t) for t in tag]
        display_all = kwargs.get('display_all', True)
        windows = kwargs.get('windows', None).copy()

        win_list_plot = self.win_list  # by default, plot everything

        if not display_all:  # only display windows with fitted data
            win_list_plot = [w for w in win_list_plot if w.in_fit]

        if tag is not None:  # user only wants some tags
            win_list_plot = [w for w in win_list_plot if w.transition.tag in tag]

        if windows is not None and len(windows) > 0:
            if isinstance(windows, dict):
                if '*' in windows.values():
                    for k, v in windows.items():
                        if v == '*':
                            windows[k] = f'1-{len(self.model_config.tr_list_by_tag[k])}'
                new_dict = utils.expand_dict(windows, expand_vals=True)
                win_names2plot = [f'{key} - {val}' for key in new_dict.keys() for val in new_dict[key]]
            else:
                win_names2plot = windows

            win_list_plot = [w for w in win_list_plot if w.name in win_names2plot]

        if len(win_list_plot) == 0:
            raise LookupError("No windows to plot. Please check your tag selection.")

        return win_list_plot

    def select_windows_other_lines(self, other_species_win_selection: str):
        """
        Select windows with other lines from this tag
        :param other_species_win_selection: desired tag
        :return:
        """

        sub_list = []
        for win in self.win_list_plot:  # check if window contains a transition from other_species_selection
            if other_species_win_selection in win.other_lines_display['tag']:
                sub_list.append(win)
        if len(sub_list) == 0:
            warn(f"No windows with transitions from {other_species_win_selection}.")
        else:
            self.win_list_plot = sub_list

    def setup_plot(self):
        """
        Prepare all data to do the plot(s), using provided keywords.
        Possible keywords are :
         tag: tag selection if do not want all the tags
         basic: do not plot other species
         other_species: list or dictionary or file with other species ;
            dictionary and file can contain their thresholds
         other_species_plot: list of other species to plot ; if None, other_species is used ;
            if other_species is provided, only these species are kept
         other_species_win_selection: select only windows with other lines from this tag.
         display_all: if False, only display windows with fitted data
        :return:

        Notes :
            - other_species_selection is deprecated, use other_species_win_selection
        """

        # Unpack keywords :
        kwargs = self.plot_kws
        tag = kwargs.get('tag')
        verbose = kwargs.get('verbose', True)
        # basic = kwargs.get('basic', False)
        other_species = kwargs.get('other_species', None)
        other_species_plot = kwargs.get('other_species_plot', 'all')
        other_species_win_selection = kwargs.get('other_species_win_selection', None)

        # set colors for model tags and components
        self.tag_colors = {t: PLOT_COLORS[itag % len(PLOT_COLORS)] for itag, t in enumerate(self.tag_list)}
        # self.cpt_cols = get_cmap('hsv')(linspace(0.1, 0.8, len(self.cpt_list)))
        self.cpt_cols = [CPT_COLORS[i % len(CPT_COLORS)] for i in range(len(self.cpt_list))]

        if 'other_species_selection' in kwargs.keys():
            warn('other_species_selection will be deprecated, use other_species_win_selection instead',
                 DeprecationWarning, stacklevel=2)
            other_species_win_selection = kwargs['other_species_selection']

        if other_species is not None:
            if other_species_plot == 'all':
                thresholds_other = get_species_thresholds(other_species)
            else:
                thresholds_other = get_species_thresholds(other_species, select_species=other_species_plot)
        else:
            thresholds_other = None

        self.thresholds_other = thresholds_other

        if self.bandwidth is None or self.model_config.fit_full_range:
            self.model_config.win_list_plot = self.win_list
            if self.model_config.plot_gui:
                self.model_config.win_list_gui = self.model_config.win_list_plot
            if self.model_config.plot_file:
                self.model_config.win_list_file = self.model_config.win_list_plot

        else:
            if self.model_config.plot_gui:
                self.model_config.win_list_gui = self.select_windows(**self.gui_kws)
                if len(self.model_config.win_list_gui) == 0:
                    print("Nothing to plot in GUI, check your selection.")

            if self.model_config.plot_file:
                if ((self.file_kws['display_all'] == self.gui_kws['display_all']) and
                        (self.file_kws['windows'] == self.gui_kws['windows']) and self.model_config.plot_gui):
                    # gui and file have the same info, nothing to do
                    self.model_config.win_list_file = self.model_config.win_list_gui

                else:
                    # select windows
                    self.model_config.win_list_file = self.select_windows(**self.file_kws)
                    if len(self.model_config.win_list_file) == 0:
                        print("Nothing to plot in file, check your selection.")

            # Disable for now, but needs to be re-written : TODO
            # if other_species_win_selection is not None:
            #     if isinstance(other_species_win_selection, int):
            #         other_species_win_selection = str(other_species_win_selection)
            #     self.select_windows_other_lines(other_species_win_selection)

    def make_plot(self, plot_type):
        """
        Do the plot(s).
        :param plot_type: gui or file
        :return:
        """

        if plot_type == 'gui':
            gui_plot(self)

        if plot_type == 'file':
            filename = self.file_kws['filename']
            dirname = self.file_kws.get('dirname', None)
            verbose = self.file_kws.get('verbose', True)
            dpi = self.file_kws.get('dpi', DPI_DEF)
            nrows = self.file_kws.get('nrows', NROWS_DEF)
            ncols = self.file_kws.get('ncols', NCOLS_DEF)

            t_start = process_time()
            file_plot(self, filename, dirname=dirname, verbose=verbose,
                      dpi=dpi, nrows=nrows, ncols=ncols)
            t_stop = process_time()
            if self.exec_time:
                print("Execution time for saving plot : {}.".format(utils.format_time(t_stop - t_start)))

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

    def update_params_dict(self, params, tag_list):
        tag_list = [str(t) for t in tag_list]  # make sure we have a list of strings
        for par in params:
            if 'ntot' in par or 'fwhm' in par:
                if par.split('_')[-1] not in tag_list:
                    params.pop(par)
        return params

    def save_ref_pixel(self):
        ref_pixel_info = {
            'params': self.params.copy(),
            'model': self.model,
            'model_fit': self.model_fit,
            'tag_list': self.model_config.tag_list,
            'windows': self.model_config.win_list
        }
        cpt_info = {
            cpt.name: {sp.tag: sp for sp in cpt.species_list}
            for cpt in self.model_config.cpt_list
        }
        self.model_config.ref_pixel_info = {**ref_pixel_info, **cpt_info}

    def use_ref_pixel(self, tag_list=None):
        params = self.ref_pixel_info['params'].copy()
        self.model_config.latest_valid_params = params
        # self.model = self.ref_pixel_info['model']
        if tag_list is not None:
            self.update_params_dict(params, tag_list)
            self.generate_lte_model()
        # self.params = params

    def save_model(self, filename, dirname=None, ext='txt', full_spectrum=True):
        """
        Save the model spectrum from self.model.
        :param filename: the name of the file
        :param dirname: the directory where to save the file
        :param ext: extension of the file : txt (default) or fits
        :param full_spectrum: save the model for the entire observed spectrum ;
        if false, only save the model spectrum for the windows in self.win_list
        :return: None
        """
        params = self.params
        if self.x_file is None:  # model only -> full spectrum
            full_spectrum = True
            x_values = self.x_mod
        else:
            x_values = [self.x_file[0]]
            for x in self.x_file[1:]:
                x_temp = np.linspace(x_values[-1], x, num=self.oversampling+1)
                x_values.extend(list(x_temp[1:]))
            x_values = np.array([x for x in x_values if utils.is_in_range(x, list(self.tuning_info['fmhz_range']))])

        if not full_spectrum:
            x_values = []
            y_values = []
            for win in self.win_list:
                if win.x_mod is None:
                    if win.x_mod_plot is not None:
                        win.x_mod = win.x_mod_plot
                        break
                    win.x_mod = np.linspace(min(win.x_file), max(win.x_file),
                                            num=self.oversampling * (len(win.x_file) - 1) + 1)
                    mdl_info = self.model_info()
                    model = Model(generate_lte_model_func(mdl_info))
                    y_mod = model.eval(params, fmhz=win.x_mod)
                    win.y_mod = y_mod
                    if len(self.cpt_list) > 1:
                        y_mod = y_mod.reshape(len(y_mod), 1)
                        for cpt in self.cpt_list:
                            mdl_info['cpt_list'] = [cpt]
                            c_best_pars = {}
                            for pname, par in params.items():
                                if cpt.name in pname:
                                    c_best_pars[pname] = par.value
                            # c_lte_func = generate_lte_model_func(mdl_info)
                            # y_cpt = c_lte_func(win.x_mod, **c_best_pars)
                            y_cpt = self.compute_model_intensities(params=self.params, x_values=win.x_mod,
                                                                   line_list=self.line_list_all, cpt=cpt)
                            y_mod = np.hstack((y_mod, y_cpt.reshape(len(y_cpt), 1)))
                            win.y_mod_cpt[cpt.name] = y_cpt
                y_mod = win.y_mod.reshape(len(win.y_mod), 1)
                for y_cpt in win.y_mod_cpt.values():
                    y_mod = np.hstack((y_mod, y_cpt.reshape(len(y_cpt), 1)))
                x_values.extend(win.x_mod)
                y_values.extend(y_mod)
        else:
            # y_values = self.compute_model_intensities(params=params, x_values=x_values)
            mdl_info = self.model_info()
            model = Model(generate_lte_model_func(mdl_info))
            y_values = model.eval(params, fmhz=x_values)
            if len(self.cpt_list) > 1:
                y_values = y_values.reshape(len(y_values), 1)
                for cpt in self.cpt_list:
                    mdl_info['cpt_list'] = [cpt]
                    c_best_pars = {}
                    for pname, par in params.items():
                        if cpt.name in pname:
                            c_best_pars[pname] = par.value
                    c_lte_func = generate_lte_model_func(mdl_info)
                    y_cpt = c_lte_func(x_values, **c_best_pars)
                    y_values = np.hstack((y_values, y_cpt.reshape(len(y_cpt), 1)))

        spec = x_values, y_values
        self.save_spectrum(filename, dirname=dirname, ext=ext, spec=spec, spectrum_type='synthetic')

    def save_spectrum(self, filename, dirname=None, ext='txt',
                      spec: None | tuple = None,
                      spectrum_type: Literal['observed', 'continuum', 'synthetic', ''] = '',
                      vlsr: None | float | int = None,
                      yunit: None | str = None):
        """
        Write a spectrum (continuum, data or model, depending on the provided parameters) on a file.

        :param filename:
        :param dirname:
        :param ext:
        :param spec: tuple of x and y values to be written ;
                     if not provided and continuum is false, stored model is written
        :param spectrum_type: 'continuum', 'observed', 'synthetic' or empty string '' (default)
        :param vlsr:
        :param yunit:
        :return: the path to the file
        """
        spec_types = ['', 'observed', 'continuum', 'synthetic']
        if spectrum_type not in spec_types:
            raise AttributeError('spectrum_type can only be one of the following:',
                                 ", ".join([f"'{t}'" for t in spec_types]))

        file_path = self.set_filepath(filename, dirname=dirname, ext=ext)
        ext = ext.strip('.')

        if spec is not None:  # should be a tuple of x and y values
            x_values, y_values = spec

        elif spectrum_type == 'continuum':
            ext = 'txt'  # force txt extension
            file_path = self.set_filepath(filename + '_cont', dirname=dirname, ext=ext)
            x_values = self.tc['f_mhz']
            y_values = self.tc['tc']

        elif spectrum_type == 'observed':
            # if self.data_file is None and self.x_file is not None:  # TODO: check whether this case is useful
            #     self.data_file = file_path
            #     x_values, y_values = self.x_file, self.y_file
            if self.model_config.bl_corr:
                x_values = np.concatenate([w.x_file for w in self.model_config.win_list_fit], axis=None)
                y_values = np.concatenate([w.y_file for w in self.model_config.win_list_fit], axis=None)
            else:
                x_values, y_values = self.x_file, self.y_file
            if yunit is None:
                yunit = self.model_config.yunit
            if self.model_config.yunit in UNITS['flux'] and yunit == 'K':
                y_values = y_values * self.model_config.jypb(x_values)

        elif spectrum_type == 'synthetic':
            x_values, y_values = self.x_mod, self.y_mod
            if y_values is None:
                y_values = self.model.eval(fmhz=self.x_mod, **self.params)

        else:
            x_values, y_values = None, None

        if x_values is None or y_values is None:
            print('Nothing to save.')
            return None

        if ext == 'fits':
            y_vals = y_values[:, 0] if len(np.shape(y_values)) > 1 else y_values
            col1 = fits.Column(name='wave', format='D', unit='MHz', array=x_values)
            col2 = fits.Column(name='flux', format='D', unit='K', array=y_vals)
            hdu = fits.BinTableHDU.from_columns([col1, col2])

            hdu.header['DATE-HDU'] = (datetime.datetime.now().strftime("%c"), 'Date of HDU creation')
            if spectrum_type == 'synthetic':
                hdu.header['DATABASE'] = ('SQLITE ({})'.format(os.path.split(SQLITE_FILE)[-1]), 'Database used')
                hdu.header['MODEL'] = ('Full LTE', 'Model used to compute this spectrum')
                hdu.header['NOISE'] = (self.noise * 1000., '[mK]Noise added to the spectrum')

            if vlsr is None:
                params = self.params
                try:
                    vlsr = params['{}_vlsr'.format(self.cpt_list[0].name)].value
                except TypeError:
                    vlsr = self.cpt_list[0].vlsr
            hdu.header['VLSR'] = (vlsr, '[km/s]')

            hdu.writeto(file_path, overwrite=True)

        if ext == 'fus':  # TODO: to be implemented
            with open(file_path, 'w') as f:
                f.writelines([f'// number of lines : {len(x_values)}\n',
                              f'// vlsr : {self.vlsr_file}\n',
                              '\t'.join(['FreqLsb', 'VeloLsb', 'FreqUsb', 'VeloUsb', 'Intensity', 'DeltaF', 'DeltaV']),
                              '\n'])
                # for i in range(len(x_values)):
                #     f.write()

        if ext == 'txt':
            with open(file_path, 'w') as f:
                if spectrum_type != 'continuum':
                    header = ['#title: ' + os.path.splitext(os.path.basename(file_path))[0] + '\n',
                              '#date: {}\n'.format(datetime.datetime.now().strftime("%c")),
                              '#database :{}\n'.format(SQLITE_FILE),
                              # '#coordinate: world\n'
                             ]
                    if len(self.model_config.tuning_info) > 1:
                        header.append('#telescopes:\n')
                        for i, row in self.model_config.tuning_info.iterrows():
                            header.append(f"#  {row['telescope']}: {row['fmhz_range']}\n")
                    else:
                        header.append('#telescope: ' + self.model_config.tuning_info.iloc[0]['telescope'] + '\n')

                    header.append(f'#vlsr [km/s]: {self.model_config.vlsr_file}\n')
                    header.extend(['#xLabel: frequency [MHz]\n',
                                   f'#yLabel: [{yunit}] Mean\n'])
                    if spectrum_type == 'synthetic':
                        header.extend(['#Columns : frequency, intensity (total), intensity of each component\n'])
                    else:
                        header.extend(['#Columns : frequency, intensity\n'])
                    f.writelines(header)

                for x, y in zip(x_values, y_values):
                    if len(np.shape(y_values)) == 1:
                        f.write('{}\t{}\n'.format(utils.format_float(x, fmt="{:.5f}"),
                                                  utils.format_float(y, fmt="{:.5e}")))
                    else:
                        line = '{}\t'.format(utils.format_float(x, fmt="{:.5f}"))
                        line += '\t'.join([utils.format_float(yy, fmt="{:.5e}") for yy in y])
                        f.write(line + '\n')

        # if self.model_config.data_file is None and spectrum_type != 'continuum' and spectrum_type != 'synthetic':
        if spectrum_type not in ['continuum', 'synthetic']:
            self.model_config.data_file = os.path.abspath(file_path)

        return os.path.abspath(file_path)

    def save_stick_spectrum(self, filename, dirname=None, ext='fits'):
        x_lines = [line['transition'].f_trans_mhz for _, line in self.line_list_all.iterrows()]
        x_lines.sort()
        x_lines = np.array(x_lines)
        for cpt in self.cpt_list:
            center_int = self.compute_model_intensities(x_values=x_lines, cpt=cpt)
            cont = self.get_tc(x_lines)
            x_vals = []
            y_vals = []
            for c, x, y in zip(cont, x_lines, center_int):
                x_vals.extend([x - 1.e-6, x, x + 1.e-6])
                y_vals.extend([c, y, c])

            x_vals = np.array([utils.velocity_to_frequency(cpt.vlsr, x, vref_kms=self.vlsr_file)
                               for x in x_vals])

            self.save_spectrum(f'{filename}_{cpt.name}', dirname=dirname, spec=(x_vals, y_vals), ext='fits',
                               vlsr=cpt.vlsr)

    def save_line_list_cassis(self, filename, dirname=None, snr_threshold=None):
        """
        Writes the list of lines for display in CASSIS. To be used when fitting the entire spectrum.
        :param filename:
        :param dirname:
        :return:
        """
        # if len(self.win_list) == 1:
        #     self.win_list_plot = self.win_list
        #     self.setup_plot_fus()
        filebase = filename
        nb_dec = '3'

        for icpt, cpt in enumerate(self.cpt_list):
            line_list = self.model_config.line_list_all[self.model_config.line_list_all['tag'].isin(cpt.tag_list)]
            line_list = line_list.sort_values('tag')
            line_list.reset_index(drop=True, inplace=True)

            if len(self.cpt_list) > 1:
                filename = f'{filebase}_{cpt.name}'

            # Compute the intensity at the *observed* frequency corresponding to the transition's frequency
            x_vals = utils.velocity_to_frequency(cpt.vlsr, line_list.fMHz.values, vref_kms=self.model_config.vlsr_file)
            line_list['x_obs'] = x_vals

            rms = []
            for i, row in line_list.iterrows():
                fmhz = row['x_obs']
                tran = row['transition']
                if fmhz < min(self.model_config.tuning_info.fmhz_min):
                    print(f"INFO save_line_list_cassis component {cpt.name} - {fmhz} "
                          f"is below the minimum telescope range ({min(self.model_config.tuning_info.fmhz_min)}) ; ")
                    rms_val = np.nan
                elif fmhz > max(self.model_config.tuning_info.fmhz_max):
                    print(f"INFO save_line_list_cassis component {cpt.name} - {fmhz} "
                          f"is above the maximum telescope range ({max(self.model_config.tuning_info.fmhz_max)}) ; ")
                    rms_val = np.nan
                else:
                    try:
                        rms_val = self.get_rms_cal(fmhz)[0]
                    except IndexError:
                        print(f"INFO save_line_list_cassis component {cpt.name} - No rms info found for {fmhz} MHz ; ")
                        rms_val = np.nan
                if np.isnan(rms_val):
                    print(f"-> ignoring the corresponding transition : {' ; '.join(tran.__str__().split(' ; ')[:3])}.")

                rms.append(rms_val)

            if np.isnan(np.array(rms)).any():
                print(" ")
            line_list['rms'] = rms
            line_list = line_list.dropna()
            x_vals = line_list.x_obs
            int0 = self.compute_model_intensities(params=self.params, x_values=x_vals,
                                                  line_list=self.line_list_all, cpt=cpt,
                                                  line_center_only=True)
            line_list['int0'] = int0
            cont = np.array([self.get_tc(fmhz) for fmhz in x_vals])
            snr = (int0 - cont) / line_list.rms
            line_list['snr'] = snr
            line_list['selected'] = np.full(len(x_vals), True)

            if snr_threshold is not None:
                line_list.loc[line_list.snr < snr_threshold, 'selected'] = False
                line_list.loc[0 < line_list.snr, 'selected'] = True  # keep lines with negative snr

            line_list = line_list[line_list.selected == True]
            line_list.sort_values(['tag', 'fMHz'], ascending=[True, True])

            with open(self.set_filepath(filename, dirname=dirname, ext='txt'), 'w') as f:
                if snr_threshold is not None:
                    f.write(f"# List of lines with S/N ratio of modelled intensity >= {snr_threshold}\n")
                f.write('# ')
                f.write('\t'.join(
                    # ['Transition', 'Tag', 'Frequency(MHz)', 'Eup(K)', 'Aij', 'Tau', 'Tex', 'Intensity(K)'])
                    ['Component', 'Tag_Transition', 'Frequency(MHz)', 'Eup(K)', 'Aij',
                     'FrequencyPlusError', 'Tex', 'Vlsr', 'Intensity(K)', 'Tau'])
                )
                f.write('\n')

                for i, row in line_list.iterrows():
                    line = row['transition']
                    # if line.tag in cpt.tag_list and selected[i]:
                    tex = self.params[f"{cpt.name}_tex"].value
                    try:
                        fwhm = self.params[f"{cpt.name}_fwhm"].value
                    except IndexError:
                        fwhm = self.params[f"{cpt.name}_fwhm_{line.tag}"].value
                    tau0 = utils.compute_tau0(line,
                                              self.params[f"{cpt.name}_ntot_{line.tag}"].value,
                                              fwhm,
                                              tex)
                    # ind0 = utils.find_nearest_id(self.win_list[0].x_mod, line.f_trans_mhz)
                    line_elements = [
                        f"{cpt.name[1:]}",
                        f"{line.tag} {line.name} ({line.qn_lo}_{line.qn_hi})",
                        f"{line.f_trans_mhz:.{nb_dec}f}",
                        f"{line.eup:.{nb_dec}f}",
                        f"{line.aij:.{nb_dec}e}",
                        f"{line.f_trans_mhz + line.f_err_mhz:.{nb_dec}f}",
                        f"{tex:.{nb_dec}f}",
                        f"{cpt.vlsr:.{nb_dec}f}",
                        f"\t{int0[i]:.{nb_dec}e}",
                        f"{tau0:.{nb_dec}e}"
                    ]
                    f.write("\t".join(line_elements) + "\n")

    def save_fit_results(self, filename, dirname=None):
        with open(self.set_filepath(filename, dirname=dirname, ext='txt'), 'w') as f:
            f.writelines(self.fit_report(report_kws={'show_correl': True}))

    def write_cassis_file(self, filename, ext, dirname=None, datafile=None):
        def lam_item(name, value):
            if isinstance(value, float) and value != 0.:
                return f'{name}={utils.format_float(value, nb_digits=4)}\n'
            return '{}={}\n'.format(name, value)

        # ext = 'ltm' if self.x_file is None else 'lam'

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
                'lineValue': 'NaN',
                'dsbSelected': 'false',
                'dsb': 'LSB',
                'loFreq': '121.5',
                'telescope': '',  # TBD when writing the ltm file
                'bandwidth': (max(self.x_mod) - min(self.x_mod)) / 1000.,
                'bandUnit': 'GHZ',
                'resolution': self.dfmhz,
                'resolutionUnit': 'MHZ',
                'freqRef': '115500.0'
            }
        else:
            if datafile is not None:
                nameData = datafile
            elif isinstance(self.data_file, str):
                nameData = os.path.abspath(self.data_file)
            else:
                nameData = ''
            tuning = {
                'nameData': nameData,
                'telescopeData': '',  # TBD when writing the lam file
                'typeFrequency': 'SKY' if self.vlsr_file == 0. else 'REST',
                'minValue': min(self.x_file) / 1000.,  # will be updated when writing lam file if multi tel
                'maxValue': max(self.x_file) / 1000.,  # will be updated when writing lam file if multi tel
                # 'minValue': min(self.x_mod) / 1000.,
                # 'maxValue': max(self.x_mod) / 1000.,
                'valUnit': 'GHZ',
                'bandValue': self.bandwidth,
                'bandUnit': 'KM_SEC_MOINS_1'
            }

        # eup_max = 1.e304
        eup_max_vals = [val['eup_max'] for val in self.thresholds.values()]
        if None in eup_max_vals:
            eup_max = '*'
        else:
            eup_max = max(eup_max_vals)
        # aij_max = 1.e304
        aij_max_vals = [val['aij_max'] for val in self.thresholds.values()]
        if None in aij_max_vals:
            aij_max = '*'
        else:
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
            'tmbBox': 'true' if self.model_config.t_a_star else 'false',
            'observing': 'PSDBS',
            'tbg': self.tcmb,
            'tbgUnit': 'K',
            'noise': np.mean(self.noise(self.x_mod)) * 1000.,  # TODO: compute at the same time as telescope
            'noiseUnit': 'mK',
            'frequency': 'SKY' if self.cpt_list[0].vlsr != 0. else 'REST'
        }
        if ext == 'lam':
            lte_radex = {'lteRadexSelected': 'true', **lte_radex, 'overSampling': self.oversampling}

        # Define continuum
        if isinstance(self.model_config.cont_info, (float, int)) or self.model_config.cont_free:
            cont_type = 'CONSTANT'
            if self.model_config.cont_free:
                cont_size = 0.0
            else:
                cont_size = f'{self.model_config.cont_info}'
            cont = 'Continuum 0 [K]'  # default
        else:  # it is a file
            cont_type = 'FILE'
            cont_size = '0.0'  # default
            cont = os.path.abspath(self.model_config.cont_info)
            if self.model_config.yunit in UNITS['flux']:
                base, ext = os.path.splitext(self.model_config.cont_info)
                cont = os.path.abspath(base + '_K' + ext)

        components = {
            '# Component parameters 1': {
                'Comp1Name': 'Continuum',
                'Comp1Enabled': 'true',
                'Comp1Interacting': 'false',
                'Comp1ContinuumSelected': cont_type,
                # 'Comp1ContinuumYUnit': self.model_config.yunit,
                'Comp1ContinuumYUnit': 'K',  # TODO: check whether to keep original unit in .ltm
                'Comp1Continuum': cont,
                'Comp1ContinuumSize': cont_size
            }
        }

        # Define other components
        params = self.params
        # if self.best_params is not None and ext == 'lam':
        #     params = self.best_params

        # Determine if some species are in comp 2+ but not in comp 1
        extra_sp = []
        try:
            for tag in self.model_config.tag_list:
                if tag not in self.cpt_list[0].tag_list:
                    extra_sp.append(Species(tag, ntot=10, tex=self.model_config.cpt_list[0].tex,
                                            fwhm=self.model_config.cpt_list[0].fwhm))
            # for cpt in self.cpt_list[1:]:
            #     for sp in cpt.species_list:
            #         if sp.tag not in self.cpt_list[0].tag_list:
            #             extra_sp.append(sp)
        except IndexError:  # do nothing
            pass

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
                    'Tex': params['{}_tex'.format(cpt.name)].value,
                    'TKin': 10.,
                    'Size': params['{}_size'.format(cpt.name)].value,
                    'NbMol': len(cpt.tag_list) if ic != 0 else len(cpt.tag_list) + len(extra_sp)
                    }
            for isp, sp in enumerate(cpt.species_list):
                molname = 'Mol{}'.format(isp + 1)
                try:
                    fwhm = params['{}_fwhm'.format(cpt.name)].value
                except KeyError:
                    fwhm = params['{}_fwhm_{}'.format(cpt.name, sp.tag)].value
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
                    'FWHM': fwhm,
                    'Size': params['{}_size'.format(cpt.name)].value
                    }
                cdic[molname] = {basename + molname + key: val for key, val in mol_dic.items()}

            if ic == 0 and len(extra_sp) > 0:  # 1st component must contain all species => add fake species if necessary
                for isp, sp in enumerate(extra_sp):
                    molname = 'Mol{}'.format(len(cpt.species_list) + isp + 1)
                    try:
                        fwhm = params['{}_fwhm'.format(cpt.name)].value
                    except KeyError:
                        fwhm = params['{}_fwhm_{}'.format(cpt.name, sp.tag)].value
                    mol_dic = {
                        'Tag': sp.tag,
                        'Species': sp.name,
                        'Database': sp.database,
                        'Collision': '-no -',
                        'Compute': 'true',
                        'NSp': 1e1,
                        'Abundance': 1e1 / cdic['Density'],
                        'Tex': params['{}_tex'.format(cpt.name)].value,
                        'TKin': '10.0',
                        'FWHM': fwhm,
                        'Size': params['{}_size'.format(cpt.name)].value
                    }
                    cdic[molname] = {basename + molname + key: val for key, val in mol_dic.items()}

            components['# Component parameters {}'.format(c_num)] = {basename + key: val for key, val in cdic.items()}

        all_lines = ['# {}\n'.format(datetime.datetime.now().strftime("%c"))]
        if ext == 'lam':
            all_lines.append('# Generals Parameters\n\n')

        for filepath, tel in zip(filepaths, tels):
            with open(filepath, 'w') as f:
                tel_path = os.path.abspath(utils.search_telescope_file(tel))
                if 'telescopeData' in tuning:
                    tuning['telescopeData'] = tel_path
                if 'telescope' in tuning:
                    tuning['telescope'] = tel_path
                lte_radex['telescope'] = tel_path
                tel_info = self.model_config.tuning_info[self.model_config.tuning_info.telescope == tel]
                if len(tels) > 1:
                    tuning['minValue'] = tel_info.fmhz_min.min() / 1000
                if len(tels) > 1:
                    tuning['maxValue'] = tel_info.fmhz_max.max() / 1000
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
        """
        Writes a LTE model configuration file for CASSIS
        :param filename: the name of the file
        :param dirname: the directory where to save the file
        :return: None
        """
        self.write_cassis_file(filename, 'ltm', dirname=dirname)

    def write_lam(self, filename, dirname=None):
        """
        Writes a line analysis configuration file for CASSIS
        :param filename: the name of the file
        :param dirname: the directory where to save the file
        :return: None
        """
        self.write_cassis_file(filename, 'lam', dirname=dirname)


class ModelCube(object):
    def __init__(self, configuration, verbose=False):

        try:
            self._data_file = configuration['data_file']
        except KeyError:
            raise KeyError("Keyword data_file is mandatory")
        if not isinstance(self._data_file, list):
            self._data_file = [self._data_file]

        self._data_path = configuration.get('data_path', None)
        if self._data_path is not None:
            self._data_file = [os.path.join(self._data_path, f) for f in self._data_file]

        self.output_dir = configuration.get('output_dir', 'outputs')
        self.log_file = os.path.join(self.output_dir, 'logfile.txt')

        self._cubes = utils.get_cubes(self._data_file)
        self.cubeshape = self._cubes[0].shape
        self.wcs = self._cubes[0].wcs
        self.hdr = self._cubes[0].header

        self.fmhz_ranges = self.read_frequencies()

        # Extract the intensity units
        yunit = self.hdr['BUNIT']
        IntensityTastar = False
        if 'K' in yunit:
             yunit = 'Kelvins'
        if 'Ta.' in yunit:
             IntensityTastar = True
        configuration['t_a*'] = IntensityTastar
        configuration['yunit'] = yunit

        configuration['continuum_free'] = configuration.get('continuum_free', False)

        # Extract the Vlsr if present in the fits files (to be check out, does not work yet)
        # --------------------------------------------------------------------------------------------------------------------------
        # if 'VELO-LSR' in hdr :
        #     vlsr_obs          = hdr['VELO-LSR'] * u.m / u.s
        #     vlsr_obs          = vlsr_obs.value / 1e3
        #     print('vlsr_obs  = ', vlsr_obs)

        tuning_info = configuration['tuning_info']
        if not isinstance(tuning_info, dict):
            configuration['tuning_info'] = {tuning_info: self.fmhz_ranges}

        self._noise_info = configuration['noise_info']
        self._cal = configuration.get('calibration', 15.)
        if isinstance(self._noise_info[0], str):
            self._cubes_noise = utils.get_cubes(self._noise_info, check_spatial_shape=self.cubeshape[-2:])
        else:
            self._cubes_noise = []

        configuration['minimize'] = False
        configuration['x_obs'] = np.concatenate([dat.spectral_axis.value / 1.e6 for dat in self._cubes])  # in MHz

        self._pix_info = configuration.get('pix_info', None)
        if self._pix_info is None:
            self.pix_list = []
        else:
            try:
                (xref, yref, delta, step) = self._pix_info
            except ValueError:
                (xref, yref, delta) = self._pix_info
                step = 1
            # Define the snake loop extent
            if delta == -1:
                ymax, xmax = self.cubeshape[-2:]
                self.pix_list = utils.pixels_snake_loop(xref, yref, xmax - 1, ymax - 1, step=step)
            else:
                self.pix_list = utils.pixels_snake_loop(xref, yref, xref + delta, yref + delta, xmin=xref - delta,
                                                        ymin=yref - delta, step=step)

        self._model_configuration_user = copy.deepcopy(configuration)
        self._model = ModelSpectrum(configuration, verbose=verbose)
        self._model.make_params()
        self._params_user = self._model.params.copy()
        # self._model_configuration = ModelConfiguration(configuration, verbose=verbose)
        self._model_configuration = self._model.model_config

        self.tags = self._model.model_config.tag_list

        # create arrays of Nans for the output parameters, ensure to make for all components
        array_dict = {'redchi2': np.full((self.cubeshape[-2], self.cubeshape[-1]), np.nan)}
        err_dict = dict()
        for param in self._model.param_names():
            array_dict['{}'.format(param)] = np.full((self.cubeshape[-2], self.cubeshape[-1]), np.nan)
            err_dict['{}'.format(param)] = np.full((self.cubeshape[-2], self.cubeshape[-1]), np.nan)
        self.array_dict = array_dict
        self.err_dict = err_dict

        # self._source = configuration.get('source', None)
        # self.win_list = None

        self.cont_info = configuration.get('cont_info', [])
        self._cont_data = []
        # retrieving the continuum
        # if isinstance(cont_info, dict):
        #     for key, val in cont_info.items():
        #         if isinstance(val, str):
        #             cont_file = os.path.join(self._data_path, val)
        #             if os.path.isfile(cont_file):  # assume it is a fits file
        #                 cont_info[key] = fits.open(cont_file)[0]
        if len(self.cont_info) > 0:
            cont_info = utils.get_cubes(self.cont_info, check_spatial_shape=self.cubeshape[-2:])
            self._cont_data = cont_info

        # conversion factors : T = (conv_fact / nu^2) * I , with :
        # conv_fact = c^2 / (2*k_B*omega) * 1.e-26 (to convert Jy to mks)
        # omega = pi*bmaj*min/(4*ln2)
        # 'jypb_MHz2': 1.e-26 * const.c.value ** 2 / 1.e12 / (2. * const.k_B.value * omega)})

        # configuration['beam_info'] = self.get_beams()

        # Mask:
        mask_info = configuration.get('mask_info', None)
        if mask_info is None:
            self.masked_pix_list = None
        else:
            mask_file = mask_info.get('file', None)
            if mask_file is None:
                raise KeyError("Missing mask file")
            if not isinstance(mask_file, list):
                mask_file = [mask_file]
            if len(mask_file) > 2:
                raise ValueError("Can only handle 1 or 2 mask files.")

            self.masked_pix_list = utils.get_mask(self.wcs, mask_file,
                                                  exclude=mask_info.get('exclude',True))


            # file1 = os.path.join(self._data_path, masks[0])
            # file2 = None
            # if len(masks) > 1:
            #     file2 = os.path.join(self._data_path, masks[1])
            # self._masked_pix_list = utils.get_valid_pixels(self._wcs, file1, file2=file2, masked=True)

        # params = ['redchi']
        # for cpt in self._model_configuration.cpt_list:
        #     params.extend([par.name for par in cpt.parameters])
        #     params.extend([par.name for sp in cpt.species_list for par in sp.parameters])
        #
        # # create arrays of zeros for the output parameters
        # self._param_arrays = dict()
        # for param in params:
        #     self._param_arrays['{}_arr'.format(param)] = np.zeros((self._nx, self._ny))

        if configuration.get('print_info', True):
            self.print_infos()

    def get_beams(self):  # keep? (can get beam from spectralCube object)
        beams = {'f_mhz': [], 'beam_omega': []}  # np.ones(len(file_list))
        for h, cube in enumerate(self._cubes):
            try:
                hdr = cube.hdu.header
            except ValueError:
                hdr = cube.hdulist[0].header
            if hdr['BUNIT'] in ['Jy/beam', 'beam-1 Jy']:  # calculate conversion factor Jy/beam to K
                try:
                    bmaj = hdr['BMAJ'] * np.pi / 180.  # major axis in radians, assuming unit = degrees
                    bmin = hdr['BMIN'] * np.pi / 180.  # major axis in radians, assuming unit = degrees
                except KeyError:
                    for hdu in cube.hdulist[1:]:
                        try:
                            unit = hdu.columns['BMAJ'].unit  # assume bmaj and bmin have the same unit
                            fact = u.Quantity("1. {}".format(unit)).to(u.rad).value
                            bmaj = np.mean(hdu.data['BMAJ']) * fact
                            bmin = np.mean(hdu.data['BMIN']) * fact
                        except KeyError:
                            raise KeyError("Beam information not found in file {}.".format(self._data_file[h]))
                omega = np.pi * bmaj * bmin / (4. * np.log(2.))
                sp = (cube.spectral_axis.to(u.MHz)).value
                beams['f_mhz'].append(sp)
                beams['beam_omega'].append([omega for _ in sp])

        return beams

    def read_frequencies(self):
        # Read the data cubes and start and end frequencies of each cube :
        fmhz_ranges = []

        for i, f in enumerate(self._data_file):
            dat = SpectralCube.read(fits.open(f))

            # Extract the min and max frequency values of each cube
            start_freq_MHz = min(dat.spectral_axis).to(u.MHz).value
            end_freq_MHz = max(dat.spectral_axis).to(u.MHz).value

            # start_freq_MHz = round(start_freq_MHz, 3)
            # end_freq_MHz = round(end_freq_MHz, 3)

            # From EC : If this is not the first cube, check whether start_freq_MHz is larger than the previous end_freq_MHz
            # if i > 0 and start_freq_MHz <= end_frequencies_MHz[i - 1]:
            #     start_freq_MHz = end_frequencies_MHz[i - 1] + 1e-3
            # SB : don't understand the use of the above lines

            # If not the first cube, do some checks :
            if i > 0:
                if end_freq_MHz <= fmhz_ranges[i - 1][0]:
                    raise IndexError(f"Cubes {f} and {self._data_file[i - 1]} are not in increasing frequency order.\n"
                                     f"Make sure all your cubes are in increasing frequency order.")
                if start_freq_MHz <= fmhz_ranges[i - 1][-1]:
                    # Issue a warning if overlap :
                    print('\n\n\n')
                    print(f'WARNING - The cube {f} starts at {start_freq_MHz} MHz '
                          f'and overlaps with the cube {self._data_file[i - 1]}, which ends {fmhz_ranges[i - 1][-1]}.')
                    print(f'          Make sure none of the selected lines are in the overlap region')
                    print('\n\n\n')

            fmhz_ranges.append([start_freq_MHz, end_freq_MHz])

        return fmhz_ranges

    def do_minimization(self, pix_list=None):
        def save_to_log(pix, res, par_names_all, append):
            par_names = [res["params"][par].name for par in res["params"]]
            if not append:
                with open(self.log_file, "w") as f:
                    cols = ["x_pix", "y_pix", "iterations", "Exec time (s)", "red_chi2", "tags_new"]
                    cols += par_names_all
                    f.write("\t".join(cols) + "\n")
            with (open(self.log_file, "a") as f):
                line = [str(p) for p in pix]
                line += [utils.format_variable(val) for key, val in res.items() if key != 'params']
                tags = model.model_config.tag_list
                line += [",".join(tags)]
                par_vals = []
                for par_name in par_names_all:
                    if par_name in par_names:
                        par_vals.append(utils.format_float(res["params"][par_name].value))
                    else:
                        par_vals.append("--")
                line += par_vals
                f.write("\t".join(line) + "\n")

        if not os.path.isdir(os.path.abspath(self.output_dir)):
            os.makedirs(os.path.abspath(self.output_dir))

        printDebug = self._model_configuration.print_debug

        if pix_list is None:
            pix_list = self.pix_list

        mask = self.masked_pix_list
        # Create the components masks (one for each component, for Tex, Vlsr, FWHM and X2, and initialize all values to False
        # --------------------------------------------------------------------------------------------------------------------------
        # mask_comp = np.full((cubeshape[-3:]), False)
        # masks_ntot = np.full((*cubeshape[-3:], len(tags)), False)

        # If one noise value per map :
        # (otherwise, rms_cal will be defined at each pixel)
        if isinstance(self._noise_info[0], (float, int)):
            rms_cal = {'[{:.1f}, {:.1f}]'.format(start, end): [noise, self._cal] for [start, end], noise in
                       zip(self.fmhz_ranges, self._noise_info)}

        # Start the snake loop
        model = self._model
        for pix in pix_list:
            t1_start = process_time()
            i, j = pix
            if mask is not None and not mask[j, i]:
                if pix == self._pix_info[:2]:  # ref pixel is masked out
                    raise IndexError("Your reference pixel is masked out.")
                # the pixel is masked, go to the next one
                continue
            data = np.concatenate([dat[:, j, i].array for dat in self._cubes])
            data = np.nan_to_num(data)
            if len(set(data)) == 1:
                print(f'Not enough data to compute the model at pixel {pix}.')
                continue

            if len(self._cubes_noise) > 0:
                noiseValues = np.concatenate([dat[:, j, i].array for dat in self._cubes_noise])
                rms_cal = {'[{:.1f}, {:.1f}]'.format(start, end): [noise, self._cal] for [start, end], noise in
                           zip(self.fmhz_ranges, noiseValues)}
            #        print('rms_cal = ', rms_cal)

            plot_name = "plot_{}_{}".format(i, j)

            # ascii files to be saved in output_dir
            config_name = "config_{}_{}".format(i, j)
            model_name = "model_{}_{}".format(i, j)
            result_name = "result_{}_{}".format(i, j)
            spec_name = "spectrum_{}_{}".format(i, j)
            output_files = {
                'results': result_name,
                'lam': config_name,
                'obs': spec_name,
                'model': model_name
            }

            cont_name = os.path.join(self.output_dir, "continuum_{}_{}".format(i, j) + ".txt")

            if len(self._cont_data) != 0:
                cont_values = []
                for k, cont_data in enumerate(self._cont_data):
                    try:
                        continuum = cont_data[0, 0, j, i].array
                    except IndexError:
                        continuum = cont_data[:, j, i][0].value
                    cont_values.append(continuum)
                if len(self.fmhz_ranges) > 1 and len(self._cont_data) == 1:  # TODO : check
                    # only one continuum value for all frequency ranges -> replicate
                    cont_values *= len(self.fmhz_ranges)
                cont_df = pd.DataFrame({
                    'fmhz_range': self.fmhz_ranges,
                    'continuum': cont_values
                })
                # print to terminal if debug mode
                if self._model_configuration.print_debug:
                    if len(cont_df['continuum'].unique()) == 1:
                        print("j =", j, " i =", i, 'continuum[j,i] =', cont_df['continuum'].unique()[0], self.yunit)
                    else:
                        for _, row in cont_df.iterrows():
                            print("j =", j, " i =", i, 'continuum[j,i] =', row['continuum'], self.yunit, ';',
                                  row['fmhz_range'])

                # save to file
                utils.write_continuum_file(cont_name, cont_df, yunit=self.yunit)
                tc = cont_name
                if self.yunit in UNITS['flux']:
                    base, ext = os.path.splitext(cont_name)
                    cont_name = base + '_K' + ext
                    mid_freq = [np.array(freq_range).mean() for freq_range in cont_df['fmhz_range']]
                    cont_df['continuum'] = cont_df['continuum'] * self._model_configuration.jypb(mid_freq)
                    utils.write_continuum_file(cont_name, cont_df, yunit='K')
                    # tc = cont_name
            else:
                tc = 0.0

            # update the observed values and name of pdf file
            # ---------------------------------------------------------------
            model.model_config.y_file = data
            model.model_config.cont_info = tc
            model.model_config.rms_cal = rms_cal
            model.model_config.output_files = output_files
            model.model_config.file_kws['filename'] = plot_name + ".pdf"
            # ---------------------------------------------------------------

            if pix == self._pix_info[:2]:

                if model.model_fit is None:  # create the model on the first (brightest) pixel
                    print("\nFitting ref pixel : ", pix)

                    # Run the model
                    model.model_config.minimize = True
                    res = model.do_minimization()

                    # Save some info
                    par_names_all = [model.params[par].name for par in model.params]
                    save_to_log(pix, res, par_names_all, append=False)

                    # Save the model
                    model.save_ref_pixel()

                    # for itag, tages in enumerate(self.tags):
                        # masks_ntot[j, i, itag] = True  # Mind, I do not mask the pixels with that !!!
                    for par in model.params:
                        param = model.params[par]
                        self.array_dict['{}'.format(param.name)][j, i] = param.value
                        self.err_dict['{}'.format(param.name)][j, i] = param.stderr
                    self.array_dict['redchi2'][j, i] = model.model_fit.redchi

                else:
                    print("Back to ref : ", pix, " - Do not fit again")
                    model.use_ref_pixel()

                continue

                # noise_pixel varying with fmhz_ranges.
            # --------------------------------------------------------------------------------------------------------------------------
            fluxes_freq_range = model.model_config.rms_cal.copy()
            for tag_i in self.tags:  # add two columns for each tag (total flux, number of points)
                fluxes_freq_range[str(tag_i) + '_Fmax'] = [0.] * len(fluxes_freq_range)
                fluxes_freq_range[str(tag_i) + '_Nwin'] = [0.] * len(fluxes_freq_range)

            for win in model.win_list_fit:
                fmhz = win.transition.f_trans_mhz
                if self._model_configuration.cont_free:
                    tc = 0.
                else:
                    tc = model.get_tc(fmhz)
                tag_i = win.transition.tag
                row = get_df_row_from_freq_range(fluxes_freq_range, fmhz)
                idx = row.index
                fluxes_freq_range.loc[idx, str(tag_i) + '_Fmax'] += max(win.y_fit - tc)
                fluxes_freq_range.loc[idx, str(tag_i) + '_Nwin'] += 1

            # create a dictionary to store the rounded SNR values
            snr_tag = {}
            for tag_i in self.tags:
                snr = fluxes_freq_range[str(tag_i) + '_Fmax'] / fluxes_freq_range[str(tag_i) + '_Nwin'] / \
                      fluxes_freq_range['rms']
                snr_tag[tag_i] = snr.mean()

            if 'constraint' in self._model_configuration_user:
                # This considers only rflux[0] values when constraints is True
                tags_new = self.tags if list(snr_tag.values())[0] >= self._model_configuration_user['snr'] else []
            else:
                # Consider all rflux values when constraints is False
                tags_new = [tages for tages, rflux in snr_tag.items() if rflux >= self._model_configuration_user['snr']]

            snr_fmt = {key: f'{val:.2f}' if abs(val) >= 0.01 else f'{val:.2e}' for key, val in snr_tag.items()}

            if len(tags_new) > 0:
                # mask_comp[j, i] = True
                # update tag list
                model.update_tag_list(tags_new)

                # update params
                if model.model_config.latest_valid_params is not None:
                    for parname in model.params:
                        model.params[parname] = model.model_config.latest_valid_params[parname]
                        if model.params[parname].vary and model.params[parname].expr is None:
                            val = model.params[parname].value
                            if (model.params[parname].user_data is not None and
                                    model.params[parname].user_data.get('factor', False)):
                                new_max = model.params[parname].user_data['max_fact'] * val
                                new_min = model.params[parname].user_data['min_fact'] * val
                            else:
                                new_max = val + self._params_user[parname].max - self._params_user[parname].value
                                new_min = val - (self._params_user[parname].value - self._params_user[parname].min)

                # update velocity ranges in windows
                for win in model.win_list_fit:
                    delta_v = [model.model_config.latest_valid_params['c1_vlsr'].init_value - v for v in win.v_range_fit]
                    win.v_range_fit = [model.params['c1_vlsr'].value - dv for dv in delta_v]
                    win.compute_f_range_fit(model.model_config.vlsr_file)

                model.model_config.get_data_to_fit()

                # fit with the updated config
                print("\nFitting pixel : ", pix)
                res = model.do_minimization()
                save_to_log(pix, res, par_names_all, append=True)

                self.array_dict['redchi2'][j, i] = model.model_fit.redchi

                for par in model.params:
                    param = model.params[par]
                    self.array_dict['{}'.format(param.name)][j, i] = param.value
                    self.err_dict['{}'.format(param.name)][j, i] = param.stderr

                t1_stop = process_time()
                print(
                    f'S/N = {", ".join(snr_fmt.values())} at pixel : {pix}, Execution time : {t1_stop - t1_start:.2f} seconds')
            else:
                print(f'S/N = {", ".join(snr_fmt.values())} at pixel : {pix}, too low to compute a model')

            print('tags_new = ', tags_new)  # new tag list with S/N ≥ signal2noise
            # --------------------------------------------------------------------------------------------------------------------------

            # Printouts for debugging
            # --------------------------------------------------------------------------------------------------------------------------
            if printDebug:
                print('plot_name = ', plot_name)
                print('config_name = ', config_name)
                print('model_name = ', model_name)
                print('result_name = ', result_name)
                print('spec_name = ', spec_name)
                print('tc = ', tc)
                print("Fitting pixel : ", pix)
                print("current list : model.tag_list = ", model.tag_list)  # current tag list
                print('tags_new = ', tags_new)  # new tag list with S/N ≥ signal2noise
                # print("mask_comp shape:", mask_comp.shape)
                # print("masks_ntot shape:", masks_ntot.shape)

    def make_maps(self):
        params = self.array_dict.keys()
        print('params = ', params)
        units = {
            'tex': 'Kelvin',
            'vlsr': 'km/s',
            'size': 'arcsec',
            'fwhm': 'km/s',
            'ntot': 'cm^-2',
            'redchi2': 'Reduced chi-squared'
        }
        for param in params:
            # Split the parameter name into a list of substrings
            if param.startswith('c'):  # we have a component
                param_type = param.split('_')[1]
            else:  # reduced chi2
                param_type = param

            for arr, ext in zip([self.array_dict, self.err_dict], ['.fits', '_err.fits']):
                try:
                    hdu = fits.PrimaryHDU(arr['{}'.format(param)], header=self.hdr)
                    hdu.header.set('BUNIT', units[param_type])
                    hdul = fits.HDUList([hdu])
                    hdul.writeto(os.path.join(self.output_dir, param + ext), overwrite=True)
                except KeyError:
                    pass  # do nothing

    def do_minimization_old(self, pix_nb=None, single_pix=True, size=None):
        if size is None:
            size = max(self._nx, self._ny)
            imin, imax = 0, self._nx
            jmin, jmax = 0, self._ny
        else:
            size2 = int(size / 2)
            pix0 = (0, 0) if pix_nb is None else pix_nb
            imin, imax = pix0[0] - size2, pix0[0] + size2 + 1
            jmin, jmax = pix0[1] - size2, pix0[1] + size2 + 1

        if pix_nb is not None:
            if single_pix:  # for one pixel
                pix_list = [pix_nb]
            else:  # spiral loop, starting from pix_nb
                ip, jp = 0, 0
                di, dj = 0, -1
                pix_list = []
                for _ in range(size ** 2):
                    if (imin <= ip + pix_nb[0] < imax) and (jmin <= jp + pix_nb[1] < jmax):
                        pix_list.append((ip + pix_nb[0], jp + pix_nb[1]))
                    if (ip == jp) or ((ip == -jp) and (ip < 0)) or ((ip == 1 - jp) and (ip > 0)):
                        # end of current segment, rotate direction
                        di, dj = -dj, di
                    ip, jp = ip + di, jp + dj

        else:  # all pixels, starting from (0, 0)
            pix_list = [(i, j) for i in range(self._nx) for j in range(self._ny)]

        # loop over all pixels and fit lines
        for (i, j) in pix_list:
            if self._masked_pix_list is not None and (j, i) in self._masked_pix_list:
                for par in self._param_arrays.keys():
                    self._param_arrays[par][j, i] = 0.
                continue

            t1_start = process_time()

            data = []
            spec = []
            # concatenate data from all files
            for dat in self._cubes:
                data.append(dat[:, j, i].array)
                spec.append((dat.spectral_axis.to(u.MHz)).value)

            self._model_configuration.get_data({'x_obs': spec, 'y_obs': data})
            self._model_configuration.get_continuum({'tc': self._cont_data})

            if self._model_configuration.win_list is None:
                self._model_configuration.get_linelist()
                self._model_configuration.get_windows()
            else:
                for win in self._model_configuration.win_list:
                    _, ywin = utils.select_from_ranges(spec, [min(win.x_file), max(win.x_file)], y_values=data)
                    win.y_file = ywin

            # Create the model ; set verbose to False to prevent printing of number of transitions
            model = ModelSpectrum(self._model_configuration, verbose=False)

            # Perform the fit ; NB :
            # max_nfev = maximum number of function evaluations
            # xtol = Relative error in the approximate solution
            # ftol = Relative error in the desired sum-of-squares
            model.fit_model(max_nfev=5000, fit_kws={'xtol': 1.e-8, 'ftol': 1.e-7},
                            print_report=False if pix_nb is None else True,
                            report_kws={'show_correl': False})

            # read out params into arrays, ensure to have for all components necessary
            for par in model.params:
                param = model.params[par]
                self._param_arrays['{}_arr'.format(param.name)][j, i] = param.value
            self._param_arrays['redchi_arr'][j, i] = model.model_fit.redchi

            t1_stop = process_time()
            if t1_stop - t1_start > 1.:
                print("Execution time : {:.2f} seconds".format(t1_stop - t1_start))

            if single_pix:
                file = self._model_configuration.base_name + "_{}_{}".format(i, j)
                model.save_model(file + "_spec", ext='txt')
                model.write_lam(file + "_lam")
                model.make_plot(filename=file + ".png", gui=False)
            # model.save_config('test-config.txt', dirname=myPath)

            # res = model.integrated_intensities()

            t2_start = process_time()
            model.make_plot(filename=self._model_configuration.base_name+"_plots_{}_{}".format(i, j)+".png", gui=False)
            t2_stop = process_time()
            print("Execution time : {:.2f} seconds".format(t2_stop - t2_start))

    def make_maps_old(self):
        # read out parameter arrays into fits files
        units = {'tex': 'K', 'vlsr': 'km/s', 'fwhm': 'km/s', 'ntot': 'cm^-2', 'size': 'arcsec'}
        hdr = utils.reduce_wcs_dim(self._wcs).to_header()

        for param in self._param_arrays.keys():
            unit = 'dimensionless'
            vary = True
            vals = param.split('_')
            if len(vals) >= 2:
                comp = vals[0]
                par = vals[1]
                unit = units[par]
                if len(vals) == 3:
                    tag = vals[2]
                    for sp in self._configuration['components'][comp]['species']:
                        if sp['tag'] == tag:
                            vary = sp[par].get('vary', True)
                else:
                    vary = self._configuration['components'][comp][par].get('vary', True)

            if not vary:
                continue
            for key, val in units.items():
                if key in param:
                    hdr['BUNIT'] = val
                    continue
            hdu = fits.PrimaryHDU(self._param_arrays['{}_arr'.format(param)], header=hdr)
            hdul = fits.HDUList([hdu])
            hdul.writeto(os.path.join(self._data_path, self._source + '_' + self._tag + '_' + param + '.fits'),
                         overwrite=True)

    def print_infos(self):
        print('output_dir = ', self.output_dir)
        # print('input_dir = ', self.input_dir)  # NB input_dir not provided in config
        print('file_cube_list = ', self._data_file)
        print('file_cont_list = ', self.cont_info)
        print('cubeshape = ', self.cubeshape)
        print('yunit = ', self.yunit)
        print('fmhz_ranges = ', self.fmhz_ranges)
        print('pix_list = ', self.pix_list)  # To check the list of pixels
        if self.masked_pix_list is not None and not self.masked_pix_list.all():
            print('mask = ', self.masked_pix_list)
        print('')
        print('tags = ', self.tags)
        print('velocity ranges = ', self._model_configuration_user['v_range'])
        print('componentConfig = ', self._model_configuration_user['components']['config'])
        try:
            print('otherSpecies = ', self._model_configuration_user['plot_kws']['gui+file']['other_species'])
        except KeyError:
            print('otherSpecies not found')
        if 'constraints' in self._model_configuration_user:
            print("constraints = ", self._model_configuration_user['constraints'])
        print('params = ', self.params)
        print('array_dict = ', self.array_dict.keys())
        print('err_dict = ', self.err_dict.keys())

    @property
    def params(self):
        return self._model.param_names()

    @property
    def yunit(self):
        return self._model.yunit

    # @property
    # def param_arrays(self):
    #     return self._param_arrays


# def frange(start, stop, step):
#     i = start
#     while i < stop:
#         yield i
#         i += step
