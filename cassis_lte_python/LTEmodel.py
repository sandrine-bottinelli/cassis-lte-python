from cassis_lte_python.utils.utils import get_telescope, get_beam_size, get_tmb2ta_factor, dilution_factor, jnu
from cassis_lte_python.utils.utils import get_valid_pixels, reduce_wcs_dim
from cassis_lte_python.utils.utils import format_float, is_in_range, select_from_ranges, find_nearest_id
from cassis_lte_python.utils.utils import velocity_to_frequency, frequency_to_velocity, \
    fwhm_to_sigma, delta_v_to_delta_f, compute_weight
from cassis_lte_python.gui.plots import file_plot, gui_plot
from cassis_lte_python.sim.model_setup import ModelConfiguration, Component
from cassis_lte_python.utils.settings import SQLITE_FILE
from cassis_lte_python.utils.constants import C_LIGHT, K_B, H, TEL_DIAM
from cassis_lte_python.utils.constants import PLOT_COLORS
from cassis_lte_python.database.species import get_species_thresholds
from cassis_lte_python.database.transitions import get_transition_df, select_transitions
from numpy import exp, sqrt, pi, array, interp, ones, linspace, mean, hstack, zeros, shape, log, concatenate
from numpy.random import normal
from lmfit import Model, Parameters
from scipy import stats, signal
import astropy.io.fits as fits
import astropy.units as u
import os
import pandas as pd
import datetime
import json
from spectral_cube import SpectralCube
from astropy.wcs import WCS
from time import process_time
from matplotlib.pyplot import get_cmap
from warnings import warn


def generate_lte_model_func(config):

    def lte_model_func(fmhz, **params):
        norm_factors = config.get('norm_factors', {key: 1. for key in params.keys()})
        tc = config['tc']
        tcmb = config['tcmb']
        vlsr_file = config['vlsr_file']
        fmhz_mod = fmhz  # config['x_mod']
        beam_sizes = config['beam_sizes']
        tmb2ta = config['tmb2ta']
        jypb2k = config['jypb2k']
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
                tran_list = []
                if isinstance(line_list, list):
                    tran_list = line_list
                else:  # assume it is a DataFrame
                    tran_list = list(line_list.loc[line_list['tag'] == tag].transition)
                ntot = params['{}_ntot_{}'.format(cpt.name, tag)] * norm_factors['{}_ntot_{}'.format(cpt.name, tag)]
                fwhm = params['{}_fwhm_{}'.format(cpt.name, tag)] * norm_factors['{}_fwhm_{}'.format(cpt.name, tag)]
                qtex = cpt.species_list[isp].get_partition_function(tex)
                for tran in tran_list:
                    # nup = ntot * tran.gup / qtex / np.exp(tran.eup_J / const.k_B.value / tex)  # [cm-2]
                    nup = ntot * tran.gup / qtex / exp(tran.eup / tex)  # [cm-2]
                    tau0 = C_LIGHT ** 3 * tran.aij * nup * 1.e4 \
                           * (exp(H * tran.f_trans_mhz * 1.e6 / K_B / tex) - 1.) \
                           / (4. * pi * (tran.f_trans_mhz * 1.e6) ** 3 * fwhm * 1.e3 * sqrt(pi / log(2.)))
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
                        sum_tau += tau0 * exp(-0.5 * (num / den) ** 2)

            ff = array([dilution_factor(size, bs) for bs in beam_sizes])
            if not cpt.isInteracting:
                intensity_cpt = jnu(fmhz_mod, tex) * (1. - exp(-sum_tau)) - \
                                intensity_before * (1. - exp(-sum_tau))
                intensity += ff * intensity_cpt
            else:
                intensity_before += intensity
                intensity_cpt = jnu(fmhz_mod, tex) * (1. - exp(-sum_tau)) - \
                                intensity_before * (1. - exp(-sum_tau))
                intensity = ff * intensity_cpt

        intensity = intensity + intensity_before - jnu(fmhz_mod, tcmb)
        intensity += normal(0., config['noise'], len(intensity))  # add gaussian noise
        intensity *= tmb2ta  # convert to Ta
        intensity /= jypb2k  # convert to Jy/beam

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
    def __init__(self, configuration: (dict, str, ModelConfiguration),
                 verbose=True, check_tel_range=False):
        if not isinstance(configuration, ModelConfiguration):
            config = configuration  # assume it is a dictionary
            if not isinstance(configuration, dict):
                try:  # assume config is a path to a file
                    config = self.load_config(configuration)
                except TypeError:
                    print("Configuration must be a dictionary or a path to a configuration file "
                          "or of type ModelConfiguration.")
            model_config = ModelConfiguration(config, verbose=verbose, check_tel_range=check_tel_range)

            if 'data_file' in model_config._configuration_dict or 'x_obs' in model_config._configuration_dict:
                model_config.get_data()

            if 'tc' in model_config._configuration_dict:
                model_config.get_continuum()

            model_config.get_linelist()
            model_config.get_windows()

        else:  # configuration is of type ModelConfiguration
            model_config = configuration

        self.output_dir = model_config.output_dir

        self.tag_list = model_config.tag_list
        self.cpt_list = model_config.cpt_list
        self.line_list_all = model_config.line_list_all

        self.cont_info = model_config.cont_info
        self.tc = model_config.tc
        self.tcmb = model_config.tcmb
        self._tuning_info_user = model_config._tuning_info_user
        self.tuning_info = model_config.tuning_info
        self._rms_cal_user = model_config._rms_cal_user
        self._v_range_user = model_config._v_range_user
        self.bandwidth = model_config.bandwidth
        self.oversampling = model_config.oversampling
        self.fmin_mhz = model_config.fmin_mhz
        self.fmax_mhz = model_config.fmax_mhz
        self.dfmhz = model_config.dfmhz
        self.noise = model_config.noise
        self.tau_max = model_config.tau_max
        self.file_rejected = model_config.file_rejected
        self.thresholds = model_config.thresholds

        self.data_file = model_config.data_file
        self.x_file = model_config.x_file
        self.y_file = model_config.y_file
        self.vlsr_file = model_config.vlsr_file
        self.vlsr_plot = model_config.vlsr_plot
        self._telescope_data = model_config._telescope_data
        self.t_a_star = model_config.t_a_star
        self.jypb = model_config.jypb

        self.x_fit = model_config.x_fit
        self.y_fit = model_config.y_fit
        self.x_mod = model_config.x_mod
        self.y_mod = model_config.y_mod

        self.win_list = model_config.win_list
        self.win_list_fit = model_config.win_list_fit
        self.win_list_plot = model_config.win_list_plot

        self.params2fit = None
        self.norm_factors = None
        self.model = None
        if self.x_file is not None:
            self.model = self.generate_lte_model()
        self.best_params = None
        self.best_model = None
        self.model_fit = None
        self.normalize = False
        self.figure = None

        self.tag_colors = None
        self.tag_other_sp_colors = None
        self.cpt_cols = None

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
            'noise': self.noise,
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

        self.y_mod = self.compute_model_intensities(params=self.params2fit, x_values=self.x_mod)

    def update_parameters(self, params=None):
        if params is None:
            params = self.best_pars()
        for cpt in self.cpt_list:
            pars = {par: value for par, value in params.items() if cpt.name in par}
            cpt.update_parameters(pars)

    def model_info(self, x_mod, line_list=None, cpt_list=None, line_center_only=False):
        tel = get_telescope(x_mod, self.tuning_info)
        tel_diam = array([TEL_DIAM[t] for t in tel])
        return {
            'tc': self.get_tc(x_mod),
            'tcmb': self.tcmb,
            'vlsr_file': self.vlsr_file,
            'norm_factors': self.norm_factors,
            'beam_sizes': get_beam_size(x_mod, tel_diam),
            'tmb2ta': [get_tmb2ta_factor(f_i, self._telescope_data[t])
                       for f_i, t in zip(x_mod, tel)] if self.t_a_star else ones(len(x_mod)),
            'jypb2k': interp(x_mod, self.x_file, self.jypb) if self.jypb is not None else ones(len(x_mod)),
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
            tc = array(tc)
        return tc

    def get_rms(self, fmhz):
        if type(fmhz) == float:
            fmhz = list(fmhz)

        rms = []
        for freq in fmhz:
            for win in self.win_list_fit:
                if min(win.f_range_fit) <= freq <= max(win.f_range_fit):
                    rms.append(win.rms)

        return rms if len(rms) > 1 else rms[0]

    def get_params2fit(self, normalize=False):
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

        norm_factors = {self.params2fit[parname].name: 1. for parname in self.params2fit}
        if normalize:
            for parname in self.params2fit:
                param = self.params2fit[parname]
                nf = abs(param.value) if param.value != 0. else 1.
                norm_factors[param.name] = nf
                if param.expr is None:
                    param.set(min=param.min / nf, max=param.max / nf, value=param.value / nf)
        self.norm_factors = norm_factors

    def generate_lte_model(self, normalize=False):
        if self.params2fit is None:
            self.get_params2fit(normalize=normalize)

        if self.x_fit is not None:
            lte_model_func = generate_lte_model_func(self.model_info(self.x_fit))
        else:
            self.x_mod = linspace(self.fmin_mhz, self.fmax_mhz,
                                  num=int((self.fmax_mhz - self.fmin_mhz) / self.dfmhz) + 1)
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

        wt = concatenate([compute_weight(win.y_fit, win.rms, win.cal) for win in self.win_list_fit], axis=None)
        # wt = None
        self.model_fit = self.model.fit(self.y_fit, params=self.params2fit, fmhz=self.x_fit,
                                        weights=wt,
                                        method=method,
                                        max_nfev=max_nfev, fit_kws=fit_kws)

        self.best_model = self.model_fit.model
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

    def compute_model_intensities(self, params=None, x_values=None, line_list=None, line_center_only=False,
                                  cpt=None):
        if self.model is None:
            self.generate_lte_model()
        if params is None:
            params = self.best_params if self.best_params is not None else self.params2fit
        if x_values is None:
            x_values = self.x_mod
        elif type(x_values) is list:
            x_values = array(x_values)

        if cpt is not None:
            c_best_pars = {}
            for pname, par in params.items():
                if cpt.name in pname:
                    c_best_pars[pname] = par.value
            params = c_best_pars
            lte_func = generate_lte_model_func(self.model_info(x_values, line_list=line_list,
                                                               cpt_list=[cpt]))
        else:
            lte_func = generate_lte_model_func(self.model_info(x_values, line_list=line_list,
                                                               line_center_only=line_center_only))

        return lte_func(x_values, **params)

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
                x_mod = linspace(fmin_mod, fmax_mod, num=npts)
                y_mod = self.compute_model_intensities(params=best_pars, x_values=x_mod, line_list=[win.transition])

                dv = 2. * fwhm / (npts - 1)
                K_kms = 0.
                for i in range(len(y_mod) - 1):
                    K_kms += mean([y_mod[i], y_mod[i+1]]) * dv

                fluxes.append([win.transition.tag, win.plot_nb, f_ref, K_kms])

            res[cpt.name] = pd.DataFrame(fluxes, columns=['tag', 'line number', 'f_mhz', 'K.km/s'])

        return res

    def setup_plot_fus(self):
        """
        Plot in full spectrum mode (self.bandwidth is None)
        :return:
        """
        pass

    def setup_plot_la(self, verbose=True, other_species_dict: dict | None = None):
        """
        Prepare all data to do the plots in line analysis mode

        :param verbose:
        :param other_species_dict: a dictionary of other species and their thresholds
        :return:
        """

        # Define some useful quantities
        plot_pars = self.plot_pars()
        vlsr = self.cpt_list[0].vlsr if self.vlsr_file == 0. else self.vlsr_file
        fwhm = max([plot_pars[par].value for par in plot_pars if 'fwhm' in par])
        padding = 0.05

        self.update_parameters(params=plot_pars)

        if other_species_dict is not None:  # list of tags for which the user wants line positions
            thresholds_other = other_species_dict
        else:
            thresholds_other = {}
        list_other_species = list(thresholds_other.keys())

        # lines from other species : if many other species, more efficient to first find all transitions
        # across entire observed range, then filter in each window
        # if len(list_other_species) > 0:
        other_species_lines = get_transition_df(list_other_species, [[min(self.x_file), max(self.x_file)]],
                                                **thresholds_other)
        # else:
        #     other_species_lines = pd.DataFrame()  # empty dataframe

        # Compute model overall model : takes longer than cycling through windows unless strong overlap of windows (TBC)
        # self.y_mod = self.compute_model_intensities(params=plot_pars, x_values=self.x_mod,
        #                                             line_list=self.line_list_all)

        # Compute model and line positions for each window
        for win in self.win_list_plot:
            tr = win.transition
            f_ref = tr.f_trans_mhz
            win.v_range_plot = [-self.bandwidth / 2 + vlsr, self.bandwidth / 2 + vlsr]
            win.f_range_plot = [velocity_to_frequency(v, f_ref, vref_kms=self.vlsr_file)
                                for v in win.v_range_plot]
            win.bottom_unit = 'km/s'
            dx1 = max(win.v_range_plot) - min(win.v_range_plot)
            win.bottom_lim = (min(win.v_range_plot) - padding * dx1, max(win.v_range_plot) + padding * dx1)
            win.top_lim = [velocity_to_frequency(v, f_ref, vref_kms=self.vlsr_file)
                           for v in win.v_range_plot]

            # all transitions in the window (no thresholds) :
            fwhm_mhz = delta_v_to_delta_f(fwhm, f_ref)
            model_lines_win = get_transition_df(self.tag_list, [min(win.f_range_plot) - 2 * fwhm_mhz,
                                                                max(win.f_range_plot) + 2 * fwhm_mhz])
            # all_lines_win = select_transitions(self.line_list_all, xrange=[min(win.f_range_plot) - 2 * fwhm_mhz,
            #                                                                max(win.f_range_plot) + 2 * fwhm_mhz])

            # compute the model :
            # win.x_mod, win.y_mod = select_from_ranges(self.x_mod, win.f_range_plot, y_values=self.y_mod)
            win.x_mod = linspace(min(win.x_file), max(win.x_file), num=self.oversampling * len(win.x_file))
            win.y_mod = self.compute_model_intensities(params=plot_pars, x_values=win.x_mod,
                                                       line_list=model_lines_win)
            if len(self.cpt_list) > 1:
                for icpt in range(len(self.cpt_list)):
                    win.y_mod_cpt.append(self.compute_model_intensities(params=plot_pars, x_values=win.x_mod,
                                                                        line_list=model_lines_win,
                                                                        cpt=self.cpt_list[icpt]))

            win.x_mod_plot = frequency_to_velocity(win.x_mod, f_ref, vref_kms=vlsr)
            win.x_file_plot = frequency_to_velocity(win.x_file, f_ref, vref_kms=vlsr)

            # transitions from model species, w/i thresholds :
            model_lines_user = select_transitions(model_lines_win,
                                                  thresholds=self.thresholds)
            # find "bright" lines (if aij_max not None and/or eup_min non-zero):
            # bright_lines = select_transitions(all_lines_win,  # xrange=[fmin, fmax],
            #                                   thresholds=self.thresholds,
            #                                   # bright_lines_only=True)

            # transitions from model species, outside thresholds :
            model_lines_other = pd.concat([model_lines_user,
                                           model_lines_win]).drop_duplicates(subset='db_id', keep=False)

            # transitions from other species :
            other_species_win_all = select_transitions(other_species_lines,
                                                       xrange=[min(win.f_range_plot), max(win.f_range_plot)])
            # remove main lines
            other_species_win_all = pd.concat([model_lines_user,
                                               other_species_win_all]).drop_duplicates(subset='db_id', keep=False)

            # concatenate with model lines outside thresholds, keeping first occurrence of duplicates
            other_species_win = pd.concat([model_lines_other,
                                           other_species_win_all]).drop_duplicates(subset='db_id', keep='first')

            for icpt, cpt in enumerate(self.cpt_list):
                # build list of dataframes containing lines to be plotted for each component
                win.main_lines_display[icpt] = self.get_lines_plot_params(
                    model_lines_user[model_lines_user['tag'].isin(cpt.tag_list)], cpt, f_ref)
                win.other_lines_display[icpt] = self.get_lines_plot_params(
                    model_lines_other[model_lines_other['tag'].isin(cpt.tag_list)], cpt, f_ref)

            # line plot parameters for other species (not component-dependent)
            tag_other_sp_colors = {t: PLOT_COLORS[(itag + len(self.tag_colors)) % len(PLOT_COLORS)]
                                   for itag, t in enumerate(other_species_win.tag)}
            if len(other_species_win) > 0:
                win.other_species_display = self.get_lines_plot_params(other_species_win, self.cpt_list[0], f_ref,
                                                                       tag_colors=tag_other_sp_colors)

    def get_lines_plot_params(self, line_list: pd.DataFrame, cpt: Component, f_ref: float,
                              tag_colors: dict = None):

        colors = self.tag_colors if tag_colors is None else tag_colors
        lines_plot_params = line_list.copy()
        lines_plot_params['x_pos'] = [frequency_to_velocity(row.fMHz, f_ref, vref_kms=cpt.vlsr)
                                      for i, row in lines_plot_params.iterrows()]
        lines_plot_params['x_pos_err'] = [delta_v_to_delta_f(row.f_err_mhz, f_ref, reverse=True)
                                          for i, row in lines_plot_params.iterrows()]
        lines_plot_params['label'] = [row.tag for i, row in lines_plot_params.iterrows()]
        lines_plot_params['color'] = [self.tag_colors[row.tag] if row.tag in self.tag_colors.keys()
                                      else colors[row.tag] for i, row in lines_plot_params.iterrows()]

        return lines_plot_params

    def select_windows(self, tag: str | None = None,
                       other_species_win_selection: str | None = None,
                       display_all=True, reset_selection=False):
        """
        Determine windows to plot
        :param tag: tag selection if do not want all the tags
        :param other_species_win_selection: select only windows with other lines from this tag.
        :param display_all: if False, only display windows with fitted data
        :param reset_selection: re-do window selection from scratch
        :return:
        """

        if len(self.win_list_plot) == 0 or reset_selection:
            self.win_list_plot = self.win_list  # by default, plot everything

        if not display_all:  # only display windows with fitted data
            self.win_list_plot = [w for w in self.win_list_plot if w.in_fit]

        if tag is not None:  # user only wants one tag
            self.win_list_plot = [w for w in self.win_list_plot if w.transition.tag == tag]

        if other_species_win_selection is not None:
            # user only wants windows with other lines from the tag given in other_species_selection
            sub_list = []
            for win in self.win_list_plot:  # check if window contains a transition from other_species_selection
                if other_species_win_selection in win.other_lines_display['tag']:
                    sub_list.append(win)
                    # f_ref = win.transition.f_trans_mhz
                    # delta_f = 3. * delta_v_to_delta_f(fwhm, fref_mhz=f_ref)
                    # res = select_transitions(tr_list_other_species_selection, thresholds=thresholds_other,
                    #                          xrange=[f_ref - delta_f, f_ref + delta_f],
                    #                          return_type='df')
                    # if len(res) > 0:
                    #     win.other_species_selection = res
                    #     self.win_list_plot.append(win)
            if len(sub_list) == 0:
                warn(f"No windows with transitions from {other_species_win_selection}.")

        if self.win_list_plot == 0:
            raise LookupError("No windows to plot. Please check your tag selection.")

    def make_plot(self, tag: str | None = None,
                  filename: str | None = None, dirname: str | os.PathLike | None = None,
                  gui=False, verbose=True, basic=False,
                  other_species: list | dict | str | os.PathLike = None,
                  other_species_plot: list | str = 'all',
                  other_species_win_selection: str | None = None,
                  display_all=True, dpi=None,
                  nrows=4, ncols=3, **kwargs):
        """
        Prepare all data to do the plot(s)

        :param tag: tag selection if do not want all the tags
        :param filename: nome of the file to be saved
        :param dirname: directory where to save the file
        :param gui: interactive display
        :param verbose:
        :param basic: do not plot other species
        :param other_species: list or dictionary or file with other species ;
            dictionary and file can contain their thresholds
        :param other_species_plot: list of other species to plot ; if None, other_species is used ;
            if other_species is provided, only these species are kept
        :param other_species_win_selection: select only windows with other lines from this tag.
        :param display_all: if False, only display windows with fitted data
        :param dpi:
        :param nrows: maximum number of rows per page
        :param ncols: maximum number of columns per page
        :return:

        Notes :
            - other_species_selection is deprecated, use other_species_win_selection
        """

        # set colors for model tags and components
        self.tag_colors = {t: PLOT_COLORS[itag % len(PLOT_COLORS)] for itag, t in enumerate(self.tag_list)}
        self.cpt_cols = get_cmap('hsv')(linspace(0.1, 0.8, len(self.cpt_list)))

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

        if self.bandwidth is None:
            self.setup_plot_fus()
        else:
            self.select_windows(tag=tag, display_all=display_all)
            self.setup_plot_la(verbose=verbose, other_species_dict=thresholds_other)
            self.select_windows(other_species_win_selection=other_species_win_selection)

        if gui:
            gui_plot(self)

        if filename:
            file_plot(self, filename, dirname=dirname, verbose=verbose,
                      dpi=dpi, nrows=nrows, ncols=ncols)

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

    def save_model(self, filename, dirname=None, ext='txt', full_spectrum=True):
        """
        Save the model spectrum from self.best_model if it exists, else from self.model.
        :param filename: the name of the file
        :param dirname: the directory where to save the file
        :param ext: extension of the file : txt (default) or fits
        :param full_spectrum: save the model for the entire observed spectrum ;
        if false, only save the model spectrum for the windows in self.win_list
        :return: None
        """
        params = self.best_params if self.best_params is not None else self.params2fit
        if self.x_file is None:  # model only -> full spectrum
            full_spectrum = True
            x_values = self.x_mod
        else:
            x_values = [self.x_file[0]]
            for x in self.x_file[1:]:
                x_temp = linspace(x_values[-1], x, num=self.oversampling+1)
                x_values.extend(list(x_temp[1:]))
            x_values = array([x for x in x_values if is_in_range(x, list(self.tuning_info['fmhz_range']))])

        if not full_spectrum:
            x_values = []
            y_values = []
            for win in self.win_list:
                if win.x_mod is None:
                    win.x_mod = linspace(min(win.x_file), max(win.x_file),
                                num=self.oversampling * (len(win.x_file) - 1) + 1)
                    mdl_info = self.model_info(win.x_mod)
                    model = Model(generate_lte_model_func(mdl_info))
                    y_mod = model.eval(params, fmhz=win.x_mod)
                    if len(self.cpt_list) > 1:
                        y_mod = y_mod.reshape(len(y_mod), 1)
                        for cpt in self.cpt_list:
                            mdl_info['cpt_list'] = [cpt]
                            c_best_pars = {}
                            for pname, par in params.items():
                                if cpt.name in pname:
                                    c_best_pars[pname] = par.value
                            c_lte_func = generate_lte_model_func(mdl_info)
                            y_cpt = c_lte_func(win.x_mod, **c_best_pars)
                            y_mod = hstack((y_mod, y_cpt.reshape(len(y_cpt), 1)))
                    win.y_mod = y_mod
                x_values.extend(win.x_mod)
                y_values.extend(win.y_mod)
        else:
            # y_values = self.compute_model_intensities(params=params, x_values=x_values)
            mdl_info = self.model_info(x_values)
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
                    y_values = hstack((y_values, y_cpt.reshape(len(y_cpt), 1)))

        spec = x_values, y_values
        self.save_spectrum(filename, dirname=dirname, ext=ext, spec=spec)

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
                    if len(shape(y_values)) == 1:
                        f.write('{}\t{}\n'.format(format_float(x), format_float(y)))
                    else:
                        line = '{}\t'.format(format_float(x)) + '\t'.join([format_float(yy) for yy in y])
                        f.write(line + '\n')

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
                'nameData': os.path.abspath(self.data_file) if isinstance(self.data_file, str) else '',
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

        components = {
            '# Component parameters 1': {
                'Comp1Name': 'Continuum',
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
                if 'telescopeData' in tuning:
                    tuning['telescopeData'] = tel
                if 'telescope' in tuning:
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
        """
        Writes a line analysis configuration file for CASSIS
        :param filename: the name of the file
        :param dirname: the directory where to save the file
        :param save_fit: whether to save the results of the fit
        :return:
        """
        if save_fit:
            self.save_fit_results(filename + '_fit_res', dirname=dirname)
        self.write_cassis_file(filename, dirname=dirname)


class ModelCube:
    def __init__(self, configuration):
        self._data_path = configuration.get('data_path', './')
        self._data_file = configuration.get('data_file', None)
        if not isinstance(self._data_file, list):
            self._data_file = [self._data_file]

        self._source = configuration.get('source', None)
        self.win_list = None

        cont_info = configuration.get('cont_info', 0.)
        # retrieving the continuum
        if isinstance(cont_info, dict):
            for key, val in cont_info.items():
                if isinstance(val, str):
                    cont_file = os.path.join(self._data_path, val)
                    if os.path.isfile(cont_file):  # assume it is a fits file
                        cont_info[key] = fits.open(cont_file)[0]
        self._cont_info = cont_info

        # retrieving the source data and information
        hduls = [fits.open(os.path.join(self._data_path, f)) for f in self._data_file]
        self._cubes = [SpectralCube.read(h) for h in hduls]
        # datfile = fits.open(os.path.join(myPath, file_list[0]))
        # dat = SpectralCube.read(datfile)
        self._wcs = WCS(hduls[0][0].header)  # RA and DEC info from WCS in header
        # hdr = datfile[0].header

        # check that cubes have the same number of pixels in RA and Dec:
        nx_list = [dat.shape[1] for dat in self._cubes]
        ny_list = [dat.shape[2] for dat in self._cubes]
        if len(set(nx_list)) > 1 or len(set(ny_list)) > 1:
            raise ValueError("The cubes do not have the same dimension(s) in RA and/or Dec.")

        self._nx = next(iter(set(nx_list)))
        self._ny = next(iter(set(ny_list)))

        # conversion factors : T = (conv_fact / nu^2) * I , with :
        # conv_fact = c^2 / (2*k_B*omega) * 1.e-26 (to convert Jy to mks)
        # omega = pi*bmaj*min/(4*ln2)
        beams = {'f_mhz': [], 'beam_omega': []}  # np.ones(len(file_list))
        for h, cube in enumerate(self._cubes):
            try:
                hdr = cube.hdu.header
            except ValueError:
                hdr = cube.hdulist[0].header
            if hdr['BUNIT'] in ['Jy/beam', 'beam-1 Jy']:  # calculate conversion factor Jy/beam to K
                try:
                    bmaj = hdr['BMAJ'] * pi / 180.  # major axis in radians, assuming unit = degrees
                    bmin = hdr['BMIN'] * pi / 180.  # major axis in radians, assuming unit = degrees
                except KeyError:
                    for hdu in cube.hdulist[1:]:
                        try:
                            unit = hdu.columns['BMAJ'].unit  # assume bmaj and bmin have the same unit
                            fact = u.Quantity("1. {}".format(unit)).to(u.rad).value
                            bmaj = mean(hdu.data['BMAJ']) * fact
                            bmin = mean(hdu.data['BMIN']) * fact
                        except KeyError:
                            raise KeyError("Beam information not found in file {}.".format(self._data_file[h]))
                omega = pi * bmaj * bmin / (4. * log(2.))
                sp = (cube.spectral_axis.to(u.MHz)).value
                beams['f_mhz'].append(sp)
                beams['beam_omega'].append([omega for _ in sp])
                # 'jypb_MHz2': 1.e-26 * const.c.value ** 2 / 1.e12 / (2. * const.k_B.value * omega)})

        configuration['beam_info'] = beams

        # Mask:
        masks = configuration.get('masks', None)
        if masks is None:
            self._masked_pix_list = None
        else:
            file1 = os.path.join(self._data_path, masks[0])
            file2 = None
            if len(masks) > 1:
                file2 = os.path.join(self._data_path, masks[1])
            self._masked_pix_list = get_valid_pixels(self._wcs, file1, file2=file2, masked=True)

        self._model_configuration = ModelConfiguration(configuration)

        params = ['redchi']
        for cpt in self._model_configuration.cpt_list:
            params.extend([par.name for par in cpt.parameters])
            params.extend([par.name for sp in cpt.species_list for par in sp.parameters])

        # create arrays of zeros for the output parameters
        self._param_arrays = dict()
        for param in params:
            self._param_arrays['{}_arr'.format(param)] = zeros((self._nx, self._ny))

    def do_minimization(self, pix_nb=None, single_pix=True, size=None):
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
            self._model_configuration.get_continuum({'tc': self._cont_info})

            if self._model_configuration.win_list is None:
                self._model_configuration.get_linelist()
                self._model_configuration.get_windows()
            else:
                for win in self._model_configuration.win_list:
                    _, ywin = select_from_ranges(spec, [min(win.x_file), max(win.x_file)], y_values=data)
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
            for par in model.best_params:
                param = model.best_params[par]
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

    def make_maps(self):
        # read out parameter arrays into fits files
        units = {'tex': 'K', 'vlsr': 'km/s', 'fwhm': 'km/s', 'ntot': 'cm^-2', 'size': 'arcsec'}
        hdr = reduce_wcs_dim(self._wcs).to_header()

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

    @property
    def param_arrays(self):
        return self._param_arrays


# def frange(start, stop, step):
#     i = start
#     while i < stop:
#         yield i
#         i += step
