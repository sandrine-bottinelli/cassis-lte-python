from __future__ import annotations

from cassis_lte_python.utils import utils
from cassis_lte_python.gui.plots import file_plot, gui_plot
from cassis_lte_python.sim.model_setup import ModelConfiguration, Component
from cassis_lte_python.utils.settings import SQLITE_FILE
from cassis_lte_python.utils.constants import TEL_DIAM, PLOT_COLORS, CPT_COLORS, UNITS
from cassis_lte_python.database.species import get_species_thresholds
from cassis_lte_python.database.transitions import get_transition_df, select_transitions
import numpy as np
from numpy.random import normal
from lmfit import Model, Parameters
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
from astropy.wcs import WCS
from time import process_time
from warnings import warn
import copy


def generate_lte_model_func(config):

    def lte_model_func(fmhz, log=False, cpt=None, line_center_only=False, **params):
        norm_factors = config.get('norm_factors', {key: 1. for key in params.keys()})
        vlsr_file = config['vlsr_file']
        tc = config['tc'](fmhz)
        beam_sizes = config['beam_sizes'](fmhz)
        tmb2ta = config['tmb2ta'](fmhz)
        jypb2k = config['jypb2k'](fmhz)
        tcmb = config['tcmb']
        line_list = config['line_list']
        cpt_list = config['cpt_list']
        if not isinstance(cpt_list, list):
            cpt_list = [cpt_list]
        if cpt is not None:
            cpt_list = [cpt]
        tau_max = config['tau_max']
        file_rejected = config['file_rejected']
        intensity_before = tc + utils.jnu(fmhz, tcmb)
        intensity = 0.
        for icpt, cpt in enumerate(cpt_list):
            tex = params['{}_tex'.format(cpt.name)] * norm_factors['{}_tex'.format(cpt.name)]
            if log:
                tex = 10. ** tex
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
                if log:
                    ntot = 10. ** ntot
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
        intensity += normal(0., config['noise'], len(intensity))  # add gaussian noise
        intensity *= tmb2ta  # convert to Ta
        intensity /= jypb2k  # convert to Jy/beam

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
                for key, val in kwargs.items():
                    if key not in config or not isinstance(val, dict):
                        config[key] = val
                    else:
                        config[key].update(val)

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

        else:  # unknown
            raise TypeError("Configuration must be a dictionary or a path to a configuration file "
                            "or of type ModelConfiguration.")

        self.model_config = model_config

        self.params = None
        self.norm_factors = None
        self.model = None
        self.log = False
        self.model_fit = None
        self.model_fit_cpt = []
        self.normalize = False
        self.figure = None

        self.tag_colors = None
        self.tag_other_sp_colors = None
        self.cpt_cols = None

        if self.minimize:
            self.log = True

        if self.modeling or self.minimize:
            self.make_params(json_params=self.model_config.jparams)
            self.generate_lte_model()

        if self.minimize:
            t_start = process_time()
            # Perform the fit
            self.fit_model(max_nfev=self.max_iter, fit_kws=self.fit_kws)
            t_stop = process_time()
            if self.exec_time:
                print("Execution time for minimization : {}.".format(utils.format_time(t_stop - t_start)))
            if self.save_results:
                filename = ''
                if self.name_lam is not None:
                    filename = self.name_lam + '_'
                filename = filename + 'fit_res'
                self.save_fit_results(filename)

            if self.save_configs:
                if self.name_lam is not None:
                    self.write_lam(self.name_lam)
                if self.name_config is not None:
                    self.save_config(self.name_config)

        if self.plot_gui or self.plot_file:
            t_start = process_time()
            self.setup_plot()
            if self.exec_time:
                print(f"Execution time for preparing plot : {utils.format_time(process_time() - t_start)}.")

            self.make_plot()

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

    def save_config(self, filename, dirname=None):
        config_save = {
            'data_file': os.path.abspath(self.data_file) if self.data_file is not None else None,
            'output_dir': os.path.abspath(self.output_dir) if self.output_dir is not None else None,
            'tc': os.path.abspath(self.cont_info) if isinstance(self.cont_info, (str, os.PathLike)) else self.cont_info,
            'tcmb': self.tcmb,
            'tuning_info': self.model_config._tuning_info_user,
            'v_range': self.model_config._v_range_user,
            'chi2_info': self.model_config._rms_cal_user,
            'bandwidth': self.bandwidth,
            'oversampling': self.oversampling,
            'fghz_min': self.fmin_mhz / 1.e3,
            'fghz_max': self.fmax_mhz / 1.e3,
            'df_mhz': self.dfmhz,
            'noise': self.noise,
            'tau_max': self.tau_max,
            'thresholds': self.thresholds,
            'minimize': False,  # by default, do not (re-)minimize
            'modeling': self.modeling or self.minimize,  # if minimization was done, want modeling too by default
            'max_iter': self.max_iter,
            'fit_kws': self.fit_kws,
            'name_lam': self.name_lam,
            'name_config': os.path.abspath(self.name_config) if os.path.isfile(self.name_config) else self.name_config,
            'save_configs': self.save_configs,
            'save_results': self.save_results,
            'plot_kws': self.plot_kws,
            'plot_gui': self.plot_gui,
            'gui_kws': self.gui_kws,
            'plot_file': os.path.abspath(self.plot_file) if os.path.isfile(self.plot_file) else self.plot_file,
            'file_kws': self.file_kws,
            'exec_time': self.exec_time,
            'components': {cpt.name: cpt.as_json() for cpt in self.cpt_list},
            'params': self.params.dumps()
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

    def model_info(self, cpt=None):

        return {
            'tc': self.tc,
            'tcmb': self.tcmb,
            'vlsr_file': self.vlsr_file,
            'norm_factors': self.norm_factors,
            'log': self.log,
            'beam_sizes': self.beam,
            'tmb2ta': self.tmb2ta,
            'jypb2k': self.jypb,
            'line_list': self.line_list_all,
            'cpt_list': self.cpt_list if cpt is None else [cpt],
            'noise': self.noise,
            'tau_max': self.tau_max,
            'file_rejected': self.file_rejected
        }

    def get_tc(self, x_mod):
        return self.tc(x_mod)

    def get_rms(self, fmhz):
        if type(fmhz) == float:
            fmhz = list(fmhz)

        rms = []
        for freq in fmhz:
            for win in self.win_list_fit:
                if min(win.f_range_fit) <= freq <= max(win.f_range_fit):
                    rms.append(win.rms)

        return rms if len(rms) > 1 else rms[0]

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

        # Check min/max Tex:
        for icpt, cpt in enumerate(self.cpt_list):
            par = params[f'{cpt.name}_tex']
            if par.min < cpt.tmin:
                print(f'Component {cpt.name} : limiting Tex search to temperatures > {cpt.tmin} '
                      f'(smallest temperature for which the partition function is defined for all species).')
            if par.max > cpt.tmax:
                print(f'Component {cpt.name} : limiting Tex search to temperatures < {cpt.tmax} '
                      f'(highest temperature for which the partition function is defined for all species).')
            par.set(min=cpt.tmin if not isinstance(par.min, (float, int)) else max(par.min, cpt.tmin),
                    max=cpt.tmax if not isinstance(par.max, (float, int)) else min(par.max, cpt.tmax))

        if self.minimize:
            if self.log:
                for par in params:
                    if 'tex' in par or 'ntot' in par:
                        params[par].user_data = {'value': params[par].value,
                                                      'min': params[par].min,
                                                      'max': params[par].max}
                        if params[par].expr is None:
                            params[par].set(value=np.log10(params[par].value),
                                            min=np.log10(params[par].min),
                                            max=np.log10(params[par].max))

        # reset bounds if a parameters contains an expression to make sure it does not interfere
        for par in params:
            if params[par].expr is not None:
                params[par].set(min=-np.inf, max=np.inf)

        self.params = params

        norm_factors = {self.params[parname].name: 1. for parname in self.params}
        if normalize:
            for parname in self.params:
                param = self.params[parname]
                nf = abs(param.value) if param.value != 0. else 1.
                norm_factors[param.name] = nf
                if param.expr is None:
                    param.set(min=param.min / nf, max=param.max / nf, value=param.value / nf)
        self.norm_factors = norm_factors

    def generate_lte_model(self, normalize=False):
        if self.params is None:
            self.make_params(normalize=normalize)

        self.model = Model(generate_lte_model_func(self.model_info()),
                           independent_vars=['fmhz', 'log', 'cpt', 'line_center_only'
                                             # 'tc', 'beam_sizes', 'tmb2ta', 'jypb2k'
                                             ]
                           )

    def fit_model(self, max_nfev=None, fit_kws=None, print_report=True, report_kws=None):
        """
        Computes weights and perform the fit.
        :param max_nfev: maximum number of iterations (default value depends on the algorithm)
        :param fit_kws: keywords for the fit function
        :param print_report: whether to print the statistics and best values (default=True)
        :param report_kws: keywords for the fit_report function
        :param method: name of the fitting method
        :return:
        """

        def fit_callback(pars, iter, resid, *args, **kws):
            # Function called after each iteration to print the iteration number every 100 iterations
            if iter % 100 == 0:
                print(f"    Iteration {int(iter // 100) * 100 + 1}...")

        if len(self.win_list_fit) > 1:
            wt = np.concatenate([utils.compute_weight(win.y_fit - self.get_tc(win.x_fit), win.rms, win.cal)
                                 for win in self.win_list_fit], axis=None)
        else:
            win = self.win_list_fit[0]
            wt = utils.compute_weight(win.y_fit - self.get_tc(win.x_fit), win.rms, win.cal)

        # wt = None
        method = fit_kws.get('method', 'leastsq')
        print(f'Performing minimization with the {method} method...')
        if 'method' in fit_kws:
            fit_kws.pop('method')

        self.model_fit = self.model.fit(self.y_fit, params=self.params, fmhz=self.x_fit, log=True,
                                        # tc=self.tc(self.x_fit), beam_sizes=self.beam(self.x_fit),
                                        # tmb2ta=self.tmb2ta(self.x_fit), jypb2k=self.jypb(self.x_fit),
                                        cpt=None, line_center_only=False,
                                        weights=wt,
                                        method=method,
                                        max_nfev=max_nfev, fit_kws=fit_kws,
                                        iter_cb=fit_callback)

        if len(self.cpt_list) > 1:
            for cpt in self.cpt_list:
                model_fit_cpt = copy.deepcopy(self.model_fit)
                c_par = Parameters()
                for par in model_fit_cpt.params:
                    if cpt.name in par:
                        c_par[par] = self.model_fit.params[par]
                model_fit_cpt.params = c_par
                indices = [i for i, var_name in enumerate(model_fit_cpt.var_names) if cpt.name in var_name]
                nvarys = len(indices)
                model_fit_cpt.var_names = [model_fit_cpt.var_names[i] for i in indices]
                model_fit_cpt.best_values = {k: v for k, v in model_fit_cpt.best_values.items() if cpt.name in k}
                model_fit_cpt.nvarys = nvarys
                covar = np.zeros((nvarys, nvarys))
                for i in range(nvarys):
                    for j in range(nvarys):
                        covar[i, j] = model_fit_cpt.covar[indices[i], indices[j]]
                model_fit_cpt.covar = covar
                model_fit_cpt.model.func = generate_lte_model_func(self.model_info(cpt=cpt))
                self.model_fit_cpt.append(model_fit_cpt)

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

            if p.user_data is not None:
                p.init_value = p.user_data['value']
                if pfit.stderr is not None:
                    p.stderr = (10 ** (pfit.value + pfit.stderr) - 10 ** (pfit.value - pfit.stderr)) / 2
                val = 10 ** pfit.value if p.vary or p.expr is not None else p.user_data['value']
                p.set(value=val, min=p.user_data['min'], max=p.user_data['max'], is_init_value=False, expr=pfit.expr)

        # update vlr_plot
        if self.vlsr_file == 0:
            self.model_config.vlsr_plot = self.cpt_list[0].vlsr

        # reset norm factors and log scale
        self.norm_factors = {key: 1 for key in self.norm_factors.keys()}
        self.log = False

        if print_report:
            print(self.fit_report(report_kws=report_kws))

    def fit_report(self, report_kws=None):
        fit_params = self.model_fit.params.copy()
        self.model_fit.params = self.params

        if report_kws is None:
            report_kws = {}

        if 'show_correl' not in report_kws:
            report_kws['show_correl'] = False

        report = self.model_fit.fit_report(**report_kws)
        pvalue = '    p-value            = {}'.format(
            utils.format_float(stats.chi2.sf(self.model_fit.chisqr, self.model_fit.nfree), nb_signif_digits=2)
        )
        lines = report.split(sep='\n')
        new_lines = []
        for line in lines:
            if "+/-" in line:  # reformat
                begin_line, end_line = line.split(sep=" +/- ")
                label, val = begin_line.rsplit(sep=' ', maxsplit=1)
                begin_line = f"{label} {utils.format_float(float(val), nb_signif_digits=2): <8}"
                err, rest = end_line.split(maxsplit=1)
                end_line = f"{utils.format_float(float(err), nb_signif_digits=2): <8} {rest}"
                new_lines.append(" +/- ".join([begin_line, end_line]))

            elif 'init' in line:  # no uncertainties, reformat as well
                begin_line, end_line = line.split(sep=" (")
                label, val = begin_line.rsplit(sep=' ', maxsplit=1)
                begin_line = f"{label} {utils.format_float(float(val), nb_signif_digits=2): <8}"
                new_lines.append(" (".join([begin_line, end_line]))

            elif "=" in line and ":" not in line and "#" not in line:  # reformat statistics if not integers
                elts = line.rsplit(sep="=", maxsplit=1)
                new_lines.append("= ".join([elts[0], utils.format_float(float(elts[1]), nb_signif_digits=2)]))

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

        for cpt in self.cpt_list:
            params = Parameters()
            for par in self.model_fit.params:
                if cpt.name in par:
                    params[par] = self.model_fit.params[par]
            indices = [i for i, var_name in enumerate(self.model_fit.var_names) if cpt.name in var_name]
            var_names = [self.model_fit.var_names[i] for i in indices]
            nvarys = len(var_names)
            cpt_model_func = generate_lte_model_func(self.model_info(cpt=cpt))

            # ensure fjac and df2 are correct size if independent var updated by kwargs
            feval = cpt_model_func(fmhz=fmhz, log=True, **params)
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
                res1 = cpt_model_func(fmhz=fmhz, log=True, **pars)

                pars[pname].value = val0 - dval
                res2 = cpt_model_func(fmhz=fmhz, log=True, **pars)

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

        return self.model.func(x_values, log=False, cpt=cpt, line_center_only=line_center_only,
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
        win = self.win_list_plot[0]
        win.x_mod = self.x_mod
        # win.f_range_plot = [min(win.x_file), max(win.x_file)]
        win.f_range_plot = [min(win.x_mod), max(win.x_mod)]
        win.bottom_unit = 'MHz'
        win.bottom_lim = utils.get_extended_limits(win.f_range_plot)
        win.top_lim = win.bottom_lim

        # compute the model :
        plot_pars = self.params
        # win.x_mod, win.y_mod = select_from_ranges(self.x_mod, win.f_range_plot, y_values=self.y_mod)
        # TODO: check if following is necessary
        if win.x_file is not None:
            x_mod = []
            for i, row in self.tuning_info.iterrows():
                x_sub = win.x_file[(win.x_file >= row['fmhz_min']) & (win.x_file <= row['fmhz_max'])]
                if len(x_sub) == 0:
                    continue
                x_mod.extend(np.linspace(min(x_sub), max(x_sub), num=self.oversampling * len(x_sub)))
            win.x_mod = np.array(x_mod)

        else:
            pass
        win.y_mod = self.compute_model_intensities(params=plot_pars, x_values=win.x_mod,
                                                   line_list=self.line_list_all)
        if len(self.cpt_list) > 1:
            for icpt in range(len(self.cpt_list)):
                win.y_mod_cpt.append(self.compute_model_intensities(params=plot_pars, x_values=win.x_mod,
                                                                    line_list=self.line_list_all,
                                                                    cpt=self.cpt_list[icpt]))

        win.x_mod_plot = win.x_mod
        win.x_file_plot = win.x_file

        if win.x_file is not None:
            win.y_res = win.y_file - self.compute_model_intensities(params=plot_pars, x_values=win.x_file,
                                                                    line_list=self.line_list_all)
            win.y_res += self.get_tc(win.x_file)

    def setup_plot_la(self, verbose=True, other_species_dict: dict | None = None, **kwargs):
        """
        Prepare all data to do the plots in line analysis mode

        :param verbose:
        :param other_species_dict: a dictionary of other species and their thresholds
        :return:
        """

        # Define some useful quantities
        plot_pars = self.params
        vlsr = self.cpt_list[0].vlsr if self.vlsr_file == 0. else self.vlsr_file
        fwhm = max([plot_pars[par].value for par in plot_pars if 'fwhm' in par])

        # self.update_parameters(params=plot_pars)

        if other_species_dict is not None:  # list of tags for which the user wants line positions
            thresholds_other = other_species_dict
        else:
            thresholds_other = {}

        for t in self.tag_list:
            if t in thresholds_other:
                thresholds_other.pop(t)
        list_other_species = list(thresholds_other.keys())

        # lines from other species : if many other species, more efficient to first find all transitions
        # across entire observed range, then filter in each window
        if len(list_other_species) > 0:
            other_species_lines = get_transition_df(list_other_species,
                                                    [[min(win.x_file), max(win.x_file)] for win in self.win_list_plot],
                                                    **thresholds_other)
        # other_species_lines = get_transition_df(list_other_species, [[min(self.x_file), max(self.x_file)]],
        #                                         **thresholds_other)
        else:
            other_species_lines = pd.DataFrame()  # empty dataframe

        # Compute model overall model : takes longer than cycling through windows unless strong overlap of windows (TBC)
        # self.y_mod = self.compute_model_intensities(params=plot_pars, x_values=self.x_mod,
        #                                             line_list=self.line_list_all)

        # Compute model and line positions for each window
        t_win = datetime.datetime.now()
        print(f"Start preparing windows : {t_win.strftime('%H:%M:%S')}", end="... ")
        t_start = process_time()
        for iwin, win in enumerate(self.win_list_plot):
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
                win.y_res += self.get_tc(win.x_file)

            if self.minimize:
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
                    win.y_mod_err_cpt = self.eval_uncertainties_components(fmhz=win.x_mod)

            if (self.modeling or self.minimize) and (len(self.cpt_list) > 1):
                for icpt in range(len(self.cpt_list)):
                    win.y_mod_cpt.append(self.compute_model_intensities(params=plot_pars, x_values=win.x_mod,
                                                                        line_list=model_lines_win,
                                                                        cpt=self.cpt_list[icpt]))

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

            # if tag of fitted species in other species, concatenate with user lines, dropping duplicates
            for tag in self.model_config.tag_list:
                if tag in other_species_win.tag.unique():
                    other_species_win = pd.concat([model_lines_user,
                                                   other_species_win]).drop_duplicates(subset='db_id', keep=False)
            # concatenate with model lines outside thresholds, keeping first occurrence of duplicates
            # other_species_win = pd.concat([model_lines_other,
            #                                other_species_win_all]).drop_duplicates(subset='db_id', keep='first')

            win_colors = {t: PLOT_COLORS[itag] for itag, t in enumerate(model_lines_user.tag.unique())}
            for icpt, cpt in enumerate(self.cpt_list):
                # build list of dataframes containing lines to be plotted for each component
                win.main_lines_display[icpt] = self.get_lines_plot_params(
                    model_lines_user[model_lines_user['tag'].isin(cpt.tag_list)], cpt, f_ref, tag_colors=win_colors)
                # win.other_lines_display[icpt] = self.get_lines_plot_params(
                #     model_lines_other[model_lines_other['tag'].isin(cpt.tag_list)], cpt, f_ref, tag_colors=win_colors)

            # line plot parameters for other lines and other species (not component-dependent)
            plot_colors_sub = PLOT_COLORS[len(win_colors):]
            icol = 0
            for itag, t in enumerate(other_species_win.tag.unique()):
                if t not in win_colors.keys():
                    win_colors[t] = plot_colors_sub[icol % len(plot_colors_sub)]
                    icol += 1
            win.tag_colors = win_colors

            if len(other_species_win) > 0:
                win.other_species_display = self.get_lines_plot_params(other_species_win, self.cpt_list[0], f_ref,
                                                                       tag_colors=win_colors)
            if iwin == 0:
                prep_time = (process_time() - t_start) * len(self.win_list_plot)
                t_win += datetime.timedelta(seconds=prep_time)
                print(f"Expected end time : {t_win.strftime('%H:%M:%S')}")

        # save line list
        # cols = ['tag', 'sp_name', 'fMHz', 'f_err_mhz', 'aij', 'elow', 'eup', 'igu', 'catdir_id', 'qn']
        cols = ['tag', 'sp_name', 'x_pos', 'fMHz', 'f_err_mhz', 'aij', 'elow', 'eup', 'igu', 'qn']
        with open(os.path.join(self.output_dir, 'linelist.txt'), "w") as f:
            for win in self.win_list_plot:
                f.write(f"{win.name} : model species within thresholds\n")
                f.writelines(win.main_lines_display[0][cols].to_string(index=False))
                f.write("\n\n")
                if len(win.other_species_display) > 0:
                    f.write(f"{win.name} : model species outside thresholds and other species within thresholds\n")
                    f.writelines(win.other_species_display[cols].to_string(index=False))
                    f.write("\n\n")

    def get_lines_plot_params(self, line_list: pd.DataFrame, cpt: Component, f_ref: float,
                              tag_colors: dict):

        colors = tag_colors
        lines_plot_params = line_list.copy()
        lines_plot_params['x_pos'] = [utils.frequency_to_velocity(row.fMHz, f_ref, vref_kms=cpt.vlsr)
                                      for i, row in lines_plot_params.iterrows()]
        lines_plot_params['x_pos_err'] = [utils.delta_v_to_delta_f(row.f_err_mhz, f_ref, reverse=True)
                                          for i, row in lines_plot_params.iterrows()]
        lines_plot_params['label'] = [row.tag for i, row in lines_plot_params.iterrows()]
        lines_plot_params['color'] = [colors[row.tag] for i, row in lines_plot_params.iterrows()]

        return lines_plot_params

    def select_windows(self, tag: list | None = None, display_all=True, windows: list | dict | None = None):
        """
        Determine windows to plot
        :param tag: tag selection if do not want all the tags
        :param display_all: if False, only display windows with fitted data
        :param windows: a dictionary of the windows to be plotted (keys=tags, vals=window numbers)
        :return:
        """

        self.win_list_plot = self.win_list  # by default, plot everything

        if not display_all:  # only display windows with fitted data
            self.win_list_plot = [w for w in self.win_list_plot if w.in_fit]

        if tag is not None:  # user only wants one tag
            self.win_list_plot = [w for w in self.win_list_plot if w.transition.tag in tag]

        if windows is not None and len(windows) > 0:
            if isinstance(windows, dict):
                new_dict = utils.expand_dict(windows, expand_vals=True)
                win_names2plot = [f'{key} - {val}' for key in new_dict.keys() for val in new_dict[key]]
            else:
                win_names2plot = windows

            self.win_list_plot = [w for w in self.win_list_plot if w.name in win_names2plot]

        if len(self.win_list_plot) == 0:
            raise LookupError("No windows to plot. Please check your tag selection.")

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
        model_err = kwargs.get('model_err', True)
        component_err = kwargs.get('component_err', True)

        # Find all windows required by the user for gui or file :
        display_all = self.gui_kws['display_all'] or self.file_kws['display_all']
        if display_all:
            win2plot = None
        else:
            win2plot = kwargs.get('windows', {})
            win2plot.update(self.gui_kws.get('windows', {}))
            win2plot.update(self.file_kws.get('windows', {}))
            if len(win2plot) > 0:
                new_dict = utils.expand_dict(win2plot, expand_vals=True)
                win2plot = [f'{key} - {val}' for key in new_dict.keys() for val in new_dict[key]]
                win2plot = list(set(win2plot))

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

        if self.bandwidth is None or self.model_config.fit_freq_except is not None:
            self.win_list_plot = self.win_list
            self.setup_plot_fus()
        else:
            self.select_windows(tag=tag, display_all=display_all, windows=win2plot)
            self.setup_plot_la(verbose=verbose, other_species_dict=thresholds_other,
                               model_err=model_err, component_err=component_err)
            if other_species_win_selection is not None:
                if isinstance(other_species_win_selection, int):
                    other_species_win_selection = str(other_species_win_selection)
                self.select_windows_other_lines(other_species_win_selection)

    def make_plot(self):
        """
        Do the plot(s), using provided keywords.
        Possible keywords are :
         filename: nome of the file to be saved
         dirname: directory where to save the file
         gui: interactive display
         verbose:
         dpi:
         nrows: maximum number of rows per page
         ncols: maximum number of columns per page
        :return:

        Notes :
            - other_species_selection is deprecated, use other_species_win_selection
        """

        if self.plot_gui:
            kwargs = self.plot_kws.copy()
            kwargs.update(self.gui_kws)
            self.select_windows(tag=kwargs['tag'], display_all=kwargs['display_all'], windows=kwargs['windows'])

            gui_plot(self)

        if self.plot_file:
            kwargs = self.plot_kws.copy()
            kwargs.update(self.file_kws)
            filename = self.file_kws['filename']
            dirname = kwargs.get('dirname', None)
            verbose = kwargs.get('verbose', True)
            dpi = kwargs.get('dpi', None)
            nrows = kwargs.get('nrows', 4)
            ncols = kwargs.get('ncols', 3)
            self.select_windows(tag=kwargs['tag'], display_all=kwargs['display_all'], windows=kwargs['windows'])

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
                    win.x_mod = np.linspace(min(win.x_file), max(win.x_file),
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
                            y_mod = np.hstack((y_mod, y_cpt.reshape(len(y_cpt), 1)))
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
                    y_values = np.hstack((y_values, y_cpt.reshape(len(y_cpt), 1)))

        spec = x_values, y_values
        self.save_spectrum(filename, dirname=dirname, ext=ext, spec=spec)

    def save_spectrum(self, filename, dirname=None, ext='txt',
                      spec: None | tuple = None, continuum=False,
                      vlsr: None | float | int = None):
        """
        Write a spectrum (continuum, data or model, depending on the provided parameters) on a file.
        :param filename:
        :param dirname:
        :param ext:
        :param spec: tuple of x and y values to be written ;
                     if not provided and continuum is false, stored model is written
        :param continuum: if True, spec is ignored
        :param vlsr:
        :return: the path to the file
        """
        file_path = self.set_filepath(filename, dirname=dirname, ext=ext)

        spectrum_type = ''
        if continuum:  # spec will be ignored
            spectrum_type = 'continuum'
            ext = 'txt'  # force txt extension
            file_path = self.set_filepath(filename + '_cont', dirname=dirname, ext=ext)
            x_values = self.tc['f_mhz']
            y_values = self.tc['tc']
        else:
            if spec is not None:  # should be a tuple of x and y values
                x_values, y_values = spec
            # elif self.data_file is None and self.x_file is not None:  # TODO: check whether this case is useful
            #     self.data_file = file_path
            #     x_values, y_values = self.x_file, self.y_file
            else:
                x_values, y_values = self.x_mod, self.y_mod
                spectrum_type = 'synthetic'

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
                    f.writelines(['#title: Spectral profile\n',
                                  '#date: {}\n'.format(datetime.datetime.now().strftime("%c")),
                                  '#coordinate: world\n',
                                  '#xLabel: frequency [MHz]\n',
                                  '#yLabel: [Kelvin] Mean\n'])
                for x, y in zip(x_values, y_values):
                    if len(np.shape(y_values)) == 1:
                        f.write('{}\t{}\n'.format(utils.format_float(x, nb_signif_digits=4), utils.format_float(y)))
                    else:
                        line = '{}\t'.format(utils.format_float(x, nb_signif_digits=4))
                        line += '\t'.join([utils.format_float(yy) for yy in y])
                        f.write(line + '\n')

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

    def save_line_list_cassis(self, filename, dirname=None):
        """
        Writes the list of lines for display in CASSIS. To be used when fitting the entire spectrum.
        :param filename:
        :param dirname:
        :return:
        """
        if len(self.win_list) == 1:
            self.win_list_plot = self.win_list
            self.setup_plot_fus()
            filebase = filename
            nb_dec = '3'
            for icpt, cpt in enumerate(self.cpt_list):
                if len(self.cpt_list) > 1:
                    filename = f'{filebase}_{cpt.name}'
                with open(self.set_filepath(filename, dirname=dirname, ext='txt'), 'w') as f:
                    f.write('# ')
                    f.write('\t'.join(
                        ['Transition', 'Tag', 'Frequency(MHz)', 'Eup(K)', 'Aij', 'Tau', 'Tex', 'Intensity(K)']))
                    f.write('\n')

                    for i, row in self.line_list_all.iterrows():
                        line = row['transition']
                        if line.tag in cpt.tag_list:
                            tex = self.params[f"{cpt.name}_tex"].value
                            tau0 = utils.compute_tau0(line,
                                                      self.params[f"{cpt.name}_ntot_{line.tag}"].value,
                                                      self.params[f"{cpt.name}_fwhm_{line.tag}"].value,
                                                      tex)
                            ind0 = utils.find_nearest_id(self.win_list[0].x_mod, line.f_trans_mhz)
                            f.write(f"{line.name} ({line.qn_lo}_{line.qn_hi})\t{line.tag}")
                            f.write(f"\t{line.f_trans_mhz:.{nb_dec}f}")
                            f.write(f"\t{line.eup:.{nb_dec}f}\t{line.aij:.{nb_dec}e}\t")
                            f.write(f"{tau0:.{nb_dec}e}\t{tex:.{nb_dec}f}")
                            f.write(f"\t{self.win_list[0].y_mod_cpt[icpt][ind0]:.{nb_dec}e}\n")

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
            lte_radex = {'lteRadexSelected': 'true', **lte_radex, 'overSampling': self.oversampling}

        # Define continuum
        if isinstance(self.model_config.cont_info, (float, int)):
            cont_type = 'CONSTANT'
            cont_size = self.tc  # in fact the continuum value
            cont = 'Continuum 0 [K]'  # default
        else:  # it is a file
            cont_type = 'FILE'
            cont_size = '0.0'  # default
            cont = os.path.abspath(self.model_config.cont_info)

        components = {
            '# Component parameters 1': {
                'Comp1Name': 'Continuum',
                'Comp1Enabled': 'true',
                'Comp1Interacting': 'false',
                'Comp1ContinuumSelected': cont_type,
                'Comp1Continuum': cont,
                'Comp1ContinuumSize': cont_size
            }
        }

        # Define other components
        params = self.params
        # if self.best_params is not None and ext == 'lam':
        #     params = self.best_params

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
                tel_path = os.path.abspath(utils.search_telescope_file(tel))
                if 'telescopeData' in tuning:
                    tuning['telescopeData'] = tel_path
                if 'telescope' in tuning:
                    tuning['telescope'] = tel_path
                lte_radex['telescope'] = tel_path
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

    def write_lam(self, filename, dirname=None):
        """
        Writes a line analysis configuration file for CASSIS
        :param filename: the name of the file
        :param dirname: the directory where to save the file
        :return: None
        """
        self.write_cassis_file(filename, dirname=dirname)


class ModelCube:  # TODO : needs to be updated
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
            self._masked_pix_list = utils.get_valid_pixels(self._wcs, file1, file2=file2, masked=True)

        self._model_configuration = ModelConfiguration(configuration)

        params = ['redchi']
        for cpt in self._model_configuration.cpt_list:
            params.extend([par.name for par in cpt.parameters])
            params.extend([par.name for sp in cpt.species_list for par in sp.parameters])

        # create arrays of zeros for the output parameters
        self._param_arrays = dict()
        for param in params:
            self._param_arrays['{}_arr'.format(param)] = np.zeros((self._nx, self._ny))

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

    def make_maps(self):
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

    @property
    def param_arrays(self):
        return self._param_arrays


# def frange(start, stop, step):
#     i = start
#     while i < stop:
#         yield i
#         i += step
