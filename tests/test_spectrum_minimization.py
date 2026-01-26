import unittest
import os
import json
import numpy as np

from cassis_lte_python.LTEmodel import ModelSpectrum


class TestSpectrumMinimization(unittest.TestCase):
    def test_spec_fit_one_cpt_one_sp(self):
        filein = os.path.abspath('./tests/inputs/test_spec_fit_one_cpt_one_sp.txt')
        with open(filein, "r") as f:
            # s = f.read()
            data = json.load(f)

        data_dir = data.pop("data_dir")
        data["data_file"] = os.path.join(data_dir, data['data_file'])

        comp = data.pop('components')
        comp_name = comp.pop("name")
        data["components"] = {comp_name: comp}

        mdl = ModelSpectrum(data)

        var_names = mdl.model_fit.var_names
        var_names.sort()
        v_names = ['c1_vlsr', 'c1_tex', 'c1_fwhm', 'c1_ntot_28503']
        v_names.sort()
        self.assertEqual(var_names, v_names)

        best_vals = {
            'c1_fwhm': 4.89145774952509,
            'c1_ntot_28503': 2.2614169966035428e+16,
            'c1_size': 30.0,
            'c1_tex': 71.87526412966713,
            'c1_vlsr': 3.8162656303833393
        }
        for name in v_names:
            if 'ntot' in name:
                self.assertAlmostEqual(np.log10(mdl.model_fit.best_values[name]), np.log10(best_vals[name]), places=5)
            else:
                self.assertAlmostEqual(mdl.model_fit.best_values[name], best_vals[name], places=3)
        pass


if __name__ == '__main__':
    unittest.main()
