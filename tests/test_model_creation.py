import unittest
import os
import json

from cassis_lte_python.sim.model_setup import Component
from cassis_lte_python.sim.parameters import parameter_infos
from cassis_lte_python.database.species import Species
from cassis_lte_python.LTEmodel import ModelSpectrum


class TestModelCreation(unittest.TestCase):
    def test_model_creation_one_cpt_one_sp(self):
        filein = os.path.abspath('./tests/inputs/test_model_creation_one_cpt_one_sp.txt')
        with open(filein, "r") as f:
            data = json.load(f)

        value = data.pop("freq_step")  # `pop` removes the key and returns its value
        data["df_mhz"] = value

        value = data.pop("freq_range")  # `pop` removes the key and returns its value
        data["franges_ghz"] = [v/1000 for v in value]

        data['tuning_info'] = {data['telescope']: data['telescope_range']}

        comp = data.pop('components')
        comp_name = comp.pop("name")
        data["components"] = {comp_name: comp}

        data['modeling'] = True
        data['exec_time'] = False  # print execution time for file saving
        data['plot_gui'] = False
        data['plot_file'] = False

        mdl = ModelSpectrum(data)
        mdl.spectrum()

        self.assertEqual(1001, len(mdl.y_mod))
        self.assertAlmostEqual(max(mdl.y_mod), 5.148319918085383, places=5)
        self.assertAlmostEqual(min(mdl.x_mod), 147090., places=5)
        self.assertAlmostEqual(max(mdl.x_mod), 147190., places=5)

if __name__ == '__main__':
    unittest.main()