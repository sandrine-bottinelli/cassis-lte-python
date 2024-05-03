from __future__ import annotations

from numpy import interp, power, log10, genfromtxt
import os
import pandas as pd
from cassis_lte_python.utils.settings import PARTITION_FUNCTION_DIR, FWHM_DEF
from cassis_lte_python.sim.parameters import create_parameter
from cassis_lte_python.database.setupdb import DATABASE_SQL
from cassis_lte_python.database.constantsdb import THRESHOLDS_DEF


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

        sp_dic = get_species_info(self._tag)
        if sp_dic is None:
            raise IndexError("Tag {} not found in the database.".format(self._tag))
        self._id = sp_dic['id']
        self._name = sp_dic['name']
        self._database = sp_dic['database_name']

        self.pf = get_partition_function(self._tag)  # (tref, qlog)

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

    @property
    def parameters(self):
        return [self._ntot, self._fwhm]

    def set_component(self, comp_name):
        self._component = comp_name
        self._ntot.name = '{}_ntot_{}'.format(comp_name, self.tag)
        self._fwhm.name = '{}_fwhm_{}'.format(comp_name, self.tag)

    def get_partition_function(self, tex):
        # tmp = interp(log10(tex), log10(self.pf[0]), self.pf[1])
        # # tmp = find_nearest_id(self.pf[0], tex)
        # # tmp = self.pf[1][tmp]
        # qex = power(10., tmp)
        return get_partition_function_tex(self.pf[0], self.pf[1], tex, tag=self.tag)


def get_species_thresholds(sp_threshold_infos: list | dict | str | os.PathLike,
                           select_species=None,
                           return_list_sp=False):
    """
    Determine thresholds for the species to be modeled or plotted.
    :param sp_threshold_infos: can either be :
        a list : thresholds will be the default ones
        a dictionary : the list of species is extracted and the thresholds completed with default values if needed
        a path : the list of species and their thresholds are read from a file
    :param select_species: a subset of species for which the thresholds are wanted
    :param return_list_sp: return the list of species as well as the thresholds
    :return:
    """
    sp_thresholds = {}
    list_species = []

    if sp_threshold_infos is type(list):
        for other_sp in sp_threshold_infos:
            if isinstance(other_sp, list):  # if list, assume it is : [tag, eup_max, aij_min, err_max]
                new_sp = str(other_sp[0])  # make sure the tag is a string
                new_th = {
                    'eup_max': other_sp[1] if other_sp[1] != '*' else THRESHOLDS_DEF['eup_max'],
                    'aij_min': other_sp[2] if other_sp[2] != '*' else THRESHOLDS_DEF['aij_min'],
                    'err_max': other_sp[3] if other_sp[3] != '*' else THRESHOLDS_DEF['err_max']
                }
            else:
                new_sp = other_sp
                new_th = THRESHOLDS_DEF
            list_species.append(new_sp)
            sp_thresholds[str(new_sp)] = new_th

    elif isinstance(sp_threshold_infos, dict):
        sp_thresholds = {str(key): val for key, val in sp_threshold_infos.items()}  # make sure tag is a string
        list_species = list(sp_threshold_infos.keys())

    elif isinstance(sp_threshold_infos, (str, os.PathLike)):
        try:
            df = pd.read_csv(sp_threshold_infos, delimiter='\t', comment='#', index_col=0, dtype=str)
            if df.index.has_duplicates:
                dup = df.index[df.index.duplicated()]
                raise ValueError('Duplicate species infos detected for tags :',
                                 ", ".join([str(val) for val in dup.values]))
            df = df[[col for col in df.columns if not col.startswith('c')]]
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.fillna('*')
            list_species = [str(t) for t in df.index]
            for index, row in df.iterrows():
                sp_thresholds[str(index)] = {c: row[c] for c in df.columns if row[c] != '*'}
        except FileNotFoundError:
            raise FileNotFoundError(f"{sp_thresholds} not found.")
    else:
        raise TypeError("Thresholds information should be a list, a dictionary or a path to a file.")

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


def get_partition_function(tag, db=DATABASE_SQL.cursor(), temp=None):
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
        return get_partition_function_tex(tref, qlog, temp, tag=tag)


def get_partition_function_tex(tref, qlog, temp, tag=''):
    """
    Interpolate the partition function at the desired temp, if temp is w/i the min/max of the database values.
    Interpolation is done on a log-log scale.
    :param tref: database values for the temperature
    :param qlog: log10 of the partition function for the values in tref
    :param temp: temperature at which the partition function is desired
    :param tag: tag of the species
    :return: interpolated partition function at temp
    """
    if temp < tref[0]:
        raise ValueError(f'Tag {tag}: {temp} is below the lowest temperature of the partition function ({tref[0]})')
    if temp > tref[-1]:
        raise ValueError(f'Tag {tag}: {temp} is above the highest temperature of the partition function ({tref[-1]})')

    for i, t in enumerate(tref[:-1]):
        if tref[i+1] >= temp >= t:
            tmp = interp(log10(temp), log10(tref[i:i+2]), qlog[i:i+2])
            qex = power(10., tmp)
            # qex = np.interp(temp, tref, qlin)
            return qex
            # return np.power(10., qlog[find_nearest_id(np.array(tref),temp)])


def get_species_info(tag: str, database=DATABASE_SQL.cursor()):
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
