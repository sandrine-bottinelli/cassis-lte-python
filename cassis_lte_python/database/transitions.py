from __future__ import annotations

import pandas as pd
from cassis_lte_python.utils.utils import delta_v_to_delta_f
from cassis_lte_python.utils.constants import C_LIGHT, K_B, H
from cassis_lte_python.database.setupdb import DATABASE_SQL
from cassis_lte_python.database.species import get_species_info
from cassis_lte_python.database.constantsdb import THRESHOLDS_DEF


# Define dictionary for query fields and corresponding column names in the dataframe for transitions
# query can be on : ['id_transitions', 'fMHz', 'err', 'aint', 'elow', 'eup', 'igu', 'catdir_id', 'qn',
# 'id_database', 'gamma_self', 'gamma_self_error']
FIELDS = {
    'id_transitions': 'db_id',
    'fMHz': 'fMHz',
    'err': 'f_err_mhz',
    'aint': 'aij',
    'elow': 'elow',
    'eup': 'eup',
    'igu': 'igu',
    'catdir_id': 'catdir_id',
    'qn': 'qn'
}
TRAN_COLNAME = 'transition'
TAG_COLNAME = 'tag'
SP_COLNAME = 'sp_name'
TRAN_DF_COLS = [TRAN_COLNAME, TAG_COLNAME, SP_COLNAME] + list(FIELDS.values())


class Transition:
    def __init__(self, tag, f_trans_mhz, aij, elo_cm, gup, sp_name='', f_err_mhz=None, db_id=None, qn=''):
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
        self.name = sp_name
        self.db_id = db_id
        self.qn = qn
        if len(qn) > 0:
            qns = qn.split(':')
            self.qn_lo = " ".join(qns[:int(len(qns)/2)])
            self.qn_hi = " ".join(qns[int(len(qns)/2):])
        else:
            self.qn_lo, self.qn_hi = '', ''

    def __str__(self):
        infos = ["{} ({}_{})".format(self.name, self.qn_lo, self.qn_hi),
                 "Tag = {}".format(self.tag),
                 "f = {} MHz (+/-{})".format(self.f_trans_mhz, self.f_err_mhz),
                 "Eup = {:.2f} K".format(self.eup),
                 "Aij = {:.2e} s-1".format(self.aij),
                 "gup = {}".format(self.gup)]
        return " ; ".join(infos)

    def __eq__(self, other):
        return True if self.db_id == other.db_id else False


def get_transition_list(species: list | str, fmhz_ranges, database=DATABASE_SQL, return_type='dict', **thresholds):
    """

    :param species:
    :param fmhz_ranges:
    :param database:
    :param return_type:
    :param thresholds:
    :return:
    """
    tran_df = get_transition_df(species, fmhz_ranges, database=database, **thresholds)
    if return_type == 'dict':  # return dictionary {tag: transition list}
        transition_dict = {}
        tag_list = list(set(tran_df.tag))
        tag_list.sort()
        for t in tag_list:
            transition_dict[t] = list(tran_df[tran_df.tag == t].transition)
        return transition_dict
    elif return_type == 'df':  # return dataframe
        return tran_df
    else:  # return list
        tr_list = list(tran_df.transition)
        tr_list.sort(key=lambda x: x.f_trans_mhz)


def get_transition_df(species: list | str, fmhz_ranges, database=DATABASE_SQL, **thresholds):
    if len(species) == 0:
        return pd.DataFrame(columns=TRAN_DF_COLS)

    species_list = [species] if type(species) is not list else species  # make sure it is a list
    tag_list = [str(sp) for sp in species_list]  # ensure it is a list of strings
    if not isinstance(fmhz_ranges[0], list):  # only one range
        fmhz_ranges = [fmhz_ranges]  # wrap it in a list

    if not thresholds:
        thresholds = {tag: {} for tag in tag_list}
    # thresholds = {tag: {label: thresholds[tag].get(label, value)
    #                     for label, value in THRESHOLDS_DEF.items()} for tag in tag_list}

    fields = FIELDS.keys()

    database = database.cursor()
    sp_transitions = []
    sp_infos = {}
    for tag in tag_list:  # NB: tag should be a string
        eup_min = thresholds[tag].get('eup_min', THRESHOLDS_DEF['eup_min'])
        eup_max = thresholds[tag].get('eup_max', THRESHOLDS_DEF['eup_max'])
        aij_min = thresholds[tag].get('aij_min', THRESHOLDS_DEF['aij_min'])
        aij_max = thresholds[tag].get('aij_max', THRESHOLDS_DEF['aij_max'])
        err_max = thresholds[tag].get('err_max', THRESHOLDS_DEF['err_max'])

        # retrieve catdir_id :
        sp_dic = get_species_info(tag)
        if sp_dic is None:
            if isinstance(species, list):
                try:
                    species.remove(tag)
                except IndexError:
                    species.remove(int(tag))
            continue
        sp_id = sp_dic['id']
        sp_infos[sp_id] = sp_dic
        # sp_name = sp_dic['name']
        # Find transitions for each range of frequencies
        for frange in fmhz_ranges:
            # cmd = "SELECT * FROM transitions WHERE catdir_id = {} and fMhz < {} and fMhz > {}" \
            #       " and eup > {} and aint > {}".format(sp_id, max(frange), min(frange), eup_min, aij_min)
            cmd = f"SELECT {', '.join(fields)} FROM transitions WHERE catdir_id = {sp_id}"
            cmd += f" and fMhz < {max(frange)} and fMhz > {min(frange)} and eup > {eup_min} and aint > {aij_min}"
            if eup_max is not None:
                cmd += " and eup < {}".format(eup_max)
            if aij_max is not None:
                cmd += " and aint < {}".format(aij_max)
            if err_max is not None:
                cmd += " and err < {}".format(err_max)

            res = database.execute(cmd)
            # col_names = [t[0] for t in res.description]
            all_rows = database.fetchall()
            if len(all_rows) == 0:
                continue

            sp_transitions.extend(all_rows)

    tran_dict = {c: [r[i] for r in sp_transitions] for i, c in enumerate(fields)}
    # tran_dict[TAG_COLNAME] = [sp_infos[cid]['speciesid'] for cid in tran_dict['catdir_id']]
    # tran_dict[SP_COLNAME] = [sp_infos[cid]['name'] for cid in tran_dict['catdir_id']]
    tran_df = pd.DataFrame(tran_dict)
    tran_df.insert(0, SP_COLNAME, [sp_infos[cid]['name'] for cid in tran_dict['catdir_id']])
    tran_df.insert(0, TAG_COLNAME, [sp_infos[cid]['speciesid'] for cid in tran_dict['catdir_id']])
    tran_df.insert(0, TRAN_COLNAME, [Transition(row.tag, row.fMHz, row.aint, row.elow, row.igu,
                                                f_err_mhz=row.err, sp_name=row.sp_name, db_id=row.id_transitions,
                                                qn=row.qn)
                                     for i, row in tran_df.iterrows()])
    tran_df.rename(columns={key: val for key, val in FIELDS.items() if key != val}, inplace=True)
    # if len(df) == 0:  # no transitions found, going to the next tag
    #     # print(f"No transitions found for tag {tag}.")
    #     continue
    #
    # if tran_df is None:
    #     tran_df = df.copy()
    # else:
    #     tran_df = pd.concat([tran_df, df])

    # if len(tran_df) == 0:
    #     raise IndexError('No transitions found. Please check your thresholds.')

    return tran_df


def is_selected(tran: Transition, sp_thresholds: dict, bright_lines_only=False):
    """
    Determine if a transition fulfills all constraints.
    :param tran: a Transition object
    :param sp_thresholds: a dictionary containing the species' thresholds
    :param bright_lines_only: TODO: TBC
    :return:
    """
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


def select_transitions(tran_df: pd.DataFrame, thresholds: dict | None = None, xrange=None,
                       bright_lines_only=False, vlsr=None):

    if not isinstance(tran_df, pd.DataFrame):
        raise TypeError("First argument must be a DataFrame.")

    if tran_df.empty:
        return tran_df

    if vlsr is not None:
        if xrange is not None:
            xrange = [x + delta_v_to_delta_f(vlsr, sum(xrange)/len(xrange)) for x in xrange]
        else:
            print("INFO - No frequency range specified, ignoring the vlsr keyword.")

    if xrange is not None:
        tran_df = tran_df[(min(xrange) <= tran_df.fMHz) & (tran_df.fMHz <= max(xrange))]

    if thresholds is None:
        thresholds = {}

    if len(thresholds) == 0:
        return tran_df
    else:
        res = pd.DataFrame(columns=tran_df.columns)
        for tag, thres in thresholds.items():
            selected = tran_df[tran_df.tag == tag]
            if bright_lines_only:
                # if user used a non-zero eup_min (and/or finite aij_max), want to find transitions with eup < eup_min
                # (and/or aij > aij_max)
                thres['eup_max'] = thres['eup_min']
                thres['eup_min'] = 0
                thres['aij_min'] = thres['aij_max']
                thres['aij_max'] = None

            for key, val in thres.items():  # thres is a dictionary containing the thresholds
                attr = key.rsplit('_', maxsplit=1)[0]  # get the attribute to be tested (eup, aij, etc)
                # if bright_lines_only and key == 'eup_min':
                #     val = 0
                # if bright_lines_only and key == 'aij_max':
                #     val = None
                if 'min' in key:
                    selected = selected[selected[attr] >= val]
                if 'max' in key and val is not None:
                    selected = selected[selected[attr if key != 'err_max' else 'f_err_mhz'] <= val]

            if res.empty:
                res = selected
            else:
                res = pd.concat([res, selected])

        return res
