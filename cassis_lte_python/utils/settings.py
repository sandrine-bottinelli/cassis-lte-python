from __future__ import annotations

import os
from cassis_lte_python import CONFIG, CONFIG_FILE, USER_CONFIG, BUILDING_DOC
from cassis_lte_python.utils.logger import CassisLogger


LOGGER = CassisLogger.create("settings")

if 'defaults' in CONFIG_FILE:
    LOGGER.warning(f'{USER_CONFIG} not found, using {CONFIG_FILE}\n')

CASSIS_DIR = CONFIG.get('GENERAL', 'CASSIS_DIR')
NB_DECIMALS = CONFIG.getint('GENERAL', 'NB_DECIMALS', fallback=2)
SQLITE_FILE = CONFIG.get('DATABASE', 'SQLITE_FILE')
PARTITION_FUNCTION_DIR = CONFIG.get('DATABASE', 'PARTITION_FUNCTION_DIR')
TELESCOPE_DIR = CONFIG.get('MODEL', 'TELESCOPE_DIR')
SIZE_DEF = CONFIG.getfloat('MODEL', 'SIZE')
VLSR_DEF = CONFIG.getfloat('MODEL', 'VLSR')
FWHM_DEF = CONFIG.getfloat('MODEL', 'FWHM')
DPI_DEF = CONFIG.getint('PLOT', 'DPI')
NROWS_DEF = CONFIG.getint('PLOT', 'NROWS', fallback=8)
NCOLS_DEF = CONFIG.getint('PLOT', 'NCOLS', fallback=3)
FONT_DEF = CONFIG.get('PLOT', 'FONT', fallback='DejaVu Sans')

if not os.path.isfile(SQLITE_FILE):
    parent_dir = os.path.dirname(SQLITE_FILE)
    # try to find a cassisYYYYMMDD.db file in the parent directory :
    if os.path.isdir(parent_dir):
        try:
            db_list = [f for f in os.listdir(parent_dir)
                       if (f.endswith('.db') and f.startswith('cassis'))]
            db_list.sort()
            SQLITE_FILE_NEW = os.path.join(parent_dir, db_list[-1])
            if 'YYYYMMDD' in SQLITE_FILE:  # generic file name
                LOGGER.info(f'No specific sqlite file provided, using sqlite file {SQLITE_FILE_NEW}.\n')
            else:
                LOGGER.warning(f'{SQLITE_FILE} not found, using {SQLITE_FILE_NEW} instead.\n')
            SQLITE_FILE = SQLITE_FILE_NEW
            # update config :
            CONFIG['DATABASE']['SQLITE_FILE'] = SQLITE_FILE
        except IndexError:
            if BUILDING_DOC:
                pass
            else:
                raise FileNotFoundError(f'No file of the form cassisYYYYMMDD.db found in {parent_dir}.')
    else:
        if BUILDING_DOC:
            pass
        else:
            raise FileNotFoundError(f'{SQLITE_FILE} not found.')

if not BUILDING_DOC:
    LOGGER.info(f"Using database : {SQLITE_FILE}")

def print_settings():
    message = ["Settings are :"]
    for section in CONFIG.sections():
        for key, val in dict(CONFIG.items(section)).items():
            unit = ""
            if "size" in key:
                unit = "arcsec"
            elif "vlsr" in key or "fwhm" in key:
                unit = "km/s"
            message.append(f"{key.upper()} = {val} {unit}")
    print("\n    ".join(message))
    print("")
