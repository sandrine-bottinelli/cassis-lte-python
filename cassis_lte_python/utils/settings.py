from __future__ import annotations

import os
import configparser


module_dir = os.path.split(os.path.dirname(__file__))[0]
user_config = os.path.join(module_dir, 'config.ini')
default_config = os.path.join(module_dir, 'config_defaults.ini')
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation(),
                                   inline_comment_prefixes=('#',))
if os.path.isfile(user_config):
    config.read(user_config)
else:
    if os.path.isfile(default_config):
        print(f'{user_config} not found, using {default_config}\n')
        config.read(default_config)
    else:
        raise FileNotFoundError('No configuration file found.')

CASSIS_DIR = config.get('GENERAL', 'CASSIS_DIR')
SQLITE_FILE = config.get('DATABASE', 'SQLITE_FILE')
PARTITION_FUNCTION_DIR = config.get('DATABASE', 'PARTITION_FUNCTION_DIR')
TELESCOPE_DIR = config.get('MODEL', 'TELESCOPE_DIR')
SIZE_DEF = config.getfloat('MODEL', 'SIZE')
VLSR_DEF = config.getfloat('MODEL', 'VLSR')
FWHM_DEF = config.getfloat('MODEL', 'FWHM')
DPI_DEF = config.getint('PLOT', 'DPI')

if not os.path.isfile(SQLITE_FILE):
    parent_dir = os.path.dirname(SQLITE_FILE)
    # try to find a cassisYYYYMMDD.db file in the parent directory :
    if os.path.isdir(parent_dir):
        try:
            db_list = [f for f in os.listdir(parent_dir)
                       if (f.endswith('.db') and f.startswith('cassis'))]
            db_list.sort()
            SQLITE_FILE_NEW = os.path.join(parent_dir, db_list[-1])
            if 'YYYYMMDD' in SQLITE_FILE:
                print(f'Using sqlite file {SQLITE_FILE_NEW}.\n')
            else:
                print(f'{SQLITE_FILE} not found, using {SQLITE_FILE_NEW} instead.\n')
            SQLITE_FILE = SQLITE_FILE_NEW
            # update config :
            config['DATABASE']['SQLITE_FILE'] = SQLITE_FILE
        except IndexError:
            raise FileNotFoundError(f'No cassisYYYYMMDD.db found in {parent_dir}.')
    else:
        raise FileNotFoundError(f'{SQLITE_FILE} not found.')


def print_settings():
    print("Settings are :")
    for section in config.sections():
        for key, val in dict(config.items(section)).items():
            unit = ""
            if "size" in key:
                unit = "arcsec"
            elif "vlsr" in key or "fwhm" in key:
                unit = "km/s"
            print(f"    {key.upper()} = {val} {unit}")
    print("")
