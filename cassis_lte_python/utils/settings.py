from __future__ import annotations

import os
import datetime
import configparser
from pathlib import Path

# NB: do not import Cassis Logger in this module to avoid loop import


class Settings:
    def __init__(self):
        LOG_PATH_DEFAULT = 'logs'

        # Determine whether to enable file logger upon import
        module_dir = str(Path(__file__).resolve().parents[1])
        self.USER_CONFIG = os.path.join(module_dir, 'settings.ini')
        self.DEFAULT_CONFIG = os.path.join(module_dir, 'settings_defaults.ini')
        CONFIG = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation(),
                                           inline_comment_prefixes=('#',))

        self.CONFIG_FILE = self.USER_CONFIG
        if not os.path.isfile(self.CONFIG_FILE):
            if os.path.isfile(self.DEFAULT_CONFIG):
                # print(f'{user_config} not found, using {default_config}\n')
                self.CONFIG_FILE = self.DEFAULT_CONFIG
            else:
                raise FileNotFoundError(f'No settings file found in {module_dir}.')

        CONFIG.read(self.CONFIG_FILE)


        # else:
        #     ENABLE_FILE_LOGGER = CONFIG.getboolean('LOGGER', 'ENABLE_FILE_LOGGER', fallback=True)
        # LOG_PATH = CONFIG.get('LOGGER', 'LOG_PATH', fallback='logs')
        # if ENABLE_FILE_LOGGER and not os.path.exists(LOG_PATH):
        #     os.makedirs(LOG_PATH)
        # Reload config_file in case it has changed
        # CONFIG.read(CONFIG_FILE)

        self.CASSIS_DIR = CONFIG.get('GENERAL', 'CASSIS_DIR')
        self.NB_DECIMALS = CONFIG.getint('GENERAL', 'NB_DECIMALS', fallback=2)
        SQLITE_FILE = CONFIG.get('DATABASE', 'SQLITE_FILE')
        self.PARTITION_FUNCTION_DIR = CONFIG.get('DATABASE', 'PARTITION_FUNCTION_DIR')
        self.TELESCOPE_DIR = CONFIG.get('MODEL', 'TELESCOPE_DIR')
        self.SIZE_DEF = CONFIG.getfloat('MODEL', 'SIZE')
        self.VLSR_DEF = CONFIG.getfloat('MODEL', 'VLSR')
        self.FWHM_DEF = CONFIG.getfloat('MODEL', 'FWHM')
        self.DPI_DEF = CONFIG.getint('PLOT', 'DPI')
        self.NROWS_DEF = CONFIG.getint('PLOT', 'NROWS', fallback=8)
        self.NCOLS_DEF = CONFIG.getint('PLOT', 'NCOLS', fallback=3)
        self.FONT_DEF = CONFIG.get('PLOT', 'FONT', fallback='DejaVu Sans')
        self.ENABLE_CONSOLE_LOGGER = CONFIG.getboolean('LOGGER', 'ENABLE_CONSOLE_LOGGER', fallback=True)
        self.ENABLE_FILE_LOGGER = CONFIG.getboolean('LOGGER', 'ENABLE_FILE_LOGGER', fallback=False)
        self.ENABLE_SUB_DIRS = CONFIG.getboolean('LOGGER', 'ENABLE_SUB_DIRS', fallback=True)
        self.LOG_PATH_USER = CONFIG.get('LOGGER', 'LOG_PATH', fallback=LOG_PATH_DEFAULT)

        # Override ENABLE_FILE_LOGGER if env variable
        if os.getenv("BUILDING_DOC") is not None and os.getenv("BUILDING_DOC").lower() == "true":
            self.BUILDING_DOC = True
            self.ENABLE_FILE_LOGGER = False
        else:
            self.BUILDING_DOC = False
        if os.getenv("FILE_LOGGER") is not None and os.getenv("FILE_LOGGER").lower() == "false":
            self.ENABLE_FILE_LOGGER = False

        self.LOG_FILE = None
        if self.ENABLE_FILE_LOGGER:
            self.LOG_FILE = 'logs_' + datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S") + '.txt'
            if self.LOG_PATH_USER == LOG_PATH_DEFAULT:
                self.LOG_PATH = os.path.abspath(LOG_PATH_DEFAULT)
            else:
                self.LOG_PATH = os.path.abspath(self.LOG_PATH_USER)

            if not os.path.exists(self.LOG_PATH):
                os.makedirs(self.LOG_PATH)

        self.SQLITE_FILE_USER = SQLITE_FILE
        if not os.path.isfile(SQLITE_FILE):
            parent_dir = os.path.dirname(SQLITE_FILE)
            # try to find a cassisYYYYMMDD.db file in the parent directory :
            if os.path.isdir(parent_dir):
                try:
                    db_list = [f for f in os.listdir(parent_dir)
                               if (f.endswith('.db') and f.startswith('cassis'))]
                    db_list.sort()
                    SQLITE_FILE_NEW = os.path.join(parent_dir, db_list[-1])
                    SQLITE_FILE = SQLITE_FILE_NEW
                    # update config :
                    CONFIG['DATABASE']['SQLITE_FILE'] = SQLITE_FILE
                except IndexError:
                    if self.BUILDING_DOC:
                        pass
                    else:
                        raise FileNotFoundError(f'No file of the form cassisYYYYMMDD.db found in {parent_dir}.')
            else:
                if self.BUILDING_DOC:
                    pass
                else:
                    raise FileNotFoundError(f'{SQLITE_FILE} not found.')
        self.SQLITE_FILE = SQLITE_FILE
        self.CONFIG = CONFIG


SETTINGS = Settings()
