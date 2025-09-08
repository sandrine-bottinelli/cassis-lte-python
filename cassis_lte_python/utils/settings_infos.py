from cassis_lte_python.utils.settings import SETTINGS
from cassis_lte_python.utils.logger import CassisLogger
import os
import datetime
import shutil


LOGGER = CassisLogger.create("Settings")
CONFIG = SETTINGS.CONFIG
SQLITE_FILE = SETTINGS.SQLITE_FILE
SQLITE_FILE_USER = SETTINGS.SQLITE_FILE_USER
USER_CONFIG = SETTINGS.USER_CONFIG
CONFIG_FILE = SETTINGS.CONFIG_FILE
ENABLE_FILE_LOGGER = SETTINGS.ENABLE_FILE_LOGGER
ENABLE_SUB_DIRS = SETTINGS.ENABLE_SUB_DIRS


if 'defaults' in CONFIG_FILE:
    LOGGER.warning(f'{USER_CONFIG} not found, using {CONFIG_FILE}.')
# else:
#     LOGGER.info(f"Using {CONFIG_FILE}.")


def print_settings_all():
    message = ["Settings are :"]
    for section in CONFIG.sections():
        for key, val in dict(CONFIG.items(section)).items():
            unit = ""
            if "size" in key:
                unit = "arcsec"
            elif "vlsr" in key or "fwhm" in key:
                unit = "km/s"
            message.append(f"{key.upper()} = {val} {unit}")
    LOGGER.info("\n    ".join(message))


def print_settings_database():
    if os.path.isfile(SQLITE_FILE):
        if not os.path.isfile(SQLITE_FILE_USER):
            if 'YYYYMMDD' in SQLITE_FILE_USER:  # generic file name
                LOGGER.info(f'No specific sqlite file provided, using sqlite file {SQLITE_FILE}.')
            else:
                LOGGER.warning(f'{SQLITE_FILE_USER} not found, using {SQLITE_FILE} instead.')
        else:
            LOGGER.info(f"Using database : {SQLITE_FILE}")


def check_log_dir(log_parent_dir: str):
    if SETTINGS.ENABLE_FILE_LOGGER:
        if os.path.dirname(SETTINGS.LOG_PATH_USER) == '':  # if LOG_PATH in user's settings is just a name
            if not os.path.dirname(SETTINGS.LOG_PATH) == log_parent_dir:
                # script not in cwd, move logs to script's directory
                shutil.move(SETTINGS.LOG_PATH, log_parent_dir)
                SETTINGS.LOG_PATH = log_parent_dir

    #     if SETTINGS.ENABLE_SUB_DIRS:
    #         SETTINGS.LOG_PATH = os.path.join(SETTINGS.LOG_PATH,
    #                                          'logs_' + datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S"))

        LOGGER.info(f"Logs will be written to: {SETTINGS.LOG_PATH}.")
