import os
import configparser


module_dir = os.path.dirname(__file__)
USER_CONFIG = os.path.join(module_dir, 'settings.ini')
DEFAULT_CONFIG = os.path.join(module_dir, 'settings_defaults.ini')
CONFIG = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation(),
                                   inline_comment_prefixes=('#',))

CONFIG_FILE = USER_CONFIG
if not os.path.isfile(CONFIG_FILE):
    if os.path.isfile(DEFAULT_CONFIG):
        # print(f'{user_config} not found, using {default_config}\n')
        CONFIG_FILE = DEFAULT_CONFIG
    else:
        raise FileNotFoundError('No settings file found.')

CONFIG.read(CONFIG_FILE)


if os.getenv("FILE_LOGGER") is not None and os.getenv("FILE_LOGGER").lower() == "false":
    ENABLE_FILE_LOGGER = False
else:
    ENABLE_FILE_LOGGER = CONFIG.getboolean('LOGGER', 'ENABLE_FILE_LOGGER', fallback=True)
LOG_PATH = CONFIG.get('LOGGER', 'LOG_PATH', fallback='logs')
if ENABLE_FILE_LOGGER and not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
