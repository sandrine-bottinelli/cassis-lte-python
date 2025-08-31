import logging
import os
from cassis_lte_python import ENABLE_FILE_LOGGER, LOG_PATH


class CustomFormatter(logging.Formatter):
    """Logging colored formatter,
    from https://alexandra-zaharia.github.io/posts/make-your-own-custom-color-formatter-with-python-logging/,
    adapted from https://stackoverflow.com/a/56944256/3638629"""

    grey = '\x1b[38;5;7m'
    # blue = '\x1b[38;5;39m'
    blue = '\x1b[38;5;27m'
    yellow = '\x1b[38;5;226m'
    orange = '\x1b[38;5;214m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self, fmt, *args, **kwargs):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.orange + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


class CassisLogger:
    @staticmethod
    def create(name: str):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
        formatter = CustomFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if ENABLE_FILE_LOGGER:
            file_handler = logging.FileHandler(os.path.join(LOG_PATH, f'{name}.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger