__all__ = ['SQLITE_FILE']

import os
import sqlite3
from cassis_lte_python.utils.settings import SETTINGS
from cassis_lte_python.utils.logger import CassisLogger


SQLITE_FILE = SETTINGS.SQLITE_FILE
LOGGER = CassisLogger.create('setupdb')

conn = None

if os.path.isfile(SQLITE_FILE):
    # LOGGER.info(f"Using database : {SQLITE_FILE}")
    conn = sqlite3.connect(SQLITE_FILE)
else:
    pass  # raise done in settings.py ; pass here to avoid FileNotFound when building doc in gitlab CI
    # raise FileNotFoundError(f'{SQLITE_FILE} not found.')

# DATABASE_SQL = conn.cursor()
DATABASE_SQL = conn
