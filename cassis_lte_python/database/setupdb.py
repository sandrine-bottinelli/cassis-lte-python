import os
import sqlite3
from cassis_lte_python.utils.settings import SQLITE_FILE
from cassis_lte_python.utils.logger import CassisLogger


LOGGER = CassisLogger.create('setupdb')

conn = None

if os.path.isfile(SQLITE_FILE):
    LOGGER.info(f"Using database : {SQLITE_FILE}")
    conn = sqlite3.connect(SQLITE_FILE)
else:
    raise FileNotFoundError(f'{SQLITE_FILE} not found.')

# DATABASE_SQL = conn.cursor()
DATABASE_SQL = conn
