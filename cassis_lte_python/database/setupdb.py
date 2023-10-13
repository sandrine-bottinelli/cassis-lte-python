import os
import sqlite3
from cassis_lte_python.utils.settings import SQLITE_FILE

conn = None

if os.path.isfile(SQLITE_FILE):
    print(f"Using database : {SQLITE_FILE}")
    conn = sqlite3.connect(SQLITE_FILE)
else:
    raise FileNotFoundError(f'{SQLITE_FILE} not found.')

# DATABASE_SQL = conn.cursor()
DATABASE_SQL = conn
