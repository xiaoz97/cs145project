
import sqlite3


def doesTableExist(TABLE_NAME, cur):
	cur.execute("SELECT 1 FROM sqlite_master WHERE name =? and type='table'", (TABLE_NAME,))
	return cur.fetchone() is not None


def delimiteDBIdentifier(identifier: str) -> str:
	return '[' + identifier + ']'


def getConnection(database):
	return sqlite3.connect(database)  # we may use ":memory:", but it may be too large, about 1.5GB
