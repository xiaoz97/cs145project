
import mysql.connector


def doesTableExist(TABLE_NAME, cur):
	cur.execute("SELECT 1 FROM information_schema.tables WHERE table_schema = 'data_mining' AND table_name = %s", (TABLE_NAME,))
	return cur.fetchone() is not None


def delimiteDBIdentifier(identifier: str) -> str:
	return '`' + identifier + '`'


def getConnection(database):
	return mysql.connector.connect(database)  # we may use ":memory:", but it may be too large, about 1.5GB
