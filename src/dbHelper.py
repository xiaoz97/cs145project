import mysql.connector


def doesTableExist(TABLE_NAME, cur):
	cur.execute("SELECT 1 FROM information_schema.tables WHERE table_schema = 'data_mining' AND table_name = %s", (TABLE_NAME,))
	return cur.fetchone() is not None


def delimiteDBIdentifier(identifier: str) -> str:
	return '`' + identifier + '`'


def getConnection():
	return mysql.connector.connect(user='root', password='root', host='127.0.0.1', database='data_mining')
