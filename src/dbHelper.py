
import pyodbc


def doesTableExist(TABLE_NAME, cur):
	cur.execute("SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'dbo' AND  TABLE_NAME = ?", (TABLE_NAME,))
	return cur.fetchone() is not None


def delimiteDBIdentifier(identifier: str) -> str:
	return '[' + identifier + ']'


def getConnection():
	return pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=data_mining;UID=sa;PWD=yourStrong(!)Password;')