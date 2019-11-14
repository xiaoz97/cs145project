
import pyodbc


def doesTableExist(TABLE_NAME, cur):
	cur.execute("SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'dbo' AND  TABLE_NAME = ?", (TABLE_NAME,))
	return cur.fetchone() is not None


def delimiteDBIdentifier(identifier: str) -> str:
	return '[' + identifier + ']'


def getConnection():
	return pyodbc.connect('DRIVER={SQL Server};SERVER=DESKTOP-FHIBPCT;DATABASE=data_mining;Trusted_Connection=yes')