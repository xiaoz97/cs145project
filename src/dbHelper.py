


def doesTableExist(TABLE_NAME, cur):
	cur.execute("SELECT 1 FROM sqlite_master WHERE name =? and type='table'", (TABLE_NAME,))
	return cur.fetchone() is not None


def delimiteDBIdentifier(identifier: str) -> str:
	return '[' + identifier + ']'
