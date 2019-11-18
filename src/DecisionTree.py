#!/usr/bin/env python3

import os
import csv
import dbHelper
import re
import sys
from types import SimpleNamespace
import bisect
from sklearn import tree
import numpy as np

import matplotlib.pyplot as plt
import time
import zipfile

import datasetHelper


def ensureMovieYearGenresFile(dataFolder, movieYearGenresFileName):
	if os.path.isfile(os.path.join(dataFolder, movieYearGenresFileName)):
		return

	movies = {}
	global ALL_GENRES
	with open(dataFolder + "/movies.csv", encoding='utf-8') as moviesFile:  # will automatically close the file when exit the with block
		reader = csv.reader(moviesFile)
		next(reader)  # skip the column headers
		for row in reader:
			id = row[0]
			title = row[1].strip()

			m = re.search('\((\d+)\)$', title)
			if m is None:
				print("Movie title doesn't have year. Id=" + id + ", title=" + title, file=sys.stderr)
				continue

			if row[2] == '(no genres listed)':
				continue

			year = int(m.group(1))
			genres = row[2].split('|')

			if (any([bisect.bisect_left(ALL_GENRES, g) < 0 for g in genres])):
				raise Exception('One of {0} is not listed in allGenres.'.format(genres))

			# print('year is %d' % year)
			# print(genres)

			item = SimpleNamespace()
			item.year = year
			item.genres = genres
			movies[id] = item

	# print(', '.join(allGenres))

	with open(dataFolder + "/" + movieYearGenresFileName, encoding='utf-8', mode='w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['id', 'year'] + ALL_GENRES)
		for id in movies:
			item = movies[id]
			map = [0] * len(ALL_GENRES)
			for i in range(len(item.genres)):
				index = bisect.bisect_left(ALL_GENRES, item.genres[i])
				map[index] = 1

			writer.writerow([id, item.year] + map)

def ensureMovieYearGenresTable(movieYearGenresFileName, dbConnection):
	cur = dbConnection.cursor()
	TABLE_NAME = 'MovieYearGenres'
	if dbHelper.doesTableExist(TABLE_NAME, cur):
		return

	# Syntax 'create table if not exists' exists, but we don't know if we need to insert rows.
	with open(os.path.join(DATA_FOLDER, movieYearGenresFileName), encoding='utf-8') as movieYearGenresFile:
		csvReader = csv.reader(movieYearGenresFile)
		headers = next(csvReader)

		headers[0] = headers[0] + ' INTEGER NOT NULL PRIMARY KEY'
		headers = headers[0:1] + [dbHelper.delimiteDBIdentifier(h) + ' INTEGER' for h in headers[1:]]
		cur.execute("CREATE TABLE {1} ({0})".format(', '.join(headers), TABLE_NAME))
		# table names can't be the target of parameter substitution
		# https://stackoverflow.com/a/3247553/746461

		to_db = [row for row in csvReader]

		cur.executemany("INSERT INTO {1} VALUES ({0});".format(','.join(['?'] * len(headers)), TABLE_NAME), to_db)
		dbConnection.commit()

	cur.execute('select * from {0} where id=131162'.format(TABLE_NAME))
	print(cur.fetchone())


def ensureRatingsTable(fileName, dbConnection):
	cur = dbConnection.cursor()
	TABLE_NAME = 'Ratings'
	if dbHelper.doesTableExist(TABLE_NAME, cur):
		return

	cur.execute("CREATE TABLE {0} (userId INTEGER NOT NULL,movieId INTEGER NOT NULL,rating INTEGER NOT NULL, PRIMARY KEY(userId,movieId))".format(TABLE_NAME))
	with open(os.path.join(DATA_FOLDER, fileName), encoding='utf-8') as f:
		csvReader = csv.reader(f)
		next(csvReader)

		to_db = [row for row in csvReader]

		cur.executemany("INSERT INTO {0} VALUES (?,?,?);".format(TABLE_NAME), to_db)
		dbConnection.commit()

	cur.execute('select * from {0} where userId=1 and movieId=151'.format(TABLE_NAME))
	print(cur.fetchone())


def ensureValidationRatingsTable(fileName, dbConnection):
	cur = dbConnection.cursor()
	TABLE_NAME = 'ValidationRatings'
	if dbHelper.doesTableExist(TABLE_NAME, cur):
		return

	cur.execute("CREATE TABLE {0} (userId INTEGER NOT NULL,movieId INTEGER NOT NULL,rating INTEGER NOT NULL, predict INTEGER, PRIMARY KEY(userId,movieId))".format(TABLE_NAME))
	with open(os.path.join(DATA_FOLDER, fileName), encoding='utf-8') as f:
		csvReader = csv.reader(f)
		next(csvReader)

		to_db = [row for row in csvReader]

		cur.executemany("INSERT INTO {0} VALUES (?,?,?,null);".format(TABLE_NAME), to_db)
		dbConnection.commit()

	cur.execute('select * from {0} where userId=1 and movieId=1653'.format(TABLE_NAME))
	print(cur.fetchone())


def ensureTestRatingTable(fileName, dbConnection):
	cur = dbConnection.cursor()
	TABLE_NAME = 'TestRatings'
	if dbHelper.doesTableExist(TABLE_NAME, cur):
		return

	cur.execute("CREATE TABLE {0} (userId INTEGER NOT NULL,movieId INTEGER NOT NULL, predict integer, PRIMARY KEY(userId,movieId))".format(TABLE_NAME))
	with open(os.path.join(DATA_FOLDER, fileName), encoding='utf-8') as f:
		csvReader = csv.reader(f)
		next(csvReader)

		to_db = [row for row in csvReader]

		cur.executemany("INSERT INTO {0} VALUES (?,?, null);".format(TABLE_NAME), to_db)
		dbConnection.commit()


def trainClassifier(cursor, userId, clf):
	cursor.execute('''
SELECT Ratings.rating, MovieYearGenres.year, {0} FROM Ratings
join MovieYearGenres on Ratings.movieId=MovieYearGenres.id
where Ratings.userId=? '''.format(','.join([dbHelper.delimiteDBIdentifier(g) for g in ALL_GENRES])), (userId,))
	trainingData = np.array(cursor.fetchall())
	if len(trainingData) == 0:
		raise Exception('User {0} does not appear in training set.'.format(userId))
	y = trainingData[:, 0]
	X = trainingData[:, 1:]
	return clf.fit(X, y)


def predictTest(cursor, userId, clf):
	cursor.execute('''
SELECT TestRatings.movieId, MovieYearGenres.year, {0} FROM TestRatings
join MovieYearGenres on TestRatings.movieId=MovieYearGenres.id
where TestRatings.userId=? '''.format(','.join([dbHelper.delimiteDBIdentifier(g) for g in ALL_GENRES])), (userId,))
	testingData = np.array(cursor.fetchall())
	predictY = clf.predict(testingData[:, 1:])

	toDB = predictY[:, None]

	toDB = np.insert(toDB, 1, userId, axis=1)
	toDB = np.insert(toDB, 2, testingData[:, 0], axis=1)
	cursor.executemany('update TestRatings set predict=? where userId=? and movieId=?', toDB.tolist())


def classifyUser(con, userId):
	cur = con.cursor()

	clf = tree.DecisionTreeClassifier()
	clf = trainClassifier(cur, userId, clf)
	cur.execute('''
SELECT ValidationRatings.movieId, MovieYearGenres.year, {0} FROM ValidationRatings
join MovieYearGenres on ValidationRatings.movieId=MovieYearGenres.id
where ValidationRatings.userId=? '''.format(','.join([dbHelper.delimiteDBIdentifier(g) for g in ALL_GENRES])), (userId,))
	validationData = np.array(cur.fetchall())
	predictY = clf.predict(validationData[:, 1:])
	toDB = predictY[:, None]
	toDB = np.insert(toDB, 1, userId, axis=1)
	toDB = np.insert(toDB, 2, validationData[:, 0], axis=1)
	cur.executemany('update ValidationRatings set predict=? where userId=? and movieId=?', toDB.tolist())
	if cur.rowcount == 0:
		raise Exception("No rows are updated.")
	# tree.plot_tree(clf)
	# plt.show()
	con.commit()
	predictTest(cur, userId, clf)
	con.commit()
	# cur.execute('select count(*) from ValidationRatings where userId=? and rating=predict', (userId,))
	# correct = cur.fetchone()[0]
	# # break
	# print('user {0}, accuracy is {1:.2f}.'.format(userId, correct / len(predictY)))  # prefer format than %.
	# print('User {0} is done.'.format(userId))


def dealWithMissingPrediction(cursor, table: str):
	cursor.execute('update {0} set predict=? where predict is null'.format(table), (1,))
	print('Fixed {0} empty prediction in table {1}.'.format(cursor.rowcount, table))


def exportTestRatings(cursor, fileName: str):
	cursor.execute('select rowid-1, predict from TestRatings order by rowid')
	data=cursor.fetchall()
	with open(os.path.join(DATA_FOLDER, fileName), 'w', newline="") as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(['Id', 'rating'])

		writer.writerows(data)


def main():
	global MAX_ROWS, ALL_GENRES, DATA_FOLDER
	try:
		i = sys.argv.index('--max-rows')
		MAX_ROWS = int(sys.argv[i + 1])
	except:
		MAX_ROWS = None

	DATA_FOLDER = datasetHelper.getDataset()

	movieYearGenresFileName = 'movies-year-genres.csv'
	ensureMovieYearGenresFile(DATA_FOLDER, movieYearGenresFileName)

	con = dbHelper.getConnection(os.path.join(DATA_FOLDER, "sqlite.db"))
	ensureMovieYearGenresTable(movieYearGenresFileName, con)
	ensureRatingsTable('train_ratings_binary.csv', con)
	ensureValidationRatingsTable('val_ratings_binary.csv', con)
	ensureTestRatingTable('test_ratings.csv', con)

	cur = con.cursor()

	# cur.execute('update ValidationRatings set predict=null')
	# cur.execute('update TestRatings set predict=null')
	# con.commit()

	cur.execute('''
SELECT userid FROM ValidationRatings
UNION
SELECT userId FROM TestRatings''')
	userIds = [row[0] for row in cur.fetchall()]

	startTime = time.time()

	lastP = 0
	total = len(userIds)
	for i in range(total):
		classifyUser(con, userIds[i])

		p = i * 100 // total
		if p > lastP:
			usedTime = time.time() - startTime
			print('User {0} is done. Progress is {1}%. Used time is {2}s, Remaining time is {3:d}s'.format(i, p, int(usedTime), int(usedTime / p * 100 - usedTime)))
			lastP = p

	dealWithMissingPrediction(cur, 'ValidationRatings')
	dealWithMissingPrediction(cur, 'TestRatings')

	print('Used time: {0}'.format(time.time() - startTime))

	bestAccuracy = 1
	try:
		with open(os.path.join(DATA_FOLDER, 'best accuracy.txt'), mode='r') as f:
			bestAccuracy = float(f.read())
	except:
		pass

	cur.execute('''select t.correct, t.total, CAST(t.correct AS float)/t.total as accuracy
from (Select 
(select count(*) from ValidationRatings where rating=predict) as correct,
(select count(*) from ValidationRatings) as total) as t''')
	row = cur.fetchone()
	print(row)
	accuracy = row[2]

	exportTestRatings(cur, 'submit.csv')
	con.close()

	print('Best accuracy is {0}. This accuracy is {1}.'.format(bestAccuracy, accuracy))
	if accuracy > bestAccuracy:
		with open(os.path.join(DATA_FOLDER, 'best accuracy.txt'), mode='w') as f:
			f.write(str(accuracy))
		if os.system('kaggle competitions submit -c uclacs145fall2019 -m "auto submission with accuracy {1}" -f "{0}"'.format(os.path.join(DATA_FOLDER, 'submit.csv'), accuracy)) != 0:
			print("Unable to submit dataset through kaggle API. Did you install the API and configure your API key properly?", file=sys.stderr)


DATA_FOLDER = os.path.normpath(os.path.dirname(os.path.realpath(__file__)) + "/../data")
ALL_GENRES = sorted(['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
MAX_ROWS = 0

if __name__ == "__main__":
	main()
