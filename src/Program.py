#!/usr/bin/env python3

import bitstring
import os
import csv
import math
import re
import sqlite3
import sys
import bisect
import time

from multiprocessing import Pool
from multiprocessing import cpu_count
from subprocess import Popen, PIPE
from types import SimpleNamespace

import datasetHelper
import dbHelper


def ensureMovieTagsFile(dbConnection, fileName: str, allTagIds, relevanceThreshold: float):
	global DATA_FOLDER

	cur = dbConnection.cursor()

	# cur.execute('drop table if exists MovieTags')
	# dbConnection.commit()
	if os.path.isfile(os.path.join(DATA_FOLDER, fileName)):
		return

	tagBitsCount = math.ceil(len(allTagIds) / 32.0)

	tagIdDict = {val: idx for idx, val in enumerate(allTagIds)}

	movies = {}
	with open(DATA_FOLDER + "/movies.csv", encoding='utf-8') as moviesFile:
		reader = csv.reader(moviesFile)
		next(reader)  # skip the column headers
		for row in reader:
			id = row[0]

			tagIds = list(row[0] for row in cur.execute('select tagId from GenomeScore where movieId=? and relevance>=?', (id, relevanceThreshold)))

			item = SimpleNamespace()
			item.tags = [None] * tagBitsCount

			for t in tagIds:
				# look up the index of the tagId
				index = tagIdDict[t]
				binIndex = math.floor(index / 32.0)

				if item.tags[binIndex] is None:
					item.tags[binIndex] = bitstring.BitArray(length=32)

				item.tags[binIndex].set(1, index % 32)

			movies[id] = item

	with open(os.path.join(DATA_FOLDER, fileName), encoding='utf-8', mode='w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['id'] + ['tagBits' + str(i) for i in range(tagBitsCount)])
		for id in movies:
			item = movies[id]
			csvRow = [id]
			for i in range(tagBitsCount):
				if item.tags[i] is None:
					csvRow.append(0)
				else:
					csvRow.append(item.tags[i].int)

			writer.writerow(csvRow)


def ensureMovieTagsTable(fileName, dbConnection):
	cur = dbConnection.cursor()
	TABLE_NAME = 'MovieTags'
	if dbHelper.doesTableExist(TABLE_NAME, cur):
		return

	# Syntax 'create table if not exists' exists, but we don't know if we need to insert rows.
	with open(os.path.join(DATA_FOLDER, fileName), encoding='utf-8') as movieYearGenresFile:
		csvReader = csv.reader(movieYearGenresFile)
		# skip header
		headers = next(csvReader)

		cur.execute("CREATE TABLE {0} (id INTEGER NOT NULL PRIMARY KEY, {1})".format(TABLE_NAME, ','.join([h + ' integer not null' for h in headers[1:]])))
		# table names can't be the target of parameter substitution
		# https://stackoverflow.com/a/3247553/746461

		to_db = list(csvReader)

		cur.executemany("INSERT INTO {0} VALUES (?,{1})".format(TABLE_NAME, ','.join(['?'] * (len(headers) - 1))), to_db)
		dbConnection.commit()

	cur.execute('select * from {0} where id=2'.format(TABLE_NAME))
	print(cur.fetchone())


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

	genresDict = {val: idx for idx, val in enumerate(ALL_GENRES)}

	with open(dataFolder + "/" + movieYearGenresFileName, encoding='utf-8', mode='w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['id', 'year', 'genreBits'])
		for id in movies:
			item = movies[id]
			map = bitstring.BitArray(length=len(ALL_GENRES))
			for i in range(len(item.genres)):
				map[genresDict[item.genres[i]]] = 1

			writer.writerow([id, item.year, map.int])


def ensureMovieYearGenresTable(movieYearGenresFileName, dbConnection):
	cur = dbConnection.cursor()
	TABLE_NAME = 'MovieYearGenres'
	if dbHelper.doesTableExist(TABLE_NAME, cur):
		return

	# Syntax 'create table if not exists' exists, but we don't know if we need to insert rows.
	with open(os.path.join(DATA_FOLDER, movieYearGenresFileName), encoding='utf-8') as movieYearGenresFile:
		csvReader = csv.reader(movieYearGenresFile)
		# skip header
		next(csvReader)

		cur.execute("CREATE TABLE {0} (id INTEGER NOT NULL PRIMARY KEY, year INTEGER NOT NULL, genreBits INTEGER NOT NULL)".format(TABLE_NAME))
		# table names can't be the target of parameter substitution
		# https://stackoverflow.com/a/3247553/746461

		to_db = list(csvReader)

		cur.executemany("INSERT INTO {0} VALUES (?,?,?)".format(TABLE_NAME), to_db)
		dbConnection.commit()

	cur.execute('select * from {0} where id=131162'.format(TABLE_NAME))
	print(cur.fetchone())


def ensureGenomeScoresTable(fileName, dbConnection):
	cur = dbConnection.cursor()
	TABLE_NAME = 'GenomeScore'
	if dbHelper.doesTableExist(TABLE_NAME, cur):
		return

	cur.execute("CREATE TABLE {0} (movieId INTEGER NOT NULL, tagId INTEGER NOT NULL, relevance REAL NOT NULL, PRIMARY KEY(movieId, tagId))".format(TABLE_NAME))
	with open(os.path.join(DATA_FOLDER, fileName), encoding='utf-8') as f:
		csvReader = csv.reader(f)
		next(csvReader)

		to_db = [row for row in csvReader]

		cur.executemany("INSERT INTO {0} VALUES (?,?,?);".format(TABLE_NAME), to_db)
		dbConnection.commit()

	cur.execute('CREATE INDEX tagId ON {0} (tagId ASC)'.format(TABLE_NAME))

	cur.execute('select * from {0} where movieId=1 and tagId=1'.format(TABLE_NAME))
	print('GenomeScore table is created.')
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
	print(TABLE_NAME)
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
	print(TABLE_NAME)
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


def ensureMoviePopularityTable(dbConnection):
	cur = dbConnection.cursor()
	TABLE_NAME = 'MoviePopularity'
	if dbHelper.doesTableExist(TABLE_NAME, cur):
		return

	cur.execute("CREATE TABLE {0} (movieId INTEGER NOT NULL, popularity integer not null, PRIMARY KEY(movieId))".format(TABLE_NAME))
	cur.execute('''
insert into {0}
select movieId, cast(sum(rating) as real) / count(*)
from Ratings
GROUP by movieId'''.format(TABLE_NAME))
	dbConnection.commit()
	cur.execute('select * from {0} where movieId=1653'.format(TABLE_NAME))
	print(TABLE_NAME)
	print(cur.fetchone())

def dealWithMissingPrediction(cursor, table: str):
	global FIRST_USERS
	if FIRST_USERS is None:
		cursor.execute('update {0} set predict=? where predict is null'.format(table), (getDefaultPrediction(),))
	else:
		cursor.execute('update {0} set predict=? where predict is null and userId<=?'.format(table), (getDefaultPrediction(), FIRST_USERS))
	print('Fixed {0} empty prediction in table {1}.'.format(cursor.rowcount, table))


def exportTestRatings(cursor, fileName: str):
	cursor.execute('select rowid-1, predict from TestRatings order by rowid')
	data = cursor.fetchall()
	with open(os.path.join(DATA_FOLDER, fileName), 'w', newline="") as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(['Id', 'rating'])

		writer.writerows(data)


def classifyForUsersInThread(threadId, classifier):
	assert threadId > 0

	# On Python 3.7.5 on Windows, DATA_FOLDER does not have the updated value,
	# so I have to set it again.
	dataFolder = datasetHelper.getDataset()
	with dbHelper.getConnection(os.path.join(dataFolder, "sqlite.db")) as con:
		startTime = time.time()
		lastP = 0
		total = len(classifier.userIds)

		for i in range(total):
			try:
				classifier.classifyForUser(con, classifier.userIds[i])
			except sqlite3.OperationalError as ex:
				print('Error on User {0}: '.format(classifier.userIds[i])+str(ex), file=sys.stderr)
				exit(1)
			except Exception as ex:
				print('Error on User {0}: '.format(classifier.userIds[i])+str(ex), file=sys.stderr)

			p = i * 100 // total
			if p > lastP:
				usedTime = time.time() - startTime
				print('[Thread {4}] User {0} is done. Progress is {1}%. Used time is {2}s. Remaining time is {3:d}s.'.
					  format(classifier.userIds[i], p, int(usedTime), int(usedTime / p * 100 - usedTime), threadId), flush=True)
				lastP = p


def chunkify(l, n):
	"""Yield n number of sequential chunks from l."""
	d, r = divmod(len(l), n)
	for i in range(n):
		si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
		yield l[si:si + (d + 1 if i < r else d)]


def main():
	global MAX_ROWS, ALL_GENRES, DATA_FOLDER, ALL_TAG_IDS, FIRST_USERS
	try:
		i = sys.argv.index('--max-rows')
		MAX_ROWS = int(sys.argv[i + 1])
	except:
		MAX_ROWS = None

	DATA_FOLDER = datasetHelper.getDataset()

	movieYearGenresFileName = 'movies-year-genres.csv'
	ensureMovieYearGenresFile(DATA_FOLDER, movieYearGenresFileName)

	con = dbHelper.getConnection(os.path.join(DATA_FOLDER, "sqlite.db"))
	ensureGenomeScoresTable('genome-scores.csv', con)
	ensureMovieYearGenresTable(movieYearGenresFileName, con)

	cur = con.cursor()
	ALL_TAG_IDS = [row[0] for row in cur.execute('select DISTINCT tagId from GenomeScore order by tagId')]

	try:
		i = sys.argv.index('--relevance')
		relevance = float(sys.argv[i + 1])
	except:
		relevance = 0.46

	movieTagsFileName = '{0:.2f}-'.format(relevance) + 'movie-tags.csv'
	ensureMovieTagsFile(con, movieTagsFileName, ALL_TAG_IDS, relevance)
	ensureMovieTagsTable(movieTagsFileName, con)

	ensureRatingsTable('train_ratings_binary.csv', con)
	ensureValidationRatingsTable('val_ratings_binary.csv', con)
	ensureTestRatingTable('test_ratings.csv', con)

	ensureMoviePopularityTable(con)

	cur.execute('update ValidationRatings set predict=null')
	cur.execute('update TestRatings set predict=null')
	con.commit()

	cur.execute('''
SELECT userid FROM ValidationRatings
UNION
SELECT userId FROM TestRatings''')
	userIds = [row[0] for row in cur.fetchall()]
	con.close()

	try:
		i = sys.argv.index('--first-users')
		FIRST_USERS = int(sys.argv[i + 1])
		userIds = list(filter(lambda x: x <= FIRST_USERS, userIds))
	except:
		pass

	startTime = time.time()

	try:
		i = sys.argv.index('--parallel')
		if sys.argv[i + 1] == 'auto':
			parallel = cpu_count()
		else:
			parallel = int(sys.argv[i + 1])
	except:
		parallel = cpu_count()

	try:
		i = sys.argv.index('--model')
		m = sys.argv[i + 1]
	except:
		m = 'RandomForest'

	if m == 'DecisionTree':
		from DecisionTree import Classifier
	elif m == 'RandomForest':
		from RandomForest import Classifier
	else:
		print('Unknown model ' + m, file=sys.stderr)
		sys.exit(1)
		
	print('Classification using {0} starts with {1} processes.'.format(m, parallel))
	
	if parallel == 1:
		classifyForUsersInThread(1, Classifier(ALL_GENRES, ALL_TAG_IDS, userIds))
	else:
		chunkedUserIds = list(chunkify(userIds, parallel))
		pool = Pool(len(chunkedUserIds))

		for i in range(len(chunkedUserIds)):
			pool.apply_async(classifyForUsersInThread, args=(i + 1, Classifier(ALL_GENRES, ALL_TAG_IDS, chunkedUserIds[i])))

		pool.close()
		pool.join()

	con = dbHelper.getConnection(os.path.join(DATA_FOLDER, "sqlite.db"))
	cur = con.cursor()
	dealWithMissingPrediction(cur, 'ValidationRatings')
	dealWithMissingPrediction(cur, 'TestRatings')

	print('Used time: {0}'.format(time.time() - startTime))

	bestAccuracy = 1
	try:
		with open(os.path.join(DATA_FOLDER, 'best accuracy.txt'), mode='r') as f:
			bestAccuracy = float(f.read())
	except:
		pass

	if FIRST_USERS is None:
		cur.execute('''select t.correct, t.total, CAST(t.correct AS float)/t.total as accuracy
from (Select 
(select count(*) from ValidationRatings where rating=predict) as correct,
(select count(*) from ValidationRatings) as total) as t''')
	else:
		cur.execute('''select t.correct, t.total, CAST(t.correct AS float)/t.total as accuracy
from (Select 
(select count(*) from ValidationRatings where rating=predict and userId<={0}) as correct,
(select count(*) from ValidationRatings where userId<={0}) as total) as t'''.format(FIRST_USERS))

	row = cur.fetchone()
	print(row)
	accuracy = row[2]

	if FIRST_USERS is None:
		exportTestRatings(cur, 'submit.csv')
	con.close()

	print('Best accuracy is {0}. This accuracy is {1}.'.format(bestAccuracy, accuracy))
	if FIRST_USERS is None and (bestAccuracy == 1 or accuracy > bestAccuracy):
		with open(os.path.join(DATA_FOLDER, 'best accuracy.txt'), mode='w') as f:
			f.write(str(accuracy))

		gitStatus = ''
		process = Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=PIPE, cwd=os.path.dirname(os.path.realpath(__file__)))
		(output, err) = process.communicate()
		exit_code = process.wait()
		if exit_code == 0:
			gitBranch = output.decode("utf-8").strip()
			gitStatus = 'Git [{0}]'.format(gitBranch)

		process = Popen(['git', 'rev-parse', '--short=8', 'HEAD'], stdout=PIPE, cwd=os.path.dirname(os.path.realpath(__file__)))
		(output, err) = process.communicate()
		exit_code = process.wait()
		if exit_code == 0:
			gitSha = output.decode("utf-8").strip()
			gitStatus += '({0})'.format(gitSha)

		if gitStatus != '':
			gitStatus += ' '

		if os.system('kaggle competitions submit -c uclacs145fall2019 -m "{3}{2} submitted with accuracy {1:.4f}" -f "{0}"'.format(os.path.join(DATA_FOLDER, 'submit.csv'), accuracy, m, gitStatus)) != 0:
			print("Unable to submit dataset through kaggle API. Did you install the API and configure your API key properly?", file=sys.stderr)


def flatNestList(a):
	return [item for sublist in a for item in sublist]


def getDefaultPrediction():
	return 1

DATA_FOLDER = os.path.normpath(os.path.dirname(os.path.realpath(__file__)) + "/../data")
ALL_GENRES = sorted(['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
MAX_ROWS = 0
FIRST_USERS = None

if __name__ == "__main__":
	main()
