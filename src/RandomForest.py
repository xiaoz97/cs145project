import bitstring
import math
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from Program import flatNestList


class Classifier(object):

	def __init__(self, ALL_GENRES, ALL_TAG_IDS, userIds):
		self.ALL_GENRES = ALL_GENRES
		self.ALL_TAG_IDS = ALL_TAG_IDS
		self.userIds=userIds
		self.tagBitsCount = math.ceil(len(self.ALL_TAG_IDS) / 32.0)



	def trainClassifier(self, cursor, userId, clf):
		cursor.execute('''
	SELECT Ratings.rating, MovieYearGenres.year, genreBits, {0} FROM Ratings
	join MovieYearGenres on Ratings.movieId=MovieYearGenres.id
	join MovieTags on Ratings.movieId=MovieTags.id
	where Ratings.userId=?'''.format(','.join(['tagBits' + str(i) for i in range(self.tagBitsCount)])), (userId,))

		trainingData = [list(row[0:2]) +
						list(bitstring.Bits(int=row[2], length=len(self.ALL_GENRES))) +
						flatNestList([list(bitstring.Bits(int=b, length=32)) for b in row[3:]])
						for row in cursor.fetchall()]

		trainingData = np.array(trainingData, dtype='int32')
		if len(trainingData) == 0:
			raise Exception('User {0} does not appear in training set.'.format(userId))
		y = trainingData[:, 0]
		X = trainingData[:, 1:]
		return clf.fit(X, y)


	def predictTest(self, cursor, userId, clf):
		cursor.execute('''
SELECT TestRatings.movieId, MovieYearGenres.year, genreBits, {0} FROM TestRatings
join MovieYearGenres on TestRatings.movieId=MovieYearGenres.id
join MovieTags on TestRatings.movieId=MovieTags.id
where TestRatings.userId=?'''.format(','.join(['tagBits' + str(i) for i in range(self.tagBitsCount)])), (userId,))

		testingData = [list(row[0:2]) +
				   		list(bitstring.Bits(int=row[2], length=len(self.ALL_GENRES))) +
				   		flatNestList([list(bitstring.Bits(int=b, length=32)) for b in row[3:]])
				   		for row in cursor.fetchall()]

		testingData = np.array(testingData, dtype='int32')
		predictY = clf.predict(testingData[:, 1:])

		toDB = predictY[:, None]

		toDB = np.insert(toDB, 1, userId, axis=1)
		toDB = np.insert(toDB, 2, testingData[:, 0], axis=1)
		cursor.executemany('update TestRatings set predict=? where userId=? and movieId=?', toDB.tolist())


	def classifyForUser(self, con, userId):
		cur = con.cursor()

		clf = RandomForestClassifier(n_estimators=100, max_depth=10)
		clf = self.trainClassifier(cur, userId, clf)
		cur.execute('''
SELECT ValidationRatings.movieId, MovieYearGenres.year, genreBits, {0} FROM ValidationRatings
join MovieYearGenres on ValidationRatings.movieId=MovieYearGenres.id
join MovieTags on ValidationRatings.movieId=MovieTags.id
where ValidationRatings.userId=?'''.format(','.join(['tagBits' + str(i) for i in range(self.tagBitsCount)])), (userId,))
		validationData = [list(row[0:2]) +
						  list(bitstring.Bits(int=row[2], length=len(self.ALL_GENRES))) +
						  flatNestList([list(bitstring.Bits(int=b, length=32)) for b in row[3:]])
						  for row in cur.fetchall()]

		validationData = np.array(validationData, dtype='int32')
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
		self.predictTest(cur, userId, clf)
		con.commit()