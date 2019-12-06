import bitstring
import math
import numpy as np
import sklearn.model_selection
import sys

from sklearn import tree
from sklearn.metrics import roc_auc_score
from traceback import format_exc

from Program import flatNestList


class Classifier(object):

	def __init__(self, ALL_GENRES, ALL_TAG_IDS, userIds):
		self.ALL_GENRES = ALL_GENRES
		self.ALL_TAG_IDS = ALL_TAG_IDS
		self.userIds = userIds
		self.tagBitsCount = math.ceil(len(self.ALL_TAG_IDS) / 32.0)


	def runKFold(self, cursor, userId):
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

		# StratifiedKFold用法类似Kfold，但是他确保训练集，测试集中各类别样本的比例与原始数据集中相同。
		# 就不会训练集中很多喜欢，导致测试集很多不喜欢。

		y = trainingData[:, 0]
		X = trainingData[:, 1:]
			
		# Make sure each fold has at least 5 samples, and we want at most 5 folds.
		n_splits = min(len(X) // 5, 10)
		bestClf = None
		bestScore = -1
		try:
			# The ratings of user 105730 have certain patterns, that the first half of ratings are 1, and the rest are 0.
			for train_index, test_index in sklearn.model_selection.KFold(n_splits, True, 1206).split(X, y):
				X_train, X_test = X[train_index], X[test_index]
				y_train, y_test = y[train_index], y[test_index]

				clf = tree.DecisionTreeClassifier(random_state=10)
				clf.fit(X_train, y_train)

				# predictY = clf.predict(X_test)
				score= clf.score(X_test, y_test)

				# Ideally we should use roc_auc_score, but there are cases where
				# only one class present in y_true. ROC AUC score is not defined in that case.
				#score = roc_auc_score(y_test, predictY)
				if score > bestScore:
					bestScore = score
					bestClf = clf
		except Exception as ex:
			# print(format_exc(), file=sys.stderr)
			print(str(ex) + ' Not use StratifiedKFold for user {0}.'.format(userId))

			bestClf = tree.DecisionTreeClassifier(random_state=10)
			bestClf.fit(X, y)
		return bestClf
		
		
	def __predictTest(self, cursor, userId, clf):
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

		clf = self.runKFold(cur, userId)

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
		self.__predictTest(cur, userId, clf)
		con.commit()

