import bitstring
import math
import numpy as np

from sklearn import tree

import ClassifierBase


class Classifier(ClassifierBase.ClassifierBase):

	def __init__(self, ALL_GENRES, ALL_TAG_IDS, userIds):
		self.ALL_GENRES = ALL_GENRES
		self.ALL_TAG_IDS = ALL_TAG_IDS
		self.userIds = userIds
		self.tagBitsCount = math.ceil(len(self.ALL_TAG_IDS) / 32.0)

	def trainClassifier(self, cursor, userId, clf):
		trainingData = self.getTrainingData(cursor, userId)
		if len(trainingData) == 0:
			raise Exception('User {0} does not appear in training set.'.format(userId))
		y = trainingData[:, 0]
		X = trainingData[:, 1:]
		return clf.fit(X, y)

	def predictTest(self, cursor, userId, clf):
		testingData = self.getTestData(cursor, userId)
		predictY = clf.predict(testingData[:, 1:])

		toDB = predictY[:, None]

		toDB = np.insert(toDB, 1, userId, axis=1)
		toDB = np.insert(toDB, 2, testingData[:, 0], axis=1)
		cursor.executemany('update TestRatings set predict=? where userId=? and movieId=?', toDB.tolist())

	def classifyForUser(self, con, userId):
		cur = con.cursor()

		clf = tree.DecisionTreeClassifier(random_state=10)
		clf = self.trainClassifier(cur, userId, clf)
		validationData = self.getValidationData(cur, userId)
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
