import bitstring
import math
import numpy as np

from abc import abstractmethod

from Program import flatNestList


class ClassifierBase(object):

	def __init__(self, ALL_GENRES, ALL_TAG_IDS, yearSize: int = 1):
		self.ALL_GENRES = ALL_GENRES
		self.ALL_TAG_IDS = ALL_TAG_IDS
		self.tagBitsCount = math.ceil(len(self.ALL_TAG_IDS) / 32.0)

		self.yearSize = yearSize
		"""
		the interval for discretizing movie year
		"""

	def getTrainingData(self, cursor, userId):
		cursor.execute('''
SELECT Ratings.rating, MovieYearGenres.year/{1}, genreBits, {0} FROM Ratings
join MovieYearGenres on Ratings.movieId=MovieYearGenres.id
join MovieTags on Ratings.movieId=MovieTags.id
where Ratings.userId=?'''.format(','.join(['tagBits' + str(i) for i in range(self.tagBitsCount)]), self.yearSize), (userId,))

		trainingData = [list(row[0:2]) +
						list(bitstring.Bits(int=row[2], length=len(self.ALL_GENRES))) +
						flatNestList([list(bitstring.Bits(int=b, length=32)) for b in row[3:]])
						for row in cursor.fetchall()]

		trainingData = np.array(trainingData, dtype='int32')
		return trainingData

	def getValidationData(self, cursor, userId):
		cursor.execute('''
SELECT ValidationRatings.movieId, MovieYearGenres.year/{1}, genreBits, {0} FROM ValidationRatings
join MovieYearGenres on ValidationRatings.movieId=MovieYearGenres.id
join MovieTags on ValidationRatings.movieId=MovieTags.id
where ValidationRatings.userId=?'''.format(','.join(['tagBits' + str(i) for i in range(self.tagBitsCount)]), self.yearSize), (userId,))
		validationData = [list(row[0:2]) +
						  list(bitstring.Bits(int=row[2], length=len(self.ALL_GENRES))) +
						  flatNestList([list(bitstring.Bits(int=b, length=32)) for b in row[3:]])
						  for row in cursor.fetchall()]

		validationData = np.array(validationData, dtype='int32')
		return validationData

	def getTestData(self, cursor, userId):
		cursor.execute('''
SELECT TestRatings.movieId, MovieYearGenres.year/{1}, genreBits, {0} FROM TestRatings
join MovieYearGenres on TestRatings.movieId=MovieYearGenres.id
join MovieTags on TestRatings.movieId=MovieTags.id
where TestRatings.userId=?'''.format(','.join(['tagBits' + str(i) for i in range(self.tagBitsCount)]), self.yearSize), (userId,))

		testingData = [list(row[0:2]) +
					   list(bitstring.Bits(int=row[2], length=len(self.ALL_GENRES))) +
					   flatNestList([list(bitstring.Bits(int=b, length=32)) for b in row[3:]])
					   for row in cursor.fetchall()]

		testingData = np.array(testingData, dtype='int32')
		return testingData
