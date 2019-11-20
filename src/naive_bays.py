#!/usr/bin/env python3

import numpy
import os
import pandas as pd
import time

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import datasetHelper
import DecisionTree
import generate_movieRatings


def trainModel(train_data):
	# seperate the independent and target variable on training data
	train_x = train_data.drop(columns=['rating'], axis=1)
	train_y = train_data['rating']
	model = GaussianNB()
	# fit the model with the training data
	model.fit(train_x, train_y)

	# predict the target on the train dataset
	# predict_train = model.predict(train_x)
	# print('Target on train data', predict_train)

	# Accuray Score on train dataset
	# accuracy_train = accuracy_score(train_y, predict_train)
	# print('accuracy_score on train dataset : ', accuracy_train)
	return model


def predictValidation(model, validation_data):
	# read the train and test dataset
	# seperate the independent and target variable on testing data
	validation_x = validation_data.drop(columns=['rating'], axis=1)
	validation_y = validation_data['rating'].values

	# predict the target on the test dataset
	predict_test = model.predict(validation_x)

	return [predict_test, validation_y]


# print('Target on validation data', predict_test)
#
# # Accuracy Score on test dataset
# accuracy_test = accuracy_score(validation_y, predict_test)
# print('accuracy_score on validation dataset : ', accuracy_test)


dataFolder = datasetHelper.getDataset()

DecisionTree.ensureMovieYearGenresFile(dataFolder, 'movies-year-genres.csv')

generate_movieRatings.ensureMergedCsv(dataFolder)

# shape of the dataset
#print('Shape of training data :',train_data.shape)
#print('Shape of testing data :',test_data.shape)

# Now, we need to predict the missing target variable in the test data
# target variable - Survived


train_data = pd.read_csv(os.path.join(dataFolder, 'train_movies_ratings.csv')).drop(['movieId'], axis='columns')
validation_data = pd.read_csv(os.path.join(dataFolder, 'val_movies_ratings.csv')).drop(['movieId'], axis='columns')
test_x = pd.read_csv(os.path.join(dataFolder, 'test_movies_ratings.csv')).drop(['movieId'], axis='columns')

userIds = pd.unique(pd.concat([train_data['userId'], validation_data['userId']]))

validation_predict_y = []
validation_y = []
predict_test = []

print('Start fitting model.')
startTime = time.time()

lastP = 0
total = len(userIds)
for i in range(len(userIds)):
	model = trainModel(train_data[train_data['userId'] == userIds[i]].drop(['userId'], axis='columns'))

	[a, b] = predictValidation(model, validation_data[validation_data['userId'] == userIds[i]].drop(['userId'], axis='columns'))

	[validation_predict_y, validation_y] = numpy.hstack([[validation_predict_y, validation_y], [a, b]])
	predict_test = numpy.hstack([predict_test, model.predict(test_x[test_x['userId'] == userIds[i]].drop(['userId'], axis='columns'))])

	p = i * 100 // total
	if p > lastP:
		usedTime = time.time() - startTime
		print('User {0} is done. Progress is {1}%. Used time is {2}s, Remaining time is {3:d}s'.format(userIds[i], p, int(usedTime), int(usedTime / p * 100 - usedTime)))
		lastP = p

print(accuracy_score(validation_y, validation_predict_y))

predict_test = predict_test[:, None]
# print(predict_test)
indexes = numpy.arange(len(predict_test))[:, None]

numpy.savetxt(os.path.join(dataFolder, 'submit.csv'), numpy.hstack((indexes, predict_test)),
			  delimiter=',', fmt='%d', header='Id,rating', comments='')
