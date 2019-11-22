#!/usr/bin/env python3

import gc
import numpy
import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import datasetHelper
import Program
import generate_movieRatings


def trainModel():
	train_data = pd.read_csv(os.path.join(dataFolder, 'train_movies_ratings.csv'))
	# seperate the independent and target variable on training data
	train_x = train_data.drop(columns=['rating'], axis=1)
	train_y = train_data['rating']
	model = GaussianNB()
	# fit the model with the training data
	model.fit(train_x, train_y)

	# predict the target on the train dataset
	predict_train = model.predict(train_x)
	print('Target on train data', predict_train)

	# Accuray Score on train dataset
	accuracy_train = accuracy_score(train_y, predict_train)
	print('accuracy_score on train dataset : ', accuracy_train)
	return model


def predictValidation(model):
	# read the train and test dataset
	validation_data = pd.read_csv(os.path.join(dataFolder, 'val_movies_ratings.csv'))
	# seperate the independent and target variable on testing data
	validation_x = validation_data.drop(columns=['rating'], axis=1)
	validation_y = validation_data['rating']

	# predict the target on the test dataset
	predict_test = model.predict(validation_x)
	print('Target on validation data', predict_test)

	# Accuracy Score on test dataset
	accuracy_test = accuracy_score(validation_y, predict_test)
	print('accuracy_score on validation dataset : ', accuracy_test)


dataFolder = datasetHelper.getDataset()

Program.ensureMovieYearGenresFile(dataFolder, 'movies-year-genres.csv')

generate_movieRatings.ensureMergedCsv(dataFolder)

# shape of the dataset
#print('Shape of training data :',train_data.shape)
#print('Shape of testing data :',test_data.shape)

# Now, we need to predict the missing target variable in the test data
# target variable - Survived

print('Start fitting model.')
model = trainModel()
predictValidation(model)

print('Running on test data...')
test_x = pd.read_csv(os.path.join(dataFolder, 'test_movies_ratings.csv'))
predict_test = model.predict(test_x)
#
# predict_test = numpy.array([1, 1, 0, 0, 0])

predict_test = predict_test[:, None]
# print(predict_test)
indexes = numpy.arange(len(predict_test))[:, None]

numpy.savetxt(os.path.join(dataFolder, 'submit.csv'), numpy.hstack((indexes, predict_test)),
			  delimiter=',', fmt='%d', header='Id,rating', comments='')
