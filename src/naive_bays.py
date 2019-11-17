#!/usr/bin/env python3

import gc
import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import datasetHelper
import DecisionTree
import generate_movieRatings


def trainModel():
	global train_x, train_y, model
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

DecisionTree.ensureMovieYearGenresFile(dataFolder, 'movies-year-genres.csv')

if os.path.isfile(os.path.join(dataFolder, 'train_movies_ratings.csv')) == False or \
		os.path.isfile(os.path.join(dataFolder, 'val_movies_ratings.csv')) == False:
	generate_movieRatings.mergeCsv(dataFolder)

# shape of the dataset
#print('Shape of training data :',train_data.shape)
#print('Shape of testing data :',test_data.shape)

# Now, we need to predict the missing target variable in the test data
# target variable - Survived

print('Start fitting model.')
model = trainModel()
predictValidation(model)

