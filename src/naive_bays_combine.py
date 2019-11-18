#!/usr/bin/env python3

import gc
import numpy
import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

import datasetHelper
import DecisionTree
import generate_movieRatings

def trainModel():
    global GAUS_MODEL, BERNL_MODEL, REFIT_MODEL

    # read the training dataset
    train_data = pd.read_csv(os.path.join(dataFolder, 'train_movies_ratings.csv'))

	# seperate the independent and target variable on training data
    train_bernl = train_data.drop(columns=['rating', 'userId', 'movieId', 'year'], axis=1)
    train_gaus = train_data[['userId', 'movieId', 'year']]
    train_class = train_data['rating']

	# fit the two separate models with the training data
    BERNL_MODEL.fit(train_bernl, train_class)
    GAUS_MODEL.fit(train_gaus, train_class)

    # predict the class assignment probabilities on the test dataset with two models
    predict_bernl = BERNL_MODEL.predict_proba(train_bernl)
    predict_gaus = GAUS_MODEL.predict_proba(train_gaus)

    # create a new dataset with the two probabilistic prediction as two new features
    train_refit = numpy.hstack((numpy.delete(predict_bernl, 1, 1), numpy.delete(predict_gaus, 1, 1)))
    
    # fit the refit model with the new dataset
    REFIT_MODEL.fit(pd.DataFrame(train_refit, columns=['Gaussian_proba', 'Bernoulli_proba']), train_class)


def predictValidation():
	# read the train and test dataset
    validation_data = pd.read_csv(os.path.join(dataFolder, 'val_movies_ratings.csv'))

	# seperate the independent and target variable on testing data
    validation_bernl = validation_data.drop(columns=['rating', 'userId', 'movieId', 'year'], axis=1)
    validation_gaus = validation_data[['userId', 'movieId', 'year']]
    validation_class = validation_data['rating']

	# predict the class assignment probabilities on the test dataset with two models
    predict_bernl = BERNL_MODEL.predict_proba(validation_bernl)
    print('Target on validation data using bernulli model', predict_bernl)

    predict_gaus = GAUS_MODEL.predict_proba(validation_gaus)
    print('Target on validation data using gaussian model', predict_gaus)

    # create a new dataset with the two probabilistic prediction as two new features
    validation_refit = numpy.hstack((numpy.delete(predict_bernl, 1, 1), numpy.delete(predict_gaus, 1, 1)))

    # predict target on the new dataset using the refit model
    predict_refit = REFIT_MODEL.predict(validation_refit)
    print('Target on validation data using refit model', predict_refit)

    # accuracy score on validation dataset
    accuracy_validation = accuracy_score(validation_class, predict_refit)
    print('Accuracy score on validation dataset : ', accuracy_validation)


GAUS_MODEL = GaussianNB()
REFIT_MODEL = GaussianNB()
BERNL_MODEL = BernoulliNB()

dataFolder = datasetHelper.getDataset()

DecisionTree.ensureMovieYearGenresFile(dataFolder, 'movies-year-genres.csv')

generate_movieRatings.ensureMergedCsv(dataFolder)

# shape of the dataset
#print('Shape of training data :',train_data.shape)
#print('Shape of testing data :',test_data.shape)

# Now, we need to predict the missing target variable in the test data
# target variable - Survived

print('Start fitting model.')
trainModel()
predictValidation()

# print('Running on test data...')
# test_x = pd.read_csv(os.path.join(dataFolder, 'test_movies_ratings.csv'))
# predict_test = model.predict(test_x)
# #
# # predict_test = numpy.array([1, 1, 0, 0, 0])

# predict_test = predict_test[:, None]
# # print(predict_test)
# indexes = numpy.arange(len(predict_test))[:, None]

# numpy.savetxt(os.path.join(dataFolder, 'submit.csv'), numpy.hstack((indexes, predict_test)),
# 			  delimiter=',', fmt='%d', header='Id,rating', comments='')
