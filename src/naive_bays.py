#!/usr/bin/env python3

import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import datasetHelper
import DecisionTree
import generate_movieRatings


dataFolder = datasetHelper.getDataset()

DecisionTree.ensureMovieYearGenresFile(dataFolder, 'movies-year-genres.csv')

if os.path.isfile(os.path.join(dataFolder, 'train_movies_ratings.csv')) == False or \
		os.path.isfile(os.path.join(dataFolder, 'val_movies_ratings.csv')) == False:
	generate_movieRatings.mergeCsv(dataFolder)


print('Reading data.')

# read the train and test dataset
train_data = pd.read_csv(os.path.join(dataFolder, 'train_movies_ratings.csv'))
test_data = pd.read_csv(os.path.join(dataFolder, 'val_movies_ratings.csv'))

# shape of the dataset
#print('Shape of training data :',train_data.shape)
#print('Shape of testing data :',test_data.shape)

# Now, we need to predict the missing target variable in the test data
# target variable - Survived

# seperate the independent and target variable on training data
train_x = train_data.drop(columns=['rating'],axis=1)
train_y = train_data['rating']

# seperate the independent and target variable on testing data
test_x = test_data.drop(columns=['rating'],axis=1)
test_y = test_data['rating']

model = GaussianNB()

print('Start fitting model.')

# fit the model with the training data
model.fit(train_x,train_y)

# predict the target on the train dataset
predict_train = model.predict(train_x)
print('Target on train data',predict_train) 

# Accuray Score on train dataset
accuracy_train = accuracy_score(train_y,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = model.predict(test_x)
print('Target on test data',predict_test) 

# Accuracy Score on test dataset
accuracy_test = accuracy_score(test_y,predict_test)
print('accuracy_score on test dataset : ', accuracy_test)
