import pandas as pd
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import os
import datasetHelper
import Program
import generate_movieRatings

# Construct input data
dataFolder = datasetHelper.getDataset()
Program.ensureMovieYearGenresFile(dataFolder, 'movies-year-genres.csv')
generate_movieRatings.ensureMergedCsv(dataFolder)

# seperate the independent and target variable on training data
train_data = pd.read_csv(os.path.join(dataFolder, 'train_movies_ratings.csv'))
train_x = train_data.drop(columns=['rating'], axis=1)
train_y = train_data['rating']

# seperate the independent and target variable on testing data
validation_data = pd.read_csv(os.path.join(dataFolder, 'val_movies_ratings.csv'))
validation_x = validation_data.drop(columns=['rating'], axis=1)
validation_y = validation_data['rating']

# define the keras model
model = Sequential()
model.add(Dense(23, input_dim=23, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(train_x, train_y, epochs=5, batch_size=50)

# evaluate the keras model
_, accuracy = model.evaluate(validation_x, validation_y)
print('Accuracy: %.2f' % (accuracy*100))