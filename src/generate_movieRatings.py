import os
import pandas as pd


def ensureMergedCsv(dataFolder):
	b = pd.read_csv(os.path.join(dataFolder, "movies-year-genres.csv"))

	if os.path.isfile(os.path.join(dataFolder, 'train_movies_ratings.csv')) == False:
		c = pd.read_csv(os.path.join(dataFolder, "train_ratings_binary.csv"))
		merged = c.merge(b, left_on='movieId', right_on='id')
		merged.to_csv(os.path.join(dataFolder, "train_movies_ratings.csv"), index=False)

	if os.path.isfile(os.path.join(dataFolder, 'val_movies_ratings.csv')) == False:
		a = pd.read_csv(os.path.join(dataFolder, "val_ratings_binary.csv"))
		merged = a.merge(b, left_on='movieId', right_on='id')
		merged.to_csv(os.path.join(dataFolder, "val_movies_ratings.csv"), index=False)

	if os.path.isfile(os.path.join(dataFolder, 'test_movies_ratings.csv')) == False:
		test = pd.read_csv(os.path.join(dataFolder, "test_ratings.csv"))
		# Some movies don't have year. They are eliminated from movies-year-genres.csv.
		# Besides, we are not sure if there are new movies in the test set.
		merged = test.merge(b, left_on='movieId', right_on='id', how='left')
		merged = merged.fillna(0)
		merged.to_csv(os.path.join(dataFolder, "test_movies_ratings.csv"), index=False)
