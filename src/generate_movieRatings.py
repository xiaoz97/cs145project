import os
import pandas as pd


def mergeCsv(dataFolder):
	a = pd.read_csv(os.path.join(dataFolder, "val_ratings_binary.csv"))
	b = pd.read_csv(os.path.join(dataFolder, "movies-year-genres.csv"))
	c = pd.read_csv(os.path.join(dataFolder, "train_ratings_binary.csv"))

	merged = a.merge(b, left_on='movieId', right_on='id')
	merged.to_csv(os.path.join(dataFolder, "val_movies_ratings.csv"), index=False)
	merged2 = c.merge(b, left_on='movieId', right_on='id')
	merged2.to_csv(os.path.join(dataFolder, "train_movies_ratings.csv"), index=False)
