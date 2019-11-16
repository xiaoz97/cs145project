import pandas as pd

a = pd.read_csv("data/val_ratings_binary.csv")
b = pd.read_csv("data/movies-year-genres.csv")
c = pd.read_csv("data/train_ratings_binary.csv")

merged = a.merge(b, on='movieId')
merged.to_csv("data/val_movies_ratings.csv", index=False)
merged2 = c.merge(b, on='movieId')
merged2.to_csv("data/train_movies_ratings.csv", index=False)