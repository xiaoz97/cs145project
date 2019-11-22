import pandas as pd
import numpy as np
import mlxtend.frequent_patterns

from mlxtend.preprocessing import TransactionEncoder

import datasetHelper

# 文件夹地址
data_folder = datasetHelper.getDataset()
# 文件地址
ratings_filename = data_folder + '/train_ratings_binary.csv'
# 给u.data 加标题行
all_ratings = pd.read_csv(ratings_filename)

# 新建一个数据集，只包括用户喜欢某部电影的数据行
favorable_ratings = all_ratings[all_ratings["rating"] == 1]

dataset = [v.values for _, v in favorable_ratings.groupby("userId")["movieId"]]
# In dataset, each row is a transaction, ie. movies liked by the same user.
# The length of each row is different.

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
encodedDataset = pd.DataFrame(te_ary, columns=te.columns_)
# Now, each row has columns of all movies.
# If a user liked one movie, the corresponding column is true.

assert len(dataset[0]) == encodedDataset.iloc[0].sum()

frequentPatterns = mlxtend.frequent_patterns.apriori(encodedDataset, min_support=0.1)

print(frequentPatterns)

np.save("fp1.npy", frequentPatterns)
