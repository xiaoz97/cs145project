import os
import pandas as pd
import numpy as np
import mlxtend.frequent_patterns

from mlxtend.preprocessing import TransactionEncoder

import datasetHelper


def getFrequentPatterns():
	# 文件夹地址
	data_folder = datasetHelper.getDataset()
	if os.path.isfile(os.path.join(data_folder, "fp1.npy")):
		frequentPatterns = np.load("fp1.npy", allow_pickle=True)
		return frequentPatterns

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
	frequentPatterns = frequentPatterns[[len(s) > 1 for s in frequentPatterns['itemsets']]]
	np.save(os.path.join(data_folder, "fp1.npy"), frequentPatterns)
	return frequentPatterns


frequentPatterns = getFrequentPatterns()
print(frequentPatterns)

# 读入之前获取的sorted rules
sorted_confidence = np.load("fp1.npy", allow_pickle=True)
# d读入validation set
validate_folder = datasetHelper.getDataset()

validate_filename = validate_folder + '/val_ratings_binary.csv'

validate = pd.read_csv(validate_filename, dtype='int32')
# print(validate)
# 读入原文件并存储进入favorable_reviews_by_users
# original_filename = validate_folder+"/train_ratings_binary.csv"
original_filename = validate_folder + "/train_ratings_binary.csv"

original = pd.read_csv(original_filename, dtype='int32')

favorable_reviews_by_users = dict((k, frozenset(v.values)) for k, v in original.groupby("userId")["movieId"])

correct = 0
total = 0

print("start")

for index, row in validate.iterrows():
	# cnt标记prediction total为总数
	predict = 0
	user_likes = favorable_reviews_by_users[row["userId"]]
	up = 0
	down = 0

	total += 1
	for confidence, rules in sorted_confidence:
		# 遍历已获取的规则，如果conclusion与当前需判断的电影相同，则判断premise是否在用户已经看过的电影中

		# 电影A属于电影集合{A,B,C,D}。一个用户看了{B,C,D}，那他也会喜欢A。
		if row["movieId"] in (rules):
			# Does the user like other movies?
			a = list(rules)
			a.remove(row["movieId"])
			# print(a)
			if set(a) < set(user_likes):
				predict = 1
				if (predict == row["rating"]):
					print("right")
				if (predict != row["rating"]):
					print("wrong")
				break

	# 如果不符合任何conclusion以及premise，暂时产生0，1随机数
	if (predict == 0):
		predict = 1
	# 与真实rating比对
	if predict == row["rating"]:
		correct += 1
		# 每出现100个正确的输出此时准确率
		if (correct % 1000 == 0):
			print(correct / total)
