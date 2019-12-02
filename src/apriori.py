import os
import pandas as pd
import numpy as np
import mlxtend.frequent_patterns

from mlxtend.preprocessing import TransactionEncoder

import datasetHelper


def getFrequentPatterns():
	# 文件夹地址
	data_folder = datasetHelper.getDataset()
	if os.path.isfile(os.path.join(data_folder, "freq.npy")):
		frequentPatterns = np.load(os.path.join(data_folder, "freq.npy"), allow_pickle=True)
		print("load")
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

	print("dataset shape is " + str(encodedDataset.shape))
	print("Run apriori with min_support=0.05")
	frequentPatterns = mlxtend.frequent_patterns.apriori(encodedDataset, min_support=0.05)
	frequentPatterns['length'] = frequentPatterns['itemsets'].apply(lambda x: len(x))
	freq = frequentPatterns[frequentPatterns['length'] >= 2]
	np.save(os.path.join(data_folder, "freq.npy"), freq)
	return freq


data_folder = datasetHelper.getDataset()
freq = getFrequentPatterns()
# print(freq)
ratings_filename = data_folder + '/train_ratings_binary.csv'
# Get favorite movies for each user
all_ratings = pd.read_csv(ratings_filename)

favorable_ratings = all_ratings[all_ratings["rating"] == 1]

candidate_rules = []

favorable_reviews_by_users = dict(
	(k, frozenset(v.values)) for k, v in favorable_ratings.groupby("userId")["movieId"])

# load sorted_confidence; If file doesn't exist, then generate a new one.
if os.path.isfile(os.path.join(data_folder, "sorted_confidence.npy")):
	sorted_confidence = np.load(os.path.join(data_folder, "sorted_confidence.npy"), allow_pickle=True)
	print("load sort")
	print(sorted_confidence)
else:
	from collections import defaultdict

	# generate potential rule candidates
	for rules in freq:
		for conclusion in rules[1]:
			# 项集中的其他电影作为前提
			temp = list(rules[1])
			temp.remove(conclusion)
			temp = tuple(temp)
			# 用前提和结论组成备选规则
			candidate_rules.append((temp, conclusion))

	correct_counts = defaultdict(int)
	incorrect_counts = defaultdict(int)
	# Traverse the whole dataset
	for user, reviews in favorable_reviews_by_users.items():
		# 遍历每条关联规则
		for candidate_rule in candidate_rules:
			candidate_rule = tuple(candidate_rule)
			# print(candidate_rule)
			premise, conclusion = candidate_rule
			# 判断用户是否喜欢前提中的电影
			if set(premise).issubset(reviews):
				# 如果前提符合，看一下用户是否喜欢结论中的电影
				if conclusion in reviews:
					correct_counts[candidate_rule] += 1
				else:
					incorrect_counts[candidate_rule] += 1
	# Decide minimum confidence; Here we set as 0.4

	min_confidence = 0.4
	rule_confidence = []
	print("start measuring")

	# Calculate the confidence of each rule candidates
	for candidate_rule in candidate_rules:
		if (correct_counts[tuple(candidate_rule)] + incorrect_counts[tuple(candidate_rule)] != 0):
			confidence = correct_counts[tuple(candidate_rule)] / (
				float(correct_counts[tuple(candidate_rule)] + incorrect_counts[tuple(candidate_rule)]))
			print(confidence)

			if confidence > min_confidence:
				rule_confidence.append((confidence, candidate_rule))
				print(confidence, candidate_rule, )

	print(rule_confidence)
	np.save(os.path.join(data_folder, "sorted_confidence.npy"), rule_confidence)
	sorted_confidence = np.load(os.path.join(data_folder, "sorted_confidence.npy"), allow_pickle=True)

# Validation Part
# 读入validation set

validate_filename = data_folder + '/val_ratings_binary.csv'

validate = pd.read_csv(validate_filename)
correct = 0
total = 0
print("start")
print(sorted_confidence)

for index, row in validate.iterrows():
	# cnt标记prediction total为总数
	cnt = 0
	total += 1
	for confidence, rules in sorted_confidence:

		if rules[1] == row["movieId"]:
			if (len(list(rules[0]) + list(favorable_reviews_by_users[row["userId"]])) - len(
					list(set(list(rules[0]) + list(favorable_reviews_by_users[row["userId"]]))))) / (
					len(rules[0])) >= 0.5:
				cnt = 1
				if cnt == row["rating"]:
					print("yes")
				else:
					print("no")
				break
	# If the new movie is not in our rules, then randomly generate 0 or 1;
	if (cnt == 0):
		cnt = np.random.randint(0, 1)
	# Compare with real rating
	if cnt == row["rating"]:
		correct += 1
		# print accuracy
		if (correct % 1000 == 0):
			print(correct / total)
