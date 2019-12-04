import os
import pandas as pd
import numpy as np
import mlxtend.frequent_patterns
import time

from mlxtend.preprocessing import TransactionEncoder

import datasetHelper

from Program import getDefaultPrediction

def getFrequentPatterns(data_folder):
	# 文件夹地址
	if os.path.isfile(os.path.join(data_folder, "freq.npy")):
		frequentPatterns = np.load(os.path.join(data_folder, "freq.npy"), allow_pickle=True)
		print("load")
		return frequentPatterns

	# 文件地址
	ratings_filename = data_folder + '/train_ratings_binary.csv'

	startTime = time.time()
	# 给u.data 加标题行
	all_ratings = pd.read_csv(ratings_filename)
	print('Loading data takes {0} seconds.'.format(time.time() - startTime))

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
	frequentPatterns = mlxtend.frequent_patterns.fpgrowth(encodedDataset, min_support=0.05)
	frequentPatterns['length'] = frequentPatterns['itemsets'].apply(lambda x: len(x))
	freq = frequentPatterns[frequentPatterns['length'] >= 2]
	np.save(os.path.join(data_folder, "freq.npy"), freq)
	return freq

def getSortedConfidence(favorable_reviews_by_users):
	global DATA_FOLDER
	# load sorted_confidence; If file doesn't exist, then generate a new one.
	if os.path.isfile(os.path.join(DATA_FOLDER, "sorted_confidence.npy")) is False:
		freq = getFrequentPatterns(DATA_FOLDER)
		print("Frequent patterns loaded")

		from collections import defaultdict

		candidate_rules = []
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

		startTime = time.time()
		lastP = -1
		total = len(favorable_reviews_by_users)

		# Traverse the whole dataset
		for i, pair in enumerate(favorable_reviews_by_users.items()):
			userId, favorableMovies = pair
			# 遍历每条关联规则
			for candidate_rule in candidate_rules:
				# print(candidate_rule)
				premise, conclusion = candidate_rule
				# 判断用户是否喜欢前提中的电影
				if set(premise).issubset(favorableMovies):
					# 如果前提符合，看一下用户是否喜欢结论中的电影
					if conclusion in favorableMovies:
						correct_counts[candidate_rule] += 1
					else:
						incorrect_counts[candidate_rule] += 1

			p = (i+1) * 100 / total
			if int(p) > lastP:
				usedTime = time.time() - startTime
				print('User {0} is done. Progress is {1}%. Used time is {2}s. Remaining time is {3}s.'.
					  format(userId, int(p), int(usedTime), int(usedTime / p * 100 - usedTime)))
				lastP = p

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
		np.save(os.path.join(DATA_FOLDER, "sorted_confidence.npy"), rule_confidence)

	sorted_confidence = np.load(os.path.join(DATA_FOLDER, "sorted_confidence.npy"), allow_pickle=True)
	return sorted_confidence


def main():
	global DATA_FOLDER

	DATA_FOLDER = datasetHelper.getDataset()
	# Get favorite movies for each user
	all_ratings = pd.read_csv(DATA_FOLDER + '/train_ratings_binary.csv')

	favorable_ratings = all_ratings[all_ratings["rating"] == 1]

	favorable_reviews_by_users = dict((k, frozenset(v.values)) for k, v in favorable_ratings.groupby("userId")["movieId"])

	sorted_confidence = getSortedConfidence(favorable_reviews_by_users)

	# Validation Part
	# 读入validation set

	validate_filename = DATA_FOLDER + '/val_ratings_binary.csv'

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
			cnt = getDefaultPrediction()
		# Compare with real rating
		if cnt == row["rating"]:
			correct += 1
			# print accuracy
			if (correct % 1000 == 0):
				print(correct / total)


DATA_FOLDER = None

if __name__ == "__main__":
	main()
