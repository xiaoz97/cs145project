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
frequentPatterns['length'] = frequentPatterns['itemsets'].apply(lambda x: len(x))
# 筛选出长度大于1的frequent pattern
freq=frequentPatterns[frequentPatterns['length']>=2]
print(freq)

np.save("freq.npy",freq)
freq = np.load("freq.npy")

freq = np.load("freq.npy")
candidate_rules = []
# 统计每个用户各喜欢哪些电影，按照UserID进行分组，并遍历每个用户看过的每一部电影
favorable_reviews_by_users = dict((k, frozenset(v.values)) for k,v in favorable_ratings.groupby("userId")["movieId"])
# 创建一个数据框，以便了解每部电影的影迷数量
# 导入字典
from collections import defaultdict
# 定义一个发现新的频繁项集的函数，参数为（每个用户喜欢哪些电影字典，上一个频繁项集，最小支持度）

for rules in freq:
    print(rules)
    for conclusion in rules[1]:
        # 项集中的其他电影作为前提
        temp=list(rules[1])

        temp.remove(conclusion)

        temp=tuple(temp)
        # 用前提和结论组成备选规则
        candidate_rules.append((temp, conclusion))

# print(candidate_rules[:5])

print(candidate_rules)
# 创建两个字典, 用来存储规则正例，和返例的次数
correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
# 遍历每个用户喜欢的电影数据
for user, reviews in favorable_reviews_by_users.items():
    # 遍历每条关联规则
    for candidate_rule in candidate_rules:
        candidate_rule=tuple(candidate_rule)

        premise,conclusion = candidate_rule
        # 判断用户是否喜欢前提中的电影
        if set(premise).issubset(reviews):
            # 如果前提符合，看一下用户是否喜欢结论中的电影
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1

min_confidence = 0.6
rule_confidence=[]
print("yes")
# 用正例次数除以前提条件出现的总次数，计算每条规则的置信度
for candidate_rule in candidate_rules:
    print(correct_counts[tuple(candidate_rule)])
    print(incorrect_counts[tuple(candidate_rule)])
    if(correct_counts[tuple(candidate_rule)]+incorrect_counts[tuple(candidate_rule)]!=0):
        confidence = correct_counts[tuple(candidate_rule)] / (
            float(correct_counts[tuple(candidate_rule)] + incorrect_counts[tuple(candidate_rule)]))
        print(confidence)
        if confidence > min_confidence:
            rule_confidence.append((confidence, candidate_rule))
# 最小支持置信度
# 筛选出执行度大于0.9的
#print(len(rule_confidence))
print(rule_confidence)

np.save('sorted_confidence.npy', rule_confidence)
# 读入之前获取的sorted rules
sorted_confidence=np.load("sorted_confidence.npy")

#d读入validation set
validate_folder="E:/pycharm/cs145project/data"

validate_filename = validate_folder+'/val_ratings_binary.csv'

validate = pd.read_csv(validate_filename)
#print(validate)
#读入原文件并存储进入favorable_reviews_by_users
#original_filename = validate_folder+"/train_ratings_binary.csv"
original_filename = validate_folder+"/train_ratings_binary.csv"

original = pd.read_csv(original_filename)

favorable_reviews_by_users = dict((k, frozenset(v.values)) for k,v in original.groupby("userId")["movieId"])

correct=0
total=0
print("start")

for index, row in validate.iterrows():
    #cnt标记prediction total为总数
    cnt=0

    total+=1
    for confidence, rules in sorted_confidence:
        #遍历已获取的规则，如果conclusion与当前需判断的电影相同，则判断premise是否在用户已经看过的电影中

        if rules[1]==row["movieId"]:
            #print("same")
            #print(rules[0])
            #print(favorable_reviews_by_users[row["userId"]])
            # a=float((len(list(rules[0])+list(favorable_reviews_by_users[row["userId"]]))-len(list(set(list(rules[0])+list(favorable_reviews_by_users[row["userId"]])))))/len(rules[0]))
            # print("a:"+str(a))

            if (len(list(rules[0]) + list(favorable_reviews_by_users[row["userId"]])) - len(list(set(list(rules[0]) + list(favorable_reviews_by_users[row["userId"]])))))/(len(rules[0]))>=0.5:
                cnt=1
                if cnt==row["rating"]:
                    print("yes")
                else:
                    print("no")
                break
    #如果不符合任何conclusion以及premise，暂时产生0，1随机数
    if(cnt==0):
        cnt=np.random.randint(0, 1)
    #与真实rating比对
    if cnt==row["rating"]:
        correct+=1
        #每出现100个正确的输出此时准确率
        if(correct%1000==0):
            print(correct/total)