import os,sys
import pandas as pd
# 文件夹地址
data_folder = '/Users/chengkai/Desktop/file/learn/project/用亲和性分析方法推荐电影/dataset/ml-100k'
# 文件地址
ratings_filename = data_folder+'/u.data'
# 给u.data 加标题行
all_ratings = pd.read_csv(ratings_filename, delimiter="\t", header=None, names= ["UserID","MovieID","Rating","Datetime"])
# 改变时间显示
all_ratings["Datetime"] = pd.to_datetime(all_ratings['Datetime'], unit='s')
# 添加一个特征，即评分大于3的定义为喜欢，小于3的定义为不喜欢
all_ratings["Favorable"] = all_ratings["Rating"]>3
# 提取前200个数据作为训练集
ratings = all_ratings[all_ratings['UserID'].isin(range(200))]
# 新建一个数据集，只包括用户喜欢某部电影的数据行
favorable_ratings = ratings[ratings["Favorable"]]
# 统计每个用户各喜欢哪些电影，按照UserID进行分组，并遍历每个用户看过的每一部电影
favorable_reviews_by_users = dict((k, frozenset(v.values)) for k,v in favorable_ratings.groupby("UserID")["MovieID"])
# 创建一个数据框，以便了解每部电影的影迷数量
num_favorable_by_movie = ratings[["MovieID", "Favorable"]].groupby("MovieID").sum()
# 初始化一个字典
frequent_itemsets = {}
# 设置最小支持度
min_support = 50
# 为每一部电影生成只包含它自己的项目，检测它是否够频繁
# numnum_favorable_by_movie.iterrows() 对数据进行遍历
frequent_itemsets[1] = dict((frozenset((movie_id,)),row["Favorable"]) for movie_id, row in num_favorable_by_movie.iterrows() if row["Favorable"] > min_support )
# 导入字典
from collections import defaultdict
# 定义一个发现新的频繁项集的函数，参数为（每个用户喜欢哪些电影字典，上一个频繁项集，最小支持度）
def find_frequent_itemsets(favorable_reviews_by_users, k_1_itemsets, min_support):
    counts = defaultdict(int)
    # 遍历每个用户以及他喜欢的电影
    for user, reviews in favorable_reviews_by_users.items():
        # 遍历上一个的项集，判断itemset是不是每个用户喜欢的电影的子集
        for itemset in k_1_itemsets:
            if itemset.issubset(reviews):
                # 遍历用户打过分却没有出现在项集里的电影，用它生成超集，更新该项集的计数
                for other_reviewed_movie in reviews - itemset:
                    current_superset = itemset | frozenset((other_reviewed_movie,))
                    #最终收集了所有项集的频率
                    counts[current_superset] +=1
    # 函数最后检测达到支持度要求的项集，看它的频繁程度够不够，并返回其中的频繁项集
    return dict([(itemset, frequency) for itemset, frequency in counts.items() if frequency >= min_support])

for k in range(2, 20):
    # 实例化函数，k表示即将发现的频繁项集的长度，用键k-1可以从frequent_itemsets字典中获取刚发现的频繁项集。
    cur_frequent_itemsets = find_frequent_itemsets(favorable_reviews_by_users, frequent_itemsets[k-1], min_support)
    # 新发现的频繁项集以长度为键，将其保存到字典中
    frequent_itemsets[k] = cur_frequent_itemsets
    # 如果在上述循环中没能找到任何新的频繁项集，就跳出循环
    if len(cur_frequent_itemsets) == 0:
        print("Did not find any frequent item")
    else:
        print("I found {} frequent of length {}".format(len(cur_frequent_itemsets), k))
        # 代码运行中输出，不写的话，代码运行结束后全部输出
        sys.stdout.flush()
# 我们对只有一个元素的项集不感兴趣，他们对生成关联规则没有用处（至少两个），所以删除
del frequent_itemsets[1]

candidate_rules = []
# 遍历不同长度的频繁项集，为每个项集生成规则
for itemset_length, itemset_counts in frequent_itemsets.items():
    for itemset in itemset_counts.keys():
        # 遍历项集中的每一部电影，把他作为结论
        for conclusion in itemset:
            # 项集中的其他电影作为前提
            premise = itemset - set((conclusion,))
            # 用前提和结论组成备选规则
            candidate_rules.append((premise, conclusion))
print(candidate_rules[:5])
# 创建两个字典, 用来存储规则正例，和返例的次数
correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
# 遍历每个用户喜欢的电影数据
for user, reviews in favorable_reviews_by_users.items():
    # 遍历每条关联规则
    for candidate_rule in candidate_rules:
        premise,conclusion = candidate_rule
        # 判断用户是否喜欢前提中的电影
        if premise.issubset(reviews):
            # 如果前提符合，看一下用户是否喜欢结论中的电影
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1
# 用正例次数除以前提条件出现的总次数，计算每条规则的置信度
rule_confidence = {candidate_rule: correct_counts[candidate_rule] / float(correct_counts[candidate_rule] + incorrect_counts[candidate_rule])for candidate_rule in candidate_rules}

# 最小支持置信度
min_confidence = 0.9
# 筛选出执行度大于0.9的
rule_confidence = {rule: confidence for rule, confidence in rule_confidence.items() if confidence > min_confidence}
#print(len(rule_confidence))

from operator import itemgetter
# 对执行度进行排序，输出置信度最高的前5条规则
sorted_confidence = sorted(rule_confidence.items(), key=itemgetter(1), reverse=True)

for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    print("Rule: If a person recommends {0} they will also recommend {1}".format(premise, conclusion))
    print(" - Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
    print("")

movie_name_filename = os.path.join(data_folder, "u.item")
movie_name_data = pd.read_csv(movie_name_filename, delimiter="|", header=None, encoding = "mac-roman")
movie_name_data.columns = ["MovieID", "Title", "Release Date", "Video Release", "IMDB", "<UNK>", "Action", "Adventure",
                           "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                           "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
def get_movie_name(movie_id):
    title_object = movie_name_data[movie_name_data["MovieID"] == movie_id]["Title"]
    title = title_object.values[0]
    return title
get_movie_name(4)
for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    premise_names = ", ".join(get_movie_name(idx) for idx in premise)
    conclusion_name = get_movie_name(conclusion)
    print("Rule: If a person recommends {0} they will also recommend {1}".format(premise_names, conclusion_name))
    print(" - Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
    print("")

# 用于测试的数据，200条以后
test_dataset = all_ratings[~all_ratings['UserID'].isin(range(200))]
# 提取用户喜欢的电影
test_favorable = test_dataset[test_dataset["Favorable"]]
#test_not_favourable = test_dataset[~test_dataset["Favourable"]]
# 统计每个用户各喜欢哪些电影，按照UserID进行分组，并遍历每个用户看过的每一部电影
test_favorable_by_users = dict((k, frozenset(v.values)) for k, v in test_favorable.groupby("UserID")["MovieID"])
#test_not_favourable_by_users = dict((k, frozenset(v.values)) for k, v in test_not_favourable.groupby("UserID")["MovieID"])
#test_users = test_dataset["UserID"].unique()

# 创建两个字典, 用来存储规则正例，和返例的次数
correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
# 遍历每个用户喜欢的电影数据
for user, reviews in test_favorable_by_users.items():
    # 遍历每条关联规则
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        # 判断用户是否喜欢前提中的电影
        if premise.issubset(reviews):
            # 如果前提符合，看一下用户是否喜欢结论中的电影
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1
# 用正例次数除以前提条件出现的总次数，计算每条规则的置信度
test_confidence = {candidate_rule: correct_counts[candidate_rule] / float(correct_counts[candidate_rule] + incorrect_counts[candidate_rule])
                   for candidate_rule in rule_confidence}
# 对执行度进行排序，输出置信度最高的前5条规则
sorted_test_confidence = sorted(test_confidence.items(), key=itemgetter(1), reverse=True)
print(sorted_test_confidence[:5])

for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    premise_names = ", ".join(get_movie_name(idx) for idx in premise)
    conclusion_name = get_movie_name(conclusion)
    print("Rule: If a person recommends {0} they will also recommend {1}".format(premise_names, conclusion_name))
    print(" - Train Confidence: {0:.3f}".format(rule_confidence.get((premise, conclusion), -1)))
    print(" - Test Confidence: {0:.3f}".format(test_confidence.get((premise, conclusion), -1)))
    print("")