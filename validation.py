import os,sys
import pandas as pd
import json
import numpy as np
#读入之前获取的sorted rules
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
    cnt=0
    total+=1
    for rules,confidence in sorted_confidence:
        if rules[1]==row["movieId"]:
            if rules[0].issubset(favorable_reviews_by_users[row["userId"]]):
                cnt=1
    if(cnt==0):
        cnt=np.random.randint(0, 1)
    if cnt==row["rating"]:
        correct+=1
        if(correct%100==0):
            print(correct/total)