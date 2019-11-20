import pandas as pd
import numpy as np

#把电影以及它的用户和rating
movie=pd.read_csv('./movies.csv')
train_rating=pd.read_csv('./train_ratings_binary.csv')
movie.tail()
print movie.shape
''''
movie['movieRow']=movie.index
movies_modfi=movie[['movieRow','movieId','title','genres']]

rating_df=pd.merge(train_rating,movies_modfi,on='movieId')
rating_df.to_csv('./rating_movies.csv')
print rating_df.head()
# 导入tags的相关文件
user_tag=pd.read_csv('./genome-tags.csv')
tag_score=pd.read_csv('./genome-scores.csv')
user_tag_movieid=pd.read_csv('./tags_shuffled_rehashed.csv')

#合并tag的相关文件，并保存成新的csv文件
movie_tag_Combin=pd.merge(user_tag,user_tag_movieid,on='tag')
print movie_tag_Combin.head()
movie_tag_Combin.to_csv('./movies_tag_Combinv2.csv')
movie_tag_user_comb=pd.merge(movie_tag_Combin,tag_score,on=['movieId','tagId'])
print movie_tag_user_comb.head()
''''''