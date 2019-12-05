import pandas as pd
import numpy as np
import math
from sklearn.cluster import SpectralBiclustering


def bicluster(userid, movie_rating, movieDatabase):
    movieDatabase = list(movieDatabase)

    movieDatabase = {v: i for i, v in enumerate(movieDatabase)}
    userid = list(userid)
    print(len(userid), len(movieDatabase))
    # ratingMatrix=[[False]*(len(userid) for i in range(len(movieid)))] # [[False]*(len(s2)+1) for i in range(len(s1)+1)]
    ratingMatrix = np.zeros((len(userid), len(movieDatabase)), dtype=int)
    print((ratingMatrix.shape))
    print(ratingMatrix[1, 2])
    movie_rating_userid = movie_rating['userId'].values.tolist()
    movie_rating_movieid = movie_rating['movieId'].values.tolist()
    distinMovie = list(set(list(movie_rating_movieid)))
    print(len(distinMovie))

    movie_rating_result = movie_rating['rating'].values.tolist()
    print(max(movie_rating_userid))
    # a=movie_rating[3, 1]
    # print(movie_rating)
    for rownumber in range(len(movie_rating_userid) - 1):
        cur_user = movie_rating_userid[rownumber]
        cur_movieid = movie_rating_movieid[rownumber]
        cur_rate = movie_rating_result[rownumber]
        movie_index = movieDatabase[cur_movieid]
        # cur_user=int(str(cur_user))
        # movie_index=int(str(cur_user))
        # print('movie_index:',movie_index,',,cur_user:',cur_user)
        rate_score= int(math.pow(3,cur_rate)-2)
        #print(cur_user,cur_movieid,cur_rate,rate_score)

        ratingMatrix[(cur_user) - 1, (movie_index)-1] = rate_score

    print('finish')
    print(ratingMatrix[1:10, 1:10])

    clustering = SpectralBiclustering(n_clusters=100, random_state=0).fit(ratingMatrix[0:1000,:])  # [:,1:len(ratingMatrix[0])])
    user_cluster = clustering.row_labels_
    #user_cluster=[1,2,3]
    #print(user_cluster)
    #movie_cluster = clustering.column_labels_
    return user_cluster


# startTime = time.time()
# cluster: int = 10
# cluster=10
# print(dataFolder)
def main():
    rating_binary = pd.read_csv('./train_ratings_binary.csv')
    print(rating_binary.tail())
    rating_binary = rating_binary.astype('int32', copy=False)
    moviesInfor = pd.read_csv('./movies.csv')
    movieid = moviesInfor['movieId']
    print(movieid.shape)

    userIds = range(0, 138493)

    userIdGroup = bicluster(userIds, rating_binary, movieid)  # kmans活bi cluster算出的结果
    file = open('userIdGroup.txt','w+')
    file.write(str(userIdGroup))
    file.close()


if __name__ == "__main__":
    main()
