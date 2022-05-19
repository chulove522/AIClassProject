import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import CountVectorizer
from itertools import permutations
import json
import requests
from tmdbv3api import Movie
from tmdbv3api import TMDb
tmdb = TMDb()
tmdb.api_key = '1807cdd6e32d6a41911062f2c73e24a9'
tmdb.language = 'en'
tmdb.debug = True

#movie_id 轉換電影名稱

#def convert_2_title(id):
#  movie = Movie()
#  title = movie.details(id).title
#  return title

#電影名稱 轉換 id
#def convert_2_id(title):
#  response = requests.get('https://api.themoviedb.org/3/search/movie?api_key={}&language=en-US&query={}&page=1&include_adult=false'.format(tmdb.api_key, movie_title))
#  data_json = response.json()
#  return data_json['results'][0]['id']

#用於讀取資料，在協同過濾和SVD會用到
def load_data():
    ratings = pd.read_csv("./Simon/csvdatas/ratings_renew.csv")   #/csvdatas/ratings_renew.csv
    # print(ratings)
    ratings = ratings.loc[:,['userId', 'title', 'rating', 'movieId']]
    user_ratings_table = ratings.pivot(index = 'userId', columns = 'movieId', values= 'rating')
# Get the average rating for each user 
    avg_ratings = user_ratings_table.mean(axis=1)

# Center each users ratings around 0
    user_ratings_table_centered = user_ratings_table.sub(avg_ratings, axis=0)

# Fill in the missing data with 0s
    user_ratings_table_normed = user_ratings_table_centered.fillna(0)

    movie_ratings = user_ratings_table_normed.T
    return user_ratings_table, avg_ratings, user_ratings_table_normed, movie_ratings


load_data()


#更新"電影-協同過濾"的資料，ratings_renew資料有變動才需更新
#耗時 2min   
def update_movie_similarity():
    user_ratings_table, avg_ratings, user_ratings_table_normed, movie_ratings = load_data() 
    # Generate the similarity matrix
    similarities_m = cosine_similarity(movie_ratings)
    # Wrap the similarities in a DataFrame
    movie_similarity = pd.DataFrame(similarities_m, index = movie_ratings.index, columns = movie_ratings.index)
    movie_similarity.to_csv("./csvdatas/movie_similarity.csv")
    

#更新"使用者-協同過濾"的資料，ratings_renew資料有變動才需更新
#耗時 5min
def update_user_similarity():
    user_ratings_table, avg_ratings, user_ratings_table_normed, movie_ratings = load_data() 
    # Generate the similarity matrix
    similarities_u = cosine_similarity(user_ratings_table_normed)
    # Wrap the similarities in a DataFrame
    user_similarity = pd.DataFrame(similarities_u, index = user_ratings_table.index, columns = user_ratings_table.index)
    user_similarity.to_csv("./csvdatas/user_similarity.csv")
    


  
#使用TMDB api去搜尋最流行的五個電影
def movie_most_popular_TMDB():
  movie = Movie()
  popular = movie.popular()
  id = []
  title = []
  for i in range(5):
    id.append(popular[i].id)
    title.append(popular[i].title)
  df = pd.DataFrame((zip(id, title)), columns = ['id', 'title'])
#input TMDB movie id >>> output TOP5 popular movie dataframe
  return df
  
  
#使用TMDB api去推薦"最相關的五個電影"
def movie_rcmd_TMDB(id):
  movie = Movie()
  recommendations = movie.recommendations(movie_id=id)
  id = []
  title = []
  for i in range(5):
    id.append(recommendations[i].id)
    title.append(recommendations[i].title)
  df = pd.DataFrame((zip(id, title)), columns = ['id', 'title'])
#input TMDB movie id >>> output TOP5 similar movie dataframe
  return df


#使用類別去推薦電影，可用於"相似的電影"
#input movie id  >> movie #搜尋耗時約5sec
def rcmd_by_genres(movie_id):
    genres = pd.read_csv('./csvdatas/genres.csv',index_col=0)
    movie = Movie()
    # targetmovie = 'war of the buttons'
    if movie_id not in genres.index:
        print("Sorry! There is no movie recommend. ")
    else:
        target = genres[genres.index==movie_id]
        sub = genres.sample(frac=0.25,axis=0)
        sub = sub.drop([movie_id],errors = 'ignore')
        mer = pd.concat([sub,target])

    # Calculate all pairwise distances
        jaccard_distances = pdist(mer.values, metric='jaccard')
    # Convert the distances to a square matrix
        jaccard_similarity_array = 1 - squareform(jaccard_distances)
    # Wrap the array in a pandas DataFrame
        jaccard_similarity_df = pd.DataFrame(jaccard_similarity_array, index=mer.index, columns=mer.index)
    # Find the values for the target movie 
        jaccard_similarity_series = jaccard_similarity_df.iloc[-1]
    # Sort these values from highest to lowest
        keys = jaccard_similarity_series.sort_values(ascending=False)[1:6].index
        dicts = {}
        for i in keys:
          dicts[i] = movie.details(i).title
        result = pd.DataFrame.from_dict(dicts, orient='index', columns=['title'])
    #input TMDB movie id >>> output TOP5 similar movie dataframe
        return result


#使用overview去推薦電影，可用於"你可能會喜歡的電影"
#輸入電影id，使用overview去推薦五則電影  #12s
#user_id為選擇性，不輸則為電影推薦，輸入則為個人化推薦
def rcmd_by_ov(movie_id = 'Waiting to Exhale', user = 1):
    meta = pd.read_csv('./csvdatas/meta_cleaned_n.csv')
    meta = meta.loc[:,['id', 'title', 'overview']]
    movie = Movie()
    result = pd.DataFrame()
    # nltk.download("stopwords")
    if user == 1:
      target = meta[meta.id==movie_id]
      n = len(meta)//10000 + 1
      indexNames = meta[(meta.id==movie_id)].index
      stopset = set(stopwords.words('english'))
      for i in range(n):
        sub = meta.iloc[i*10000:(i+1)*10000]
        sub = sub.drop(indexNames,errors = 'ignore')
        mer = pd.concat([sub,target])
      # Instantiate the vectorizer object to the vectorizer variable
        vectorizer = TfidfVectorizer(max_df=0.7, min_df=2, stop_words=stopset)
      # Fit and transform the plot column
        vectorized_data = vectorizer.fit_transform(mer['overview'])
      # Create Dataframe from TF-IDFarray and Assign the movie titles to the index and inspect
        tfidf_df = pd.DataFrame(vectorized_data.toarray(), columns=vectorizer.get_feature_names_out(), index = mer['id'])

        target_df = tfidf_df.loc[movie_id, :]

      #   # Calculate the cosine_similarity and wrap it in a DataFrame
        similarity_array = cosine_similarity(target_df.values.reshape(1, -1), tfidf_df)
        similarity_df = pd.DataFrame(similarity_array.T, index=tfidf_df.index, columns=["similarity_score"])

      # # Sort the values from high to low by the values in the similarity_score
        sorted_similarity_df = similarity_df.sort_values(by="similarity_score", ascending=False).iloc[1:6]
        keys = pd.concat([result,sorted_similarity_df]).index

      dicts = {}
      for i in keys:
        dicts[i] = movie.details(i).title
      result = pd.DataFrame.from_dict(dicts, orient='index', columns=['title'])
    
      return result
    #input TMDB movie id >>> output TOP5 similar movie dataframe by overview
    else:
    #have a list
      list_of_movies_enjoyed = [862, 8844, 15602, 31357, 96823]
      target = meta.loc[meta['id'].isin(list_of_movies_enjoyed), :]
      result = pd.DataFrame()
      n = len(meta)//10000 + 1
      indexNames = meta[(meta.id.isin(list_of_movies_enjoyed))].index
    # nltk.download("stopwords")
      stopset = set(stopwords.words('english'))
      for i in range(n):
        sub = meta.iloc[i*10000:(i+1)*10000]
      
        sub = sub.drop(indexNames,errors = 'ignore')
      
        mer = pd.concat([sub,target])
      
      # Instantiate the vectorizer object to the vectorizer variable
        vectorizer = TfidfVectorizer(max_df=0.7, min_df=2, stop_words=stopset)
      # Fit and transform the plot column
        vectorized_data = vectorizer.fit_transform(mer['overview'])
      # Create Dataframe from TF-IDFarray
        tfidf_df = pd.DataFrame(vectorized_data.toarray(), columns=vectorizer.get_feature_names_out(), index = mer['id'])
        movies_enjoyed_df = tfidf_df.loc[list_of_movies_enjoyed, :]
        user_prof = movies_enjoyed_df.mean()

      # Find subset of tfidf_df that does not include movies in list_of_movies_enjoyed
        tfidf_subset_df = tfidf_df.drop(list_of_movies_enjoyed, axis=0)

      # Calculate the cosine_similarity and wrap it in a DataFrame
        similarity_array = cosine_similarity(user_prof.values.reshape(1, -1), tfidf_subset_df)
        similarity_df = pd.DataFrame(similarity_array.T, index=tfidf_subset_df.index, columns=["similarity_score"])

      # Sort the values from high to low by the values in the similarity_score
        sorted_similarity_df = similarity_df.sort_values(by="similarity_score", ascending=False).iloc[0:5]
        keys = pd.concat([result,sorted_similarity_df]).index

      dicts = {}
      for i in keys:
        dicts[i] = movie.details(i).title
      result = pd.DataFrame.from_dict(dicts, orient='index', columns=['title'])
    #input TMDB movie id and user id >>> output TOP5 similar movie dataframe by overview
      return result






  
  
#某人對某電影之預測 使用相似的人的平均  user, movie >> rate  #60s
def rcmd_user(u, m):
    user_ratings_table, avg_ratings, user_ratings_table_normed, movie_ratings = load_data() 
    #相似使用者 使用者id7
    similarities_u = cosine_similarity(user_ratings_table_normed)
    user_similarities = pd.DataFrame(similarities_u, index = user_ratings_table.index, columns = user_ratings_table.index)
    #跟我"使用者id7"相近的十個人
    # Isolate the similarity scores for user_7 and sort
    user_similarity_series = user_similarities.loc[u]
    ordered_similarities = user_similarity_series.sort_values(ascending=False)
    # Find the top 10 most similar users
    nearest_neighbors = ordered_similarities[0:11].index
    # Extract the ratings of the neighbors
    neighbor_ratings = user_ratings_table.reindex(nearest_neighbors)
    #對某電影"電影id1"的評分
    # Calculate the mean rating given by the users nearest neighbors
    return neighbor_ratings[m].mean()



#CbtF_user 和 CbtF_movie 基本上功用是一致的，只是做法不同，用於推測某使用者未看過電影的評分，output為0~5之數值
#user, movie >>> rate  #13s
def CbtF_user(u, m):
    user_ratings_table, avg_ratings, user_ratings_table_normed, movie_ratings = load_data() 

  # Drop the column you are trying to predict #答案 丟掉想預測的電影

    droped_user_ratings_table_normed = user_ratings_table_normed.drop(m, axis=1, inplace=False)

  # Get the data for the user you are predicting for  #想預測對象
    target_user_x = droped_user_ratings_table_normed.loc[[u]]
  # Get the target data from user_ratings_table   #有評分該電影的分數
    other_users_y = user_ratings_table[m]
  # Get the data for only those that have seen the movie   #有預測該電影的人
    other_users_x = droped_user_ratings_table_normed[other_users_y.notnull()]
  # Remove those that have not seen the movie from the target #有預測該電影的人的分數
    other_users_y.dropna(inplace=True)

  # Instantiate the user KNN model
    user_knn = KNeighborsRegressor(metric='cosine', n_neighbors=3)

  # Fit the model and predict the target user
    user_knn.fit(other_users_x, other_users_y)
    user_user_pred = user_knn.predict(target_user_x)

    return user_user_pred


#user, movie >>> rate #9s
def CbtF_movie(u, m):
  user_ratings_table, avg_ratings, user_ratings_table_normed, movie_ratings = load_data() 

  # Drop the column you are trying to predict #答案 丟掉想預測的人
  movie_ratings_droped = movie_ratings.drop(u, axis=1, inplace=False)
  # Get the data for the user you are predicting for  #想預測的電影
  target_movie_x = movie_ratings_droped.loc[m]
  # Get the target data from user_ratings_table   #此人的觀影資料
  other_movies_y = user_ratings_table.loc[u]
  # Get the data for only those that have seen the movie   #有預測該電影的人
  other_movies_x = movie_ratings_droped[other_movies_y.notnull()]
  # Remove those that have not seen the movie from the target #有預測該電影的人的分數
  other_movies_y.dropna(inplace=True)
  # Instantiate the user KNN model
  movie_knn= KNeighborsRegressor(metric='cosine', n_neighbors=3)
  # Fit the model and predict the target user
  movie_knn.fit(other_movies_x, other_movies_y)
  movie_movie_pred = movie_knn.predict(target_movie_x.values.reshape(1, -1))

  return movie_movie_pred


# user >>> movie #8s
def SVD_rcmd(user_num):
    user_ratings_table, avg_ratings, user_ratings_table_normed, movie_ratings = load_data() 
    movie = Movie()
    avg_ratings_pd = pd.Series(avg_ratings, index =user_ratings_table.index) 
    U = pd.read_csv('./csvdatas/U_50.csv',index_col=0).to_numpy()
    sigma = pd.read_csv('./csvdatas/sigma_50.csv',index_col=0).to_numpy()
    Vt = pd.read_csv('./csvdatas/Vt_50.csv',index_col=0).to_numpy()
  # Dot product of U and sigma
    U_sigma = np.dot(U, sigma)

  # Dot product of result and Vt
    U_sigma_Vt = np.dot(U_sigma, Vt)

  # Add back on the row means contained in avg_ratings
    uncentered_ratings = U_sigma_Vt + avg_ratings_pd.values.reshape(-1, 1)

  # Create DataFrame of the results
    calc_pred_ratings_df = pd.DataFrame(uncentered_ratings,index=user_ratings_table.index,columns=user_ratings_table.columns)

  # calc_pred_ratings_df = pd.read_csv('svd_50.csv')
    hasbeenseen = user_ratings_table.loc[user_num,:].dropna(inplace=False).index
    rcmd = calc_pred_ratings_df.loc[user_num,:][~user_ratings_table.columns.isin(hasbeenseen)].sort_values(ascending=False)
    keys = rcmd.head().index
    
    link = pd.read_csv('./csvdatas/link.csv', index_col=1)
    dicts = {}
    for i in keys:
      dicts[i] = link.loc[i].title
    result = pd.DataFrame.from_dict(dicts, orient='index', columns=['title'])
    return result
  #input user id >>> output TOP5 rcmd movie dataframe by SVD
    


    






























