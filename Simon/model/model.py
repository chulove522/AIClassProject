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


    
def load_data():
    ratings = pd.read_csv('ratings_renew.csv')
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
    
    
def update_movie_most_popular():
    ratings = pd.read_csv('ratings_t.csv')
    movie_popularity = ratings["title"].value_counts()
    popular_movies = movie_popularity[movie_popularity > 100].index
    popular_movies_rankings =  ratings[ratings["title"].isin(popular_movies)]
    result = popular_movies_rankings[["title", "rating"]].groupby('title').mean().sort_values(ascending=False, by="rating").head(10)
    result.to_csv("result_of_mp.csv")

        
        
        
def update_movie_similarity():
    user_ratings_table, avg_ratings, user_ratings_table_normed, movie_ratings = load_data() 
    # Generate the similarity matrix
    similarities_m = cosine_similarity(movie_ratings)
    # Wrap the similarities in a DataFrame
    movie_similarity = pd.DataFrame(similarities_m, index = movie_ratings.index, columns = movie_ratings.index)
    movie_similarity.to_csv("movie_similarity.csv")
  
def update_user_similarity():
    user_ratings_table, avg_ratings, user_ratings_table_normed, movie_ratings = load_data() 
    # Generate the similarity matrix
    similarities_u = cosine_similarity(user_ratings_table_normed)
    # Wrap the similarities in a DataFrame
    user_similarity = pd.DataFrame(similarities_u, index = user_ratings_table.index, columns = user_ratings_table.index)
    user_similarity.to_csv("user_similarity.csv")
    
def rcmd_pairs(x):
    pairs = pd.DataFrame(list(permutations(x.values, 2)),
                       columns=['movie_a', 'movie_b'])
    return pairs        

  

#more than 100 ratings & TOP10 average rating movies  # >>>moive
def most_popular():
    result = pd.read_csv('result_of_mp.csv')
    return result.title


#input movie name and number of recommend >> movie #搜尋耗時約5sec
def rcmd_by_genres(targetmovie,k):
    genres = pd.read_csv('genres.csv',index_col=0)
    
# targetmovie = 'war of the buttons'
    if targetmovie not in genres.index:
        print("Sorry! There is no movie recommend. ")
    else:
        target = genres[genres.index==targetmovie]
        sub = genres.sample(frac=0.25,axis=0)
        sub = sub.drop([targetmovie],errors = 'ignore')
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
        ordered_similarities = jaccard_similarity_series.sort_values(ascending=False)[1:k+1]
      
        return ordered_similarities
        


# 1.movie >> movie   2.user data >>> movie  #12s
def rcmd_by_ov(targetmovie = 'Waiting to Exhale', user = 1):
    meta = pd.read_csv('meta_cleaned.csv')
    meta = meta.loc[:,['id', 'title', 'overview']]
    result = pd.DataFrame()
    nltk.download("stopwords")
    if user == 1:
    # targetmovie = 'Waiting to Exhale'
      target = meta[meta.title==targetmovie]
      n = len(meta)//10000 + 1
      indexNames = meta[(meta.title==targetmovie)].index
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
        tfidf_df = pd.DataFrame(vectorized_data.toarray(), columns=vectorizer.get_feature_names_out(), index = mer['title'])


        target_df = tfidf_df.loc[targetmovie, :]

      #   # Calculate the cosine_similarity and wrap it in a DataFrame
        similarity_array = cosine_similarity(target_df.values.reshape(1, -1), tfidf_df)
        similarity_df = pd.DataFrame(similarity_array.T, index=tfidf_df.index, columns=["similarity_score"])

      # # Sort the values from high to low by the values in the similarity_score
        sorted_similarity_df = similarity_df.sort_values(by="similarity_score", ascending=False).iloc[1:5]
        result = pd.concat([result,sorted_similarity_df])
      return (result.sort_values(by = 'similarity_score', ascending = False).head())
    else:
    #have a list
      list_of_movies_enjoyed = ['Waiting to Exhale', 'Ruthless People', 'Robin and Marian']
      target = meta.loc[meta['title'].isin(list_of_movies_enjoyed), :]
      result = pd.DataFrame()
      n = len(meta)//10000 + 1
      indexNames = meta[(meta.title.isin(list_of_movies_enjoyed))].index
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
        tfidf_df = pd.DataFrame(vectorized_data.toarray(), columns=vectorizer.get_feature_names_out(), index = mer['title'])
        movies_enjoyed_df = tfidf_df.loc[list_of_movies_enjoyed, :]
        user_prof = movies_enjoyed_df.mean()

      # Find subset of tfidf_df that does not include movies in list_of_movies_enjoyed
        tfidf_subset_df = tfidf_df.drop(list_of_movies_enjoyed, axis=0)

      # Calculate the cosine_similarity and wrap it in a DataFrame
        similarity_array = cosine_similarity(user_prof.values.reshape(1, -1), tfidf_subset_df)
        similarity_df = pd.DataFrame(similarity_array.T, index=tfidf_subset_df.index, columns=["similarity_score"])

      # Sort the values from high to low by the values in the similarity_score
        sorted_similarity_df = similarity_df.sort_values(by="similarity_score", ascending=False).iloc[0:5]
        result = pd.concat([result,sorted_similarity_df])
      return (result.sort_values(by = "similarity_score",ascending = False).head())




# 28s  movie >> movie
def pair_rcmd(x):  
    ratings = pd.read_csv('ratings_renew.csv', index_col=0)
  # Apply the function to the title column and reset the index
    movie_combinations = ratings.groupby('userId')['title'].apply(find_movie_pairs).reset_index(drop=True)

  # Calculate how often each item in movies_a occurs with the items in movies_b
    combination_counts = movie_combinations.groupby(['movie_a', 'movie_b']).size()



  # Convert the results to a DataFrame and reset the index
    combination_counts_df = combination_counts.to_frame(name='size').reset_index()
 

  # Sort the counts from highest to lowest
    combination_counts_df.sort_values('size', ascending=False, inplace=True)

  # Find the movies most frequently watched by people who watched "target"
    df = combination_counts_df[combination_counts_df['movie_a'] == x]



    return df.head(10)



#Collaborative Filtering #52s
def CbtF_byMovie(x):
    user_ratings_table, avg_ratings, user_ratings_table_normed, movie_ratings = load_data() 

  #相似電影預測 "電影id1"

    movie_similarity = pd.read_csv('movie_similarity.csv', index_col=0)

  # Find the similarity values for a specific movie
    cosine_similarity_series = movie_similarity.loc[x]

  # Sort these values highest to lowest
    ordered_similarities = cosine_similarity_series.sort_values(ascending=False)

    return ordered_similarities.iloc[1:6]
  
  
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


user, movie >>> rate #9s
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
    avg_ratings_pd = pd.Series(avg_ratings, index =user_ratings_table.index) 
    U = pd.read_csv('U_50.csv',index_col=0).to_numpy()
    sigma = pd.read_csv('sigma_50.csv',index_col=0).to_numpy()
    Vt = pd.read_csv('Vt_50.csv',index_col=0).to_numpy()
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

    return rcmd.head()
































