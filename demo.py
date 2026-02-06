import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

ratings = pd.read_csv("ml-100k/u.data", sep='\t', names=["userId", "movieId", "rating", "timestamp"])
movies = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", header=None)
movies = movies[[0,1]]
movies.columns = ["movieId", "title"]
data = pd.merge(ratings, movies, on="movieId")

user_movie_matrix = data.pivot_table(index="userId", columns="title", values="rating")
user_movie_filled = user_movie_matrix.fillna(0)

user_similarity = cosine_similarity(user_movie_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_filled.index, columns=user_movie_filled.index)
user_similarity_df.head()

def get_similar_users(user_id, top_n=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).iloc[1:top_n+1]
    return similar_users

def recommend_movies(user_id, top_n=5):
    similar_users = get_similar_users(user_id)

    user_rated_movies = user_movie_matrix.loc[user_id]
    unseen_movies = user_rated_movies[user_rated_movies.isna()].index

    scores = {}

    for sim_user, similarity in similar_users.items():
        sim_user_ratings = user_movie_matrix.loc[sim_user, unseen_movies]

        for movie, rating in sim_user_ratings.dropna().items():
            scores[movie] = scores.get(movie,0) + similarity * rating

    ranked_movies = sorted(scores.items(), key = lambda x: x[1], reverse=True)

    return ranked_movies[:top_n]

def show_recommendation(user_id, top_n=5):
    recs = recommend_movies(user_id, top_n)

    print(f"\nTop {top_n} recommendations for User {user_id}: \n")
    for movie, score in recs:
         print(f"{movie} --> score: {round(score, 2)}")