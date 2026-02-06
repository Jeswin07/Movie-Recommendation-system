import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import streamlit as st

import os
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

import requests

@st.cache_data
def fetch_poster(movie_title):
    cleaned_title = clean_movie_title(movie_title)

    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": cleaned_title
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get("results"):
        for result in data["results"]:
            poster_path = result.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"

    return None




# Load ratings
ratings = pd.read_csv(
    r"C:\Users\anasb\OneDrive\Desktop\Data Science Btype\Week29PredNLP\Recommendation-project\ml-100k\u.data",
    sep="\t",
    names=["userId", "movieId", "rating", "timestamp"]
)

# Load movie titles
movies = pd.read_csv(
    r"C:\Users\anasb\OneDrive\Desktop\Data Science Btype\Week29PredNLP\Recommendation-project\ml-100k\u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1]
)

movies.columns = ["movieId", "title"]

# Merge
data = pd.merge(ratings, movies, on="movieId")

# User–Movie matrix
user_movie_matrix = data.pivot_table(
    index="userId",
    columns="title",
    values="rating"
)

# Fill missing values for similarity computation
user_movie_filled = user_movie_matrix.fillna(0)



# User–User similarity
user_similarity = cosine_similarity(user_movie_filled)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_filled.index,
    columns=user_movie_filled.index
)

def get_similar_users(user_id, top_n=5):
    return (
        user_similarity_df[user_id]
        .sort_values(ascending=False)
        .iloc[1 : top_n + 1]
    )

def get_users_who_liked_movie(movie_title, min_rating=4):
    return data[
        (data["title"] == movie_title) &
        (data["rating"] >= min_rating)
    ]["userId"].unique()



# Load full movie metadata
movies_full = pd.read_csv(
    r"C:\Users\anasb\OneDrive\Desktop\Data Science Btype\Week29PredNLP\Recommendation-project\ml-100k\u.item",
    sep="|",
    encoding="latin-1",
    header=None
)

genre_cols = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

movies_full.columns = (
    ["movieId", "title", "release_date", "video_release", "imdb_url"]
    + genre_cols
)

def combine_genres(row):
    return " ".join([genre for genre in genre_cols if row[genre] == 1])

movies_full["genres_text"] = movies_full.apply(combine_genres, axis=1)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies_full["genres_text"])

content_similarity = cosine_similarity(tfidf_matrix)

content_similarity_df = pd.DataFrame(
    content_similarity,
    index=movies_full["title"],
    columns=movies_full["title"]
)


#Content based Recommendation
def content_based_recommend(movie_title, top_n=10):
    if movie_title not in content_similarity_df:
        return {}

    scores = (
        content_similarity_df[movie_title]
        .drop(movie_title)
        .sort_values(ascending=False)
        .head(top_n)
    )

    return scores.to_dict()

#Collaborative Recommendation
def collaborative_recommend(movie_title, top_n=10):
    users = get_users_who_liked_movie(movie_title)

    scores = {}

    for user_id in users:
        similar_users = get_similar_users(user_id)

        user_ratings = user_movie_matrix.loc[user_id]
        unseen_movies = user_ratings[user_ratings.isna()].index

        for sim_user, similarity in similar_users.items():
            sim_ratings = user_movie_matrix.loc[sim_user, unseen_movies]

            for movie, rating in sim_ratings.dropna().items():
                if movie != movie_title:
                    scores[movie] = scores.get(movie, 0) + similarity * rating

    return dict(
        sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    )




def is_content_similar(base_movie, candidate_movie, threshold=0.15):
    try:
        score = content_similarity_df.loc[base_movie, candidate_movie]

        # If pandas Series or numpy array, extract scalar
        if hasattr(score, "__len__") and not isinstance(score, (float, int)):
            score = score.iloc[0]

        return float(score) >= threshold # type: ignore

    except Exception:
        return False


def normalize_scores(scores_dict):
    if not scores_dict:
        return {}

    max_score = max(scores_dict.values())
    return {k: v / max_score for k, v in scores_dict.items()}


#Hybrid Recommendation
def hybrid_recommend(movie_title, top_n=5, alpha=0.3, beta=0.7):
    collab_raw = collaborative_recommend(movie_title, top_n=30)
    content_raw = content_based_recommend(movie_title, top_n=30)

    collab_scores = normalize_scores(collab_raw)
    content_scores = normalize_scores(content_raw)

    final_scores = {}

    for movie in set(collab_scores) | set(content_scores):

        # Gate using content similarity
        if not is_content_similar(movie_title, movie):
            continue

        final_scores[movie] = (
            alpha * collab_scores.get(movie, 0) +
            beta * content_scores.get(movie, 0)
        )

    ranked = sorted(
        final_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_n]



#Movie name cleaning and matching with poster names
import re

def clean_movie_title(title):
    title = re.sub(r"\(\d{4}\)", "", title)
    
    if "," in title:
        parts = title.split(",")
        title = parts[1].strip() + " " + parts[0].strip()
    
    return title.strip()




#streamlit
#--------------

import streamlit as st

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# -------------------------------
# Title / Hero
# -------------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🎬 Movie Recommendation System 🍿</h1>
    <p style='text-align: center; font-size:18px; color:gray;'>
    Hybrid recommender using Collaborative + Content-Based Filtering
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------------------
# MOVIE SELECTION SECTION
# -------------------------------
st.subheader("🔍 Choose a movie you like")

movie_list = sorted(user_movie_matrix.columns.tolist())

search_text = st.text_input(
    "Type a movie name",
    placeholder="e.g. Titanic"
)

if search_text:
    filtered_movies = [
        m for m in movie_list if search_text.lower() in m.lower()
    ]
else:
    filtered_movies = movie_list

selected_movie = st.selectbox(
    "Select from matching movies",
    filtered_movies
)

recommend_from_search = st.button("🎯 Recommend from selected movie")

st.divider()

# -------------------------------
# QUICK PICK: CLICK A POSTER
# -------------------------------
st.subheader("🍿 Popular Picks")

popular_movies = [
    "Titanic (1997)",
    "Toy Story (1995)",
    "Jurassic Park (1993)",
    "Forrest Gump (1994)",
    "The Matrix (1999)",
    "Pulp Fiction (1994)",
    "Star Wars (1977)",
    "The Lion King (1994)",
]

cols = st.columns(len(popular_movies))
clicked_movie = None

for i, movie in enumerate(popular_movies):
    with cols[i]:
        poster = fetch_poster(movie)
        if poster:
            st.image(poster, use_container_width=True)
        else:
            st.write("🎬")

        if st.button(movie, key=f"popular_{i}"):
            clicked_movie = movie

# -------------------------------
# DECIDE WHICH MOVIE TO USE
# -------------------------------
final_movie = None

if recommend_from_search:
    final_movie = selected_movie
elif clicked_movie:
    final_movie = clicked_movie

# -------------------------------
# SHOW RECOMMENDATIONS
# -------------------------------
if final_movie:
    st.divider()
    st.subheader(f"🎯 Recommendations based on **{final_movie}**")

    with st.spinner("Finding great movies for you..."):
        recommendations = hybrid_recommend(final_movie, top_n=6)

    if not recommendations:
        st.warning(
            "😕 Couldn't find strong recommendations for this movie.\n"
            "Try another movie or a more popular one."
        )
    else:
        rec_cols = st.columns(len(recommendations))

        for idx, (movie, score) in enumerate(recommendations):
            with rec_cols[idx]:
                poster = fetch_poster(movie)

                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.write("🎬 Poster not available")

                st.markdown(f"**{movie}**")
                st.caption(f"⭐ Similarity score: {round(score, 2)}")

