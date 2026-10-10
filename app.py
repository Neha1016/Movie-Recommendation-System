
# TASK 6: STREAMLIT APP

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommendation System")
st.write("Select a movie to find similar movies.")

# Load movie dataset
@st.cache_data
def load_movies():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    movies["overview"] = movies["overview"].fillna("")
    movies = movies.dropna(subset=["title"]).reset_index(drop=True)
    return movies

# Create TF-IDF vectors
@st.cache_resource
def create_tfidf(movies):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["overview"])
    return tfidf_matrix

movies = load_movies()
tfidf_matrix = create_tfidf(movies)

# Create movie title index
indices = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()

# Recommendation function
def recommend_movies(movie_title):
    idx = indices[movie_title]

    # Calculate similarity only for the selected movie
    scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    similar_indices = scores.argsort()[::-1]
    similar_indices = [
        i for i in similar_indices if i != idx
    ][:5]

    return movies["title"].iloc[similar_indices].tolist()

# Movie selection dropdown
movie_title = st.selectbox(
    "Choose a movie:",
    sorted(movies["title"].unique())
)

# Show recommendations
if st.button("Recommend Movies"):
    recommendations = recommend_movies(movie_title)

    st.subheader("Recommended Movies")

    if recommendations:
        for movie in recommendations:
            st.write("🎥", movie)
    else:
        st.write("No recommendations found.")
