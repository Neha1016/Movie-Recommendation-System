
# TASK 6: STREAMLIT APP

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page title
st.title("🎬 Movie Recommendation System")
st.write("Select a movie to find similar movies.")

# Load movie dataset
movies = pd.read_csv("tmdb_5000_movies.csv")

# Handle missing descriptions
movies["overview"] = movies["overview"].fillna("")

# Create TF-IDF vectors
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["overview"])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create movie title index
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# Recommendation function
def recommend_movies(movie_title):
    idx = indices[movie_title]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:6]

    movie_indices = [i[0] for i in scores]
    return movies["title"].iloc[movie_indices].tolist()

# Movie selection dropdown
movie_title = st.selectbox(
    "Choose a movie:",
    sorted(movies["title"].dropna().unique())
)

# Show recommendations
if st.button("Recommend Movies"):
    recommendations = recommend_movies(movie_title)

    st.subheader("Recommended Movies")
    for movie in recommendations:
        st.write("🎥", movie)
