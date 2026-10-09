import pandas as pd
import re


# TASK 1: DATASET COLLECTION
# Dataset: TMDB 5000 Movie Dataset
# Source: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
# Dataset is stored inside the Dataset folder.


# TASK 2: DATA PREPROCESSING

# Load the movie dataset
movies = pd.read_csv("Dataset/tmdb_5000_movies.csv")

# Display the first five rows
print("Movie Dataset:")
print(movies.head())

# Check missing values
print("\nMissing Values:")
print(movies.isnull().sum())

# Handle missing movie descriptions
movies["overview"] = movies["overview"].fillna("")

# Clean movie descriptions
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

movies["overview"] = movies["overview"].apply(clean_text)

# Remove extra spaces from movie titles
movies["title"] = movies["title"].fillna("").str.strip()

# Display cleaned movie descriptions
print("\nCleaned Movie Descriptions:")
print(movies[["title", "overview"]].head())

# Check missing descriptions after cleaning
print("\nMissing Descriptions After Cleaning:")
print(movies["overview"].isnull().sum())

print("\nTask 2: Data Preprocessing Completed!")


# TASK 3: TEXT VECTORIZATION USING TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

# Convert movie descriptions into numerical vectors
tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(movies["overview"])

# Display the shape of the TF-IDF matrix
print("\nTF-IDF Matrix Shape:")
print(tfidf_matrix.shape)

print("\nTask 3: Text Vectorization Completed!")



# TASK 4: SIMILARITY CALCULATION


from sklearn.metrics.pairwise import cosine_similarity

# Calculate similarity between movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Display similarity matrix shape
print("\nCosine Similarity Matrix Shape:")
print(cosine_sim.shape)

# Display similarity scores for the first movie
print("\nSimilarity Scores for the First Movie:")
print(cosine_sim[0][:10])

print("\nTask 4: Similarity Calculation Completed!")


# TASK 5: RECOMMENDATION LOGIC

# Create a mapping of movie titles to their indexes
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# Function to recommend similar movies
def recommend_movies(movie_title):
    if movie_title not in indices:
        print("Movie not found!")
        return

    # Get the index of the selected movie
    idx = indices[movie_title]

    # Get similarity scores for the selected movie
    scores = list(enumerate(cosine_sim[idx]))

    # Sort movies by similarity score
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Select the top 5 similar movies, excluding itself
    scores = scores[1:6]

    # Get movie indexes
    movie_indices = [i[0] for i in scores]

    # Display recommended movie titles
    print("\nMovies recommended for:", movie_title)
    print(movies["title"].iloc[movie_indices].to_string(index=False))


# Test the recommendation function
recommend_movies("Avatar")

print("\nTask 5: Recommendation Logic Completed!")


