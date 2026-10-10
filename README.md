# 🎬 Movie Recommendation System

## About the Project

The Movie Recommendation System is a content-based recommendation application built using Python, Pandas, Scikit-learn, and Streamlit. It recommends the top 5 similar movies based on the movie selected by the user.

## Features

- Select a movie from the dropdown menu.
- Clean and preprocess movie descriptions.
- Convert movie overviews into numerical features using TF-IDF.
- Calculate similarity between movies using Cosine Similarity.
- Display the top 5 similar movie recommendations.
- Simple and user-friendly web interface.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit

## Dataset

This project uses the TMDB 5000 Movie Dataset available on Kaggle.

Dataset Link: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

## Project Structure

- `app.py` — Streamlit web application and recommendation logic.
- `task20.py` — Data preprocessing and recommendation implementation.
- `tmdb_5000_movies.csv` — Movie dataset used by the application.
- `requirements.txt` — Required Python libraries.

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Neha1016/Movie-Recommendation-System.git
```

### Step 2: Open the Project Folder

```bash
cd Movie-Recommendation-System
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

Make sure `tmdb_5000_movies.csv` is present in the project root folder.

## How to Use

1. Open the application.
2. Select a movie from the dropdown menu.
3. Click the **Recommend Movies** button.
4. View the similar movie recommendations displayed on the screen.

## Deployment

The application is deployed using Render.

**Live Application:** https://movie-recommendation-system-ximq.onrender.com

## Testing

The deployed application was tested successfully.

- The live website opens.
- The movie selection dropdown works.
- The Recommend Movies button works.
- Similar movie recommendations are displayed for the selected movie.

## GitHub Repository

https://github.com/Neha1016/Movie-Recommendation-System

## Conclusion

This project demonstrates how TF-IDF and Cosine Similarity can be used to build a content-based movie recommendation system. The Streamlit interface allows users to select a movie and explore similar movies easily.
