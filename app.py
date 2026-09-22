import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write("Select a movie and get personalized movie recommendations.")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_resource
def load_data():
    movie_list = joblib.load("movie_list.joblib")
    similarity = joblib.load("similarity.joblib")
    return movie_list, similarity


new_df, similarity = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("⚙️ Recommendation Settings")

# Number of recommendations
num_recommendations = st.sidebar.slider(
    "Number of movies to recommend",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

st.sidebar.write(
    f"Selected recommendation count: **{num_recommendations}**"
)

# --------------------------------------------------
# CUSTOM MOVIE SELECTION
# --------------------------------------------------

st.subheader("🔎 Select Your Movie")

movie_list = sorted(new_df["title"].dropna().unique().tolist())

selected_movie = st.selectbox(
    "Choose a movie",
    movie_list,
    index=movie_list.index("Iron Man 2")
    if "Iron Man 2" in movie_list else 0
)

# --------------------------------------------------
# RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend(movie, number_of_movies=5):

    # Find movie index
    movie_index = new_df[
        new_df["title"] == movie
    ].index[0]

    # Get similarity scores
    distances = list(
        enumerate(similarity[movie_index])
    )

    # Sort highest similarity first
    distances = sorted(
        distances,
        reverse=True,
        key=lambda x: x[1]
    )

    recommendations = []

    # Skip selected movie itself
    for index, score in distances[1:number_of_movies + 1]:

        recommendations.append({
            "title": new_df.iloc[index]["title"],
            "similarity": score
        })

    return recommendations


# --------------------------------------------------
# RECOMMEND BUTTON
# --------------------------------------------------

if st.button("🎯 Get Recommendations", type="primary"):

    recommendations = recommend(
        selected_movie,
        num_recommendations
    )

    st.success(
        f"Recommendations based on: **{selected_movie}**"
    )

    st.subheader(
        f"🎬 Top {num_recommendations} Recommended Movies"
    )

    # Display recommendations
    for i, movie in enumerate(recommendations, start=1):

        similarity_percentage = movie["similarity"] * 100

        st.markdown(
            f"""
            ### {i}. {movie['title']}

            **Similarity:** {similarity_percentage:.2f}%

            ---
            """
        )

# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Dataset")

st.sidebar.metric(
    "Total Movies",
    len(new_df)
)

st.sidebar.metric(
    "Features",
    len(new_df.columns)
)

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------

with st.expander("📋 View Movie Dataset"):

    st.dataframe(
        new_df,
        use_container_width=True
    )

# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

with st.expander("ℹ️ About This Project"):

    st.write("""
    This project is a Content-Based Movie Recommendation System.

    The system uses:

    • Movie overview
    • Genres
    • Keywords
    • Cast
    • Director

    These text features are combined into movie tags.

    CountVectorizer is used to convert movie tags into numerical
    vectors, and Cosine Similarity is used to find movies that
    are most similar to the selected movie.
    """)

    st.write(
        f"Total movies available for selection: **{len(new_df)}**"
    )