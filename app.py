import streamlit as st
import sqlite3
import pandas as pd
import joblib
import numpy as np
from auth import login_user, register_user
from aptitude_test import run_aptitude_test  # Import Aptitude Test
from sklearn.metrics.pairwise import cosine_similarity

# Load the trained model, feature names, and label encoder
model = joblib.load('model.pkl')
feature_columns = joblib.load('features.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load the dataset
df = pd.read_csv("stud.csv")  # Ensure the correct path

# Extract feature columns (excluding the 'Courses' column)
feature_columns = df.columns[df.columns != "Courses"]
career_features = df[feature_columns].astype(float)  # Ensure it's numeric
career_labels = df["Courses"]

# Compute similarity matrix directly using cosine similarity
career_sim_matrix = cosine_similarity(career_features)

def predict_career(user_features):

    # Convert user input to NumPy array and reshape for prediction
    user_features = np.array(user_features).reshape(1, -1)

    # Predict career label
    predicted_label = model.predict(user_features)[0]

    # Decode label back to career name
    predicted_career = label_encoder.inverse_transform([predicted_label])[0]

    return predicted_career


# Function to get serendipity-based career suggestions
def get_serendipity_careers(predicted_career, num_options=5):
    # Load dataset
    df = pd.read_csv("stud.csv")

    # Ensure 'Courses' column exists
    if "Courses" not in df.columns:
        return ["Error: 'Courses' column not found in dataset"]

    # Get unique careers
    unique_careers = df["Courses"].unique()

    # Check if predicted career exists in dataset
    if predicted_career not in unique_careers:
        return ["Error: Predicted career not found in dataset"]

    # Extract feature columns (excluding 'Courses')
    feature_columns = df.columns[df.columns != "Courses"]
    career_features = df[feature_columns]

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(career_features)

    # Get index of the predicted career
    predicted_idx = df[df["Courses"] == predicted_career].index[0]

    # Get similarity scores for the predicted career
    similarity_scores = similarity_matrix[predicted_idx]

    # Sort careers based on similarity (excluding itself)
    sorted_indices = np.argsort(similarity_scores)[::-1]  # Sort in descending order
    similar_careers = df.iloc[sorted_indices]["Courses"].unique().tolist()

    # Remove the predicted career from the list
    similar_careers = [career for career in similar_careers if career != predicted_career]

    return similar_careers[:num_options]  # Return the required number of diverse careers



# Initialize Session State
if "page" not in st.session_state:
    st.session_state.page = "login"
if "username" not in st.session_state:
    st.session_state.username = None
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "recommended_career" not in st.session_state:
    st.session_state.recommended_career = None
if "aptitude_result" not in st.session_state:
    st.session_state.aptitude_result = None
if "serendipity_careers" not in st.session_state:
    st.session_state.serendipity_careers = []


# --- LOGIN PAGE ---
def login_page():
    st.title("Career Recommendation System")
    option = st.radio("Select an option:", ["Login", "Register"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Register":
        if st.button("Register"):
            if register_user(username, password):
                st.success("Registered successfully! Please log in.")
            else:
                st.error("Username already exists.")

    elif option == "Login":
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.username = username
                st.session_state.page = "landing"
                st.rerun()
            else:
                st.error("Invalid username or password.")


# --- LANDING PAGE ---
def landing_page():
    st.title("Welcome to Career Recommendation System")
    st.write(f"Hello, {st.session_state.username}!")

    if st.button("Take Career Test"):
        st.session_state.page = "quiz"
        st.rerun()

    if st.button("Take Aptitude Test"):
        st.session_state.page = "aptitude"
        st.rerun()

    if (st.session_state.recommended_career or st.session_state.aptitude_result) and st.button(
            "View Career Recommendation"):
        st.session_state.page = "serendipity"
        st.rerun()

    if st.button("View Past Recommendations"):
        st.session_state.page = "history"
        st.rerun()

    if st.button("Logout"):
        st.session_state.clear()
        st.session_state.page = "login"
        st.rerun()





# --- QUIZ PAGE ---
def quiz_page():
    st.title("Career Test")

    user_features = []  # Store user responses

    # Loop through feature questions
    for feature in feature_columns:
        response = st.radio(f"Do you have an interest in {feature}?", ("Yes", "No"))
        user_features.append(1 if response == "Yes" else 0)

    if st.button("Submit Test"):
        st.session_state.recommended_career = predict_career(user_features)
        st.success(f"🎯 Your Predicted Career: {st.session_state.recommended_career}")
        st.session_state.page = "result"
        st.rerun()

# --- APTITUDE TEST PAGE ---
def aptitude_test_page():
    st.title("Aptitude Test")
    run_aptitude_test()

    if st.button("Back to Landing Page"):
        st.session_state.page = "landing"
        st.rerun()


# --- HISTORY PAGE ---
def history_page():
    st.title("Your Past Recommendations")
    conn = sqlite3.connect("career_recommendation.db")
    cursor = conn.cursor()
    cursor.execute("SELECT final_career, serendipity_career FROM final_recommendations WHERE username = ?",
                   (st.session_state.username,))
    results = cursor.fetchall()
    conn.close()

    if results:
        st.write("Your past career recommendations:")
        for row in results:
            st.write(f"- **Final Recommendation:** {row[0]} | ✨ **Serendipity Career:** {row[1]}")
    else:
        st.write("No past recommendations found.")

    if st.button("Go Back to Landing Page"):
        st.session_state.page = "landing"
        st.rerun()

def result_page():
    st.title("Career Recommendation Result")

    if st.session_state.recommended_career:
        st.write(f"🎯 *Recommended Career:* {st.session_state.recommended_career}")

    else:
        st.warning("No recommendation available. Please take the quiz first.")

    if st.button("Back to Landing Page"):
        st.session_state.page = "landing"
        st.rerun()


def serendipity_result_page():
    st.title("Career Recommendation with Serendipity")

    if st.session_state.recommended_career:
        st.write(f"🎯 *Predicted Career:* {st.session_state.recommended_career}")

        # Dropdown for user to select number of recommendations (1 to 5)
        num_options = st.selectbox("Select number of serendipity suggestions:", [1, 2, 3, 4, 5], index=2)

        # Get top 10 similar careers first
        all_similar_careers = get_serendipity_careers(st.session_state.recommended_career, num_options=10)

        # Display only the number selected by user
        st.write("✨ *Serendipity-Based Suggestions:* ")
        for career in all_similar_careers[:num_options]:
            st.write(f"- {career}")

    else:
        st.warning("No career recommendation available. Please complete the tests first.")

    if st.button("Back to Landing Page"):
        st.session_state.page = "landing"
        st.rerun()



# --- PAGE NAVIGATION ---
if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "landing":
    landing_page()
elif st.session_state.page == "quiz":
    quiz_page()
elif st.session_state.page == "aptitude":
    aptitude_test_page()
elif st.session_state.page == "result":
    result_page()  # Assuming result_page() is missing, redirecting to landing_page
elif st.session_state.page == "history":
    history_page()
elif st.session_state.page == "serendipity":
    serendipity_result_page()


