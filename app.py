import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer from pickle files
with open('pickle files/random_forest_model.pkl', 'rb') as model_file:
    trained_model = pickle.load(model_file)

with open('pickle files/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Check if vectorizer is fitted
if not hasattr(vectorizer, 'vocabulary_'):
    st.error("The vectorizer was not fitted. Please make sure the model and vectorizer were correctly trained and saved.")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Spam or Ham Email Classifier",
    page_icon="📧",
    layout="wide",
)

# Custom styles
st.markdown(
    """
    <style>
    body {
        background-color: #f4f6f9;
        font-family: 'Roboto', sans-serif;
    }
    .title {
        font-size: 2em;
        color: #2e3b4e;
        text-align: center;
        font-weight: bold;
    }
    .header {
        font-size: 1.5em;
        color: #2e3b4e;
        text-align: center;
    }
    .text-area {
        font-size: 1em;
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.2em;
        border-radius: 8px;
        padding: 10px;
        width: 100%;
        text-align: center;
        cursor: pointer;
    }
    .result {
        font-size: 1.5em;
        text-align: center;
        font-weight: bold;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header section
st.markdown('<div class="title">Spam or Ham Email Classifier</div>', unsafe_allow_html=True)

# Description
st.markdown("""This app uses a **Random Forest model** to classify emails as either **SPAM** or **HAM** (non-spam). Just enter the email content below, and click **Classify** to see the result!""", unsafe_allow_html=True)

# User input text for classification
st.markdown("<div class='header'>Paste the email content here:</div>", unsafe_allow_html=True)

# Text area for input email
input_text = st.text_area("Email Content", height=200, placeholder="Enter email content...")

# Classify button logic
if st.button("Classify Email", key="classify_button", use_container_width=True):
    if input_text:
        # Transform the input text using the TF-IDF vectorizer
        input_vectorized = vectorizer.transform([input_text])
        
        # Make prediction using the trained Random Forest model
        prediction = trained_model.predict(input_vectorized)
        
        # Display the result
        if prediction == 0:
            st.markdown('<div class="result" style="color: #2e8b57;">This email is HAM (not spam).</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result" style="color: #f44336;">This email is SPAM.</div>', unsafe_allow_html=True)
    else:
        st.error("Please enter the text of the email to classify.")

# Footer Section
st.markdown("""<br><br><footer style="text-align: center;"><p style="font-size: 0.9em; color: #2e3b4e;">Built by Karthikeya MV</p></footer>""", unsafe_allow_html=True)
