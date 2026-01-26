# ===============================
# Step 1: Import Libraries
# ===============================
import numpy as np
import tensorflow as tf
import streamlit as st
import re

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# ===============================
# Step 2: Load IMDB Word Index
# ===============================
word_index = imdb.get_word_index()

# Reserve special tokens
# 0 = padding
# 1 = start
# 2 = OOV
# 3 = unused
INDEX_FROM = 3


# ===============================
# Step 3: Load Trained Model
# ===============================
model = load_model("simple_rnn_imdb.h5")


# ===============================
# Step 4: Preprocessing Function
# ===============================
def preprocess_text(text):
    # Clean text (same style IMDB expects)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    words = text.split()

    encoded_review = []
    for word in words:
        if word in word_index:
            encoded_review.append(word_index[word] + INDEX_FROM)
        else:
            encoded_review.append(2)  # OOV token

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500,
        padding="pre",
        truncating="pre"
    )

    return padded_review


# ===============================
# Step 5: Streamlit App UI
# ===============================
st.set_page_config(page_title="IMDB Sentiment Analysis", layout="centered")

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review and the model will classify it as **Positive** or **Negative**.")

user_input = st.text_area("✍️ Movie Review", height=150)


# ===============================
# Step 6: Prediction Logic
# ===============================
if st.button("Classify"):

    if not user_input.strip():
        st.warning("⚠️ Please enter a valid movie review.")
    else:
        preprocessed_input = preprocess_text(user_input)

        prediction = model.predict(preprocessed_input)[0][0]

        sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"

        st.subheader("Result")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Prediction Score:** {prediction:.4f}")

        # Debug (optional – uncomment if needed)
        # st.write("Encoded Input:", preprocessed_input)
