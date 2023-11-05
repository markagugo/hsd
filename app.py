import joblib
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
import warnings

# Suppress the warning and capture the original scikit-learn version
warnings.simplefilter("error", InconsistentVersionWarning)

try:
    with open('model_and_vectorizer.pkl', 'rb') as model_file:
        loaded_model, loaded_cv = joblib.load(model_file)
except InconsistentVersionWarning as w:
    original_version = w.original_sklearn_version
    st.error(f"Original scikit-learn version: {original_version}")
    st.stop()

post_text = st.text_input('Enter Sentence')

if st.button('Check'):
    # Transform the input text using the loaded CountVectorizer
    text_array = loaded_cv.transform([post_text]).toarray()

    # Predict using the loaded model
    prediction = loaded_model.predict(text_array)

    # Display the prediction
    st.text(prediction)
