import joblib
import streamlit as st

with open('model_and_vectorizer.pkl', 'rb') as model_file:
    loaded_model, loaded_cv = joblib.load(model_file)

post_text = st.text_input('Enter Sentence')

if st.button('Check'):
    # Transform the input text using the loaded CountVectorizer
    text_array = loaded_cv.transform([post_text]).toarray()

    # Predict using the loaded model
    prediction = loaded_model.predict(text_array)

    # Display the prediction
    st.text(prediction)
