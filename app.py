import joblib
import streamlit as st

with open('model_and_vectorizer.pkl', 'rb') as model_file:
    loaded_model, loaded_cv = joblib.load(model_file)

post_text = st.text_input('post_input')

if st.button('check'):
    text_array = loaded_cv.transform([post_text]).toarray()

    prediction = loaded_model.predict(text_array)

    st.text(prediction)
