import streamlit as st
import tensorflow as tf 
import numpy as np 
from tensorflow import keras 
from PIL import Image
from tensorflow.keras.preprocessing import image 

MODEL_PATH   = "final_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

class_names =['Normal', 'TB']

def predict_image(img):
    img = img.resize([224,224])
    img_array = image.img_to_array(img) / 255.0 
    img_array = np.expand_dims(img_array,axis=0)
    
    predict = model.predict(img_array)
    predict_class = class_names[np.argmax(predict)]
    return predict_class


st.set_page_config(page_title="🩻 Tuberculosis Detection", layout="centered")

st.title("🩻 Tuberculosis Detection from Chest X-ray")
st.write("Upload a chest X-ray image and the AI model will predict if TB is present.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image",use_column_width=True)
    st.write("")

    with st.spinner("Predicting..."):
        predicted_disease = predict_image(img)  
    st.success(f"**Prediction: {predicted_disease}**")
    
    
    