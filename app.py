import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

model_cnn = load_model("cnn_model.h5")

img_width, img_height = 150, 150

st.title("Car Accident Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    
    img = load_img(uploaded_file, target_size=(img_width, img_height))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.  

    
    if st.button("Predict with CNN"):
        prediction = model_cnn.predict(img)
        if prediction[0][0] > 0.5:
            st.write("CNN Prediction: Non-Accidental")
        else:
            st.write("CNN Prediction: Accidental")