import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = keras.models.load_model("flower_model.h5")

st.set_page_config(page_title="Flower-Type-Classification",layout="wide")

st.title("Flower Type Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.

    # Make a prediction
    prediction = model.predict(img)
    flower_class = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
    st.write(prediction)
    st.image(uploaded_file,caption =flower_class[np.argmax(prediction)] )

    # # Display the result
    # if prediction[0][0] > 0.5:
    #     st.image(uploaded_file, caption='This is a Dog', use_column_width=True)
    # else:
    #     st.image(uploaded_file, caption='This is a Cat', use_column_width=True)
       
