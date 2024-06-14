import streamlit as st
import PIL.Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import os
import pickle

# Load label data for landmark detection
@st.cache_data
def load_labels():
    label_url = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_asia_V1_label_map.csv'
    df = pd.read_csv(label_url)
    return dict(zip(df.id, df.name))

labels = load_labels()

@st.cache_resource
def load_landmark_model():
    model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
    model = hub.KerasLayer(model_url, output_key='predictions:logits')
    return model

@st.cache_resource
def load_language_model():
    with open('model.pckl', 'rb') as file:
        model = pickle.load(file)
    return model

def image_processing(image_path, model):
    img_shape = (321, 321)
    img = PIL.Image.open(image_path)
    img = img.resize(img_shape)
    img1 = img.copy()  # Keep a copy of the original image for display
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    result = model(img)
    prediction_logits = result.numpy()[0]  # Convert tensor to numpy array
    return labels[np.argmax(prediction_logits)], img1

def get_map(loc):
    geolocator = Nominatim(user_agent="landmark_recognition_app")
    location = geolocator.geocode(loc)
    return location.address, location.latitude, location.longitude

def landmark_recognition():
    st.title("Landmark Recognition")

    # Display logo
    if os.path.exists('logo.png'):
        logo_image = PIL.Image.open('logo.png')
        logo_image = logo_image.resize((256, 256))
        st.image(logo_image)

    img_file = st.file_uploader("Choose your Image", type=['png', 'jpg', 'jpeg'])
    
    if img_file is not None:
        save_image_path = './Uploaded_Images/' + img_file.name
        os.makedirs(os.path.dirname(save_image_path), exist_ok=True)
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        
        model = load_landmark_model()
        prediction, image = image_processing(save_image_path, model)
        st.image(image)
        st.header(f"📍 **Predicted Landmark is: {prediction}**")
        
        try:
            address, latitude, longitude = get_map(prediction)
            st.success(f'Address: {address}')
            
            loc_dict = {'Latitude': latitude, 'Longitude': longitude}
            st.subheader(f'✅ **Latitude & Longitude of {prediction}**')
            st.json(loc_dict)
            
            data = pd.DataFrame([[latitude, longitude]], columns=['lat', 'lon'])
            st.subheader(f'✅ **{prediction} on the Map** 🗺️')
            st.map(data)
        except Exception as e:
            st.warning("No address found!!")

def language_detection():
    st.title("Language Detection Tool")
    st.write("Provide a text input to detect the language.")
    
    input_test = st.text_input("Provide your text input here", 'Hello, my name is Jay.')

    if st.button("Detect Language"):
        if input_test.strip():
            model = load_language_model()
            prediction = model.predict([input_test])[0]
            st.write(f"Detected Language: {prediction}")
        else:
            st.warning("Please enter some text for detection.")

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Landmark Recognition", "Language Detection"])

    if app_mode == "Landmark Recognition":
        landmark_recognition()
    elif app_mode == "Language Detection":
        language_detection()

if __name__ == "__main__":
    main()
