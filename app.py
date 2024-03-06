import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load feature list and filenames from saved files
feature_list = np.array(pickle.load(open('/content/embedd/embeddings.pkl', 'rb')))
filenames = pickle.load(open('/content/filenames.pkl', 'rb'))

# Define the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    path = os.path.join('uploads', uploaded_file.name)
    with open(path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return path

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])

    return indices

# File upload step
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    # Save and display the uploaded file
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        st.success("Image uploaded successfully!")
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption='Uploaded Fashion Item')

        # Feature extraction
        features = feature_extraction(file_path, model)

        # Recommendations
        indices = recommend(features, feature_list)
        st.write("Recommendations:")

        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                recommended_image = Image.open(filenames[indices[0][i]])
                st.image(recommended_image, use_column_width=True)
else:
    st.write("Please upload an image to get recommendations.")
