import streamlit as st
import numpy as np
import cv2
import requests
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from PIL import Image

def sharpen_image(image):
    # Convert image to numpy array
    img_np = np.array(image)

    # Convert RGB image to BGR (OpenCV uses BGR format)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Define the sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    
    # Apply the sharpening kernel
    sharpened_image = cv2.filter2D(img_bgr, -1, kernel)
    
    # Convert back to RGB format for display
    sharpened_image_rgb = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)

    return sharpened_image_rgb
# Load pre-trained VGG16 model
vgg16 = load_model('vgg19.h5')

# Function to load and preprocess image
def load_and_preprocess_image(image):
    # Use OpenCV to read the image as a numpy array
    image = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)
    # Resize image to 224x224 (VGG16 input size) and expand dimensions
    processed_image = cv2.resize(image, (64, 64))
    processed_image = np.expand_dims(processed_image, axis=0)
    # Preprocess the image to the format VGG16 expects
    processed_image = preprocess_input(processed_image)
    return processed_image

# Function to make predictions
def predict(image):
    # Preprocess the input image
    processed_image = load_and_preprocess_image(image)
    # Make prediction using the VGG16 model
    predictions = vgg16.predict(processed_image)
    return predictions

# Load ImageNet class labels
def load_class_labels():
    class_labels = {0:'AnnualCrop',1:'Forest',2:'HerbaceousVegetation',3: 'Highway',4:'Industrial',5:'Pasture',6:'PermanentCrop',7:'Residential',8:'River',9:'SeaLake'}
    return {int(key): value for key, value in class_labels.items()}

# Streamlit app
def main():
    st.title("Satellite Image")
    st.write("Upload an image ")

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Make predictions
        if st.button("Classify"):
            with st.spinner('Classifying...'):
                predictions = predict(uploaded_image)
            st.success("Prediction completed!")
            # Load class labels
            class_labels = load_class_labels()

            # Display the top 5 predicted classes and probabilities
            st.subheader("Top 5 Predictions:")
            for i in range(5):
                class_index = np.argmax(predictions)
                class_name = class_labels[class_index]
                probability = predictions[0][class_index]
                st.write(f"{i+1}. {class_name} - Probability: {probability:.2f}")
                predictions[0][class_index] = 0.0  # Set the probability of the predicted class to 0
        if st.button("Sharpen Image"):
            with st.spinner('Sharpening...'):
                # Open and preprocess the image
                image = Image.open(uploaded_image)
                # Sharpen the image
                sharpened_image = sharpen_image(image)
            st.success("Image sharpening completed!")
            st.image(sharpened_image, caption="Sharpened Image", use_column_width=True)

if __name__ == "__main__":
    main()
