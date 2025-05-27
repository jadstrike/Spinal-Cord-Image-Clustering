import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io
import zipfile
import pandas as pd
import os

# Function to resize large images while preserving aspect ratio
def resize_image(image, max_size=1000):
    h, w = image.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

# Function to preprocess the image with CLAHE
def preprocess_image(image):
    # Apply CLAHE for adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return enhanced

# Function to apply K-Means clustering for image enhancement
def enhance_image_kmeans(image, n_clusters=8):
    pixel_values = image.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixel_values)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    segmented_pixels = centers[labels].reshape(image.shape)
    return cv2.normalize(segmented_pixels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Function to blend original and clustered image
def blend_images(original, clustered, alpha=0.7):
    original = original.astype(np.float32)
    clustered = clustered.astype(np.float32)
    blended = alpha * clustered + (1 - alpha) * original
    blended = cv2.normalize(blended, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return blended

# Process a single image and return all stages
def process_image(image, n_clusters, optimize_large_images):
    img_array = np.array(image.convert('L'))
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    original_img = img_array.copy()  # Store original image
    if optimize_large_images:
        original_img = resize_image(original_img, max_size=1000)
        img_array = resize_image(img_array, max_size=1000)
    # Preprocessed image
    preprocessed_img = preprocess_image(img_array)
    # Enhanced image using K-Means
    clustered_img = enhance_image_kmeans(preprocessed_img, n_clusters)
    # Blend the clustered image with the preprocessed image
    enhanced_img = blend_images(preprocessed_img, clustered_img, alpha=0.7)
    return original_img, preprocessed_img, enhanced_img

# Streamlit app
st.title("X-ray Image Enhancement with K-Means Clustering")

# File uploader
uploaded_file = st.file_uploader("Upload an X-ray image or a ZIP file containing X-ray images (JPG/PNG)", type=["jpg", "png", "zip"])

# Number of clusters input
n_clusters = st.slider("Number of Clusters", min_value=2, max_value=12, value=8)

# Option to optimize for large images
optimize_large_images = st.checkbox("Optimize for Large Images (faster processing)", value=False)
if optimize_large_images:
    st.info("Images will be resized to a maximum dimension of 1000 pixels to improve performance while preserving quality.")

if uploaded_file is not None:
    # Check if the uploaded file is a ZIP file
    if uploaded_file.name.endswith('.zip'):
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall('temp_images')
        image_files = [os.path.join('temp_images', f) for f in os.listdir('temp_images') if f.endswith(('.jpg', '.png'))]
        images_to_process = [Image.open(f) for f in image_files]
        image_names = [os.path.basename(f) for f in image_files]
    else:
        images_to_process = [Image.open(uploaded_file)]
        image_names = ["Uploaded Image"]
    
    # Process and display images
    with st.spinner("Enhancing images..."):
        for i, (image, name) in enumerate(zip(images_to_process, image_names)):
            st.write(f"Processing {name}...")
            original_img, preprocessed_img, enhanced_img = process_image(image, n_clusters, optimize_large_images)
            # Display final summary in one line
            st.subheader(f"Summary: All Stages for {name}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(original_img, caption="Original", width=None)
            with col2:
                st.image(preprocessed_img, caption="Preprocessed", width=None)
            with col3:
                st.image(enhanced_img, caption="Enhanced", width=None)
else:
    st.info("Please upload an X-ray image or a ZIP file containing X-ray images to begin.")
