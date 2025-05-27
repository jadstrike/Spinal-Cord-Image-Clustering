import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io
import zipfile
import pandas as pd
import os
import base64

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

# Function to convert image to bytes for download
def image_to_bytes(image):
    img_pil = Image.fromarray(image)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()

# Process a single image and return all stages
def process_image(image, n_clusters, optimize_large_images):
    img_array = np.array(image.convert('L'))
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    original_img = img_array.copy()
    if optimize_large_images:
        original_img = resize_image(original_img, max_size=1000)
        img_array = resize_image(img_array, max_size=1000)
    preprocessed_img = preprocess_image(img_array)
    clustered_img = enhance_image_kmeans(preprocessed_img, n_clusters)
    enhanced_img = blend_images(preprocessed_img, clustered_img, alpha=0.7)
    return original_img, preprocessed_img, enhanced_img

# Streamlit app
st.set_page_config(page_title="X-ray Image Enhancer", layout="wide")

# Custom CSS for improved styling
st.markdown("""
    <style>
    .main { padding: 20px; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; }
    .stSlider { margin-bottom: 20px; }
    .stCheckbox { margin-bottom: 20px; }
    .image-container { text-align: center; }
    .caption { font-size: 14px; color: #555; margin-top: 5px; }
    .header { font-size: 24px; font-weight: bold; margin-bottom: 20px; }
    .subheader { font-size: 18px; font-weight: bold; margin-top: 20px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.header("X-ray Image Enhancer")
    st.markdown("Upload an X-ray image or a ZIP file and adjust settings to enhance images.")
    uploaded_file = st.file_uploader("Upload X-ray Image or ZIP (JPG/PNG)", type=["jpg", "png", "zip"], help="Upload a single image or a ZIP file containing multiple X-ray images.")
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=12, value=8, help="Adjust the number of clusters for K-Means clustering.")
    optimize_large_images = st.checkbox("Optimize for Large Images", value=False, help="Resize large images to speed up processing while preserving quality.")
    if optimize_large_images:
        st.info("Images will be resized to a maximum dimension of 1000 pixels.")

# Main content
st.markdown('<div class="header">X-ray Image Enhancement with K-Means Clustering</div>', unsafe_allow_html=True)
st.markdown("Enhance X-ray images using CLAHE and K-Means clustering for better visualization.")

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
    progress_bar = st.progress(0)
    for i, (image, name) in enumerate(zip(images_to_process, image_names)):
        st.markdown(f'<div class="subheader">Processing: {name}</div>', unsafe_allow_html=True)
        with st.spinner(f"Enhancing {name}..."):
            original_img, preprocessed_img, enhanced_img = process_image(image, n_clusters, optimize_large_images)
        
        # Display images in columns
        cols = st.columns(3)
        with cols[0]:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(original_img, use_column_width=True)
            st.markdown('<div class="caption">Original Image</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(preprocessed_img, use_column_width=True)
            st.markdown('<div class="caption">Preprocessed (CLAHE)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(enhanced_img, use_column_width=True)
            st.markdown('<div class="caption">Enhanced (K-Means)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Download button for enhanced image
        enhanced_bytes = image_to_bytes(enhanced_img)
        st.download_button(
            label=f"Download Enhanced {name}",
            data=enhanced_bytes,
            file_name=f"enhanced_{name}",
            mime="image/png",
            help="Download the enhanced image as a PNG file."
        )
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(images_to_process))
    
    st.success("Image processing completed!")
else:
    st.info("Please upload an X-ray image or a ZIP file to start enhancing.")
