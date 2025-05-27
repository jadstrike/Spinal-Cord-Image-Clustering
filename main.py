import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io
import zipfile
import pandas as pd
import os
from scipy.ndimage import label as scipy_label

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

# Function to detect and label fracture lines
def detect_and_label_fracture_lines(image):
    # Simplified Canny edge detection with adjusted thresholds
    edges = cv2.Canny(image, 80, 200)  # Increased lower threshold for better sensitivity
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)  # Increased dilation for thicker lines
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
    return edges, num_labels, labels, stats, centroids

# Function to overlay fracture lines with improved labels
def overlay_fracture_lines(enhanced_img, fracture_lines, num_labels, labels, stats, centroids):
    enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)
    label_positions = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 50:
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            overlap = False
            for pos in label_positions:
                dist = np.sqrt((cx - pos[0])**2 + (cy - pos[1])**2)
                if dist < 40:  # Increased distance to reduce overlap
                    overlap = True
                    cy += 30  # Increased offset for clarity
                    break
            label_positions.append((cx, cy))
            # Highlight fracture lines with higher intensity
            enhanced_rgb[labels == i] = [255, 50, 50]  # Brighter red for better visibility
            # Add label with improved visibility
            severity = "Mild" if stats[i, cv2.CC_STAT_AREA] < 500 else "Moderate" if stats[i, cv2.CC_STAT_AREA] < 1000 else "Severe"
            label_text = f"Fracture {i} (Severity: {severity})"
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3)
            # Semi-transparent black background for better contrast
            overlay = enhanced_rgb.copy()
            cv2.rectangle(overlay, (cx, cy - text_height - baseline), (cx + text_width, cy + baseline), (0, 0, 0), -1)
            alpha = 0.6  # Transparency for background
            cv2.addWeighted(overlay, alpha, enhanced_rgb, 1 - alpha, 0, enhanced_rgb)
            cv2.putText(enhanced_rgb, label_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3, cv2.LINE_AA)
    return enhanced_rgb

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
    # Detect and label fracture lines
    fracture_lines, num_labels, fracture_labels, stats, centroids = detect_and_label_fracture_lines(enhanced_img)
    final_img_with_fractures = overlay_fracture_lines(enhanced_img, fracture_lines, num_labels, fracture_labels, stats, centroids)
    return original_img, preprocessed_img, enhanced_img, final_img_with_fractures

# Streamlit app
st.title("X-ray Image Enhancement with K-Means Clustering (Fracture Line Detection)")

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
    with st.spinner("Enhancing images and detecting fracture lines..."):
        for i, (image, name) in enumerate(zip(images_to_process, image_names)):
            st.write(f"Processing {name}...")
            original_img, preprocessed_img, enhanced_img, final_img_with_fractures = process_image(image, n_clusters, optimize_large_images)
            # Display all stages
            st.image(original_img, caption=f"Original {name}", width=None)
            st.image(preprocessed_img, caption=f"Preprocessed {name}", width=None)
            st.image(enhanced_img, caption=f"Enhanced {name} (Before Labeling)", width=None)
            st.image(final_img_with_fractures, caption=f"Enhanced {name} with Labeled Fracture Lines", width=None)
else:
    st.info("Please upload an X-ray image or a ZIP file containing X-ray images to begin.")
