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

# Function to detect spaces between bones (disc spaces)
def detect_disc_spaces(image, min_space_height=5, max_space_height=20):
    # Convert to binary image for segmentation
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert binary image (bones are white, spaces are black)
    binary = cv2.bitwise_not(binary)
    # Use Canny edge detection to find bone edges
    edges = cv2.Canny(image, 50, 150)
    # Dilate edges to connect small gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours of the bones
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spaces = []
    
    # Analyze gaps between contours (potential disc spaces)
    for i in range(len(contours) - 1):
        # Get bounding boxes of consecutive contours
        x1, y1, w1, h1 = cv2.boundingRect(contours[i])
        x2, y2, w2, h2 = cv2.boundingRect(contours[i + 1])
        
        # Check if contours are vertically aligned (potential disc space)
        if abs(x1 - x2) < max(w1, w2) * 0.5:  # Ensure some overlap in x-axis
            space_height = y2 - (y1 + h1)
            if min_space_height < space_height < max_space_height:
                # Calculate the y-position and width of the space
                y_space = y1 + h1 + space_height // 2
                x_start = max(x1, x2)
                x_end = min(x1 + w1, x2 + w2)
                spaces.append((y_space, x_start, x_end, space_height))
    
    return spaces

# Function to overlay detected spaces on the image
def overlay_disc_spaces(image, spaces):
    # Convert to RGB for coloring
    img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for y, x_start, x_end, space_height in spaces:
        # Color based on space height (narrowed vs normal)
        color = (0, 255, 255) if space_height > 10 else (255, 0, 0)  # Cyan for normal, red for narrowed
        thickness = 2
        cv2.line(img_rgb, (x_start, y), (x_end, y), color, thickness)
    return img_rgb

# Function to convert image to bytes
def image_to_bytes(image):
    img_pil = Image.fromarray(image)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()

# Function to create a ZIP file of all images
def create_zip_file(images_data, image_names):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for name, (orig, prep, enh) in zip(image_names, images_data):
            zip_file.writestr(f"{name}_original.png", image_to_bytes(orig))
            zip_file.writestr(f"{name}_preprocessed.png", image_to_bytes(prep))
            zip_file.writestr(f"{name}_enhanced.png", image_to_bytes(enh))
    zip_buffer.seek(0)
    return zip_buffer

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

# Custom CSS for improved styling with even bolder captions
st.markdown("""
    <style>
    .main { padding: 20px; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; }
    .stSlider { margin-bottom: 20px; }
    .stCheckbox { margin-bottom: 20px; }
    .image-container { text-align: center; }
    .caption { font-size: 16px; color: #000; margin-top: 5px; font-weight: 800; }
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
        image_names = ["uploaded_image"]
    
    # Store all processed images for ZIP download
    images_data = []
    
    # Process and display images
    progress_bar = st.progress(0)
    for i, (image, name) in enumerate(zip(images_to_process, image_names)):
        st.markdown(f'<div class="subheader">Processing: {name}</div>', unsafe_allow_html=True)
        with st.spinner(f"Enhancing {name}..."):
            original_img, preprocessed_img, enhanced_img = process_image(image, n_clusters, optimize_large_images)
            images_data.append((original_img, preprocessed_img, enhanced_img))
        
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
            # Placeholder for the enhanced image (to be updated with spaces if toggled)
            placeholder = st.empty()
            placeholder.image(enhanced_img, use_column_width=True)
            st.markdown('<div class="caption">Enhanced (K-Means)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Button to toggle disc space visualization
        if st.button(f"Show Detected Spaces Between Bones for {name}", key=f"disc_space_{i}"):
            spaces = detect_disc_spaces(enhanced_img)
            enhanced_with_spaces = overlay_disc_spaces(enhanced_img, spaces)
            placeholder.image(enhanced_with_spaces, use_column_width=True)
        
        # Download button for individual enhanced image
        enhanced_bytes = image_to_bytes(enhanced_img)
        st.download_button(
            label=f"Download Enhanced {name}",
            data=enhanced_bytes,
            file_name=f"enhanced_{name}.png",
            mime="image/png",
            help="Download the enhanced image as a PNG file."
        )
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(images_to_process))
    
    # Download all images as a ZIP file
    if images_data:
        zip_buffer = create_zip_file(images_data, image_names)
        st.download_button(
            label="Download All Images as ZIP",
            data=zip_buffer,
            file_name="processed_images.zip",
            mime="application/zip",
            help="Download all processed images (original, preprocessed, enhanced) as a ZIP file."
        )
    
    st.success("Image processing completed!")
else:
    st.info("Please upload an X-ray image or a ZIP file to start enhancing.")
