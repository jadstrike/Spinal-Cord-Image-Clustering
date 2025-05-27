import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io
import zipfile
import pandas as pd
import os
import shutil # Added for directory cleanup

# Function to resize large images while preserving aspect ratio
def resize_image(image, max_size=1000):
    """Resizes large images while preserving aspect ratio."""
    h, w = image.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

# Function to preprocess the image with CLAHE
def preprocess_image(image):
    """Applies CLAHE for adaptive contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return enhanced

# Function to apply K-Means clustering for image enhancement
def enhance_image_kmeans(image, n_clusters=8):
    """Applies K-Means clustering for image segmentation/enhancement."""
    pixel_values = image.reshape(-1, 1)
    unique_pixels = np.unique(pixel_values)
    actual_n_clusters = min(n_clusters, len(unique_pixels))
    if actual_n_clusters < n_clusters:
        st.warning(f"Reduced number of clusters to {actual_n_clusters} due to limited unique pixel values.")
    if actual_n_clusters < 2:
        st.error("Image has too few unique pixel values for clustering.")
        return image
    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10) # Set n_init
    kmeans.fit(pixel_values)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    segmented_pixels = centers[labels].reshape(image.shape)
    return cv2.normalize(segmented_pixels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Function to blend original and clustered image
def blend_images(original, clustered, alpha=0.7):
    """Blends the original and clustered images."""
    original = original.astype(np.float32)
    clustered = clustered.astype(np.float32)
    blended = alpha * clustered + (1 - alpha) * original
    blended = cv2.normalize(blended, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return blended

# Function to detect feature lines using Canny edge detection
def detect_feature_lines(image):
    """Detects edges using Canny edge detection."""
    edges = cv2.Canny(image, 80, 200)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    return edges

# Function to overlay feature boxes with color-coded labels
def overlay_feature_boxes(enhanced_img, feature_lines, narrow_threshold=10):
    """Finds contours, measures width, and draws color-coded boxes."""
    enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(feature_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for contour in contours:
        # Filter out small contours (noise) - adjust 100 as needed
        if cv2.contourArea(contour) > 100:
            count += 1
            x, y, w, h = cv2.boundingRect(contour)

            # --- Color-Coding Logic ---
            # Use the smaller dimension as a proxy for "narrowness"
            min_dimension = min(w, h)

            if min_dimension < narrow_threshold:
                color = (0, 0, 255)  # Red for Critical (BGR)
                label = "Critical"
            else:
                color = (255, 0, 0)  # Blue for Normal (BGR)
                label = "Normal"
            # --- End Color-Coding Logic ---

            # Draw the bounding box
            cv2.rectangle(enhanced_rgb, (x, y), (x + w, y + h), color, 2)

            # Add the label
            label_text = f"{label} {count}"
            text_pos_x = x
            text_pos_y = y - 10
            # Ensure text is within image bounds
            if text_pos_y < 10:
                text_pos_y = y + h + 20

            cv2.putText(enhanced_rgb, label_text, (text_pos_x, text_pos_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    return enhanced_rgb

# Process a single image and return all stages
def process_image(image, n_clusters, optimize_large_images, narrow_threshold):
    """Processes a single image through all stages."""
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
    feature_lines_edges = detect_feature_lines(enhanced_img)
    # Pass narrow_threshold to the overlay function
    final_img_with_boxes = overlay_feature_boxes(enhanced_img, feature_lines_edges, narrow_threshold)
    return original_img, preprocessed_img, enhanced_img, final_img_with_boxes

# --- Streamlit app ---
st.set_page_config(layout="wide")
st.title("X-ray Image Enhancement & Gap/Distance Analysis")

# Sidebar for controls
st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("Upload X-ray Image(s) (JPG/PNG/ZIP)", type=["jpg", "png", "zip"])
n_clusters = st.sidebar.slider("Number of Clusters (K-Means)", min_value=2, max_value=12, value=8)

# Add the new slider for narrow threshold
narrow_threshold = st.sidebar.slider(
    "Critical Width Threshold (pixels)",
    min_value=1, max_value=50, value=10,
    help="Regions with a minimum dimension below this value will be 'Critical' (Red), others 'Normal' (Blue)."
)

optimize_large_images = st.sidebar.checkbox("Optimize for Large Images", value=True)
if optimize_large_images:
    st.sidebar.info("Resizes images >1000px for faster processing.")

# Main area for display
if uploaded_file is not None:
    images_to_process = []
    image_names = []
    temp_dir = 'temp_images'

    # Ensure temp dir is clean before use
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    # Process uploads
    if uploaded_file.name.endswith('.zip'):
        try:
            os.makedirs(temp_dir)
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            image_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.lower().endswith(('.jpg', '.png'))]
            images_to_process = [Image.open(f) for f in image_files]
            image_names = [os.path.basename(f) for f in image_files]
            if not images_to_process:
                st.error("No valid JPG or PNG images found in the ZIP file.")
        except zipfile.BadZipFile:
            st.error("The uploaded file is not a valid ZIP archive.")
        except Exception as e:
            st.error(f"An error occurred while processing the ZIP file: {e}")
    else:
        try:
            images_to_process = [Image.open(uploaded_file)]
            image_names = [uploaded_file.name]
        except Exception as e:
            st.error(f"An error occurred while opening the image: {e}")

    # Process and display images if any were loaded
    if images_to_process:
        with st.spinner("Processing images..."):
            for i, (image, name) in enumerate(zip(images_to_process, image_names)):
                st.markdown(f"---")
                st.write(f"### Processing: {name}")
                try:
                    # Pass narrow_threshold to process_image
                    original_img, preprocessed_img, enhanced_img, final_img_with_boxes = process_image(image, n_clusters, optimize_large_images, narrow_threshold)

                    st.subheader(f"Processing Stages for {name}")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.image(original_img, caption="1. Original", use_column_width=True)
                    with col2:
                        st.image(preprocessed_img, caption="2. Preprocessed (CLAHE)", use_column_width=True)
                    with col3:
                        st.image(enhanced_img, caption="3. Enhanced (K-Means)", use_column_width=True)
                    with col4:
                        st.image(final_img_with_boxes, caption="4. Final with Color-Coded Boxes", use_column_width=True)
                except Exception as e:
                    st.error(f"Failed to process {name}: {e}")
                    st.image(np.array(image.convert('L')), caption=f"Original {name} (Processing Failed)", use_column_width=True)

        # Clean up temp_images after processing
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
else:
    st.info("Please upload an X-ray image or a ZIP file to begin.")
