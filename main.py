import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io
import zipfile
import os
import shutil

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

# Function to apply K-Means clustering (can help with segmentation)
def enhance_image_kmeans(image, n_clusters=8):
    """Applies K-Means clustering."""
    pixel_values = image.reshape(-1, 1)
    unique_pixels = np.unique(pixel_values)
    actual_n_clusters = min(n_clusters, len(unique_pixels))
    if actual_n_clusters < n_clusters:
        st.warning(f"Reduced clusters to {actual_n_clusters}.")
    if actual_n_clusters < 2:
        st.error("Too few unique pixels for clustering.")
        return image
    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixel_values)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    segmented_pixels = centers[labels].reshape(image.shape)
    return cv2.normalize(segmented_pixels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Function to blend original and clustered image
def blend_images(original, clustered, alpha=0.7):
    """Blends images."""
    original = original.astype(np.float32)
    clustered = clustered.astype(np.float32)
    blended = alpha * clustered + (1 - alpha) * original
    blended = cv2.normalize(blended, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return blended

# --- NEW FUNCTION: Segment and Color Spaces ---
def segment_and_color_spaces(enhanced_img, binarization_threshold, narrow_distance_threshold):
    """Segments image, finds spaces, and colors them based on width."""

    # 1. Binarization: Separate bones (white) from spaces (black)
    # This is a CRITICAL step. Otsu tries to find an optimal threshold automatically.
    # You might need to use the manual `binarization_threshold` or
    # even `cv2.adaptiveThreshold` for better results.
    # _, binary_img = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary_img = cv2.threshold(enhanced_img, binarization_threshold, 255, cv2.THRESH_BINARY)

    # 2. Morphology: Clean up the binary image (optional, tune as needed)
    kernel = np.ones((3, 3), np.uint8)
    # Opening: Remove small noise/islands (potential false spaces)
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)
    # Closing: Fill small holes in bones
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3. Invert for Distance Transform (Spaces = white = 255)
    spaces_img = cv2.bitwise_not(binary_img)

    # 4. Distance Transform: Calculate distance from each space pixel to nearest bone
    dist_transform = cv2.distanceTransform(spaces_img, cv2.DIST_L2, 5)

    # 5. Create Color Overlay
    enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)
    overlay = np.zeros_like(enhanced_rgb, dtype=np.uint8)

    # Find where distance is below threshold (critical) - exclude 0 distance (bones)
    critical_mask = (dist_transform > 0) & (dist_transform < narrow_distance_threshold)
    # Find where distance is above or equal (normal)
    normal_mask = (dist_transform >= narrow_distance_threshold)

    # Apply colors (BGR format)
    overlay[normal_mask] = [255, 0, 0]  # Blue for Normal
    overlay[critical_mask] = [0, 0, 255]  # Red for Critical

    # 6. Blend the overlay with the enhanced image
    alpha = 0.6  # Transparency - adjust as needed
    final_img = cv2.addWeighted(overlay, alpha, enhanced_rgb, 1 - alpha, 0)

    # Optionally, draw bone contours for clarity
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(final_img, contours, -1, (220, 220, 220), 1) # Light gray outlines

    # Return the binary mask too, for debugging/understanding
    return final_img, binary_img, dist_transform

# Process a single image and return all stages
def process_image(image, n_clusters, optimize_large_images, binarization_threshold, narrow_distance_threshold):
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

    # Call the new function
    final_img_with_spaces, binary_debug, dist_debug = segment_and_color_spaces(
        enhanced_img, binarization_threshold, narrow_distance_threshold
    )

    return original_img, enhanced_img, binary_debug, final_img_with_spaces


# --- Streamlit app ---
st.set_page_config(layout="wide")
st.title("X-ray Image Enhancement & Inter-Bone Space Analysis")

# Sidebar for controls
st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("Upload X-ray Image(s) (JPG/PNG/ZIP)", type=["jpg", "png", "zip"])
n_clusters = st.sidebar.slider("Number of Clusters (K-Means)", min_value=2, max_value=12, value=4, help="Adjusts K-Means enhancement, can aid segmentation.")

# --- NEW SLIDERS ---
st.sidebar.markdown("---")
st.sidebar.subheader("Space Analysis Settings")
binarization_threshold = st.sidebar.slider(
    "Binarization Threshold (Bone Intensity)",
    min_value=1, max_value=254, value=128,
    help="Pixels above this value are considered 'Bone'. Adjust this to correctly separate bones from spaces."
)
narrow_distance_threshold = st.sidebar.slider(
    "Critical Distance Threshold (pixels)",
    min_value=1, max_value=50, value=5,
    help="Spaces narrower than this (in pixels) will be 'Critical' (Red)."
)
# --- END NEW SLIDERS ---

st.sidebar.markdown("---")
optimize_large_images = st.sidebar.checkbox("Optimize for Large Images", value=True)
if optimize_large_images:
    st.sidebar.info("Resizes images >1000px for faster processing.")

# Main area for display
if uploaded_file is not None:
    images_to_process = []
    image_names = []
    temp_dir = 'temp_images'

    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

    if uploaded_file.name.endswith('.zip'):
        try:
            os.makedirs(temp_dir)
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            image_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.lower().endswith(('.jpg', '.png'))]
            images_to_process = [Image.open(f) for f in image_files]
            image_names = [os.path.basename(f) for f in image_files]
            if not images_to_process: st.error("No valid images found in ZIP.")
        except Exception as e: st.error(f"ZIP error: {e}")
    else:
        try:
            images_to_process = [Image.open(uploaded_file)]
            image_names = [uploaded_file.name]
        except Exception as e: st.error(f"Image open error: {e}")

    if images_to_process:
        with st.spinner("Processing images..."):
            for i, (image, name) in enumerate(zip(images_to_process, image_names)):
                st.markdown(f"---")
                st.write(f"### Processing: {name}")
                try:
                    original_img, enhanced_img, binary_debug, final_img_with_spaces = process_image(
                        image, n_clusters, optimize_large_images, binarization_threshold, narrow_distance_threshold
                    )

                    st.subheader(f"Processing Stages for {name}")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.image(original_img, caption="1. Original", use_column_width=True)
                    with col2:
                        st.image(enhanced_img, caption="2. Enhanced", use_column_width=True)
                    with col3:
                        st.image(binary_debug, caption="3. Bone Segmentation (Debug)", use_column_width=True)
                    with col4:
                        st.image(final_img_with_spaces, caption="4. Final with Color-Coded Spaces", use_column_width=True)
                except Exception as e:
                    st.error(f"Failed to process {name}: {e}")
                    st.image(np.array(image.convert('L')), caption=f"Original {name} (Failed)", use_column_width=True)

        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
else:
    st.info("Please upload an X-ray image or a ZIP file to begin.")
