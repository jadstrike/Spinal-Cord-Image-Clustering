import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io

# Function to resize large images while preserving aspect ratio
def resize_image(image, max_size=1000):
    h, w = image.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

# Function to preprocess the image with edge-preserving filter
def preprocess_image(image):
    # Step 1: Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Step 2: Apply histogram equalization to enhance contrast
    equalized = cv2.equalizeHist(filtered)
    
    return equalized

# Function to apply K-Means clustering for image enhancement
def enhance_image_kmeans(image, n_clusters=8):
    # Reshape image to a 1D array of pixels
    pixel_values = image.reshape(-1, 1)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixel_values)
    
    # Get cluster labels and centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # Create segmented image by mapping pixels to cluster centers
    segmented_pixels = centers[labels].reshape(image.shape)
    segmented_image = cv2.normalize(segmented_pixels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return segmented_image, labels

# Function to blend original and clustered image for better detail preservation
def blend_images(original, clustered, alpha=0.7):
    # Ensure both images are in the same format
    original = original.astype(np.float32)
    clustered = clustered.astype(np.float32)
    
    # Blend: alpha * clustered + (1 - alpha) * original
    blended = alpha * clustered + (1 - alpha) * original
    blended = cv2.normalize(blended, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return blended

# Function to detect and label fracture lines
def detect_and_label_fracture_lines(image):
    # Apply Canny edge detection to find potential fracture lines
    edges = cv2.Canny(image, 100, 200)
    
    # Dilate edges to make fracture lines more visible and connect small gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Perform connected component analysis to label distinct fracture lines
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
    
    return edges, num_labels, labels, stats, centroids

# Function to overlay fracture lines with labels on the enhanced image
def overlay_fracture_lines(enhanced_img, fracture_lines, num_labels, labels, stats, centroids):
    # Create a color version of the enhanced image (grayscale to RGB)
    enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)
    
    # Highlight fracture lines in red and add labels
    for i in range(1, num_labels):  # Skip background (label 0)
        # Check if the component is significant (e.g., area > 50 pixels to filter noise)
        if stats[i, cv2.CC_STAT_AREA] > 50:
            # Get centroid of the fracture line
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            
            # Highlight the fracture line in red
            enhanced_rgb[labels == i] = [255, 0, 0]  # Red color for fracture lines
            
            # Add a label near the centroid
            label_text = f"Fracture {i}"
            cv2.putText(enhanced_rgb, label_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    return enhanced_rgb

# Streamlit app
st.title("X-ray Image Enhancement with K-Means Clustering (Fracture Line Detection)")

# File uploader
uploaded_file = st.file_uploader("Upload an X-ray image (JPG/PNG)", type=["jpg", "png"])

# Number of clusters input
n_clusters = st.slider("Number of Clusters", min_value=2, max_value=12, value=8)

# Option to optimize for large images
optimize_large_images = st.checkbox("Optimize for Large Images (faster processing)", value=False)
if optimize_large_images:
    st.info("Images will be resized to a maximum dimension of 1000 pixels to improve performance while preserving quality.")

if uploaded_file is not None:
    # Read and display original image
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert('L'))  # Convert to grayscale
    
    # Ensure img_array is uint8
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    
    # Resize if optimize option is selected
    if optimize_large_images:
        img_array = resize_image(img_array, max_size=1000)
    
    # Display original image
    st.image(img_array, caption="Original X-ray Image", width=None)
    
    # Enhance image when button is clicked
    if st.button("Enhance Image"):
        with st.spinner("Enhancing image and detecting fracture lines..."):
            # Preprocess image
            preprocessed_img = preprocess_image(img_array)
            
            # Ensure preprocessed_img is uint8
            if preprocessed_img.dtype != np.uint8:
                preprocessed_img = preprocessed_img.astype(np.uint8)
            
            # Display preprocessed image
            st.image(preprocessed_img, caption="Preprocessed (Contrast Enhanced)", width=None)
            
            # Enhance image using K-Means
            clustered_img, labels = enhance_image_kmeans(preprocessed_img, n_clusters)
            
            # Blend the clustered image with the preprocessed image
            blended_img = blend_images(preprocessed_img, clustered_img, alpha=0.7)
            
            # Detect and label fracture lines
            fracture_lines, num_labels, fracture_labels, stats, centroids = detect_and_label_fracture_lines(blended_img)
            
            # Overlay fracture lines with labels on the enhanced image
            final_img_with_fractures = overlay_fracture_lines(blended_img, fracture_lines, num_labels, fracture_labels, stats, centroids)
            
            # Display enhanced image with labeled fracture lines
            st.image(final_img_with_fractures, caption=f"Enhanced X-ray Image with Labeled Fracture Lines ({n_clusters} Clusters)", width=None)
            
            # Display fracture lines separately
            st.image(fracture_lines, caption="Detected Fracture Lines", width=None)
            
            # Provide download link
            enhanced_image = Image.fromarray(final_img_with_fractures)
            img_byte_arr = io.BytesIO()
            enhanced_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            st.download_button(
                label="Download Enhanced Image with Labeled Fracture Lines",
                data=img_byte_arr,
                file_name="enhanced_xray_with_fractures.png",
                mime="image/png"
            )
else:
    st.info("Please upload an X-ray image to begin.")
