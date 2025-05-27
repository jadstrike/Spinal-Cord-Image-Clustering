import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io
import matplotlib.pyplot as plt

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
    filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    equalized = cv2.equalizeHist(filtered)
    return equalized

# Function to apply K-Means clustering for image enhancement
def enhance_image_kmeans(image, n_clusters=8):
    pixel_values = image.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixel_values)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    segmented_pixels = centers[labels].reshape(image.shape)
    segmented_image = cv2.normalize(segmented_pixels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return segmented_image, labels

# Function to blend original and clustered image
def blend_images(original, clustered, alpha=0.7):
    original = original.astype(np.float32)
    clustered = clustered.astype(np.float32)
    blended = alpha * clustered + (1 - alpha) * original
    blended = cv2.normalize(blended, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return blended

# Function to detect and label fracture lines
def detect_and_label_fracture_lines(image):
    edges = cv2.Canny(image, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
    return edges, num_labels, labels, stats, centroids

# Function to overlay fracture lines with labels
def overlay_fracture_lines(enhanced_img, fracture_lines, num_labels, labels, stats, centroids):
    enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 50:
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            enhanced_rgb[labels == i] = [255, 0, 0]
            label_text = f"Fracture {i}"
            cv2.putText(enhanced_rgb, label_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return enhanced_rgb

# Function to detect vertebrae and measure intervertebral distances
def detect_vertebrae_and_distances(image):
    # Apply adaptive thresholding to isolate bone regions
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours (vertebrae)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area (vertebrae are large structures)
    vertebrae = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:  # Adjust based on image resolution
            vertebrae.append(cnt)
    
    # Sort vertebrae by y-coordinate (top to bottom)
    vertebrae.sort(key=lambda c: cv2.boundingRect(c)[1])
    
    # Measure distances between adjacent vertebrae
    distances = []
    for i in range(len(vertebrae) - 1):
        x1, y1, w1, h1 = cv2.boundingRect(vertebrae[i])
        x2, y2, w2, h2 = cv2.boundingRect(vertebrae[i + 1])
        distance = y2 - (y1 + h1)  # Distance between bottom of one vertebra and top of the next
        distances.append((distance, (x1 + w1 // 2, y1 + h1), (x2 + w2 // 2, y2)))
    
    return distances

# Function to overlay intervertebral distances with color coding
def overlay_distances(image_rgb, distances):
    for distance, (x1, y1), (x2, y2) in distances:
        # Color coding based on distance
        if 5 <= distance <= 15:  # Normal
            color = (0, 0, 255)  # Blue
            label = "Normal"
        elif distance > 15:  # Large
            color = (0, 255, 0)  # Green
            label = "Large"
        else:  # Very Narrow
            color = (255, 0, 0)  # Red
            label = "Very Narrow"
        
        # Draw a line between the points
        cv2.line(image_rgb, (x1, y1), (x2, y2), color, 2)
        
        # Add a label with the distance and classification
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(image_rgb, f"{distance}px ({label})", (mid_x + 10, mid_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    return image_rgb

# Function to calculate fracture severity score
def calculate_fracture_severity(image, fracture_lines, stats):
    severity = 0
    total_length = 0
    for i in range(1, len(stats)):
        if stats[i, cv2.CC_STAT_AREA] > 50:
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            perimeter = 2 * (width + height)
            total_length += perimeter
            fracture_region = image[fracture_lines > 0]
            if len(fracture_region) > 0:
                intensity_diff = np.mean(image) - np.mean(fracture_region)
                severity += perimeter * intensity_diff / 255
    
    max_score = np.sqrt(image.size) * 10
    if total_length > 0:
        severity = min(10, (severity / total_length) * 10)
    else:
        severity = 0
    return severity

# Function to plot histogram
def plot_histogram(image, title, filename):
    plt.figure(figsize=(6, 4))
    plt.hist(image.ravel(), bins=256, range=(0, 255), color='gray')
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.savefig(filename)
    plt.close()

# Streamlit app
st.title("X-ray Image Enhancement with K-Means Clustering (Spinal Analysis)")

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
    
    # Plot histogram for original image
    plot_histogram(img_array, "Histogram (Original)", "original_hist.png")
    st.image("original_hist.png", caption="Histogram of Original Image", width=None)
    
    # Calculate and display contrast ratio
    contrast_ratio = np.std(img_array)
    st.write(f"Contrast Ratio (Original): {contrast_ratio:.2f}")
    
    # Enhance image when button is clicked
    if st.button("Enhance Image"):
        with st.spinner("Enhancing image and analyzing spinal structure..."):
            # Preprocess image
            preprocessed_img = preprocess_image(img_array)
            
            # Ensure preprocessed_img is uint8
            if preprocessed_img.dtype != np.uint8:
                preprocessed_img = preprocessed_img.astype(np.uint8)
            
            # Display preprocessed image
            st.image(preprocessed_img, caption="Preprocessed (Contrast Enhanced)", width=None)
            
            # Plot histogram for preprocessed image
            plot_histogram(preprocessed_img, "Histogram (Preprocessed)", "preprocessed_hist.png")
            st.image("preprocessed_hist.png", caption="Histogram of Preprocessed Image", width=None)
            
            # Calculate and display contrast ratio
            contrast_ratio_preprocessed = np.std(preprocessed_img)
            st.write(f"Contrast Ratio (Preprocessed): {contrast_ratio_preprocessed:.2f}")
            
            # Enhance image using K-Means
            clustered_img, labels = enhance_image_kmeans(preprocessed_img, n_clusters)
            
            # Blend the clustered image with the preprocessed image
            blended_img = blend_images(preprocessed_img, clustered_img, alpha=0.7)
            
            # Detect and label fracture lines
            fracture_lines, num_labels, fracture_labels, stats, centroids = detect_and_label_fracture_lines(blended_img)
            
            # Overlay fracture lines with labels on the enhanced image
            final_img_with_fractures = overlay_fracture_lines(blended_img, fracture_lines, num_labels, fracture_labels, stats, centroids)
            
            # Detect vertebrae and measure intervertebral distances
            distances = detect_vertebrae_and_distances(blended_img)
            
            # Overlay intervertebral distances with color coding
            final_img_with_distances = overlay_distances(final_img_with_fractures, distances)
            
            # Display enhanced image with fracture lines and distances
            st.image(final_img_with_distances, caption=f"Enhanced X-ray Image with Fracture Lines and Intervertebral Distances ({n_clusters} Clusters)", width=None)
            
            # Plot histogram for enhanced image
            plot_histogram(blended_img, "Histogram (Enhanced)", "enhanced_hist.png")
            st.image("enhanced_hist.png", caption="Histogram of Enhanced Image", width=None)
            
            # Calculate and display contrast ratio and edge strength
            contrast_ratio_enhanced = np.std(blended_img)
            edge_strength = np.mean(fracture_lines[fracture_lines > 0]) if np.sum(fracture_lines) > 0 else 0
            st.write(f"Contrast Ratio (Enhanced): {contrast_ratio_enhanced:.2f}")
            st.write(f"Edge Strength (Fracture Lines): {edge_strength:.2f}")
            
            # Calculate and display fracture severity score
            severity_score = calculate_fracture_severity(blended_img, fracture_lines, stats)
            st.write(f"Fracture Severity Score: {severity_score:.2f}/10")
            st.write("Interpretation: 0 = No significant fractures, 10 = Severe fractures")
            
            # Provide download link
            enhanced_image = Image.fromarray(final_img_with_distances)
            img_byte_arr = io.BytesIO()
            enhanced_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            st.download_button(
                label="Download Enhanced Image with Analysis",
                data=img_byte_arr,
                file_name="enhanced_xray_with_analysis.png",
                mime="image/png"
            )
else:
    st.info("Please upload an X-ray image to begin.")
