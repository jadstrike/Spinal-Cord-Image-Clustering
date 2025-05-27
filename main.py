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

# Function to preprocess the image conservatively
def preprocess_image(image):
    # Denoise to reduce noise while preserving details
    denoised = cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)
    # Apply histogram equalization for balanced contrast
    equalized = cv2.equalizeHist(denoised)
    return equalized

# Function to enhance with K-Means clustering
def enhance_image_kmeans(image, n_clusters=8):
    pixel_values = image.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixel_values)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    segmented_pixels = centers[labels].reshape(image.shape)
    return cv2.normalize(segmented_pixels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Function to blend images
def blend_images(original, clustered, alpha=0.3):
    original = original.astype(np.float32)
    clustered = clustered.astype(np.float32)
    blended = alpha * clustered + (1 - alpha) * original
    return cv2.normalize(blended, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Function for subtle sharpening
def sharpen_image(image, amount=1.0):
    kernel = np.array([[0, -1, 0], [-1, 5 + amount, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

# Streamlit app
st.title("X-ray Image Enhancement for Medical Readability")

# File uploader
uploaded_file = st.file_uploader("Upload an X-ray image (JPG/PNG)", type=["jpg", "png"])

# Option to optimize for large images
optimize_large_images = st.checkbox("Optimize for Large Images (faster processing)", value=False)
if optimize_large_images:
    st.info("Images will be resized to a maximum dimension of 1000 pixels for better performance.")

# Adjustable parameters
n_clusters = st.slider("Number of Clusters", min_value=3, max_value=10, value=8, step=1)
sharpen_amount = st.slider("Sharpening Amount", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

if uploaded_file is not None:
    # Load and convert image to grayscale
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert('L')).astype(np.uint8)

    # Resize if selected
    if optimize_large_images:
        img_array = resize_image(img_array, max_size=1000)

    # Display original image
    st.image(img_array, caption="Original X-ray Image")

    # Enhance image on button click
    if st.button("Enhance Image"):
        with st.spinner("Enhancing image..."):
            # Preprocess
            preprocessed_img = preprocess_image(img_array)
            st.image(preprocessed_img, caption="Preprocessed Image")

            # Enhance with K-Means
            clustered_img = enhance_image_kmeans(preprocessed_img, n_clusters)
            blended_img = blend_images(preprocessed_img, clustered_img, alpha=0.3)
            final_img = sharpen_image(blended_img, amount=sharpen_amount)

            # Display final enhanced image
            st.image(final_img, caption=f"Final Enhanced Image ({n_clusters} Clusters)")

            # Show all stages side by side
            st.subheader("Summary: All Stages in One Line")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(img_array, caption="Original")
            with col2:
                st.image(preprocessed_img, caption="Preprocessed")
            with col3:
                st.image(final_img, caption="Final Enhanced")

            # Download option
            enhanced_image = Image.fromarray(final_img)
            img_byte_arr = io.BytesIO()
            enhanced_image.save(img_byte_arr, format='PNG')
            st.download_button(
                label="Download Final Enhanced Image",
                data=img_byte_arr.getvalue(),
                file_name="enhanced_xray.png",
                mime="image/png"
            )
else:
    st.info("Please upload an X-ray image to begin.")
