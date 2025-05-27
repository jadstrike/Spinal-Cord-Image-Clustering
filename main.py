import streamlit as st
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
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

# Function to preprocess the image with advanced techniques
def preprocess_image(image):
    # Apply non-local means denoising to reduce noise while preserving details
    denoised = cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    return enhanced

# Function to apply Gaussian Mixture Model for adaptive clustering
def enhance_image_gmm(image, n_components=5):
    pixel_values = image.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(pixel_values)
    labels = gmm.predict(pixel_values)
    centers = gmm.means_
    segmented_pixels = centers[labels].reshape(image.shape)
    segmented_image = cv2.normalize(segmented_pixels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return segmented_image

# Function to blend original and clustered image
def blend_images(original, clustered, alpha=0.5):
    original = original.astype(np.float32)
    clustered = clustered.astype(np.float32)
    blended = alpha * clustered + (1 - alpha) * original
    blended = cv2.normalize(blended, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return blended

# Function to apply multi-scale unsharp masking
def multi_scale_unsharp_mask(image):
    # Small scale (fine details)
    blurred_small = cv2.GaussianBlur(image, (3, 3), 0)
    sharpened_small = image + (image - blurred_small) * 1.5
    
    # Medium scale (broader edges)
    blurred_medium = cv2.GaussianBlur(image, (5, 5), 0)
    sharpened_medium = image + (image - blurred_medium) * 1.5
    
    # Combine scales with weighting
    combined = cv2.addWeighted(sharpened_small, 0.6, sharpened_medium, 0.4, 0)
    return np.clip(combined, 0, 255).astype(np.uint8)

# Streamlit app
st.title("Advanced X-ray Image Enhancement")

# File uploader
uploaded_file = st.file_uploader("Upload an X-ray image (JPG/PNG)", type=["jpg", "png"])

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
    
    # Step 1: Display original image
    st.image(img_array, caption="Original X-ray Image", width=None)
    
    # Enhance image when button is clicked
    if st.button("Enhance Image"):
        with st.spinner("Enhancing image..."):
            # Step 2: Preprocess image
            preprocessed_img = preprocess_image(img_array)
            
            # Ensure preprocessed_img is uint8
            if preprocessed_img.dtype != np.uint8:
                preprocessed_img = preprocessed_img.astype(np.uint8)
            
            # Display preprocessed image
            st.image(preprocessed_img, caption="Preprocessed Image (Advanced Algorithm)", width=None)
            
            # Step 3: Enhance image using GMM
            clustered_img = enhance_image_gmm(preprocessed_img)
            
            # Blend the clustered image with the preprocessed image
            blended_img = blend_images(preprocessed_img, clustered_img, alpha=0.5)
            
            # Apply multi-scale unsharp masking
            final_img = multi_scale_unsharp_mask(blended_img)
            
            # Display final enhanced image
            st.image(final_img, caption="Final Enhanced Image", width=None)
            
            # Step 4: Display all images in one line
            st.subheader("Summary: All Stages in One Line")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(img_array, caption="Original", width=None)
            with col2:
                st.image(preprocessed_img, caption="Preprocessed", width=None)
            with col3:
                st.image(final_img, caption="Final Enhanced", width=None)
            
            # Provide download link for the final enhanced image
            enhanced_image = Image.fromarray(final_img)
            img_byte_arr = io.BytesIO()
            enhanced_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            st.download_button(
                label="Download Final Enhanced Image",
                data=img_byte_arr,
                file_name="enhanced_xray.png",
                mime="image/png"
            )
else:
    st.info("Please upload an X-ray image to begin.")
