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

# Modified process_image function to include disc space detection and overlay
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
    # Detect spaces and create overlaid image
    spaces = detect_disc_spaces(enhanced_img)
    overlaid_img = overlay_disc_spaces(enhanced_img, spaces)
    return original_img, preprocessed_img, enhanced_img, overlaid_img, spaces  # Added overlaid_img and spaces

# ... [Keep all existing functions unchanged until Streamlit app] ...

# Modified create_zip_file to include overlaid image
def create_zip_file(images_data, image_names):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for name, (orig, prep, enh, overlaid, _) in zip(image_names, images_data):  # Unpack new structure
            zip_file.writestr(f"{name}_original.png", image_to_bytes(orig))
            zip_file.writestr(f"{name}_preprocessed.png", image_to_bytes(prep))
            zip_file.writestr(f"{name}_enhanced.png", image_to_bytes(enh))
            zip_file.writestr(f"{name}_overlaid.png", image_to_bytes(overlaid))  # Add overlaid image
    zip_buffer.seek(0)
    return zip_buffer

# Streamlit app modifications
# ... [Keep initial setup and CSS unchanged] ...

if uploaded_file is not None:
    # ... [Keep ZIP extraction logic unchanged] ...
    
    # Store all processed images and data for ZIP download
    images_data = []  # Now stores (orig, prep, enh, overlaid, spaces)
    
    # Process and display images
    progress_bar = st.progress(0)
    for i, (image, name) in enumerate(zip(images_to_process, image_names)):
        st.markdown(f'<div class="subheader">Processing: {name}</div>', unsafe_allow_html=True)
        with st.spinner(f"Enhancing {name}..."):
            # Updated to receive overlaid_img and spaces
            original_img, preprocessed_img, enhanced_img, overlaid_img, spaces = process_image(image, n_clusters, optimize_large_images)
            images_data.append((original_img, preprocessed_img, enhanced_img, overlaid_img, spaces))
        
        # Display images in columns
        cols = st.columns(3)
        with cols[0]:
            st.image(original_img, use_column_width=True)
            st.markdown('<div class="caption">Original Image</div>', unsafe_allow_html=True)
        with cols[1]:
            st.image(preprocessed_img, use_column_width=True)
            st.markdown('<div class="caption">Preprocessed (CLAHE)</div>', unsafe_allow_html=True)
        with cols[2]:
            placeholder = st.empty()
            status_placeholder = st.empty()
            if st.session_state["enable_disc_space_feature"]:
                # Use precomputed overlaid image and spaces
                if st.session_state.get(f"show_spaces_{i}"):
                    placeholder.image(overlaid_img, use_column_width=True, channels="RGB")
                    if spaces:
                        status_placeholder.markdown(f'<div class="status">Detected {len(spaces)} disc spaces (cyan: normal, red: narrowed).</div>', unsafe_allow_html=True)
                    else:
                        status_placeholder.warning("No disc spaces detected.")
                else:
                    placeholder.image(enhanced_img, use_column_width=True)
                    status_placeholder.markdown('<div class="status">Click the button to detect disc spaces.</div>', unsafe_allow_html=True)
                # Toggle button
                if st.button(f"Show Detected Spaces for {name}", key=f"disc_space_btn_{i}"):
                    st.session_state[f"show_spaces_{i}"] = not st.session_state.get(f"show_spaces_{i}", False)
            else:
                placeholder.image(enhanced_img, use_column_width=True)
                status_placeholder.markdown('<div class="status">Disc space detection is disabled.</div>', unsafe_allow_html=True)
            st.markdown('<div class="caption">Enhanced (K-Means)</div>', unsafe_allow_html=True)
        
        # Download button for individual enhanced image (unchanged)
        enhanced_bytes = image_to_bytes(enhanced_img)
        st.download_button(
            label=f"Download Enhanced {name}",
            data=enhanced_bytes,
            file_name=f"enhanced_{name}.png",
            mime="image/png"
        )
        progress_bar.progress((i + 1) / len(images_to_process))
    
    # Download ZIP now includes overlaid images
    if images_data:
        zip_buffer = create_zip_file(images_data, image_names)
        st.download_button(
            label="Download All Images as ZIP",
            data=zip_buffer,
            file_name="processed_images.zip",
            mime="application/zip"
        )
    
    st.success("Processing complete!")
else:
    st.info("Please upload an image or ZIP file.")
