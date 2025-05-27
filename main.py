import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io
import zipfile
import os
import pandas as pd

# Function to resize large images
def resize_image(image, max_size=1000):
    h, w = image.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

# CLAHE preprocessing
def preprocess_image(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# K-Means enhancement
def enhance_image_kmeans(image, n_clusters=8):
    pixel_values = image.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixel_values)
    return centers[kmeans.labels_].reshape(image.shape).astype(np.uint8)

# Image blending
def blend_images(original, clustered, alpha=0.7):
    return cv2.addWeighted(clustered, alpha, original, 1-alpha, 0)

# Enhanced disc space detection
def detect_disc_spaces(image):
    # Preprocessing
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 21, 5)
    
    # Morphological operations
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find and sort contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    
    spaces = []
    for i in range(len(contours)-1):
        x1, y1, w1, h1 = cv2.boundingRect(contours[i])
        x2, y2, w2, h2 = cv2.boundingRect(contours[i+1])
        
        if abs(x1-x2) < max(w1,w2)*1.5 and y2 > y1+h1:
            space_height = y2 - (y1+h1)
            if 5 < space_height < 40:  # Reasonable disc space range
                space = {
                    'y': y1+h1 + space_height//2,
                    'x_start': min(x1,x2),
                    'x_end': max(x1+w1,x2+w2),
                    'height': space_height,
                    'width': max(x1+w1,x2+w2) - min(x1,x2),
                    'type': classify_space(image, y1+h1, y2, min(x1,x2), max(x1+w1,x2+w2))
                }
                spaces.append(space)
    return spaces

def classify_space(image, top, bottom, left, right):
    roi = image[top:bottom, left:right]
    edges = cv2.Canny(roi, 50, 150)
    edge_density = np.sum(edges)/(roi.size*255)
    
    if edge_density > 0.2:
        return "Osteophytes"
    elif np.std(roi) < 25:
        return "Sclerotic"
    elif (bottom-top) < 15:
        return "Narrowed"
    else:
        return "Normal"

def overlay_spaces(image, spaces):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    colors = {
        "Normal": (0,255,0),
        "Narrowed": (255,0,0),
        "Osteophytes": (255,165,0),
        "Sclerotic": (0,255,255)
    }
    
    for space in spaces:
        color = colors.get(space['type'], (0,255,255))
        cv2.line(img_rgb, (space['x_start'], space['y']), 
                (space['x_end'], space['y']), color, 3)
        cv2.putText(img_rgb, space['type'], 
                   (space['x_start']+10, space['y']-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img_rgb

def image_to_bytes(image):
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format='PNG')
    return buf.getvalue()

# Streamlit UI
st.set_page_config(page_title="Spine Analyzer", layout="wide")

st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload spine image", type=["jpg","png"])
n_clusters = st.sidebar.slider("Clusters", 2, 12, 8)
enable_detection = st.sidebar.checkbox("Enable disc space detection", True)

st.title("Spinal Disc Space Analysis")

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')
    img_array = np.array(image)
    
    # Processing pipeline
    processed = preprocess_image(img_array)
    clustered = enhance_image_kmeans(processed, n_clusters)
    enhanced = blend_images(processed, clustered)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_array, caption="Original", use_column_width=True)
    with col2:
        st.image(processed, caption="Preprocessed", use_column_width=True)
    with col3:
        if enable_detection:
            spaces = detect_disc_spaces(enhanced)
            overlaid = overlay_spaces(enhanced, spaces)
            st.image(overlaid, caption="Analysis", use_column_width=True)
            
            # Show measurements
            st.subheader("Disc Space Measurements")
            data = [[f"Space {i+1}", s['type'], f"{s['height']:.1f}px", f"{s['width']:.1f}px"] 
                   for i,s in enumerate(spaces)]
            st.table(pd.DataFrame(data, columns=["Space", "Type", "Height", "Width"]))
        else:
            st.image(enhanced, caption="Enhanced", use_column_width=True)
    
    # Download options
    st.download_button(
        label="Download Analysis",
        data=image_to_bytes(overlaid if enable_detection else enhanced),
        file_name="spine_analysis.png",
        mime="image/png"
    )
else:
    st.info("Please upload a spinal X-ray image to begin analysis")
