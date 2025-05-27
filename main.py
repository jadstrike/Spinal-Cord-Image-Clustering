import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io
import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
from scipy import ndimage

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
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    segmented_pixels = centers[labels].reshape(image.shape)
    return cv2.normalize(segmented_pixels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Image blending
def blend_images(original, clustered, alpha=0.7):
    original = original.astype(np.float32)
    clustered = clustered.astype(np.float32)
    blended = alpha * clustered + (1 - alpha) * original
    return cv2.normalize(blended, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Enhanced disc space detection with curvature analysis
def detect_disc_spaces(image):
    # Enhanced preprocessing
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 21, 5)
    
    # Improved morphological operations
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

# Advanced classification with curvature analysis
def classify_space(image, top, bottom, left, right):
    roi = image[top:bottom, left:right]
    
    # Edge and texture analysis
    edges = cv2.Canny(roi, 50, 150)
    edge_density = np.sum(edges)/(roi.size*255)
    
    # Curvature analysis
    contours, _ = cv2.findContours(cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1], 
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    curvature = 0
    if len(contours) > 0:
        perimeter = cv2.arcLength(contours[0], True)
        if perimeter > 0:
            curvature = 4*np.pi*(cv2.contourArea(contours[0]))/(perimeter**2)
    
    # Intensity analysis
    intensity_std = np.std(roi)
    
    # Classification rules
    if edge_density > 0.2 and curvature < 0.6:
        return "Osteophytes"
    elif intensity_std < 25:
        return "Sclerotic"
    elif (bottom-top) < 15:
        return "Narrowed"
    else:
        return "Normal"

# Enhanced visualization with measurements
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
        
        # Enhanced annotations
        label = f"{space['type']} ({space['height']:.1f}px)"
        cv2.putText(img_rgb, label, (space['x_start']+10, space['y']-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add measurement lines
        cv2.line(img_rgb, (space['x_start'], space['y']-20), 
                (space['x_start'], space['y']+20), (255,255,255), 1)
        cv2.line(img_rgb, (space['x_end'], space['y']-20), 
                (space['x_end'], space['y']+20), (255,255,255), 1)
    return img_rgb

# Generate spine curvature analysis plot
def plot_spine_curvature(image, spaces):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image, cmap='gray')
    
    if spaces:
        # Plot center points
        x_centers = [(s['x_start'] + s['x_end'])/2 for s in spaces]
        y_centers = [s['y'] for s in spaces]
        ax.plot(x_centers, y_centers, 'r-', linewidth=2, label='Spine Centerline')
        ax.scatter(x_centers, y_centers, c='yellow', s=50, label='Disc Centers')
        
        # Calculate curvature metrics
        if len(spaces) > 2:
            # Fit polynomial to centerline
            coeffs = np.polyfit(y_centers, x_centers, 2)
            poly = np.poly1d(coeffs)
            y_fit = np.linspace(min(y_centers), max(y_centers), 100)
            x_fit = poly(y_fit)
            ax.plot(x_fit, y_fit, 'b--', linewidth=1, label='Curvature Fit')
            
            # Calculate curvature
            dy = np.gradient(y_fit)
            dx = np.gradient(x_fit)
            d2y = np.gradient(dy)
            d2x = np.gradient(dx)
            curvature = np.abs(d2x * dy - dx * d2y) / (dx**2 + dy**2)**1.5
            max_curvature = np.max(curvature)
            ax.set_title(f"Spine Curvature Analysis (Max Curvature: {max_curvature:.3f})")
    
    ax.legend()
    ax.axis('off')
    return fig

def image_to_bytes(image):
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format='PNG')
    return buf.getvalue()

# Streamlit UI
st.set_page_config(page_title="Spinal Cord Image Clustering", layout="wide")

# Custom CSS
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
    .status { font-size: 14px; margin-top: 10px; }
    .stDownloadButton { margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.header("Spinal Cord Image Clustering")
uploaded_file = st.sidebar.file_uploader("Upload spine image", type=["jpg","png","jpeg"])
n_clusters = st.sidebar.slider("Number of Clusters", 2, 12, 8)
enable_detection = st.sidebar.checkbox("Enable disc space detection", True)
enable_curvature = st.sidebar.checkbox("Enable spine curvature analysis", True)

st.title("Spinal Cord Image Clustering and Analysis")

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')
    img_array = np.array(image)
    filename = os.path.splitext(uploaded_file.name)[0]
    
    # Processing pipeline
    processed = preprocess_image(img_array)
    clustered = enhance_image_kmeans(processed, n_clusters)
    enhanced = blend_images(processed, clustered)
    
    # Initialize session state for downloads
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = {
            'original': img_array,
            'preprocessed': processed,
            'enhanced': enhanced
        }
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_array, caption="Original", use_column_width=True)
        st.download_button(
            label="Download Original",
            data=image_to_bytes(img_array),
            file_name=f"{filename}_original.png",
            mime="image/png",
            key="dl_original"
        )
    with col2:
        st.image(processed, caption="Preprocessed (CLAHE)", use_column_width=True)
        st.download_button(
            label="Download Preprocessed",
            data=image_to_bytes(processed),
            file_name=f"{filename}_preprocessed.png",
            mime="image/png",
            key="dl_preprocessed"
        )
    with col3:
        if enable_detection:
            spaces = detect_disc_spaces(enhanced)
            overlaid = overlay_spaces(enhanced, spaces)
            st.session_state.processed_images['analyzed'] = overlaid
            st.image(overlaid, caption="Disc Space Analysis", use_column_width=True)
            
            # Show measurements
            st.subheader("Disc Space Measurements")
            data = [[f"Space {i+1}", s['type'], f"{s['height']:.1f}px", f"{s['width']:.1f}px"] 
                   for i,s in enumerate(spaces)]
            df = pd.DataFrame(data, columns=["Space", "Type", "Height", "Width"])
            st.dataframe(df)
            
            st.download_button(
                label="Download Analysis",
                data=image_to_bytes(overlaid),
                file_name=f"{filename}_analysis.png",
                mime="image/png",
                key="dl_analysis"
            )
        else:
            st.image(enhanced, caption="Enhanced (K-Means)", use_column_width=True)
            st.download_button(
                label="Download Enhanced",
                data=image_to_bytes(enhanced),
                file_name=f"{filename}_enhanced.png",
                mime="image/png",
                key="dl_enhanced"
            )
    
    # Curvature analysis
    if enable_detection and enable_curvature and spaces:
        st.subheader("Spine Curvature Analysis")
        curvature_fig = plot_spine_curvature(enhanced, spaces)
        st.pyplot(curvature_fig)
        
        # Save curvature plot to bytes
        buf = io.BytesIO()
        curvature_fig.savefig(buf, format='png')
        buf.seek(0)
        st.download_button(
            label="Download Curvature Plot",
            data=buf,
            file_name=f"{filename}_curvature.png",
            mime="image/png",
            key="dl_curvature"
        )
    
    # Download all images as ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for name, img in st.session_state.processed_images.items():
            zip_file.writestr(f"{filename}_{name}.png", image_to_bytes(img))
        if enable_detection and enable_curvature and spaces:
            zip_file.writestr(f"{filename}_curvature.png", buf.getvalue())
    zip_buffer.seek(0)
    
    st.download_button(
        label="Download All Results as ZIP",
        data=zip_buffer,
        file_name=f"{filename}_results.zip",
        mime="application/zip",
        key="dl_all"
    )
    
    # Innovative Features Section
    st.subheader("Advanced Analysis")
    with st.expander("Show 3D Spine Reconstruction (Experimental)"):
        st.info("This feature would use multiple X-ray views to reconstruct a 3D model of the spine")
        st.image("https://via.placeholder.com/600x300?text=3D+Reconstruction+Placeholder", 
                caption="Future Feature: 3D Spine Reconstruction")
    
    with st.expander("Show Degeneration Scoring"):
        if enable_detection and spaces:
            normal_count = sum(1 for s in spaces if s['type'] == "Normal")
            score = (normal_count / len(spaces)) * 100 if len(spaces) > 0 else 100
            st.metric("Spine Health Score", f"{score:.1f}%")
            st.progress(int(score))
            st.caption("Higher scores indicate better spinal health with more normal disc spaces")
    
    with st.expander("Show Longitudinal Comparison"):
        st.info("This feature would compare current scan with previous scans to track progression")
        st.image("https://via.placeholder.com/600x200?text=Comparison+Over+Time+Placeholder",
                caption="Future Feature: Longitudinal Analysis")
    
else:
    st.info("Please upload a spinal X-ray image to begin analysis")
    st.image("https://via.placeholder.com/800x400?text=Upload+a+spinal+X-ray+image",
            caption="Example Spinal X-ray")
