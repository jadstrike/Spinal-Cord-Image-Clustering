import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFont
import io
import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import base64

# Function to check font availability
def check_fonts():
    try:
        test_img = np.zeros((100,100,3), dtype=np.uint8)
        cv2.putText(test_img, "Test", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return True
    except:
        return False

# Initialize font availability
FONT_AVAILABLE = check_fonts()

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

# Enhanced disc space detection
def detect_disc_spaces(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 21, 5)
    
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    
    spaces = []
    for i in range(len(contours)-1):
        x1, y1, w1, h1 = cv2.boundingRect(contours[i])
        x2, y2, w2, h2 = cv2.boundingRect(contours[i+1])
        
        if abs(x1-x2) < max(w1,w2)*1.5 and y2 > y1+h1:
            space_height = y2 - (y1+h1)
            if 5 < space_height < 40:
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

# Robust text overlay using Pillow as fallback
def overlay_spaces(image, spaces):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    colors = {
        "Normal": (0,255,0),
        "Narrowed": (255,0,0),
        "Osteophytes": (255,165,0),
        "Sclerotic": (0,255,255)
    }
    
    if FONT_AVAILABLE:
        # Use OpenCV if fonts are available
        font = cv2.FONT_HERSHEY_SIMPLEX
        for space in spaces:
            color = colors.get(space['type'], (0,255,255))
            cv2.line(img_rgb, (space['x_start'], space['y']), 
                    (space['x_end'], space['y']), color, 3)
            
            label = f"{space['type']} {space['height']:.1f}px"
            text_size = cv2.getTextSize(label, font, 0.5, 2)[0]
            text_x = space['x_start'] + 10
            text_y = space['y'] - 10
            
            if text_y - text_size[1] < 0:
                text_y = space['y'] + text_size[1] + 10
            if text_x + text_size[0] > img_rgb.shape[1]:
                text_x = img_rgb.shape[1] - text_size[0] - 10
                
            cv2.putText(img_rgb, label, (text_x, text_y),
                       font, 0.5, color, 2)
    else:
        # Fallback to Pillow
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            font = ImageFont.load_default()
        
        for space in spaces:
            color = colors.get(space['type'], (0,255,255))
            draw.line([(space['x_start'], space['y']), 
                     (space['x_end'], space['y'])], 
                     fill=color, width=3)
            draw.text((space['x_start']+10, space['y']-15), 
                     f"{space['type']} {space['height']:.1f}px", 
                     fill=color, font=font)
        
        img_rgb = np.array(img_pil)
    
    return img_rgb

def image_to_base64(image_array):
    buf = io.BytesIO()
    Image.fromarray(image_array).save(buf, format='PNG')
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode('utf-8')

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
    .modal {
      display: none; position: fixed; z-index: 9999; left: 0; top: 0; width: 100vw; height: 100vh;
      overflow: auto; background-color: rgba(0,0,0,0.8); justify-content: center; align-items: center;
    }
    .modal-content {
      margin: auto; display: block; max-width: 90vw; max-height: 90vh; border-radius: 10px;
      box-shadow: 0 0 20px #000;
    }
    .close {
      position: absolute; top: 30px; right: 50px; color: #fff; font-size: 40px; font-weight: bold; cursor: pointer;
      z-index: 10000;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.header("Spinal Cord Image Clustering")
uploaded_file = st.sidebar.file_uploader("Upload spine image", type=["jpg","png","jpeg"])
n_clusters = st.sidebar.slider("Number of Clusters", 2, 12, 8)
enable_detection = st.sidebar.checkbox("Enable disc space detection", False)

st.title("Spinal Cord Image Clustering and Analysis")

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')
    img_array = np.array(image)
    filename = os.path.splitext(uploaded_file.name)[0]
    
    # Processing pipeline
    processed = preprocess_image(img_array)
    clustered = enhance_image_kmeans(processed, n_clusters)
    enhanced = blend_images(processed, clustered)
    
    # Prepare images for download
    images_dict = {
        'Original': image_to_base64(img_array),
        'Preprocessed': image_to_base64(processed),
        'Enhanced': image_to_base64(enhanced)
    }
    if enable_detection:
        spaces = detect_disc_spaces(enhanced)
        overlaid = overlay_spaces(enhanced, spaces)
        images_dict['Analysis'] = image_to_base64(overlaid)
    
    # Prepare ZIP for download
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for key, img_bytes in images_dict.items():
            zip_file.writestr(f"{filename}_{key.lower()}.png", img_bytes)
    zip_buffer.seek(0)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div style="text-align:center">', unsafe_allow_html=True)
        st.image(img_array, caption="Original", use_column_width=True)
        st.download_button(
            label="Download Original Image",
            data=image_to_base64(img_array),
            file_name=f"{filename}_original.png",
            mime="image/png",
            key="download1"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="text-align:center">', unsafe_allow_html=True)
        st.image(processed, caption="Preprocessed (CLAHE)", use_column_width=True)
        st.download_button(
            label="Download Preprocessed Image",
            data=image_to_base64(processed),
            file_name=f"{filename}_preprocessed.png",
            mime="image/png",
            key="download2"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div style="text-align:center">', unsafe_allow_html=True)
        if enable_detection:
            st.image(overlaid, caption="Disc Space Analysis", use_column_width=True)
            st.download_button(
                label="Download Analysis Image",
                data=image_to_base64(overlaid),
                file_name=f"{filename}_analysis.png",
                mime="image/png",
                key="download3"
            )
        else:
            st.image(enhanced, caption="Enhanced (K-Means)", use_column_width=True)
            st.download_button(
                label="Download Enhanced Image",
                data=image_to_base64(enhanced),
                file_name=f"{filename}_enhanced.png",
                mime="image/png",
                key="download4"
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show measurements
    if enable_detection:
        st.subheader("Disc Space Measurements")
        data = [[f"Space {i+1}", s['type'], f"{s['height']:.1f}px", f"{s['width']:.1f}px"] 
               for i,s in enumerate(spaces)]
        st.table(pd.DataFrame(data, columns=["Space", "Type", "Height", "Width"]))
    
    # Health score
    if enable_detection and spaces:
        with st.expander("Spine Health Score"):
            normal_count = sum(1 for s in spaces if s['type'] == "Normal")
            score = (normal_count / len(spaces)) * 100 if len(spaces) > 0 else 100
            st.metric("Spine Health Score", f"{score:.1f}%")
            st.progress(int(score))
            st.caption("Higher scores indicate better spinal health with more normal disc spaces")
else:
    st.info("Please upload a spinal X-ray image to begin analysis")
