import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io
import zipfile
import os
import base64

st.set_page_config(page_title="Spinal Cord Image Clustering", layout="wide")

# Custom CSS for gradient sidebar background and download button color
# Removed the global .stImage > img CSS rule to allow images to display in full quality
# without being forced into specific dimensions or object-fit: cover, which could crop.
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #007BFF 0%, #6BCBFF 100%) !important;
    }
    .stDownloadButton > button {
        background-color: #007BFF !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
        margin-left: auto !important;
        margin-right: auto !important;
        display: block !important;
        max-width: 350px !important;
        width: 100% !important;
        transition: background 0.2s;
    }
    .stDownloadButton > button:hover {
        background-color: #0056b3 !important;
        color: #fff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

# Function to resize image to consistent dimensions for DISPLAY ONLY
def resize_for_display(image, target_width=400, target_height=400):
    """Resize image to standard dimensions for display purposes only"""
    if len(image.shape) == 2:  # Grayscale
        h, w = image.shape
    else:  # Color
        h, w = image.shape[:2]
    
    # Calculate scaling factor to fit within target dimensions
    scale = min(target_width/w, target_height/h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize the image using high-quality interpolation
    if len(image.shape) == 2:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create a canvas of target size and center the resized image
    if len(image.shape) == 2:
        canvas = np.zeros((target_height, target_width), dtype=image.dtype)
    else:
        canvas = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
    
    # Calculate position to center the image
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    
    # Place the resized image on the canvas
    if len(image.shape) == 2:
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    else:
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

# Function to maintain original image dimensions during processing
def process_at_original_size(image, max_processing_size=2048):
    """
    Process image at original size, but limit extremely large images for performance
    Returns the processed image at original dimensions
    """
    h, w = image.shape[:2] if len(image.shape) > 2 else image.shape
    
    # Only resize if image is extremely large (for performance)
    if max(h, w) > max_processing_size:
        scale = max_processing_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    return image

# Function to resize large images
def resize_image(image, max_size=1000):
    h, w = image.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

# CLAHE preprocessing - enhanced for quality
def preprocess_image(image):
    """Apply CLAHE preprocessing while maintaining image quality"""
    # Ensure we're working with the full resolution image
    image = process_at_original_size(image)
    
    # Apply CLAHE with optimized parameters for medical images
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = clahe.apply(image)
    
    # Optional: Apply slight gaussian blur to reduce noise while preserving edges
    processed = cv2.bilateralFilter(processed, 5, 75, 75)
    
    return processed

# K-Means enhancement - optimized for quality
def enhance_image_kmeans(image, n_clusters=8):
    """Apply K-means clustering while preserving image quality"""
    # Ensure we're working with full resolution
    original_shape = image.shape
    
    # Reshape for clustering
    pixel_values = image.reshape(-1, 1).astype(np.float32)
    
    # Apply K-means with better initialization
    kmeans = KMeans(
        n_clusters=n_clusters,  
        random_state=42,  
        init='k-means++',   # Better initialization
        n_init=10,          # More initializations for stability
        max_iter=300        # More iterations for convergence
    )
    kmeans.fit(pixel_values)
    
    # Get cluster centers and labels
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # Reconstruct the image
    segmented_pixels = centers[labels].reshape(original_shape)
    
    # Normalize to preserve dynamic range
    return cv2.normalize(segmented_pixels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Image blending - enhanced for quality preservation
def blend_images(original, clustered, alpha=0.7):
    """Blend images while preserving quality and dynamic range"""
    # Ensure both images are the same size and type
    if original.shape != clustered.shape:
        clustered = cv2.resize(clustered, (original.shape[1], original.shape[0]), 
                               interpolation=cv2.INTER_LANCZOS4)
    
    # Convert to float32 for precision
    original = original.astype(np.float32)
    clustered = clustered.astype(np.float32)
    
    # Perform blending
    blended = alpha * clustered + (1 - alpha) * original
    
    # Normalize back to uint8 while preserving full dynamic range
    blended = np.clip(blended, 0, 255)
    return blended.astype(np.uint8)

# High-quality image conversion function
def image_to_bytes(image_array, quality=95):
    """Convert image array to high-quality PNG bytes"""
    # Ensure the image is in uint8 format
    if image_array.dtype != np.uint8:
        image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Convert to PIL Image with high quality settings
    if len(image_array.shape) == 2:  # Grayscale
        img = Image.fromarray(image_array, mode='L')
    else:  # Color
        img = Image.fromarray(image_array)
    
    # Save to PNG with maximum quality (lossless compression)
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=False, compress_level=1)  # Low compression for max quality
    return buf.getvalue()

# Function to convert image array to base64 string for display (if needed)
def image_to_base64(image_array):
    img_bytes = image_to_bytes(image_array)
    return base64.b64encode(img_bytes).decode('utf-8')

# Track which image is clicked for full view
if "full_image" not in st.session_state:
    st.session_state.full_image = None
if "full_image_caption" not in st.session_state:
    st.session_state.full_image_caption = None

# Show full image in the center if set
if st.session_state.full_image is not None:
    st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 60vh;">
    """, unsafe_allow_html=True)
    st.image(st.session_state.full_image, use_column_width=False, caption=st.session_state.full_image_caption)
    st.markdown("""
        </div>
    """, unsafe_allow_html=True)
    if st.button("Close Full Image", key="close_full_image", help="Close the full image view"):
        st.session_state.full_image = None
        st.session_state.full_image_caption = None
    st.markdown("---")

# Streamlit UI
st.sidebar.header("Spinal Cord Image Clustering")
uploaded_file = st.sidebar.file_uploader("Upload spine image", type=["jpg","png","jpeg"])
show_team = st.sidebar.checkbox("Show Team Info")

# Layout: main content (left), team info (right if checked)
if show_team:
    main_col, team_col = st.columns([3, 1])
else:
    main_col = st.container()
    team_col = None

with main_col:
    st.title("Spinal Cord Image Clustering and Analysis")

    if uploaded_file:
        # Keep original image for display
        original_image = Image.open(uploaded_file)           # Do NOT convert to grayscale here
        original_array = np.array(original_image)

        # Convert to grayscale for processing - preserve original resolution
        gray_image = original_image.convert('L')
        img_array = np.array(gray_image)
        
        filename = os.path.splitext(uploaded_file.name)[0]
        
        # Processing pipeline - ALL AT ORIGINAL RESOLUTION
        st.info("Processing images at original resolution for maximum quality...")
        
        # Process at full resolution
        processed = preprocess_image(img_array)
        clustered = enhance_image_kmeans(processed, 8)
        enhanced = blend_images(processed, clustered)
        
        # Create display versions ONLY for showing on screen
        target_width, target_height = 400, 400
        original_display = resize_for_display(original_array, target_width, target_height)
        processed_display = resize_for_display(processed, target_width, target_height)
        clustered_display = resize_for_display(clustered, target_width, target_height)
        enhanced_display = resize_for_display(enhanced, target_width, target_height)
        
        # Show image dimensions info
        st.success(f"✅ Original image dimensions: {img_array.shape[1]} × {img_array.shape[0]} pixels")
        st.success(f"📱 Display size: {target_width} × {target_height} pixels (for viewing only)")
        
        # Prepare FULL RESOLUTION images for download
        images_dict = {
            'Original': img_array,           # Full resolution grayscale
            'Preprocessed': processed,       # Full resolution processed
            'Clustered': clustered,          # Full resolution clustered
            'Enhanced': enhanced             # Full resolution enhanced
        }
        
        # Prepare ZIP for download
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for key, img_array in images_dict.items():
                img_bytes = image_to_bytes(img_array)  # Get raw PNG bytes
                zip_file.writestr(f"{filename}_{key.lower()}.png", img_bytes)
        zip_buffer.seek(0)
        
        # Display images with consistent sizes
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(original_display, caption="Original", width=300)
        with col2:
            st.image(processed_display, caption="Preprocessed (CLAHE)", width=300)
        with col3:
            st.image(clustered_display, caption="Clustered (K-Means)", width=300)
        with col4:
            st.image(enhanced_display, caption="Enhanced (Blended)", width=300)

        # Download buttons under each image
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.download_button(
                label="Download Original Image",
                data=image_to_bytes(img_array),
                file_name=f"{filename}_original.png",
                mime="image/png",
                key="download1"
            )
        with col2:
            st.download_button(
                label="Download Preprocessed Image",
                data=image_to_bytes(processed),
                file_name=f"{filename}_preprocessed.png",
                mime="image/png",
                key="download2"
            )
        with col3:
            st.download_button(
                label="Download Clustered Image",
                data=image_to_bytes(clustered),
                file_name=f"{filename}_clustered.png",
                mime="image/png",
                key="download3"
            )
        with col4:
            st.download_button(
                label="Download Enhanced Image",
                data=image_to_bytes(enhanced),
                file_name=f"{filename}_enhanced.png",
                mime="image/png",
                key="download4"
            )

        # Download all images as ZIP
        st.download_button(
            label="Download All Images as ZIP",
            data=zip_buffer,
            file_name=f"{filename}_images.zip",
            mime="application/zip",
            key="download_zip"
        )

    else:
        st.info("Please upload a spinal X-ray image to begin analysis")

if show_team and team_col is not None:
    with team_col:
        st.markdown("---")
        st.header("Meet the Team")
        team = [
            {
                "name": "Ye Lin Soe",
                "role": "Team Leader",
                "img": "yls.jpeg"
            },
            {
                "name": "Khant Zwe Naing",
                "role": "Lead Developer",
                "img": "./kzn.jpg"
            },
            {
                "name": "A. Traore",
                "role": "Lead Developer",
                "img": "./abu.jpeg"
            },
            {
                "name": "Dr Dumisani Mbib",
                "role": "Clinical Consultant",
                "img": "./dr.jpeg"
            },
            {
                "name": "Rishika",
                "role": "Poster and Analyst",
                "img": "./rishika.jpeg"
            },
            
            
        ]
        for member in team:
            # Setting a consistent width for team images to make them appear the "same size"
            # while allowing Streamlit to maintain their aspect ratio and quality.
            st.image(member["img"], width=200) # You can adjust this width as needed
            st.subheader(member["name"])
            st.caption(member["role"])
