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

# Enhanced CSS for better styling and layout
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .image-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .image-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        justify-content: center;
        margin: 20px 0;
    }
    
    .image-card {
        background: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        text-align: center;
        min-width: 280px;
        max-width: 320px;
    }
    
    .image-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .image-title {
        font-weight: bold;
        color: #2E86AB;
        font-size: 1.1rem;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .processing-info {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        font-weight: bold;
    }
    
    .download-section {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 5px solid #28a745;
    }
    
    .team-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stButton button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .fullscreen-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(0,0,0,0.9);
        z-index: 9999;
        display: flex;
        justify-content: center;
        align-items: center;
        backdrop-filter: blur(5px);
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .image-grid {
            flex-direction: column;
            align-items: center;
        }
        
        .image-card {
            min-width: 90%;
            max-width: 95%;
        }
        
        .main-title {
            font-size: 2rem;
        }
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

# Enhanced function to resize image with better quality
def resize_for_display(image, target_width=350, target_height=350, maintain_aspect=True):
    """Resize image with enhanced quality preservation"""
    if len(image.shape) == 2:  # Grayscale
        h, w = image.shape
    else:  # Color
        h, w = image.shape[:2]
    
    if maintain_aspect:
        # Calculate scaling factor to fit within target dimensions while maintaining aspect ratio
        scale = min(target_width/w, target_height/h)
        new_w, new_h = int(w * scale), int(h * scale)
    else:
        new_w, new_h = target_width, target_height
    
    # Use LANCZOS for high-quality downsampling
    if len(image.shape) == 2:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Apply sharpening filter to enhance details after resizing
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    if len(image.shape) == 2:
        sharpened = cv2.filter2D(resized, -1, kernel)
        # Blend original with sharpened (subtle sharpening)
        resized = cv2.addWeighted(resized, 0.8, sharpened, 0.2, 0)
    
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

# Enhanced preprocessing with better quality preservation
def preprocess_image(image):
    """Enhanced CLAHE preprocessing with noise reduction"""
    # Apply bilateral filter first to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Apply CLAHE with optimized parameters
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Apply histogram equalization for better contrast
    equalized = cv2.equalizeHist(enhanced)
    
    # Blend CLAHE and equalized versions
    result = cv2.addWeighted(enhanced, 0.7, equalized, 0.3, 0)
    
    return result

# Enhanced K-Means clustering
def enhance_image_kmeans(image, n_clusters=8):
    """Enhanced K-means with better cluster selection"""
    original_shape = image.shape
    
    # Apply Gaussian blur to reduce noise before clustering
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Reshape for clustering
    pixel_values = blurred.reshape(-1, 1).astype(np.float32)
    
    # Apply K-means with improved parameters
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=42, 
        init='k-means++',
        n_init=20,  # More initializations for better stability
        max_iter=500,
        tol=1e-6
    )
    kmeans.fit(pixel_values)
    
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # Sort centers for better visual appeal
    centers = np.sort(centers.flatten())
    
    # Create mapping from old to new labels
    old_to_new = {old_idx: new_idx for new_idx, old_idx in 
                  enumerate(np.argsort(kmeans.cluster_centers_.flatten()))}
    new_labels = np.array([old_to_new[label] for label in labels])
    
    # Reconstruct image
    segmented_pixels = centers[new_labels].reshape(original_shape)
    
    return cv2.normalize(segmented_pixels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Enhanced blending function
def blend_images(original, clustered, alpha=0.6):
    """Enhanced blending with edge preservation"""
    if original.shape != clustered.shape:
        clustered = cv2.resize(clustered, (original.shape[1], original.shape[0]), 
                              interpolation=cv2.INTER_LANCZOS4)
    
    # Convert to float32
    original_f = original.astype(np.float32)
    clustered_f = clustered.astype(np.float32)
    
    # Edge-aware blending
    edges = cv2.Canny(original, 50, 150)
    edges_normalized = edges.astype(np.float32) / 255.0
    
    # Adaptive alpha based on edge strength
    adaptive_alpha = alpha + (1 - alpha) * edges_normalized * 0.3
    
    # Perform blending
    blended = adaptive_alpha[..., np.newaxis] * clustered_f + (1 - adaptive_alpha[..., np.newaxis]) * original_f
    blended = blended.squeeze()
    
    return np.clip(blended, 0, 255).astype(np.uint8)

# High-quality image conversion
def image_to_bytes(image_array, format='PNG', quality=100):
    """Convert image to high-quality bytes"""
    if image_array.dtype != np.uint8:
        image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    if len(image_array.shape) == 2:
        img = Image.fromarray(image_array, mode='L')
    else:
        img = Image.fromarray(image_array)
    
    buf = io.BytesIO()
    if format.upper() == 'PNG':
        img.save(buf, format='PNG', optimize=False, compress_level=1)
    else:
        img.save(buf, format='JPEG', quality=quality, optimize=True)
    
    return buf.getvalue()

# Session state for full image view
if "full_image" not in st.session_state:
    st.session_state.full_image = None
if "full_image_caption" not in st.session_state:
    st.session_state.full_image_caption = None

# Full image overlay
if st.session_state.full_image is not None:
    st.markdown(f"""
    <div class="fullscreen-overlay">
        <div style="text-align: center; color: white;">
            <h2>{st.session_state.full_image_caption}</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(st.session_state.full_image, use_column_width=True, 
                caption=st.session_state.full_image_caption)
        if st.button("✖ Close Full Image", key="close_full", 
                    help="Close the full image view", use_container_width=True):
            st.session_state.full_image = None
            st.session_state.full_image_caption = None
            st.rerun()
    st.markdown("---")

# Sidebar
st.sidebar.header("🔬 Spinal Cord Analysis")
uploaded_file = st.sidebar.file_uploader("📁 Upload spine image", type=["jpg","png","jpeg"])

# Processing parameters
st.sidebar.subheader("⚙️ Processing Parameters")
n_clusters = st.sidebar.slider("Number of K-means clusters", 4, 12, 8)
blend_alpha = st.sidebar.slider("Blend intensity", 0.1, 1.0, 0.6, 0.1)
display_size = st.sidebar.selectbox("Display size", [300, 350, 400, 450], index=1)

show_team = st.sidebar.checkbox("👥 Show Team Info")

# Main layout
if show_team:
    main_col, team_col = st.columns([4, 1])
else:
    main_col = st.container()
    team_col = None

with main_col:
    st.markdown('<h1 class="main-title">🏥 Spinal Cord Image Clustering & Analysis</h1>', 
                unsafe_allow_html=True)

    if uploaded_file:
        # Load and process image
        original_image = Image.open(uploaded_file)
        original_array = np.array(original_image)

        # Convert to grayscale for processing
        gray_image = original_image.convert('L')
        img_array = np.array(gray_image)
        
        filename = os.path.splitext(uploaded_file.name)[0]
        
        # Processing info
        st.markdown(f"""
        <div class="processing-info">
            🔄 Processing high-resolution image: {img_array.shape[1]} × {img_array.shape[0]} pixels
        </div>
        """, unsafe_allow_html=True)
        
        # Process images at full resolution
        with st.spinner("Processing images..."):
            processed = preprocess_image(img_array)
            clustered = enhance_image_kmeans(processed, n_clusters)
            enhanced = blend_images(processed, clustered, blend_alpha)
        
        # Create display versions
        original_display = resize_for_display(original_array, display_size, display_size)
        processed_display = resize_for_display(processed, display_size, display_size)
        clustered_display = resize_for_display(clustered, display_size, display_size)
        enhanced_display = resize_for_display(enhanced, display_size, display_size)
        
        # Image display container
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        
        # Create responsive grid
        col1, col2, col3, col4 = st.columns(4)
        
        images_data = [
            (col1, original_display, "📸 Original", "original"),
            (col2, processed_display, "🔧 Preprocessed", "preprocessed"),
            (col3, clustered_display, "🎯 Clustered", "clustered"),
            (col4, enhanced_display, "✨ Enhanced", "enhanced")
        ]
        
        for col, img_display, title, key in images_data:
            with col:
                st.markdown(f'<div class="image-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="image-title">{title}</div>', unsafe_allow_html=True)
                
                # Make images clickable for full view
                if st.button(f"🔍 View Full", key=f"view_{key}", use_container_width=True):
                    st.session_state.full_image = img_display
                    st.session_state.full_image_caption = title
                    st.rerun()
                
                st.image(img_display, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Download section
        st.markdown('<div class="download-section">', unsafe_allow_html=True)
        st.markdown("### 📥 Download Processed Images")
        
        # Prepare full resolution images for download
        images_dict = {
            'Original': img_array,
            'Preprocessed': processed,
            'Clustered': clustered,
            'Enhanced': enhanced
        }
        
        # Individual download buttons
        download_cols = st.columns(4)
        download_data = [
            ("📸 Original", img_array, f"{filename}_original.png"),
            ("🔧 Preprocessed", processed, f"{filename}_preprocessed.png"),
            ("🎯 Clustered", clustered, f"{filename}_clustered.png"),
            ("✨ Enhanced", enhanced, f"{filename}_enhanced.png")
        ]
        
        for i, (label, img_data, filename_dl) in enumerate(download_data):
            with download_cols[i]:
                st.download_button(
                    label=label,
                    data=image_to_bytes(img_data),
                    file_name=filename_dl,
                    mime="image/png",
                    key=f"download_{i}",
                    use_container_width=True
                )
        
        # ZIP download
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for key, img_array in images_dict.items():
                img_bytes = image_to_bytes(img_array)
                zip_file.writestr(f"{filename}_{key.lower()}.png", img_bytes)
        zip_buffer.seek(0)
        
        st.download_button(
            label="📦 Download All Images (ZIP)",
            data=zip_buffer,
            file_name=f"{filename}_complete_analysis.zip",
            mime="application/zip",
            key="download_zip",
            use_container_width=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Processing stats
        with st.expander("📊 Processing Statistics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Size", f"{img_array.shape[1]}×{img_array.shape[0]}")
            with col2:
                st.metric("K-means Clusters", n_clusters)
            with col3:
                st.metric("Blend Intensity", f"{blend_alpha:.1f}")

    else:
        st.markdown("""
        <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; border-radius: 15px; margin: 20px 0;">
            <h2>🏥 Welcome to Spinal Cord Image Analysis</h2>
            <p style="font-size: 1.2rem; margin: 20px 0;">
                Upload a spinal X-ray image to begin advanced clustering analysis
            </p>
            <p>📁 Supported formats: JPG, PNG, JPEG</p>
        </div>
        """, unsafe_allow_html=True)

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
            # Load image with high quality
            try:
                # Use PIL to load and ensure high quality
                team_img = Image.open(member["img"])
                # Convert to RGB if needed
                if team_img.mode != 'RGB':
                    team_img = team_img.convert('RGB')
                # Display with high quality
                st.image(team_img, width=100)
            except:
                st.image(member["img"], width=100)
            st.subheader(member["name"])
            st.caption(member["role"])