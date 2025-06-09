import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io
import zipfile
import os
import base64

# Set Streamlit page configuration to wide
st.set_page_config(page_title="Spinal Cord Image Clustering", layout="wide")

# --- Custom CSS for maximum blank space removal and styling ---
st.markdown(
    """
    <style>
    /* General body/html styling for full height */
    html, body {
        height: 100%;
        margin: 0;
        padding: 0;
        overflow-x: hidden; /* Prevent horizontal scroll */
    }

    /* Target the main Streamlit container (often the biggest culprit for top padding) */
    .st-emotion-cache-1jmve30 { /* This is a common class for the main content area with layout="wide" */
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        padding-top: 0rem !important; /* Critical: remove top padding */
        padding-bottom: 0rem !important; /* Critical: remove bottom padding */
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        min-height: 100vh; /* Ensure it tries to take full viewport height */
        display: flex;
        flex-direction: column; /* Ensure content stacks vertically */
        justify-content: flex-start; /* Start content from the top */
        width: 100%;
    }

    /* Target the inner block container where most user content sits */
    div.block-container {
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        padding-top: 0rem !important; /* Critical: remove top padding */
        padding-bottom: 0rem !important; /* Critical: remove bottom padding */
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        flex-grow: 1; /* Allow this to grow and fill remaining space */
        width: 100%;
    }

    /* Specific overrides for common Streamlit wrapper divs */
    /* These class names are auto-generated and might change. */
    /* Use browser's inspect element (F12) to find the correct classes if still not working */
    .st-emotion-cache-1sddxrb { /* Example of another wrapper div class */
        padding-top: 0 !important;
        margin-top: 0 !important;
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
    }
    .st-emotion-cache-z5fcl4 { /* Another common div that might add padding */
        padding-top: 0 !important;
        margin-top: 0 !important;
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
    }
    /* Add more .st-emotion-cache-XXXXXX { padding: 0 !important; margin: 0 !important; } rules here if needed */


    /* Sidebar styling with gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #007BFF 0%, #6BCBFF 100%) !important;
        color: white; /* Ensure text is visible on the gradient */
    }
    /* Adjust sidebar header color for contrast */
    [data-testid="stSidebar"] .st-emotion-cache-16txt4v {
        color: white !important;
    }

    /* Download button styling */
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

    /* --- Team Info Section Styling --- */
    /* This container will now be the first child of a zero-padded block-container */
    .team-info-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
        box-sizing: border-box; /* Include padding in width calculation */
        background-color: #f0f2f6; /* Light background for the team page */
        color: #333;
        text-align: center;
        flex-grow: 1; /* Ensure it grows to fill height */

        /* Restore desired internal padding after removing outer padding */
        padding-top: 40px !important;
        padding-bottom: 40px !important;
        padding-left: 20px;
        padding-right: 20px;

        /* If you still see a small gap at the very top, try commenting out the min-height: 100vh on .st-emotion-cache-1jmve30
           and use this instead: */
        min-height: calc(100vh - 56px); /* Roughly viewport height minus Streamlit's header/toolbar */
        /* If Streamlit's top elements are completely removed, then just use 100vh */
    }

    .team-info-container h1 {
        color: #007BFF;
        margin-bottom: 40px;
        font-size: 3em;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        padding-top: 0;
    }

    .team-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 30px;
        width: 100%;
        max-width: 1200px;
        margin-top: 30px;
    }

    .team-member-card {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        padding: 25px;
        display: flex;
        flex-direction: column;
        align-items: center;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }

    .team-member-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }

    .team-member-card img {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        object-fit: cover;
        margin-bottom: 15px;
        border: 5px solid #007BFF;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
    }

    .team-member-card h3 {
        color: #007BFF;
        margin-bottom: 5px;
        font-size: 1.6rem;
        font-weight: 700;
    }

    .team-member-card p.role {
        color: #555;
        font-size: 1.1rem;
        font-style: italic;
        margin-top: 0;
    }

    /* Full image view styling (for when a processed image is clicked) */
    .full-image-view {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 60vh;
        padding: 20px;
        box-sizing: border-box;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Utility Functions ---

# Function to check font availability (for OpenCV text rendering)
def check_fonts():
    try:
        test_img = np.zeros((100,100,3), dtype=np.uint8)
        cv2.putText(test_img, "Test", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return True
    except:
        return False

# Initialize font availability (global flag)
FONT_AVAILABLE = check_fonts()

# Function to resize large images (for image processing, not display of team photos)
def resize_image(image, max_size=1000):
    h, w = image.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

# CLAHE preprocessing for medical images
def preprocess_image(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# K-Means enhancement
def enhance_image_kmeans(image, n_clusters=8):
    pixel_values = image.reshape(-1, 1)
    # Added n_init='auto' to suppress FutureWarning in newer scikit-learn versions
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(pixel_values)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    segmented_pixels = centers[labels].reshape(image.shape)
    return cv2.normalize(segmented_pixels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Image blending for enhanced visualization
def blend_images(original, clustered, alpha=0.7):
    original = original.astype(np.float32)
    clustered = clustered.astype(np.float32)
    blended = alpha * clustered + (1 - alpha) * original
    return cv2.normalize(blended, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Function to convert image array to raw PNG bytes for download
def image_to_bytes(image_array):
    # Ensure the image is in uint8 format (0-255)
    if image_array.dtype != np.uint8:
        image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Convert to PIL Image and save to PNG bytes
    img = Image.fromarray(image_array)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

# --- Session State for Full Image View ---
if "full_image" not in st.session_state:
    st.session_state.full_image = None
if "full_image_caption" not in st.session_state:
    st.session_state.full_image_caption = None

# Show full image in the center if set (this overlays other content)
if st.session_state.full_image is not None:
    st.markdown("""<div class="full-image-view">""", unsafe_allow_html=True)
    st.image(st.session_state.full_image, use_column_width=False, caption=st.session_state.full_image_caption)
    st.markdown("""</div>""", unsafe_allow_html=True)
    if st.button("Close Full Image", key="close_full_image", help="Close the full image view"):
        st.session_state.full_image = None
        st.session_state.full_image_caption = None
    st.markdown("---") # Separator

# --- Streamlit UI Layout ---
st.sidebar.header("Spinal Cord Image Clustering")
uploaded_file = st.sidebar.file_uploader("Upload spine image", type=["jpg","png","jpeg"])
# Use a key to ensure this checkbox doesn't interfere with other widgets' state
show_team = st.sidebar.checkbox("Show Team Info", help="Tick to view information about the development team.", key="show_team_checkbox")

# --- Conditional Content Display ---

if show_team:
    # --- Team Info Page ---
    # The div will now fill the available space due to the CSS changes
    st.markdown('<div class="team-info-container">', unsafe_allow_html=True)
    st.markdown('<h1>Meet Our Amazing Team</h1>', unsafe_allow_html=True)
    st.markdown('<div class="team-grid">', unsafe_allow_html=True)
    
    # Define team members data
    team = [
        {
            "name": "Ye Lin Soe",
            "role": "Team Leader",
            "img": "./yls.jpeg" # Local path (ensure file is in same directory)
        },
        {
            "name": "Khant Zwe Naing",
            "role": "Lead Developer",
            "img": "./kzn.jpg" # Local path
        },
        {
            "name": "A. Traore",
            "role": "Lead Developer",
            "img": "./abu.jpeg" # Local path
        },
        {
            "name": "Dr. D. Mbiba",
            "role": "Project Advisor",
            "img": "https://randomuser.me/api/portraits/men/68.jpg" # Example URL (can be replaced with local file)
        },
        {
            "name": "Rishika",
            "role": "Data Analyst",
            "img": "https://randomuser.me/api/portraits/women/68.jpg" # Example URL
        },
    ]

    # Render each team member card
    for member in team:
        image_source = ""
        try:
            # Check if it's a local file path (doesn't start with http)
            if not member["img"].startswith("http"):
                local_img_path = member["img"].replace("./", "") # Remove leading './' if present
                if os.path.exists(local_img_path):
                    with open(local_img_path, "rb") as f:
                        # Encode local image to Base64 for embedding in HTML
                        image_source = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"
                else:
                    # Fallback to placeholder if local file not found
                    st.warning(f"Local image not found for {member['name']}: {local_img_path}. Using placeholder.")
                    # Use a default placeholder that is distinct if name is empty
                    placeholder_text = member['name'].split()[0] if member['name'] else 'Team'
                    image_source = f"https://via.placeholder.com/150/007BFF/FFFFFF?text={placeholder_text}"
            else: # Assume it's a URL
                image_source = member["img"]
        except Exception as e:
            st.error(f"Error loading image for {member['name']}: {e}. Using placeholder.")
            # Use a default placeholder that is distinct if name is empty
            placeholder_text = member['name'].split()[0] if member['name'] else 'Team'
            image_source = f"https://via.placeholder.com/150/007BFF/FFFFFF?text={placeholder_text}" # Fallback

        st.markdown(f"""
            <div class="team-member-card">
                <img src="{image_source}" alt="{member['name']}">
                <h3>{member["name"]}</h3>
                <p class="role">{member["role"]}</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True) # Close team-grid
    st.markdown('</div>', unsafe_allow_html=True) # Close team-info-container

else:
    # --- Main Image Clustering UI ---
    st.title("Spinal Cord Image Clustering and Analysis")

    if uploaded_file:
        # Read image using PIL (handles various formats)
        original_image = Image.open(uploaded_file)
        # Convert to numpy array for OpenCV processing, ensure RGB for display
        original_array = np.array(original_image)
        if original_array.ndim == 2: # If grayscale, convert to 3 channels for st.image display
            original_array = cv2.cvtColor(original_array, cv2.COLOR_GRAY2RGB)
        elif original_array.shape[2] == 4: # If RGBA, convert to RGB
            original_array = original_array[:, :, :3]

        # Convert to grayscale for processing
        gray_image = original_image.convert('L')
        img_array = np.array(gray_image) # This is the array for processing pipeline
        
        filename = os.path.splitext(uploaded_file.name)[0]
        
        # Apply image processing pipeline
        processed = preprocess_image(img_array)
        clustered = enhance_image_kmeans(processed, 8)
        enhanced = blend_images(processed, clustered)
        
        # Prepare images for download
        images_dict = {
            'Original': img_array, # Use the grayscale array for original in dict, but original_array for display
            'Preprocessed': processed,
            'Clustered': clustered,
            'Enhanced': enhanced
        }
        
        # Prepare ZIP for download
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for key, img_arr_to_save in images_dict.items():
                img_bytes = image_to_bytes(img_arr_to_save) # Get raw PNG bytes
                zip_file.writestr(f"{filename}_{key.lower()}.png", img_bytes)
        zip_buffer.seek(0)
        
        # Display images in 4 columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(original_array, caption="Original", use_column_width=True) # Display RGB original
        with col2:
            st.image(processed, caption="Preprocessed (CLAHE)", use_column_width=True)
        with col3:
            st.image(clustered, caption="Clustered (K-Means)", use_column_width=True)
        with col4:
            st.image(enhanced, caption="Enhanced (Blended)", use_column_width=True)

        # Download buttons under each image
        col1_dl, col2_dl, col3_dl, col4_dl = st.columns(4) # Use new column names for clarity
        with col1_dl:
            st.download_button(
                label="Download Original",
                data=image_to_bytes(img_array), # Download processed grayscale original
                file_name=f"{filename}_original.png",
                mime="image/png",
                key="download1"
            )
        with col2_dl:
            st.download_button(
                label="Download Preprocessed",
                data=image_to_bytes(processed),
                file_name=f"{filename}_preprocessed.png",
                mime="image/png",
                key="download2"
            )
        with col3_dl:
            st.download_button(
                label="Download Clustered",
                data=image_to_bytes(clustered),
                file_name=f"{filename}_clustered.png",
                mime="image/png",
                key="download3"
            )
        with col4_dl:
            st.download_button(
                label="Download Enhanced",
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
        st.info("Please upload a spinal X-ray image to begin analysis.")