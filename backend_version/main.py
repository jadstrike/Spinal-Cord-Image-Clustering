import io
import base64
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from sklearn.cluster import KMeans, OPTICS
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def enhance_image_kmeans(image, n_clusters=8):
    pixel_values = image.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixel_values)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    segmented_pixels = centers[labels].reshape(image.shape)
    return cv2.normalize(segmented_pixels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def blend_images(original, clustered, alpha=0.7):
    original = original.astype(np.float32)
    clustered = clustered.astype(np.float32)
    blended = alpha * clustered + (1 - alpha) * original
    return cv2.normalize(blended, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def detect_disc_spaces_optics(image):
    if len(image.shape) == 2:
        color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = image
    else:
        color_img = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            points.append([cx, cy])
    points = np.array(points)
    if len(points) < 2:
        return [], color_img
    optics = OPTICS(min_samples=2, xi=0.05, min_cluster_size=0.05)
    labels = optics.fit_predict(points)
    output = color_img.copy()
    unique_labels = sorted(set(labels))
    base_cmap = plt.colormaps.get_cmap("tab10")
    color_list = [base_cmap(i % 10) for i in range(len(unique_labels))]
    cluster_centers = []
    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = points[labels == label]
        center = np.mean(cluster_points, axis=0).astype(int)
        cluster_centers.append(center)
        color = color_list[label % len(color_list)]
        for pt in cluster_points:
            cv2.circle(output, tuple(pt), 3, (
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255)
            ), -1)
        cv2.circle(output, tuple(center), 6, (255, 255, 255), 2)
    cluster_centers.sort(key=lambda x: x[1])
    spaces = []
    for i in range(len(cluster_centers) - 1):
        pt1 = tuple(cluster_centers[i])
        pt2 = tuple(cluster_centers[i + 1])
        dist = euclidean(pt1, pt2)
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        cv2.line(output, pt1, pt2, (0, 0, 255), 1)
        cv2.putText(output, f"{int(dist)}px", (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        spaces.append({
            'Space': f'Space {i+1}',
            'Type': 'Cluster',
            'Height': f"{dist:.1f}px",
            'Width': f"-"
        })
    return spaces, output

def image_to_base64(image_array):
    buf = io.BytesIO()
    img = Image.fromarray(image_array)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(buf, format='PNG')
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode('utf-8')

class AnalyzeResponse(BaseModel):
    original: str
    preprocessed: str
    enhanced: str
    analysis: str
    # Optionally: analysis_data: list

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('L')
    img_array = np.array(image)
    processed = preprocess_image(img_array)
    clustered = enhance_image_kmeans(processed, 8)
    enhanced = blend_images(processed, clustered)
    orig_color = np.array(Image.open(io.BytesIO(contents)).convert('RGB'))
    _, overlaid = detect_disc_spaces_optics(orig_color)
    return AnalyzeResponse(
        original=image_to_base64(img_array),
        preprocessed=image_to_base64(processed),
        enhanced=image_to_base64(enhanced),
        analysis=image_to_base64(overlaid)
    ) 