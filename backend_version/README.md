# Spinal Cord Disc Space Detection Backend (FastAPI)

This is a FastAPI backend for spinal cord image analysis and disc space detection (OPTICS). It is designed to be used with a React (or any) frontend.

## Features

- Accepts an uploaded X-ray image
- Returns processed images (original, preprocessed, enhanced, analysis) as base64 strings
- CORS enabled for frontend integration

## Setup

1. **Install dependencies**

```sh
pip install -r requirements.txt
```

2. **Run the server**

```sh
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

## API Usage

### POST `/analyze`

- **Body:** `multipart/form-data` with a file field (the X-ray image)
- **Response:** JSON with base64-encoded images:
  - `original`: original grayscale
  - `preprocessed`: CLAHE preprocessed
  - `enhanced`: K-means enhanced
  - `analysis`: OPTICS disc space detection overlay

#### Example (using `curl`):

```sh
curl -X POST -F "file=@your_image.png" http://localhost:8000/analyze
```

#### Example Response

```json
{
  "original": "iVBORw0KGgoAAAANSUhEUgAA...",
  "preprocessed": "iVBORw0KGgoAAAANSUhEUgAA...",
  "enhanced": "iVBORw0KGgoAAAANSUhEUgAA...",
  "analysis": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

## Frontend Integration

- Use `fetch` or `axios` in React to POST an image file to `/analyze`.
- Display the returned images by setting the `src` of an `<img>` tag to `data:image/png;base64, ...`.

---

**This backend is stateless and does not store images.**
