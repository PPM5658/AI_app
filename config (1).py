# config.py
# This file stores all our configuration variables

# Video Processing
VIDEO_PATH = "/content/drive/MyDrive/Object tracking/test 3.mp4"
YOLO_MODEL = 'yolov8l.pt'  # Using 's' for a good balance of speed and accuracy
YOLO_CONFIDENCE_THRESHOLD = 0.3
FRAME_SKIP = 10
# --- Distance Estimation Config ---
# You MUST calibrate this value.
# See instructions on how to do this.
FOCAL_LENGTH = 1000 # Example value, REPLACE THIS!

# Average real-world width of objects (in meters)
# These are ESTIMATES. For best results, measure your
# own objects and update these values.
KNOWN_WIDTHS_M = {
    "person": 0.45,       # Average shoulder width
    "chair": 0.5,         # Standard dining chair
    "sofa": 2.1,          # Standard 3-seater sofa
    "dining table": 0.95, # Width of a rectangular table
    "potted plant": 0.3,  # A medium-sized pot
    "tv": 1.2,            # Width of a ~55-inch TV screen
    "laptop": 0.35,       # A 14-15 inch laptop
    "bottle": 0.07,       # A standard water/soda bottle
    "cell phone": 0.075,  # An average smartphone
    "remote": 0.05,       # A standard TV remote
    "book": 0.15,         # A standard hardcover book (portrait)
    "cup": 0.08,          # A coffee mug
    "keyboard": 0.4,      # A standard computer keyboard
    "mouse": 0.06         # A computer mouse
}
# Memory Bank
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
FAISS_INDEX_PATH = "video_memory.index"

# QA System
GEMINI_API_KEY = "AIzaSyCXUJZgYlbmkU9m6f8HpIn2dxrwJ6VtuoM"
GEMINI_MODEL = "gemini-pro-latest"