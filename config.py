"""
Configuration file for Live-Object-Detection-and-Tracking

Modify these settings to customize the application behavior.
"""

# Camera Settings
CAMERA_INDEX = 0              # Camera index (0 for default, 1 for external, etc.)
CAMERA_WIDTH = 640            # Camera resolution width
CAMERA_HEIGHT = 480           # Camera resolution height
CAMERA_FPS = 30               # Target FPS for camera capture

# YOLOv8 Model Settings
YOLO_MODEL = 'yolov8n.pt'     # Model size: yolov8n.pt (fastest) to yolov8x.pt (most accurate)
CONFIDENCE_THRESHOLD = 0.5    # Minimum confidence for detections (0.0 to 1.0)

# Deep SORT Tracking Settings
MAX_AGE = 50                  # Maximum frames to keep track alive without detections
N_INIT = 3                    # Consecutive detections needed before track confirmation
NMS_MAX_OVERLAP = 1.0         # Non-maxima suppression threshold
MAX_COSINE_DISTANCE = 0.4     # Cosine distance threshold for track matching

# Display Settings
TRAJECTORY_LENGTH = 30        # Number of points to show in object trajectory
BBOX_THICKNESS = 2            # Thickness of bounding box lines
TEXT_SCALE = 0.6              # Scale of text labels
TEXT_THICKNESS = 2            # Thickness of text

# Web Server Settings
HOST = '0.0.0.0'              # Server host (0.0.0.0 for all interfaces)
PORT = 5000                   # Server port
DEBUG = False                 # Flask debug mode (set to True for development)

# Streaming Settings
JPEG_QUALITY = 85             # JPEG compression quality (1-100)
STREAM_FPS = 30               # Target FPS for video streaming

# Performance Settings
PROCESSING_THREADS = 1        # Number of processing threads
FRAME_BUFFER_SIZE = 2         # Number of frames to buffer
