"""
Object Detection and Tracking Module using YOLOv8 and Deep SORT

This module handles:
1. YOLOv8 object detection
2. Deep SORT multi-object tracking
3. Frame processing and annotation
4. Webcam capture and management
"""

import os
# Fix OpenMP runtime conflict - must be set before importing other libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
from collections import defaultdict
import math
try:
    from config import *
except ImportError:
    # Default values if config.py is not available
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    YOLO_MODEL = 'yolov8n.pt'
    CONFIDENCE_THRESHOLD = 0.5
    MAX_AGE = 50
    N_INIT = 3
    NMS_MAX_OVERLAP = 1.0
    MAX_COSINE_DISTANCE = 0.4
    TRAJECTORY_LENGTH = 30
    BBOX_THICKNESS = 2
    TEXT_SCALE = 0.6
    TEXT_THICKNESS = 2

class SimpleTracker:
    """
    A simple object tracker using IoU (Intersection over Union) matching
    Compatible with Python 3.13.5
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_id_count = 0
    
    def update(self, detections):
        """
        Update tracks with new detections
        detections: list of [x1, y1, x2, y2, confidence, class_id, class_name]
        """
        # Predict new locations of existing tracks
        for track in self.tracks:
            track['age'] += 1
            track['hits_since_update'] += 1
        
        # Match detections to existing tracks
        matched_tracks = []
        unmatched_detections = list(range(len(detections)))
        
        if len(self.tracks) > 0 and len(detections) > 0:
            # Calculate IoU matrix
            iou_matrix = self._calculate_iou_matrix(detections)
            
            # Find matches using Hungarian algorithm (simplified)
            matches = self._associate_detections_to_tracks(iou_matrix)
            
            # Update matched tracks
            for match in matches:
                det_idx, track_idx = match
                if det_idx in unmatched_detections:
                    unmatched_detections.remove(det_idx)
                
                track = self.tracks[track_idx]
                det = detections[det_idx]
                
                # Update track with detection
                track['bbox'] = det[:4]
                track['confidence'] = det[4]
                track['class_id'] = det[5]
                track['class_name'] = det[6]
                track['hits_since_update'] = 0
                track['hit_streak'] += 1
                matched_tracks.append(track)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            new_track = {
                'id': self.track_id_count,
                'bbox': det[:4],
                'confidence': det[4],
                'class_id': det[5],
                'class_name': det[6],
                'age': 0,
                'hit_streak': 1,
                'hits_since_update': 0
            }
            self.tracks.append(new_track)
            self.track_id_count += 1
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks 
                      if track['hits_since_update'] < self.max_age]
        
        # Return confirmed tracks
        return [track for track in self.tracks 
                if track['hit_streak'] >= self.min_hits or track['age'] < self.min_hits]
    
    def _calculate_iou_matrix(self, detections):
        """Calculate IoU between detections and existing tracks"""
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        
        for d, det in enumerate(detections):
            for t, track in enumerate(self.tracks):
                iou_matrix[d, t] = self._calculate_iou(det[:4], track['bbox'])
        
        return iou_matrix
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _associate_detections_to_tracks(self, iou_matrix):
        """Simple greedy matching based on IoU"""
        matches = []
        used_detections = set()
        used_tracks = set()
        
        # Sort by IoU score (highest first)
        indices = np.unravel_index(np.argsort(-iou_matrix.ravel()), iou_matrix.shape)
        
        for det_idx, track_idx in zip(indices[0], indices[1]):
            if (det_idx not in used_detections and 
                track_idx not in used_tracks and 
                iou_matrix[det_idx, track_idx] > self.iou_threshold):
                
                matches.append((det_idx, track_idx))
                used_detections.add(det_idx)
                used_tracks.add(track_idx)
        
        return matches


class ObjectDetectorTracker:
    def __init__(self, model_name=None, confidence_threshold=None):
        """
        Initialize the object detector and tracker
        
        Args:
            model_name (str): YOLOv8 model name/path (defaults to config value)
            confidence_threshold (float): Minimum confidence for detections (defaults to config value)
        """
        # Use config values as defaults
        model_name = model_name or YOLO_MODEL
        confidence_threshold = confidence_threshold or CONFIDENCE_THRESHOLD
        
        # Initialize YOLOv8 model
        print("Loading YOLOv8 model...")
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        
        # Initialize Deep SORT tracker
        print("Initializing Simple Tracker...")
        # Use our custom simple tracker instead of Deep SORT
        self.tracker = SimpleTracker(
            max_age=MAX_AGE,
            min_hits=N_INIT,
            iou_threshold=0.3
        )
        
        # Video capture object
        self.cap = None
        self.is_running = False
        self.frame_lock = threading.Lock()
        self.current_frame = None
        
        # Track statistics
        self.track_history = defaultdict(list)
        self.track_colors = {}
        
        # COCO class names for YOLOv8
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def generate_color(self, track_id):
        """Generate a unique color for each track ID"""
        if track_id not in self.track_colors:
            # Generate a random color based on track ID
            np.random.seed(track_id)
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            self.track_colors[track_id] = color
        return self.track_colors[track_id]
    
    def start_webcam(self, camera_index=0):
        """
        Start webcam capture
        
        Args:
            camera_index (int): Camera index (usually 0 for default camera)
        """
        print(f"Starting webcam capture from camera {camera_index}...")
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open camera {camera_index}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        self.is_running = True
        print("Webcam started successfully!")
    
    def stop_webcam(self):
        """Stop webcam capture and release resources"""
        print("Stopping webcam...")
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Webcam stopped successfully!")
    
    def detect_and_track(self, frame):
        """
        Perform object detection and tracking on a single frame
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            annotated_frame: Frame with bounding boxes, labels, and track IDs
        """
        # Run YOLOv8 inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        # Extract detections and convert to tracking format
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else f"object_{class_id}"
                    
                    # Create detection in format: [x1, y1, x2, y2, confidence, class_id, class_name]
                    detection = [x1, y1, x2, y2, confidence, class_id, class_name]
                    detections.append(detection)
        
        # Update tracker with detections
        tracks = self.tracker.update(detections)
        
        # Draw tracks on frame
        annotated_frame = frame.copy()
        
        # Draw confirmed tracks
        for track in tracks:
            # Get track bounding box
            x1, y1, x2, y2 = track['bbox']
            
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Generate color for this track
            track_id = track['id']
            color = self.generate_color(track_id)
            
            # Get track info
            class_name = track['class_name']
            confidence = track['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, BBOX_THICKNESS)
            
            # Create label with class name, confidence, and track ID
            label = f"{class_name} #{track_id} ({confidence:.2f})"
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS)
            
            # Store track history for trajectory
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.track_history[track_id].append(center)
            
            # Keep only recent history points
            if len(self.track_history[track_id]) > TRAJECTORY_LENGTH:
                self.track_history[track_id] = self.track_history[track_id][-TRAJECTORY_LENGTH:]
            
            # Draw trajectory
            points = self.track_history[track_id]
            for j in range(1, len(points)):
                cv2.line(annotated_frame, points[j-1], points[j], color, BBOX_THICKNESS)
        
        # If no tracks but we have detections, show raw detections
        if len(tracks) == 0 and len(detections) > 0:
            for detection in detections:
                x1, y1, x2, y2, confidence, class_id, class_name = detection
                
                # Convert to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Use a default color
                color = (0, 255, 0)  # Green
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, BBOX_THICKNESS)
                
                # Create label
                label = f"{class_name} ({confidence:.2f})"
                
                # Draw label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS)
        
        # Add frame information
        fps_text = f"Objects: {len(detections)} | Tracks: {len(tracks)}"
        cv2.putText(annotated_frame, fps_text, (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_frame
    
    def get_frame(self):
        """
        Get the current processed frame (thread-safe)
        
        Returns:
            processed frame or None if no frame available
        """
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def process_video_stream(self):
        """
        Main processing loop for video stream
        This runs in a separate thread to continuously process frames
        """
        print("Starting video processing loop...")
        
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue
            
            # Read frame from webcam
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from webcam")
                time.sleep(0.1)
                continue
            
            try:
                # Process frame with detection and tracking
                processed_frame = self.detect_and_track(frame)
                
                # Update current frame (thread-safe)
                with self.frame_lock:
                    self.current_frame = processed_frame
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
                # Fall back to original frame
                with self.frame_lock:
                    self.current_frame = frame
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)
        
        print("Video processing loop ended")
    
    def start_processing_thread(self):
        """Start the video processing in a separate thread"""
        processing_thread = threading.Thread(target=self.process_video_stream, daemon=True)
        processing_thread.start()
        return processing_thread


# Global detector instance (will be initialized by Flask app)
detector = None

def initialize_detector():
    """Initialize the global detector instance"""
    global detector
    if detector is None:
        detector = ObjectDetectorTracker()
    return detector

def get_detector():
    """Get the global detector instance"""
    return detector
