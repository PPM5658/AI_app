# video_processor.py
# Handles loading video, running tracking, and calculating distance.
# --- IMPROVED with Frame Skipping ---

import cv2
from ultralytics import YOLO
from tqdm import tqdm
import config
import numpy as np

# --- 1. IMPORT ALL CONFIGS (FRAME_SKIP + DISTANCE) ---
from config import FRAME_SKIP, YOLO_CONFIDENCE_THRESHOLD, FOCAL_LENGTH, KNOWN_WIDTHS_M
from ultralytics.engine.results import Results

class VideoProcessor:
    def __init__(self):
        print("Loading YOLO model...")
        self.model = YOLO(config.YOLO_MODEL)
        print("YOLO model loaded.")
        
        # --- 2. ADD CALIBRATION WARNING ---
        if FOCAL_LENGTH == 1000: # Your default example value
            print("WARNING: FOCAL_LENGTH is set to the default (1000).")
            print("Distance estimates will be inaccurate until you calibrate it.")

    # --- 3. RE-ADD _calculate_distance METHOD ---
    def _calculate_distance(self, pixel_width, object_class_name):
        """Calculates distance using the triangle similarity formula."""
        if object_class_name in KNOWN_WIDTHS_M:
            known_width_m = KNOWN_WIDTHS_M[object_class_name]
            if pixel_width > 0:
                distance_m = (known_width_m * FOCAL_LENGTH) / pixel_width
                return distance_m
        return None # Return None if class is unknown or width is 0
    
    def process_video(self, video_path):
        """
        Processes a video file frame by frame using YOLO tracking.
        Applies frame skipping for performance.
        
        Yields:
            tuple: (frame, timestamp_sec, tracks)
                   'tracks' is an 8-column numpy array:
                   [x1, y1, x2, y2, track_id, class_id, confidence, distance_m]
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Error opening video file: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames <= 0 or fps <= 0:
                print("Warning: Could not get video properties (frames/fps).")
                fps = 30 
                total_frames = 1 

            print(f"Video loaded: {total_frames} frames @ {fps:.2f} FPS")
            
            frame_count = 0 # <-- 4. ADD FRAME COUNTER
            
            with tqdm(total=total_frames, desc="Processing Video") as pbar:
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break

                    # --- 5. FRAME SKIPPING LOGIC ---
                    # Only process 1 frame every FRAME_SKIP frames
                    if frame_count % FRAME_SKIP == 0:
                        
                        timestamp_sec = pbar.n / fps
                        
                        # Run YOLOv8 tracking
                        results_list = self.model.track(
                            frame, 
                            persist=True, 
                            conf=YOLO_CONFIDENCE_THRESHOLD,
                            verbose=False
                        )
                        
                        results = results_list[0] if results_list else None
                        
                        # --- 6. PROCESS TRACKS (WITH DISTANCE) ---
                        tracks_list = []

                        if isinstance(results, Results) and results.boxes.id is not None:
                            
                            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
                            track_ids = results.boxes.id.cpu().numpy().astype(int)
                            class_ids = results.boxes.cls.cpu().numpy().astype(int)
                            confidences = results.boxes.conf.cpu().numpy()
                            
                            for xyxy, track_id, class_id, conf in zip(boxes_xyxy, track_ids, class_ids, confidences):
                                
                                # --- 7. RE-ADD DISTANCE CALCULATION ---
                                class_name = self.model.model.names[class_id]
                                pixel_width = xyxy[2] - xyxy[0] # x2 - x1
                                distance_m = self._calculate_distance(pixel_width, class_name)
                                # ---
                                
                                # Append all 8 columns of data
                                tracks_list.append([
                                    xyxy[0], xyxy[1], xyxy[2], xyxy[3], # xyxy
                                    track_id,
                                    class_id,
                                    conf,
                                    distance_m if distance_m is not None else -1 # Use -1 for unknown
                                ])

                        # Convert to numpy array
                        if len(tracks_list) > 0:
                               tracks = np.array(tracks_list)
                        else:
                               # --- 8. Array is now 8 columns ---
                               tracks = np.empty((0, 8)) 

                        # --- 9. YIELD THE 8-COLUMN 'tracks' ARRAY ---
                        yield frame, timestamp_sec, tracks

                    # --- 10. UPDATE PBAR AND FRAME COUNT ON *EVERY* FRAME ---
                    # This ensures the progress bar and frame count are correct
                    pbar.update(1)
                    frame_count += 1

        except Exception as e:
            print(f"Error in video thread processing: {e}")
        finally:
            cap.release()
            print("Video processing complete.")