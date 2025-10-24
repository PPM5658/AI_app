# video_processor.py
# Handles reading video frames, detecting/tracking objects using YOLOv8,
# calculating distances, and yielding structured detection results.
# Includes frame skipping for performance optimization.

import cv2
from ultralytics import YOLO
from tqdm import tqdm
import config
import numpy as np
from config import FRAME_SKIP, YOLO_CONFIDENCE_THRESHOLD, FOCAL_LENGTH, KNOWN_WIDTHS_M
from ultralytics.engine.results import Results

class VideoProcessor:
    def __init__(self):
        """Initialize the YOLO model and warn if calibration is not set."""
        print("Loading YOLO model...")
        self.model = YOLO(config.YOLO_MODEL)
        print("YOLO model loaded.")

        # Warn if using default focal length, since it affects distance accuracy
        if FOCAL_LENGTH == 1000:
            print("WARNING: FOCAL_LENGTH is set to the default (1000).")
            print("Distance estimates will be inaccurate until you calibrate it.")

    def _calculate_distance(self, pixel_width, object_class_name):
        """
        Estimate the real-world distance to an object using the triangle similarity formula:
            distance = (known_width * focal_length) / pixel_width
        Returns None if class is unknown or pixel width is invalid.
        """
        if object_class_name in KNOWN_WIDTHS_M:
            known_width_m = KNOWN_WIDTHS_M[object_class_name]
            if pixel_width > 0:
                distance_m = (known_width_m * FOCAL_LENGTH) / pixel_width
                return distance_m
        return None

    def process_video(self, video_path, stop_event=None):
        """
        Process a video file (or live stream) frame by frame.
        Performs YOLO tracking and distance estimation, with frame skipping for speed.

        Yields:
            frame (ndarray): the current video frame
            timestamp_sec (float): the current time in seconds
            tracks (ndarray): 8-column array containing:
                [x1, y1, x2, y2, track_id, class_id, confidence, distance_m]
        """
        try:
            # Open video file or camera stream
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Error opening video file: {video_path}")

            # Get video metadata (total frames, FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Handle cases where FPS or frame count is not available
            if total_frames <= 0 or fps <= 0:
                print("Warning: Could not get video properties (frames/fps).")
                fps = 30
                total_frames = 1

            print(f"Video loaded: {total_frames} frames @ {fps:.2f} FPS")

            frame_count = 0  # Counts every frame read

            # Show a progress bar for video processing
            with tqdm(total=total_frames, desc="Processing Video") as pbar:
                while cap.isOpened():
                    # Gracefully stop if event is triggered (used for threading)
                    if stop_event is not None and stop_event.is_set():
                        print("Stop signal received. Ending video processing loop.")
                        break

                    success, frame = cap.read()
                    if not success:
                        break  # End of video or camera feed

                    # --- Frame skipping for speed ---
                    if frame_count % FRAME_SKIP == 0:
                        timestamp_sec = pbar.n / fps  # Approximate video time

                        # Perform object tracking using YOLOv8
                        results_list = self.model.track(
                            frame,
                            persist=True,
                            conf=YOLO_CONFIDENCE_THRESHOLD,
                            verbose=False
                        )

                        # Extract YOLO results from list
                        results = results_list[0] if results_list else None
                        tracks_list = []

                        # Process results if detections exist
                        if isinstance(results, Results) and results.boxes.id is not None:
                            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
                            track_ids = results.boxes.id.cpu().numpy().astype(int)
                            class_ids = results.boxes.cls.cpu().numpy().astype(int)
                            confidences = results.boxes.conf.cpu().numpy()

                            # Build detection rows
                            for xyxy, track_id, class_id, conf in zip(boxes_xyxy, track_ids, class_ids, confidences):
                                class_name = self.model.model.names[class_id]
                                pixel_width = xyxy[2] - xyxy[0]  # box width in pixels
                                distance_m = self._calculate_distance(pixel_width, class_name)

                                # Each detection = one row (8 columns)
                                tracks_list.append([
                                    xyxy[0], xyxy[1], xyxy[2], xyxy[3],
                                    track_id, class_id, conf,
                                    distance_m if distance_m is not None else -1
                                ])

                        # Convert detections to numpy array for consistency
                        tracks = np.array(tracks_list) if len(tracks_list) > 0 else np.empty((0, 8))

                        # Yield the frame and detection data for other modules
                        yield frame, timestamp_sec, tracks

                    # Update progress bar and frame counter on every frame
                    pbar.update(1)
                    frame_count += 1

        except Exception as e:
            # Log any runtime error during processing
            if stop_event is None or not stop_event.is_set():
                print(f"Error in video thread processing: {e}")
        finally:
            # Always release video capture object
            cap.release()
            print("Video processing complete.")
