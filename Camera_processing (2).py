# video_processor.py
# ------------------------------------------------------------------
# Handles live video capture from a webcam, performs YOLO-based
# object detection and tracking, and estimates object distances.
#
# This version is optimized for *real-time camera processing*
# rather than offline video files.
# ------------------------------------------------------------------

import cv2
from ultralytics import YOLO
import config
import time
import numpy as np  # Used for efficient numerical array handling

# Import key configuration parameters and result object type
from config import FOCAL_LENGTH, KNOWN_WIDTHS_M, YOLO_CONFIDENCE_THRESHOLD
from ultralytics.engine.results import Results


class VideoProcessor:
    """
    The VideoProcessor class manages real-time frame processing:
      - Loads YOLOv8 model from configuration
      - Continuously reads frames from the webcam
      - Detects and tracks objects
      - Calculates approximate object distances
      - Yields detection results frame by frame for downstream modules
    """

    def __init__(self):
        """Initializes YOLO model and prints configuration info."""
        print("Loading YOLO model...")
        self.model = YOLO(config.YOLO_MODEL)
        print("YOLO model loaded successfully.")
        
        # Warn user if FOCAL_LENGTH has not been calibrated
        if FOCAL_LENGTH == 1000:
            print("WARNING: FOCAL_LENGTH is set to the default (1000).")
            print("Distance estimates will be inaccurate until you calibrate it.")

    # ------------------------------------------------------------------
    # Helper: Distance Estimation
    # ------------------------------------------------------------------
    def _calculate_distance(self, pixel_width, object_class_name):
        """
        Calculate real-world distance (in meters) to a detected object
        using the triangle similarity formula.

        Formula:
            distance = (known_object_width * focal_length) / object_pixel_width

        Args:
            pixel_width (float): The width of the object's bounding box (pixels).
            object_class_name (str): Name of the detected object (used to look up width).

        Returns:
            float | None: Estimated distance in meters, or None if class not found.
        """
        if object_class_name in KNOWN_WIDTHS_M:
            known_width_m = KNOWN_WIDTHS_M[object_class_name]
            if pixel_width > 0:
                distance_m = (known_width_m * FOCAL_LENGTH) / pixel_width
                return distance_m
        return None  # Return None if data insufficient for calculation

    # ------------------------------------------------------------------
    # Main Method: Process Live Camera Stream
    # ------------------------------------------------------------------
    def process_video(self, camera_index, stop_event):
        """
        Continuously processes a live camera feed frame by frame.

        Performs object tracking using YOLO and returns the raw frame
        along with a structured NumPy array containing detection info.

        Args:
            camera_index (int): The index of the camera (0 = default webcam).
            stop_event (threading.Event): Shared flag to stop loop gracefully.

        Yields:
            tuple: (frame, timestamp_sec, tracks)
                - frame: current frame (np.ndarray)
                - timestamp_sec: time elapsed since start of stream
                - tracks: np.ndarray of detections with 8 columns:
                    [x1, y1, x2, y2, track_id, class_id, confidence, distance_m]
        """
        try:
            # Try to open webcam or video stream
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise IOError(f"Error opening video stream: {camera_index}")

            print(f"Video stream loaded successfully: {camera_index}")
            start_time = time.time()

            # Main processing loop
            while cap.isOpened() and not stop_event.is_set():
                success, frame = cap.read()
                if not success:
                    # If frame capture fails, retry after short delay
                    print("Warning: Failed to grab frame.")
                    time.sleep(0.1)
                    continue

                # Calculate timestamp (seconds since stream start)
                timestamp_sec = time.time() - start_time

                # ------------------------------------------------------
                # Run YOLO tracking on the current frame
                # ------------------------------------------------------
                results_list = self.model.track(
                    frame,
                    persist=True,                  # Keep object IDs consistent
                    conf=YOLO_CONFIDENCE_THRESHOLD, # Minimum confidence threshold
                    verbose=False                  # Suppress detailed logs
                )

                # Get first (and only) result object
                results = results_list[0] if results_list else None

                # ------------------------------------------------------
                # Convert detections to structured NumPy array
                # ------------------------------------------------------
                tracks_list = []
                if isinstance(results, Results) and results.boxes.id is not None:
                    boxes_xyxy = results.boxes.xyxy.cpu().numpy()
                    track_ids = results.boxes.id.cpu().numpy().astype(int)
                    class_ids = results.boxes.cls.cpu().numpy().astype(int)
                    confidences = results.boxes.conf.cpu().numpy()

                    # Iterate through each detection and compute distance
                    for xyxy, track_id, class_id, conf in zip(
                        boxes_xyxy, track_ids, class_ids, confidences
                    ):
                        class_name = self.model.model.names[class_id]
                        pixel_width = xyxy[2] - xyxy[0]  # (x2 - x1)
                        distance_m = self._calculate_distance(pixel_width, class_name)

                        # Append formatted detection data
                        tracks_list.append([
                            xyxy[0], xyxy[1], xyxy[2], xyxy[3],  # Bounding box
                            track_id,                             # Unique ID per object
                            class_id,                             # Class index
                            conf,                                 # Detection confidence
                            distance_m if distance_m is not None else -1  # Distance (or -1)
                        ])

                # Convert detections list into a NumPy array
                tracks = np.array(tracks_list) if len(tracks_list) > 0 else np.empty((0, 8))

                # Yield frame + detections to caller (e.g., MemoryBank)
                yield frame, timestamp_sec, tracks

        except Exception as e:
            # Handle unexpected errors without breaking the main app
            if not stop_event.is_set():
                print(f"Error in VideoProcessor: {e}")

        finally:
            # Always release camera resources
            cap.release()
            print("Video stream closed.")
