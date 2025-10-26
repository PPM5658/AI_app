# video_processor.py
# ------------------------------------------------------------------
# Handles reading video frames, detecting/tracking objects using YOLOv8,
# calculating distances, and yielding structured detection results.
# Extended with multi-object highlight, distance labeling, smart frame search,
# and color-coded bounding boxes for better visualization.
# ------------------------------------------------------------------

import cv2
import datetime
import shutil
import os
import threading
import numpy as np
import time
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results
import config
from config import (
    FRAME_SKIP,
    YOLO_CONFIDENCE_THRESHOLD,
    FOCAL_LENGTH,
    KNOWN_WIDTHS_M,
)


class VideoProcessor:
    def __init__(self):
        """Initialize YOLO model and internal state for video and highlights."""
        print("Loading YOLO model...")
        self.model = YOLO(config.YOLO_MODEL)
        print("YOLO model loaded.")

        if FOCAL_LENGTH == 1000:
            print("⚠️ WARNING: FOCAL_LENGTH is set to default (1000).")
            print("Distance estimates may be inaccurate until calibrated.")

        # Shared state for highlights
        self.last_frame = None
        self.last_results = None
        self.frame_lock = threading.Lock()
        self.results_buffer = []  # store last few YOLO frames for smart search

        os.makedirs(config.CAPTURE_DIR, exist_ok=True)

    # ------------------------------------------------------------
    # Distance Estimation
    # ------------------------------------------------------------
    def _calculate_distance(self, pixel_width, object_class_name):
        """Estimate real-world distance (m) from bounding box width."""
        if object_class_name in KNOWN_WIDTHS_M and pixel_width > 0:
            known_width_m = KNOWN_WIDTHS_M[object_class_name]
            return (known_width_m * FOCAL_LENGTH) / pixel_width
        return None

    # ------------------------------------------------------------
    # Main Video Processing Loop
    # ------------------------------------------------------------
    def process_video(self, video_path, stop_event=None):
        """
        Read video frames, perform YOLO detection/tracking, and yield structured results.
        Yields (frame, timestamp_sec, tracks) tuples.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Error opening video file: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            print(f"🎥 Video loaded: {total_frames} frames @ {fps:.2f} FPS")

            frame_count = 0
            with tqdm(total=total_frames, desc="Processing Video") as pbar:
                while cap.isOpened():
                    if stop_event is not None and stop_event.is_set():
                        print("🛑 Stop signal received. Exiting video loop.")
                        break

                    success, frame = cap.read()
                    if not success:
                        break

                    if frame_count % FRAME_SKIP == 0:
                        timestamp_sec = pbar.n / fps
                        results_list = self.model.track(
                            frame, persist=True, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False
                        )
                        results = results_list[0] if results_list else None
                        tracks_list = []

                        if isinstance(results, Results) and results.boxes.id is not None:
                            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
                            track_ids = results.boxes.id.cpu().numpy().astype(int)
                            class_ids = results.boxes.cls.cpu().numpy().astype(int)
                            confidences = results.boxes.conf.cpu().numpy()

                            for xyxy, track_id, class_id, conf in zip(
                                boxes_xyxy, track_ids, class_ids, confidences
                            ):
                                class_name = self.model.model.names[class_id]
                                pixel_width = xyxy[2] - xyxy[0]
                                distance_m = self._calculate_distance(pixel_width, class_name)
                                tracks_list.append(
                                    [
                                        xyxy[0], xyxy[1], xyxy[2], xyxy[3],
                                        track_id, class_id, conf,
                                        distance_m if distance_m else -1,
                                    ]
                                )

                        tracks = np.array(tracks_list) if len(tracks_list) > 0 else np.empty((0, 8))

                        # --- Keep a rolling buffer for smart frame search ---
                        if results is not None:
                            self.results_buffer.append(results)
                            if len(self.results_buffer) > 30:  # keep only last 30 frames
                                self.results_buffer.pop(0)

                        with self.frame_lock:
                            self.last_frame = frame.copy()
                            self.last_results = results

                        yield frame, timestamp_sec, tracks

                    pbar.update(1)
                    frame_count += 1

        except Exception as e:
            if stop_event is None or not stop_event.is_set():
                print(f"❌ Error in video processing: {e}")
        finally:
            cap.release()
            print("✅ Video processing complete.")

    # ------------------------------------------------------------
    # Select multiple detections for highlight
    # ------------------------------------------------------------
    def _select_detections(self, labels=None):
        """Select detections by one or more class names."""
        with self.frame_lock:
            results = self.last_results

        if not results or len(results.boxes) == 0:
            return []

        boxes = results.boxes
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
        all_detections = []

        if labels:
            labels = [lbl.lower() for lbl in labels]
            for xyxy, cls_id, conf in zip(boxes.xyxy, cls_ids, confs):
                label_name = self.model.names[int(cls_id)].lower()
                if label_name in labels:
                    all_detections.append((xyxy, cls_id, conf))
        else:
            for xyxy, cls_id, conf in zip(boxes.xyxy, cls_ids, confs):
                all_detections.append((xyxy, cls_id, conf))
        return all_detections

    # ------------------------------------------------------------
    # Capture Highlighted Frame with Smart Search + Distance
    # ------------------------------------------------------------
    def capture_highlight(self, target_labels):
        """
        Capture a frame with all objects matching the user's query (target_labels).
        Searches recent frames if current one has no match.
        Draws bounding boxes with confidence and distance above each box.
        Saves annotated frame both locally and to Google Drive.
        """
        import datetime

        # Normalize query labels
        target_labels = [lbl.lower().strip() for lbl in target_labels]

        # Validate frame availability
        if self.last_frame is None:
            return None, "No recent frame available yet."

        # ---------------------------------------------------------------------
        # SMART SEARCH: find recent frames where query objects appeared
        # ---------------------------------------------------------------------
        recent_results = []
        try:
            candidate_results = list(reversed(self.results_buffer[-10:])) if self.results_buffer else []
            found_in_recent = False

            for result in candidate_results:
                for box in getattr(result, "boxes", []):
                    cls_id = int(box.cls[0]) if hasattr(box, "cls") else None
                    if cls_id is not None:
                        class_name = self.model.names.get(cls_id, "unknown").lower()
                        if class_name in target_labels:
                            recent_results = [result]
                            found_in_recent = True
                            break
                if found_in_recent:
                    break

            # fallback if none found
            if not found_in_recent:
                recent_results = [self.last_results] if self.last_results else []
        except Exception as e:
            print(f"[Visual Q&A] Smart frame search failed: {e}")
            recent_results = [self.last_results] if self.last_results else []

        if not recent_results:
            return None, f"No detection results found for {target_labels}"

        frame = self.last_frame.copy()

        # Prepare folders
        os.makedirs("captures", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"highlight_{'_'.join(target_labels)}_{timestamp}.jpg"
        save_path = os.path.join("captures", filename)

        objects_found = 0

        # ------------------------------------------------------------
        # DRAW DETECTIONS
        # ------------------------------------------------------------
        color_map = {
            "bottle": (0, 255, 0),   # green
            "chair": (255, 0, 0),    # blue
            "person": (0, 165, 255), # orange
            "knife": (0, 0, 255),    # red
        }

        for r in recent_results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0]) if hasattr(box, "cls") else None
                if cls_id is None:
                    continue

                class_name = self.model.names.get(cls_id, "unknown").lower()
                if class_name not in target_labels:
                    continue

                # Coordinates and confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0

                # Distance estimate
                distance_m = self._calculate_distance(abs(x2 - x1), class_name)
                distance_text = f"{distance_m:.2f} m" if distance_m else "?"

                # Bounding box + labels
                color = color_map.get(class_name, (0, 255, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{class_name} ({conf*100:.0f}%)"
                cv2.putText(frame, label_text, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"Dist: {distance_text}",
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                objects_found += 1

        if objects_found == 0:
            return None, f"No objects found for labels: {target_labels}"

        # ------------------------------------------------------------
        # SAVE ANNOTATED FRAME LOCALLY
        # ------------------------------------------------------------
        cv2.imwrite(save_path, frame)
        print(f"💾 Multi-object highlighted frame saved: {save_path}")

        # ------------------------------------------------------------
        # SAVE TO GOOGLE DRIVE (if mounted)
        # ------------------------------------------------------------
        try:
            drive_dir = "/content/drive/MyDrive/ObjectTrackingCaptures"
            if os.path.exists("/content/drive/MyDrive"):
                os.makedirs(drive_dir, exist_ok=True)
                dest = os.path.join(drive_dir, filename)
                shutil.copy(save_path, dest)
                print(f"📂 Also saved to Drive: {dest}")
        except Exception as e:
            print(f"⚠️ Drive save failed: {e}")

        return save_path, None
