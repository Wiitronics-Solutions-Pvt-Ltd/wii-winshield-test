import os
import time
import math
import threading
import cv2
import numpy as np

class VideoWindshieldExtractor:
    """
    Manages the video analysis session, running frame-by-frame windshield detection
    and coordinate extraction in a background thread.
    Supports ROI selected area filtering and custom frame counts/ranges.
    """

    def __init__(self, detector, base_dir="."):
        self.detector = detector
        self.base_dir = os.path.abspath(base_dir)
        self.output_dir = os.path.join(self.base_dir, "output")
        self.crops_dir = os.path.join(self.output_dir, "crops")
        self.frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(self.crops_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)

        self.video_path = None
        self.video_meta = None

        # State
        self.is_processing = False
        self.is_paused = False
        self.should_stop = False
        self.progress = 0.0
        self.current_frame = 0
        self.start_frame = 0
        self.end_frame = 0
        self.total_frames_target = 0
        self.processed_count = 0
        self.fps = 0.0
        self.eta_seconds = 0
        self.status_message = "Ready"

        # ROI state
        self.roi_polygon = []  # Normalized points [[x, y], ...]
        self.roi_match = "windshield_center"

        # Results
        self.lock = threading.RLock()
        self.windshield_records = []  # Flat list of all extracted windshield dicts
        self.records_by_frame = {}  # { frame_idx: [detection_dict, ...] }
        self.saved_frames = {}  # { frame_idx: frame_path }
        self.cropped_images = {}  # { filename: numpy_image }

        # Latest annotated frame buffer for live web stream
        self.latest_live_frame = None
        self._thread = None

    def set_roi(self, polygon, mode="windshield_center"):
        """Sets the active ROI polygon (normalized coordinates [[x,y], ...])."""
        with self.lock:
            self.roi_polygon = polygon or []
            self.roi_match = mode or "windshield_center"

    def clear_results(self):
        """Resets all extracted records and output caches."""
        with self.lock:
            self.windshield_records.clear()
            self.records_by_frame.clear()
            self.saved_frames.clear()
            self.cropped_images.clear()
            self.progress = 0.0
            self.processed_count = 0
            self.status_message = "Results cleared"

    def delete_detection(self, det_id):
        """Deletes a detection item by ID."""
        with self.lock:
            target = None
            for d in self.windshield_records:
                if d.get("id") == det_id:
                    target = d
                    break
            if target:
                self.windshield_records.remove(target)
                f_idx = target.get("frame_idx")
                if f_idx in self.records_by_frame:
                    self.records_by_frame[f_idx] = [
                        x for x in self.records_by_frame[f_idx] if x.get("id") != det_id
                    ]
                    if not self.records_by_frame[f_idx]:
                        del self.records_by_frame[f_idx]
                return True
        return False

    def start_extraction(self, video_path, video_meta, config):
        """
        Launches the extraction thread.
        config keys:
          - start_frame (int)
          - end_frame (int)
          - frame_step (int, e.g. 1, 5, 15)
          - vehicle_conf (float)
          - windshield_conf (float)
          - roi_polygon (list of [x, y])
          - roi_match (str)
          - save_frames (bool)
          - vehicle_types (list of str)
        """
        if self.is_processing:
            return False, "An extraction task is already running."

        self.video_path = video_path
        self.video_meta = video_meta

        self.start_frame = max(0, int(config.get("start_frame", 0)))
        total_in_vid = video_meta.get("total_frames", 1000)
        self.end_frame = min(total_in_vid, int(config.get("end_frame", total_in_vid)))
        self.frame_step = max(1, int(config.get("frame_step", 5)))
        self.vehicle_conf = float(config.get("vehicle_conf", 0.30))
        self.windshield_conf = float(config.get("windshield_conf", 0.15))
        self.save_frames = bool(config.get("save_frames", True))
        self.vehicle_types = config.get("vehicle_types", ["Car", "Bus", "Truck"])

        if "roi_polygon" in config:
            self.set_roi(config["roi_polygon"], config.get("roi_match", "windshield_center"))

        self.total_frames_target = len(range(self.start_frame, self.end_frame, self.frame_step))
        self.processed_count = 0
        self.is_processing = True
        self.is_paused = False
        self.should_stop = False
        self.status_message = "Starting windshield extraction..."

        # Clear existing records for fresh run
        self.clear_results()

        self._thread = threading.Thread(target=self._run_extraction, daemon=True)
        self._thread.start()
        return True, "Extraction started successfully."

    def pause(self):
        self.is_paused = True
        self.status_message = "Extraction paused."

    def resume(self):
        self.is_paused = False
        self.status_message = "Extraction resumed."

    def stop(self):
        self.should_stop = True
        self.status_message = "Stopping extraction..."

    def _run_extraction(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.is_processing = False
            self.status_message = "Error: Could not open video file."
            return

        frame_rate = self.video_meta.get("fps", 30.0) or 30.0
        start_time = time.time()
        last_calc_time = start_time
        frames_since_calc = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        current_idx = self.start_frame

        det_seq_id = 0

        try:
            while current_idx < self.end_frame and not self.should_stop:
                while self.is_paused and not self.should_stop:
                    time.sleep(0.1)

                if self.should_stop:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                timestamp = round(current_idx / frame_rate, 2)

                # Execute detection
                detections = self.detector.detect_frame(
                    frame=frame,
                    vehicle_conf=self.vehicle_conf,
                    windshield_conf=self.windshield_conf,
                    roi_polygon=self.roi_polygon,
                    roi_match=self.roi_match,
                    vehicle_types=self.vehicle_types
                )

                # Process detections
                annotated_frame = frame.copy()

                # Draw ROI Polygon on annotated live frame
                if self.roi_polygon and len(self.roi_polygon) >= 3:
                    pts = np.asarray(self.roi_polygon, dtype=np.float32)
                    if np.all((pts >= 0) & (pts <= 1)):
                        pts = pts * np.array([w, h], dtype=np.float32)
                    pts = pts.astype(np.int32).reshape((-1, 1, 2))

                    # Cyan translucent fill
                    overlay = annotated_frame.copy()
                    cv2.fillPoly(overlay, [pts], (255, 220, 0))  # BGR Cyan/Blue
                    cv2.addWeighted(overlay, 0.20, annotated_frame, 0.80, 0, annotated_frame)
                    cv2.polylines(annotated_frame, [pts], True, (255, 255, 0), 2, cv2.LINE_AA)

                frame_detections = []

                for d in detections:
                    det_seq_id += 1
                    wb = d["windshield_bbox"]
                    wx1, wy1, wx2, wy2 = wb
                    w_crop = frame[wy1:wy2, wx1:wx2]

                    crop_filename = f"windshield_f{current_idx:06d}_id{det_seq_id}.jpg"
                    crop_path = os.path.join(self.crops_dir, crop_filename)

                    if w_crop.size > 0:
                        cv2.imwrite(crop_path, w_crop)
                        with self.lock:
                            self.cropped_images[crop_filename] = w_crop

                    rec = {
                        "id": det_seq_id,
                        "frame_idx": current_idx,
                        "timestamp_sec": timestamp,
                        "crop_filename": crop_filename,
                        "crop_url": f"/api/crops/{crop_filename}",
                        "vehicle_type": d["vehicle_type"],
                        "vehicle_confidence": d["vehicle_confidence"],
                        "vehicle_bbox": d["vehicle_bbox"],
                        "windshield_bbox": wb,
                        "windshield_w": d["windshield_w"],
                        "windshield_h": d["windshield_h"],
                        "corners_px": d.get("corners_px", []),
                        "corners_norm": d.get("corners_norm", []),
                        "confidence": d["confidence"],
                        "yolo_bbox": d["yolo_bbox"],
                        "yolo_line": d["yolo_line"],
                        "extraction_method": d.get("extraction_method", "owlvit")
                    }
                    frame_detections.append(rec)

                    # Draw on annotated stream
                    # 1. Vehicle box (thin muted cyan/grey)
                    vx1, vy1, vx2, vy2 = d["vehicle_bbox"]
                    cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), (180, 140, 80), 1)

                    # 2. Windshield box (neon emerald green)
                    cv2.rectangle(annotated_frame, (wx1, wy1), (wx2, wy2), (0, 255, 128), 2)

                    # Label pill
                    conf_pct = int(d["confidence"] * 100)
                    lbl = f"Windshield ({d['vehicle_type']}) {conf_pct}%"
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(
                        annotated_frame,
                        (wx1, max(0, wy1 - th - 6)),
                        (wx1 + tw + 6, max(0, wy1)),
                        (0, 255, 128),
                        -1
                    )
                    cv2.putText(
                        annotated_frame,
                        lbl,
                        (wx1 + 3, max(0, wy1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA
                    )

                with self.lock:
                    if frame_detections:
                        self.windshield_records.extend(frame_detections)
                        self.records_by_frame[current_idx] = frame_detections

                        # Save full frame if requested
                        if self.save_frames:
                            frame_filename = f"frame_{current_idx:06d}.jpg"
                            frame_path = os.path.join(self.frames_dir, frame_filename)
                            cv2.imwrite(frame_path, frame)
                            self.saved_frames[current_idx] = frame_path

                    # Update live preview frame (downscaled for fast SSE/MJPEG transmission)
                    preview_w = min(1280, w)
                    preview_h = int(h * (preview_w / w))
                    self.latest_live_frame = cv2.resize(annotated_frame, (preview_w, preview_h))

                self.processed_count += 1
                frames_since_calc += 1
                current_time = time.time()
                elapsed_since = current_time - last_calc_time

                if elapsed_since >= 0.5:
                    self.fps = round(frames_since_calc / elapsed_since, 1)
                    frames_since_calc = 0
                    last_calc_time = current_time

                    rem_frames = self.total_frames_target - self.processed_count
                    self.eta_seconds = max(0, int(rem_frames / self.fps)) if self.fps > 0 else 0

                self.progress = round((self.processed_count / max(1, self.total_frames_target)) * 100, 1)
                self.current_frame = current_idx
                current_idx += self.frame_step

        except Exception as e:
            self.status_message = f"Error during extraction: {str(e)}"
            print(f"[Extractor] Error: {e}")
        finally:
            cap.release()
            self.is_processing = False
            self.is_paused = False
            if self.should_stop:
                self.status_message = "Extraction stopped by user."
            else:
                self.status_message = f"Completed: {len(self.windshield_records)} windshields extracted."

    def get_live_frame_bytes(self):
        """Returns the current live preview frame as JPEG bytes."""
        with self.lock:
            if self.latest_live_frame is not None:
                ret, jpeg = cv2.imencode(".jpg", self.latest_live_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret:
                    return jpeg.tobytes()
        return None

    def get_telemetry(self):
        """Returns real-time processing statistics."""
        with self.lock:
            return {
                "is_processing": self.is_processing,
                "is_paused": self.is_paused,
                "progress": self.progress,
                "current_frame": self.current_frame,
                "processed_count": self.processed_count,
                "total_frames_target": self.total_frames_target,
                "windshields_count": len(self.windshield_records),
                "fps": self.fps,
                "eta_seconds": self.eta_seconds,
                "status_message": self.status_message
            }

    def get_results(self):
        """Returns all extracted detections and metadata."""
        with self.lock:
            return {
                "total_windshields": len(self.windshield_records),
                "total_annotated_frames": len(self.records_by_frame),
                "windshield_records": list(self.windshield_records),
                "records_by_frame": dict(self.records_by_frame)
            }
