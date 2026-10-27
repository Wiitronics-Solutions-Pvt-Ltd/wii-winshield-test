import os
import math
import threading
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

SUPPORTED_VEHICLES = {
    "car": "Car",
    "motorcycle": "Motorcycle",
    "bus": "Bus",
    "truck": "Truck",
    "bicycle": "Bicycle"
}

CORNER_NAMES = ["top_left", "top_right", "bottom_right", "bottom_left"]

def box_iou(box1, box2):
    """Computes IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def is_box_inside_roi(box, roi_polygon, frame_shape, mode="windshield_center"):
    """
    Checks if a bounding box [x1, y1, x2, y2] is inside the ROI polygon.
    Modes:
      - 'windshield_center' / 'plate_center': Center point of box is inside ROI
      - 'contact': Bottom-center point of box is inside ROI (road contact)
      - 'windshield_inside' / 'plate_inside': All 4 corners of box are inside ROI
    """
    if not roi_polygon or len(roi_polygon) < 3:
        return True

    h, w = frame_shape[:2]
    pts = np.asarray(roi_polygon, dtype=np.float32)
    if np.all((pts >= 0) & (pts <= 1)):
        pts = pts * np.array([w, h], dtype=np.float32)
    pts = pts.astype(np.float32)

    x1, y1, x2, y2 = box

    if mode == "contact":
        point = (float((x1 + x2) / 2.0), float(y2))
        return cv2.pointPolygonTest(pts, point, False) >= 0
    elif mode in ("windshield_inside", "plate_inside"):
        corners = [
            (float(x1), float(y1)),
            (float(x2), float(y1)),
            (float(x2), float(y2)),
            (float(x1), float(y2))
        ]
        return all(cv2.pointPolygonTest(pts, c, False) >= 0 for c in corners)
    else:  # 'windshield_center' / default
        center = (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))
        return cv2.pointPolygonTest(pts, center, False) >= 0


def select_windshield(predictions, width, height, threshold=0.08):
    """
    Ported from wind_shield/dist/detection.js:
    Selects the best qualifying rectangular windshield region from zero-shot predictions.
    """
    if width <= 0 or height <= 0 or not predictions:
        return None

    def clamp(v):
        return max(0.0, min(1.0, float(v)))

    # Sort predictions by score descending
    sorted_preds = sorted(predictions, key=lambda hit: hit.get("score", 0.0), reverse=True)

    for hit in sorted_preds:
        score = hit.get("score", 0.0)
        box = hit.get("box")
        if not math.isfinite(score) or score < threshold or not box:
            continue

        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        if not all(math.isfinite(v) for v in (xmin, ymin, xmax, ymax)):
            continue

        x1 = clamp(xmin / width)
        y1 = clamp(ymin / height)
        x2 = clamp(xmax / width)
        y2 = clamp(ymax / height)

        if (x2 - x1) * (y2 - y1) < 0.002:
            continue

        points = [
            {"x": round(x1, 6), "y": round(y1, 6)},
            {"x": round(x2, 6), "y": round(y1, 6)},
            {"x": round(x2, 6), "y": round(y2, 6)},
            {"x": round(x1, 6), "y": round(y2, 6)}
        ]

        bbox_px = [
            int(round(x1 * width)),
            int(round(y1 * height)),
            int(round(x2 * width)),
            int(round(y2 * height))
        ]

        return {
            "points_norm": points,
            "bbox_norm": [x1, y1, x2, y2],
            "bbox_px": bbox_px,
            "score": round(float(score), 4)
        }

    return None


class WindshieldAndVehicleDetector:
    """
    Detects vehicles using YOLO and extracts windshield bounding boxes and coordinates
    using the zero-shot OWL-ViT algorithm from wind_shield.
    """

    def __init__(self, model_path="yolov8n.pt", owlvit_model_id="google/owlvit-base-patch32"):
        self.model_path = os.path.abspath(model_path)
        self.model_name = os.path.basename(model_path)
        self._lock = threading.RLock()

        if not os.path.isfile(self.model_path):
            alt_model = "/Users/admin/VSCODE/WII-AI/Video Analytics v4/yolov8n.pt"
            if os.path.isfile(alt_model):
                self.model_path = alt_model
            else:
                raise FileNotFoundError(f"YOLO model file not found: {self.model_path}")

        print(f"[Detector] Loading YOLO from: {self.model_path}")
        self.yolo = YOLO(self.model_path)

        # Vehicle classes
        names = self.yolo.names
        names_dict = names if isinstance(names, dict) else dict(enumerate(names))
        self.target_classes = {
            idx: SUPPORTED_VEHICLES[name.lower()]
            for idx, name in names_dict.items()
            if name.lower() in SUPPORTED_VEHICLES
        }

        # Windshield Zero-Shot queries from wind_shield
        self.queries = ["a car windshield", "a front windscreen", "a rear windshield"]

        # Load OWL-ViT Processor & Model
        print(f"[Detector] Loading OWL-ViT ({owlvit_model_id})...")
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        self.processor = OwlViTProcessor.from_pretrained(owlvit_model_id)
        self.owlvit = OwlViTForObjectDetection.from_pretrained(owlvit_model_id)
        self.owlvit.eval()
        print("[Detector] OWL-ViT model ready for windshield localization.")

    def extract_windshield(self, vehicle_crop, min_confidence=0.08):
        """
        Runs OWL-ViT zero-shot windshield detection on a cropped vehicle.
        Matches the wind_shield algorithm logic.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        h, w = vehicle_crop.shape[:2]
        if w < 24 or h < 24:
            return None

        pil_img = Image.fromarray(cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB))
        inputs = self.processor(text=[self.queries], images=pil_img, return_tensors="pt")

        with self._lock:
            with torch.no_grad():
                outputs = self.owlvit(**inputs)

        target_sizes = torch.Tensor([[h, w]])
        # Use post_process_grounded_object_detection or post_process_object_detection
        if hasattr(self.processor, "post_process_grounded_object_detection"):
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=min_confidence
            )
        else:
            results = self.processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=min_confidence
            )

        boxes = results[0]["boxes"].tolist()
        scores = results[0]["scores"].tolist()

        predictions = []
        for b, s in zip(boxes, scores):
            predictions.append({
                "score": float(s),
                "box": {
                    "xmin": float(b[0]),
                    "ymin": float(b[1]),
                    "xmax": float(b[2]),
                    "ymax": float(b[3])
                }
            })

        match = select_windshield(predictions, w, h, threshold=min_confidence)
        if match:
            return {
                "windshield_bbox": match["bbox_px"],
                "confidence": match["score"],
                "points_norm": match["points_norm"],
                "method": "owlvit"
            }

        return None

    def detect_frame(self, frame, vehicle_conf=0.30, windshield_conf=0.15,
                     roi_polygon=None, roi_match="windshield_center", vehicle_types=None):
        """
        Processes a full video frame:
        1. Detects vehicles using YOLO.
        2. Filters with ROI polygon if in 'contact' mode.
        3. Localizes windshields using the wind_shield OWL-ViT algorithm.
        4. Calculates exact pixel coordinates and normalized YOLO bounding boxes.
        """
        h, w = frame.shape[:2]
        allowed_types = vehicle_types or ["Car", "Bus", "Truck", "Motorcycle"]
        classes = [idx for idx, name in self.target_classes.items() if name in allowed_types]

        if not classes:
            return []

        with self._lock:
            results = self.yolo.predict(
                frame,
                verbose=False,
                imgsz=960,
                conf=vehicle_conf,
                classes=classes,
                iou=0.45,
                agnostic_nms=True
            )[0]

        detections = []
        for box in results.boxes:
            v_conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            v_type = self.target_classes.get(cls_id, "Vehicle")

            coords = box.xyxy[0].tolist()
            vx1, vy1 = max(0, int(coords[0])), max(0, int(coords[1]))
            vx2, vy2 = min(w, math.ceil(coords[2])), min(h, math.ceil(coords[3]))

            # Filter tiny vehicle detections
            if (vx2 - vx1) < 25 or (vy2 - vy1) < 25:
                continue

            # First-pass vehicle ROI check if in 'contact' mode
            if roi_polygon and roi_match == "contact":
                if not is_box_inside_roi([vx1, vy1, vx2, vy2], roi_polygon, frame.shape, mode="contact"):
                    continue

            v_crop = frame[vy1:vy2, vx1:vx2]
            if v_crop.size == 0:
                continue

            # Extract windshield inside vehicle crop using OWL-ViT algorithm
            ws_info = self.extract_windshield(v_crop, min_confidence=windshield_conf)

            if ws_info and ws_info["confidence"] >= windshield_conf:
                wx1, wy1, wx2, wy2 = ws_info["windshield_bbox"]

                # Map to full frame coordinates
                fx1 = max(0, min(w, vx1 + wx1))
                fy1 = max(0, min(h, vy1 + wy1))
                fx2 = max(0, min(w, vx1 + wx2))
                fy2 = max(0, min(h, vy1 + wy2))

                box_w = fx2 - fx1
                box_h = fy2 - fy1

                if box_w < 12 or box_h < 8:
                    continue

                ws_frame_box = [fx1, fy1, fx2, fy2]

                # STRICT ROI CHECK FOR THE WINDSHIELD:
                if roi_polygon and len(roi_polygon) >= 3 and roi_match != "contact":
                    if not is_box_inside_roi(ws_frame_box, roi_polygon, frame.shape, mode=roi_match):
                        continue

                # Normalized YOLO coordinates: 0 x_center y_center width height
                x_center = (fx1 + fx2) / 2.0 / w
                y_center = (fy1 + fy2) / 2.0 / h
                norm_w = box_w / w
                norm_h = box_h / h

                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                norm_w = max(0.001, min(1.0, norm_w))
                norm_h = max(0.001, min(1.0, norm_h))

                # 4 Corner coordinates (matching wind_shield schema)
                corners_px = [
                    {"corner": "top_left", "x": fx1, "y": fy1},
                    {"corner": "top_right", "x": fx2, "y": fy1},
                    {"corner": "bottom_right", "x": fx2, "y": fy2},
                    {"corner": "bottom_left", "x": fx1, "y": fy2}
                ]
                corners_norm = [
                    {"corner": "top_left", "x": round(fx1 / w, 6), "y": round(fy1 / h, 6)},
                    {"corner": "top_right", "x": round(fx2 / w, 6), "y": round(fy1 / h, 6)},
                    {"corner": "bottom_right", "x": round(fx2 / w, 6), "y": round(fy2 / h, 6)},
                    {"corner": "bottom_left", "x": round(fx1 / w, 6), "y": round(fy2 / h, 6)}
                ]

                detections.append({
                    "vehicle_type": v_type,
                    "vehicle_confidence": round(v_conf, 3),
                    "vehicle_bbox": [vx1, vy1, vx2, vy2],
                    "windshield_bbox": ws_frame_box,
                    "windshield_w": box_w,
                    "windshield_h": box_h,
                    "corners_px": corners_px,
                    "corners_norm": corners_norm,
                    "yolo_bbox": {
                        "class_id": 0,
                        "class_name": "windshield",
                        "x_center": round(x_center, 6),
                        "y_center": round(y_center, 6),
                        "width": round(norm_w, 6),
                        "height": round(norm_h, 6)
                    },
                    "yolo_line": f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}",
                    "confidence": ws_info["confidence"],
                    "extraction_method": ws_info["method"]
                })

        # Remove overlapping windshield boxes
        unique = []
        for detection in sorted(detections, key=lambda d: d["confidence"], reverse=True):
            if not any(box_iou(detection["windshield_bbox"], existing["windshield_bbox"]) > 0.5 for existing in unique):
                unique.append(detection)

        return unique
