# WII-AI Windshield Extractor & YOLO Dataset Studio

A web application designed for traffic intelligence and machine learning engineering. Ingest traffic footage, isolate specific road lanes or areas alone, choose exact frame counts and intervals, extract vehicle windshield bounding boxes and 4-corner coordinates using the zero-shot algorithm from `wind_shield`, and export ready-to-train **YOLO Object Detection Datasets** (normalized coordinates, train/val splits, images, labels, `data.yaml`, CSV, and JSON).

---

## 🚀 The 5-Step Workflow

```
[ 01: Region of Interest ] ➔ [ 02: Frame Selection ] ➔ [ 03: Live Extraction ] ➔ [ 04: Inspector ] ➔ [ 05: YOLO Export ]
  • Draw polygon on video      • Quick presets (50..all)     • Live stream with ROI       • Review boxes      • Train/val ZIP
  • Isolate lanes / areas      • Sampling interval (1..60f)  • Bounding boxes             • Verify coordinates• Labels & YAML
  • Ignore outside traffic     • Time & frame estimation     • Real-time telemetry        • Delete false pos  • CSV & JSON
```

### 1. Region of Interest (Selected Area Alone) (`Tab 01`)
* **Isolate Selected Area**: Detect vehicle windshields strictly inside your chosen polygon (e.g. entrance lane, specific highway lane, toll gate).
* **Interactive Polygon Studio**:
  * Click anywhere on the video frame canvas to add vertices.
  * Drag any vertex point to fine-tune boundaries.
* **Quick Presets**:
  * `Main Road (Bottom 65%)`
  * `Left Lane`
  * `Right Lane`
  * `Full Frame (Reset)`
* **Area Matching Rule**:
  * `Windshield Center Inside Area` (Default & Recommended)
  * `Vehicle Road Contact Point (Bottom-Center)`
  * `Entire Windshield Strictly Inside Area (All 4 Corners)`

### 2. Frame Selection & Estimation (`Tab 02`)
* **How Many Frames to Process**:
  * **Quick Presets**: `⚡ 50 Frames`, `⚡ 100 Frames`, `⚡ 300 Frames`, `⚡ 500 Frames`, `⚡ 1,000 Frames`, `🎬 Full Video`.
  * **Custom Range**: Start Frame and End Frame inputs with timeline scrubbing.
* **Sampling Interval**:
  * `Every Frame`: 100% full scan (maximum annotations).
  * `Every 5 frames (~0.16s)`: Recommended balance between dataset diversity and processing speed.
  * `Every 15 frames (~0.5s)` / `Every 30 frames (~1.0s)`: Fast scan.
* **Windshield Confidence Threshold**:
  * Slider from 0.05 to 0.60 (default 0.15) for OWL-ViT zero-shot scoring.
* **Live Estimation Card**:
  * Dynamic calculation of **Total Frames to Scan**.
  * Real-time calculation of **Estimated Total Duration** and **Active Area**.

### 3. Live Extraction Stream (`Tab 03`)
* **Live Annotated Stream**: Shows the active ROI boundary in neon cyan with translucent highlight, vehicle boxes in muted steel, plus detected windshield bounding boxes in bright neon green in real time.
* **Telemetry & Speed**: Real-time FPS, elapsed time, ETA countdown, and windshield counters.
* **Controls**: Pause, Resume, or Stop extraction.

### 4. Frame & Coordinates Inspector (`Tab 04`)
* **Dual-View Studio**:
  * **Left**: Canvas overlaying pixel-perfect colored bounding boxes with labels, confidence, and 4 corner keypoints.
  * **Right**: Extracted windshield gallery showing crop thumbnails, vehicle types, exact pixel bounds (`x1, y1, x2, y2`), 4 corner points (`TL, TR, BR, BL`), and normalized YOLO strings.
* **One-Click Actions**:
  * `Copy YOLO`: Copies the exact YOLO line (`0 0.348958 0.502778 0.053125 0.027778`) to clipboard.
  * `Delete`: Remove false positives before dataset export.
  * `Frame Cycling`: Jump directly between frames where windshields are present.

### 5. YOLO Dataset & Export Center (`Tab 05`)
* **1. Full YOLO Training Dataset (`.zip`)**:
  Contains standard Ultralytics YOLO structure with train/val splits (e.g. 80% Train / 20% Val):
  ```
  yolo_windshield_dataset/
  ├── data.yaml
  ├── README.md
  ├── train/
  │   ├── images/  (e.g., frame_000138.jpg)
  │   └── labels/  (e.g., frame_000138.txt)
  └── val/
      ├── images/  (e.g., frame_000180.jpg)
      └── labels/  (e.g., frame_000180.txt)
  ```
* **2. Coordinates CSV Spreadsheet (`.csv`)**:
  Columns: `frame_index, timestamp_sec, image_filename, class_id, class_name, x1_pixel, y1_pixel, x2_pixel, y2_pixel, box_width_px, box_height_px, frame_width, frame_height, x_center_norm, y_center_norm, width_norm, height_norm, top_left_x, top_left_y, top_right_x, top_right_y, bottom_right_x, bottom_right_y, bottom_left_x, bottom_left_y, confidence, vehicle_type, vehicle_bbox, extraction_method`.
* **3. Structured JSON (`.json`)**:
  Complete machine-readable JSON containing video metadata, frame dimensions, 4 corners, and bounding box coordinates for custom ML pipelines.
* **4. YOLO Labels Only (`.zip`)**:
  Text label files (`frame_XXXXXX.txt`) and `data.yaml` for users who already have frame images.
* **5. Cropped Windshields (`.zip`)**:
  Extracted glass images for model validation and specialized glass classification.

---

## 🏃 Quickstart & Running

### 1. Start the Studio
From the terminal:
```bash
cd "/Users/admin/VSCODE/WII-AI/VA_winshield v1"
./run.sh
```
Then open your browser at **http://localhost:5056**.

Run regression checks with `python3 -m unittest discover -s tests -v`.

---

## 🧠 Training a YOLO Model on Your Exported Dataset

Once you download and extract `yolo_windshield_dataset.zip`:

### 1. Install Ultralytics
```bash
pip install ultralytics
```

### 2. Train with YOLOv8 or YOLOv11 CLI
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=16
```
