import os
import io
import glob
import json
import math
import time
import cv2
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from flask_cors import CORS

from backend.video_info import get_video_metadata
from backend.detector import WindshieldAndVehicleDetector
from backend.extractor import VideoWindshieldExtractor
from backend.yolo_exporter import YOLODatasetExporter

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB max

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize Detector and Extractor
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
if not os.path.exists(MODEL_PATH):
    alt_model = "/Users/admin/VSCODE/WII-AI/Video Analytics v4/yolov8n.pt"
    if os.path.exists(alt_model):
        MODEL_PATH = alt_model

print("[App] Initializing Windshield & Vehicle Detector...")
detector = WindshieldAndVehicleDetector(
    model_path=MODEL_PATH,
    owlvit_model_id="google/owlvit-base-patch32"
)
extractor = VideoWindshieldExtractor(detector=detector, base_dir=BASE_DIR)
exporter = YOLODatasetExporter(output_dir=OUTPUT_FOLDER)

# Global active video state
current_video_path = None
current_video_meta = None

def get_available_videos():
    """Scans for video files in workspace, uploads, and parent folders."""
    videos = []
    # 1. Local workspace and uploads
    patterns = [
        os.path.join(BASE_DIR, "*.mp4"),
        os.path.join(BASE_DIR, "*.mov"),
        os.path.join(UPLOAD_FOLDER, "*.*"),
        os.path.join(BASE_DIR, "..", "VideoAnalytics W1", "sample_video.mp4"),
        os.path.join(BASE_DIR, "..", "Video Analytics v4", "*.mp4"),
    ]
    seen = set()
    for pat in patterns:
        for p in glob.glob(pat):
            ap = os.path.abspath(p)
            if ap not in seen and os.path.isfile(ap) and allowed_file(ap):
                seen.add(ap)
                name = os.path.basename(ap)
                if "sample_video" in name:
                    display = f"🎬 Sample Video ({name})"
                else:
                    display = f"📹 {name}"
                videos.append({
                    "name": name,
                    "path": ap,
                    "display": display,
                    "size_mb": round(os.path.getsize(ap) / (1024 * 1024), 1)
                })
    return videos

# Auto-set initial video if available
videos_list = get_available_videos()
if videos_list:
    current_video_path = videos_list[0]["path"]
    try:
        current_video_meta = get_video_metadata(current_video_path)
    except Exception as e:
        print(f"[App] Video metadata warning: {e}")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/video/list", methods=["GET"])
def api_video_list():
    videos = get_available_videos()
    return jsonify({
        "videos": videos,
        "current_video": current_video_path
    })


@app.route("/api/video/upload", methods=["POST"])
def api_video_upload():
    global current_video_path, current_video_meta
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        current_video_path = save_path
        current_video_meta = get_video_metadata(current_video_path)

        return jsonify({
            "message": "Upload successful",
            "metadata": current_video_meta,
            "path": current_video_path
        })
    return jsonify({"error": "File type not supported"}), 400


@app.route("/api/video/info", methods=["GET", "POST"])
def api_video_info():
    global current_video_path, current_video_meta
    video_path = request.args.get("path") or (request.json.get("path") if request.is_json else None)

    if not video_path:
        video_path = current_video_path

    if not video_path or not os.path.isfile(video_path):
        videos = get_available_videos()
        if videos:
            video_path = videos[0]["path"]
        else:
            return jsonify({"error": "No video available. Please upload a video."}), 404

    try:
        current_video_path = video_path
        current_video_meta = get_video_metadata(video_path)
        return jsonify(current_video_meta)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/frame", methods=["GET"])
def api_video_frame():
    """Extracts a specific frame image as JPEG for scrubbing & ROI drawing."""
    global current_video_path
    video_path = request.args.get("path") or current_video_path
    frame_idx = int(request.args.get("frame_idx", 0))

    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "Video file not found"}), 404

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Could not open video file"}), 500

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(total_frames - 1, frame_idx))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return jsonify({"error": "Could not read frame"}), 500

    # Optional resize for faster transfer if requested
    max_dim = int(request.args.get("max_dim", 0))
    if max_dim > 0:
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ret:
        return jsonify({"error": "Could not encode frame"}), 500

    return Response(buffer.tobytes(), mimetype="image/jpeg")


@app.route("/api/video/extract", methods=["POST"])
def api_video_extract():
    global current_video_path, current_video_meta
    data = request.json or {}

    video_path = data.get("video_path") or current_video_path
    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "Valid video path is required."}), 400

    if not current_video_meta or current_video_meta.get("filepath") != video_path:
        current_video_meta = get_video_metadata(video_path)

    success, msg = extractor.start_extraction(
        video_path=video_path,
        video_meta=current_video_meta,
        config=data
    )

    if not success:
        return jsonify({"error": msg}), 400

    return jsonify({"message": msg, "target_frames": extractor.total_frames_target})


@app.route("/api/video/pause", methods=["POST"])
def api_video_pause():
    extractor.pause()
    return jsonify({"message": "Paused"})


@app.route("/api/video/resume", methods=["POST"])
def api_video_resume():
    extractor.resume()
    return jsonify({"message": "Resumed"})


@app.route("/api/video/stop", methods=["POST"])
def api_video_stop():
    extractor.stop()
    return jsonify({"message": "Stopped"})


@app.route("/api/video/telemetry", methods=["GET"])
def api_video_telemetry():
    return jsonify(extractor.get_telemetry())


@app.route("/api/video/stream")
def api_video_stream():
    """MJPEG stream of live annotated extraction view."""
    def generate():
        last_sent = None
        while True:
            frame_bytes = extractor.get_live_frame_bytes()
            if frame_bytes is not None and frame_bytes != last_sent:
                last_sent = frame_bytes
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.04)  # ~25 fps cap

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/video/results", methods=["GET"])
def api_video_results():
    return jsonify(extractor.get_results())


@app.route("/api/delete-detection/<int:det_id>", methods=["DELETE", "POST"])
def api_delete_detection(det_id):
    success = extractor.delete_detection(det_id)
    if success:
        return jsonify({"message": f"Detection {det_id} deleted."})
    return jsonify({"error": "Detection not found"}), 404


@app.route("/api/clear-detections", methods=["POST"])
def api_clear_detections():
    extractor.clear_results()
    return jsonify({"message": "All detections cleared."})


@app.route("/api/crops/<filename>")
def api_serve_crop(filename):
    return send_from_directory(extractor.crops_dir, filename)


# Export Endpoints
@app.route("/api/export/yolo", methods=["GET"])
def api_export_yolo():
    """Generates and downloads full YOLO Training Dataset ZIP."""
    train_split = float(request.args.get("split", 0.8))
    results = extractor.get_results()
    records_by_frame = results["records_by_frame"]

    if not records_by_frame:
        return jsonify({"error": "No windshields extracted yet to export."}), 400

    zip_buffer = exporter.export_full_yolo_zip(
        records_by_frame=records_by_frame,
        frame_images=extractor.saved_frames,
        train_ratio=train_split,
        class_names=["windshield"]
    )

    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="yolo_windshield_dataset.zip"
    )


@app.route("/api/export/csv", methods=["GET"])
def api_export_csv():
    """Generates and downloads Coordinates CSV."""
    results = extractor.get_results()
    records = results["windshield_records"]

    if not records:
        return jsonify({"error": "No windshields extracted yet to export."}), 400

    csv_data = exporter.export_csv(records, video_meta=current_video_meta)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=windshield_coordinates.csv"}
    )


@app.route("/api/export/json", methods=["GET"])
def api_export_json():
    """Generates and downloads rich JSON dataset."""
    results = extractor.get_results()
    records = results["windshield_records"]

    if not records:
        return jsonify({"error": "No windshields extracted yet to export."}), 400

    json_data = exporter.export_json(records, video_meta=current_video_meta)
    return Response(
        json_data,
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=windshield_annotations.json"}
    )


@app.route("/api/export/labels-zip", methods=["GET"])
def api_export_labels_zip():
    results = extractor.get_results()
    records_by_frame = results["records_by_frame"]

    if not records_by_frame:
        return jsonify({"error": "No windshields extracted yet to export."}), 400

    zip_buffer = exporter.export_labels_only_zip(records_by_frame, class_names=["windshield"])
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="windshield_yolo_labels.zip"
    )


@app.route("/api/export/crops-zip", methods=["GET"])
def api_export_crops_zip():
    if not extractor.cropped_images:
        return jsonify({"error": "No cropped windshield images available to export."}), 400

    zip_buffer = exporter.export_crops_zip(extractor.cropped_images)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="cropped_windshields.zip"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5056))
    print(f"============================================================")
    print(f"  WII-AI · Windshield YOLO Extractor & Dataset Studio")
    print(f"  Server running on http://localhost:{port}")
    print(f"============================================================")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
