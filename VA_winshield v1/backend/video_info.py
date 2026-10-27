import os
import json
import subprocess
import cv2

def get_video_metadata(video_path):
    """
    Extracts comprehensive video metadata:
    - FPS, format, file size, resolution, duration, total frames, codec, bitrate.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    file_size_bytes = os.path.getsize(video_path)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    file_size_gb = round(file_size_bytes / (1024 * 1024 * 1024), 2)

    filename = os.path.basename(video_path)
    extension = os.path.splitext(filename)[1].lstrip('.').lower()

    metadata = {
        "filename": filename,
        "filepath": os.path.abspath(video_path),
        "filesize_bytes": file_size_bytes,
        "filesize_mb": file_size_mb,
        "filesize_formatted": f"{file_size_mb} MB" if file_size_mb < 1024 else f"{file_size_gb} GB",
        "format": extension.upper(),
        "width": 0,
        "height": 0,
        "resolution": "Unknown",
        "fps": 0.0,
        "total_frames": 0,
        "duration_seconds": 0.0,
        "duration_formatted": "00:00:00",
        "codec": "Unknown",
        "bitrate_kbps": 0,
        "aspect_ratio": "Unknown"
    }

    # Attempt ffprobe first for rich stream metadata
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name,bit_rate:format=duration,size,bit_rate,format_long_name",
            "-of", "json",
            video_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe = json.loads(res.stdout)

        streams = probe.get("streams", [])
        fmt = probe.get("format", {})

        if streams:
            vstream = streams[0]
            w = int(vstream.get("width", 0))
            h = int(vstream.get("height", 0))
            codec = vstream.get("codec_name", "Unknown")

            fps_str = vstream.get("r_frame_rate", "0/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 0.0
            else:
                fps = float(fps_str)

            nb_frames = int(vstream.get("nb_frames", 0))
            duration = float(vstream.get("duration", 0) or fmt.get("duration", 0))
            bitrate = int(vstream.get("bit_rate", 0) or fmt.get("bit_rate", 0))

            metadata.update({
                "width": w,
                "height": h,
                "resolution": f"{w} x {h}",
                "fps": round(fps, 2),
                "total_frames": nb_frames,
                "duration_seconds": round(duration, 2),
                "codec": codec.upper(),
                "bitrate_kbps": round(bitrate / 1000, 1) if bitrate else 0,
                "format_details": fmt.get("format_long_name", extension.upper())
            })
    except Exception:
        pass

    # Fallback / Supplement with OpenCV VideoCapture
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            if not metadata["width"]:
                metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                metadata["resolution"] = f"{metadata['width']} x {metadata['height']}"
            if not metadata["fps"] or metadata["fps"] == 0:
                metadata["fps"] = round(float(cap.get(cv2.CAP_PROP_FPS)), 2)
            if not metadata["total_frames"]:
                metadata["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not metadata["duration_seconds"] and metadata["fps"] > 0 and metadata["total_frames"] > 0:
                metadata["duration_seconds"] = round(metadata["total_frames"] / metadata["fps"], 2)
            if metadata["codec"] == "Unknown":
                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                metadata["codec"] = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]).strip()
            cap.release()
    except Exception:
        pass

    # Format duration
    total_secs = int(metadata["duration_seconds"])
    mins, secs = divmod(total_secs, 60)
    hrs, mins = divmod(mins, 60)
    metadata["duration_formatted"] = f"{hrs:02d}:{mins:02d}:{secs:02d}"

    # Aspect ratio
    w, h = metadata["width"], metadata["height"]
    if w and h:
        import math
        gcd = math.gcd(w, h)
        metadata["aspect_ratio"] = f"{w//gcd}:{h//gcd}"

    return metadata
