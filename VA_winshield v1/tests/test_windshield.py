import os
import unittest
import numpy as np
import json
import zipfile

from backend.detector import (
    select_windshield,
    is_box_inside_roi,
    box_iou
)
from backend.yolo_exporter import YOLODatasetExporter
from app import app

class TestWindshieldStudio(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_select_windshield_algorithm(self):
        """Tests the ported wind_shield algorithm."""
        preds = [
            {"score": 0.45, "box": {"xmin": 10.0, "ymin": 20.0, "xmax": 90.0, "ymax": 70.0}},
            {"score": 0.12, "box": {"xmin": 5.0, "ymin": 5.0, "xmax": 20.0, "ymax": 15.0}},
            {"score": 0.02, "box": {"xmin": 0.0, "ymin": 0.0, "xmax": 10.0, "ymax": 10.0}}
        ]
        res = select_windshield(preds, width=100, height=100, threshold=0.10)
        self.assertIsNotNone(res)
        self.assertEqual(res["score"], 0.45)
        self.assertEqual(res["bbox_px"], [10, 20, 90, 70])
        self.assertEqual(len(res["points_norm"]), 4)

    def test_box_inside_roi(self):
        """Tests ROI polygon containment checks."""
        polygon = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
        frame_shape = (1000, 1000, 3)

        # Center inside
        box_in = [400, 400, 600, 600]
        self.assertTrue(is_box_inside_roi(box_in, polygon, frame_shape, mode="windshield_center"))

        # Center outside
        box_out = [10, 10, 50, 50]
        self.assertFalse(is_box_inside_roi(box_out, polygon, frame_shape, mode="windshield_center"))

    def test_box_iou(self):
        box1 = [0, 0, 100, 100]
        box2 = [50, 50, 150, 150]
        iou = box_iou(box1, box2)
        self.assertGreater(iou, 0.1)
        self.assertLess(iou, 0.3)

    def test_yolo_exporter_csv_and_json(self):
        exporter = YOLODatasetExporter(output_dir="/tmp/test_ws_output")
        flat_records = [
            {
                "frame_idx": 10,
                "timestamp_sec": 0.33,
                "vehicle_type": "Car",
                "vehicle_bbox": [100, 200, 400, 500],
                "windshield_bbox": [150, 220, 350, 320],
                "windshield_w": 200,
                "windshield_h": 100,
                "corners_px": [
                    {"corner": "top_left", "x": 150, "y": 220},
                    {"corner": "top_right", "x": 350, "y": 220},
                    {"corner": "bottom_right", "x": 350, "y": 320},
                    {"corner": "bottom_left", "x": 150, "y": 320}
                ],
                "corners_norm": [
                    {"corner": "top_left", "x": 0.15, "y": 0.22},
                    {"corner": "top_right", "x": 0.35, "y": 0.22},
                    {"corner": "bottom_right", "x": 0.35, "y": 0.32},
                    {"corner": "bottom_left", "x": 0.15, "y": 0.32}
                ],
                "confidence": 0.85,
                "yolo_bbox": {
                    "class_id": 0,
                    "class_name": "windshield",
                    "x_center": 0.25,
                    "y_center": 0.27,
                    "width": 0.20,
                    "height": 0.10
                },
                "yolo_line": "0 0.250000 0.270000 0.200000 0.100000",
                "extraction_method": "owlvit"
            }
        ]

        csv_str = exporter.export_csv(flat_records)
        self.assertIn("windshield", csv_str)
        self.assertIn("0.25", csv_str)
        self.assertIn("frame_index", csv_str)

        json_str = exporter.export_json(flat_records)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["metadata"]["classes"], ["windshield"])
        self.assertEqual(len(parsed["annotations"]), 1)

    def test_yolo_zip_export(self):
        exporter = YOLODatasetExporter(output_dir="/tmp/test_ws_output")
        records_by_frame = {
            10: [
                {
                    "yolo_bbox": {
                        "class_id": 0,
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "width": 0.2,
                        "height": 0.2
                    }
                }
            ]
        }
        dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame_images = {10: dummy_frame}

        zip_buf = exporter.export_full_yolo_zip(records_by_frame, frame_images, train_ratio=0.8)
        with zipfile.ZipFile(zip_buf, "r") as zf:
            namelist = zf.namelist()
            self.assertTrue(any("data.yaml" in name for name in namelist))
            self.assertTrue(any("labels" in name for name in namelist))
            self.assertTrue(any("images" in name for name in namelist))

    def test_app_index_and_endpoints(self):
        res = self.app.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"WII-AI Windshield Studio", res.data)

        res_list = self.app.get("/api/video/list")
        self.assertEqual(res_list.status_code, 200)

        res_telem = self.app.get("/api/video/telemetry")
        self.assertEqual(res_telem.status_code, 200)

if __name__ == "__main__":
    unittest.main()
