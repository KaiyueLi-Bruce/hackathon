import numpy as np
import pytest


def test_detect_yolo_no_model_returns_empty(tmp_path, monkeypatch):
    """With no .onnx file, detect_yolo must return empty SpotsResult."""
    import chromalog_cv.yolo as Y
    monkeypatch.setattr(Y, "YOLO_ONNX_PATH", tmp_path / "nonexistent.onnx")
    monkeypatch.setattr(Y, "_YOLO_SESSION", None)
    from chromalog_cv.config import Config
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    result = Y.detect_yolo(img, Config())
    assert result.spots == []


def test_nms_removes_overlapping_boxes():
    from chromalog_cv.yolo import _nms
    boxes = np.array([
        [10, 10, 50, 50],
        [12, 12, 52, 52],   # heavily overlaps with box 0 → should be suppressed
        [200, 200, 240, 240],  # no overlap → kept
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    kept = _nms(boxes, scores, iou_thr=0.45)
    assert 0 in kept
    assert 1 not in kept
    assert 2 in kept


def test_nms_empty_input():
    from chromalog_cv.yolo import _nms
    kept = _nms(np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32))
    assert kept == []
