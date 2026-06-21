"""Tests for YOLO server endpoints."""
import json
from pathlib import Path
from fastapi.testclient import TestClient

from chromalog_cv.server import app, _YOLO_LOCK, _YOLO_ONNX


def test_yolo_model_endpoint_not_trained():
    """Test /yolo-model returns not_trained when no model exists."""
    client = TestClient(app)
    # Ensure model doesn't exist
    if _YOLO_ONNX.exists():
        _YOLO_ONNX.unlink()
    if _YOLO_LOCK.exists():
        _YOLO_LOCK.unlink()

    response = client.get("/yolo-model")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "not_trained"
    assert data["trained_at"] is None


def test_yolo_model_endpoint_training():
    """Test /yolo-model returns training when lockfile exists."""
    client = TestClient(app)
    # Clean up any model files
    if _YOLO_ONNX.exists():
        _YOLO_ONNX.unlink()

    # Create lockfile
    _YOLO_LOCK.parent.mkdir(exist_ok=True)
    _YOLO_LOCK.touch()

    try:
        response = client.get("/yolo-model")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "training"
        assert data["trained_at"] is None
    finally:
        if _YOLO_LOCK.exists():
            _YOLO_LOCK.unlink()


def test_yolo_model_endpoint_ready():
    """Test /yolo-model returns ready when ONNX file exists."""
    client = TestClient(app)
    # Clean up lockfile
    if _YOLO_LOCK.exists():
        _YOLO_LOCK.unlink()

    # Create dummy ONNX file
    _YOLO_ONNX.parent.mkdir(exist_ok=True)
    _YOLO_ONNX.touch()

    try:
        response = client.get("/yolo-model")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["trained_at"] is not None
        # Check it's a valid ISO timestamp
        assert "T" in data["trained_at"]
    finally:
        if _YOLO_ONNX.exists():
            _YOLO_ONNX.unlink()


def test_train_yolo_endpoint_already_training():
    """Test /train-yolo returns error when already training."""
    client = TestClient(app)
    # Create lockfile to simulate ongoing training
    _YOLO_LOCK.parent.mkdir(exist_ok=True)
    _YOLO_LOCK.touch()

    try:
        response = client.post("/train-yolo")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is False
        assert data["error"] == "already training"
    finally:
        if _YOLO_LOCK.exists():
            _YOLO_LOCK.unlink()


def test_train_yolo_endpoint_starts_training():
    """Test /train-yolo starts training and returns success."""
    client = TestClient(app)
    # Ensure no lockfile exists
    if _YOLO_LOCK.exists():
        _YOLO_LOCK.unlink()

    response = client.post("/train-yolo")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["status"] == "training_started"

    # Clean up the lockfile that may have been created
    if _YOLO_LOCK.exists():
        _YOLO_LOCK.unlink()
