import tempfile
from pathlib import Path
import numpy as np
import cv2

def test_generate_creates_expected_structure():
    from chromalog_cv.yolo import generate_synthetic_dataset
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "synth"
        # Pass a non-existent real_images_dir — fallback to solid colour bg
        generate_synthetic_dataset(real_images_dir="/no/such/dir", out_dir=out, n=6, seed=0)
        assert (out / "images" / "train").exists()
        assert (out / "images" / "val").exists()
        assert (out / "labels" / "train").exists()
        assert (out / "labels" / "val").exists()
        assert (out / "dataset.yaml").exists()
        # n=6 → 5 train + 1 val (90/10 split)
        train_imgs = list((out / "images" / "train").glob("*.jpg"))
        val_imgs   = list((out / "images" / "val").glob("*.jpg"))
        assert len(train_imgs) == 5
        assert len(val_imgs)   == 1

def test_labels_are_valid_yolo_format():
    from chromalog_cv.yolo import generate_synthetic_dataset
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "synth"
        generate_synthetic_dataset(real_images_dir="/no/such/dir", out_dir=out, n=4, seed=1)
        for lbl in (out / "labels" / "train").glob("*.txt"):
            for line in lbl.read_text().strip().splitlines():
                parts = line.split()
                assert len(parts) == 5, f"bad line: {line}"
                assert parts[0] == "0"                        # class id
                vals = [float(p) for p in parts[1:]]
                assert all(0.0 <= v <= 1.0 for v in vals)    # normalised
                assert vals[2] > 0 and vals[3] > 0           # w, h > 0

def test_images_are_readable_rgb():
    from chromalog_cv.yolo import generate_synthetic_dataset
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "synth"
        generate_synthetic_dataset(real_images_dir="/no/such/dir", out_dir=out, n=2, seed=2)
        for img_path in (out / "images" / "train").glob("*.jpg"):
            img = cv2.imread(str(img_path))
            assert img is not None
            assert img.shape == (640, 640, 3)
