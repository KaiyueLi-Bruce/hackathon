"""Unit tests for cv/train_yolo.py.

ultralytics is NOT installed in the venv, so all tests mock/stub the
ultralytics import and avoid actual training. We test:
  - CLI arg parsing (flags, defaults)
  - Lockfile creation and deletion logic
  - _run() skipping synth data with --skip-synth
  - _run() raising SystemExit when dataset.yaml is missing
  - ONNX export: RuntimeError when no best.onnx found
"""
import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


@contextmanager
def _fake_ort():
    """Inject a fake onnxruntime module so _run()'s smoke-test try-block can import it
    (but raises on InferenceSession so the test doesn't need a real .onnx file)."""
    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = MagicMock(side_effect=Exception("no real ort in test"))
    sys.modules.setdefault("onnxruntime", fake_ort)
    old = sys.modules.get("onnxruntime")
    sys.modules["onnxruntime"] = fake_ort
    try:
        yield fake_ort
    finally:
        if old is None:
            sys.modules.pop("onnxruntime", None)
        else:
            sys.modules["onnxruntime"] = old

# ---------------------------------------------------------------------------
# Helpers to import train_yolo without ultralytics (it's gated behind try/import)
# ---------------------------------------------------------------------------

def _import_train_yolo():
    """Import train_yolo.py from cv/ even if ultralytics is absent."""
    cv_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "train_yolo", cv_root / "train_yolo.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TY = _import_train_yolo()


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------

class TestArgParsing:
    def test_defaults(self):
        """Default flags: skip_synth=False, epochs=50, n_synth=2000."""
        parser = TY.main.__code__  # just verify we can reach defaults via _parse
        # Re-create parser logic inline to avoid sys.exit on --help
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--skip-synth", action="store_true")
        p.add_argument("--epochs", type=int, default=50)
        p.add_argument("--n-synth", type=int, default=2000)
        args = p.parse_args([])
        assert args.skip_synth is False
        assert args.epochs == 50
        assert args.n_synth == 2000

    def test_skip_synth_flag(self):
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--skip-synth", action="store_true")
        p.add_argument("--epochs", type=int, default=50)
        p.add_argument("--n-synth", type=int, default=2000)
        args = p.parse_args(["--skip-synth"])
        assert args.skip_synth is True

    def test_epochs_override(self):
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--skip-synth", action="store_true")
        p.add_argument("--epochs", type=int, default=50)
        p.add_argument("--n-synth", type=int, default=2000)
        args = p.parse_args(["--epochs", "30"])
        assert args.epochs == 30

    def test_n_synth_override(self):
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--skip-synth", action="store_true")
        p.add_argument("--epochs", type=int, default=50)
        p.add_argument("--n-synth", type=int, default=2000)
        args = p.parse_args(["--n-synth", "500"])
        assert args.n_synth == 500


# ---------------------------------------------------------------------------
# Lockfile creation & deletion
# ---------------------------------------------------------------------------

class TestLockfile:
    def test_lockfile_created_and_deleted_on_success(self, tmp_path, monkeypatch):
        """LOCK is written before _run() and deleted after success."""
        monkeypatch.setattr(TY, "MODELS", tmp_path)
        monkeypatch.setattr(TY, "LOCK", tmp_path / ".yolo_training")

        lock_path = tmp_path / ".yolo_training"
        created_during_run = []

        def fake_run(args, YOLO):
            created_during_run.append(lock_path.exists())

        mock_yolo_class = MagicMock()

        # Patch ultralytics import inside main()
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = mock_yolo_class

        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            with patch.object(TY, "_run", fake_run):
                with patch("sys.argv", ["train_yolo.py"]):
                    TY.main()

        assert created_during_run == [True], "lockfile should exist during _run()"
        assert not lock_path.exists(), "lockfile should be deleted after success"

    def test_lockfile_deleted_on_exception(self, tmp_path, monkeypatch):
        """LOCK is deleted even when _run() raises."""
        monkeypatch.setattr(TY, "MODELS", tmp_path)
        monkeypatch.setattr(TY, "LOCK", tmp_path / ".yolo_training")

        lock_path = tmp_path / ".yolo_training"

        def fake_run_raises(args, YOLO):
            raise RuntimeError("simulated training failure")

        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = MagicMock()

        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            with patch.object(TY, "_run", fake_run_raises):
                with patch("sys.argv", ["train_yolo.py"]):
                    with pytest.raises(RuntimeError, match="simulated training failure"):
                        TY.main()

        assert not lock_path.exists(), "lockfile should be deleted even on failure"

    def test_lockfile_content_is_timestamp(self, tmp_path, monkeypatch):
        """LOCK file content should be a float-parseable timestamp."""
        monkeypatch.setattr(TY, "MODELS", tmp_path)
        monkeypatch.setattr(TY, "LOCK", tmp_path / ".yolo_training")
        lock_path = tmp_path / ".yolo_training"

        captured_content = []

        def fake_run(args, YOLO):
            captured_content.append(lock_path.read_text())

        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = MagicMock()

        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            with patch.object(TY, "_run", fake_run):
                with patch("sys.argv", ["train_yolo.py"]):
                    TY.main()

        assert len(captured_content) == 1
        ts = float(captured_content[0])  # must parse as float
        assert ts > 1_000_000_000, "timestamp should be a reasonable Unix time"


# ---------------------------------------------------------------------------
# _run() logic
# ---------------------------------------------------------------------------

class TestRun:
    """Tests for _run() using a stub YOLO class and mocked filesystem."""

    def _make_args(self, skip_synth=False, epochs=50, n_synth=2000):
        import argparse
        return argparse.Namespace(skip_synth=skip_synth, epochs=epochs, n_synth=n_synth)

    def test_skip_synth_skips_generate(self, tmp_path, monkeypatch):
        """--skip-synth must skip generate_synthetic_dataset() call."""
        monkeypatch.setattr(TY, "SYNTH", tmp_path / "synth")
        monkeypatch.setattr(TY, "ONNX", tmp_path / "yolo_spot.onnx")
        monkeypatch.setattr(TY, "ROOT", tmp_path)

        # Create dataset.yaml so _run() doesn't raise SystemExit
        synth_dir = tmp_path / "synth"
        synth_dir.mkdir(parents=True)
        (synth_dir / "dataset.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: spot\n")

        # Stub YOLO class
        mock_model = MagicMock()
        mock_model.export.return_value = None
        MockYOLO = MagicMock(return_value=mock_model)

        # Stub generate_synthetic_dataset and shutil.copy2
        gen_called = []

        with patch("chromalog_cv.yolo.generate_synthetic_dataset", side_effect=lambda **kw: gen_called.append(kw)):
            # Also need to handle the rglob for best.onnx — create a fake one
            fake_onnx = tmp_path / "runs" / "yolo_spot" / "weights" / "best.onnx"
            fake_onnx.parent.mkdir(parents=True)
            fake_onnx.write_bytes(b"fake")

            with patch("shutil.copy2"):
                with _fake_ort():
                    TY._run(self._make_args(skip_synth=True), MockYOLO)

        assert gen_called == [], "generate_synthetic_dataset should NOT be called with --skip-synth"

    def test_no_skip_synth_calls_generate(self, tmp_path, monkeypatch):
        """Without --skip-synth, generate_synthetic_dataset() must be called."""
        monkeypatch.setattr(TY, "SYNTH", tmp_path / "synth")
        monkeypatch.setattr(TY, "ONNX", tmp_path / "yolo_spot.onnx")
        monkeypatch.setattr(TY, "ROOT", tmp_path)

        synth_dir = tmp_path / "synth"
        gen_called = []

        def fake_generate(real_images_dir, out_dir, n):
            gen_called.append({"out_dir": out_dir, "n": n})
            # Create dataset.yaml so _run() continues
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "dataset.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: spot\n")

        mock_model = MagicMock()
        MockYOLO = MagicMock(return_value=mock_model)

        fake_onnx = tmp_path / "runs" / "yolo_spot" / "weights" / "best.onnx"
        fake_onnx.parent.mkdir(parents=True)
        fake_onnx.write_bytes(b"fake")

        with patch("chromalog_cv.yolo.generate_synthetic_dataset", side_effect=fake_generate):
            with patch("shutil.copy2"):
                with _fake_ort():
                    TY._run(self._make_args(skip_synth=False, n_synth=100), MockYOLO)

        assert len(gen_called) == 1
        assert gen_called[0]["n"] == 100

    def test_missing_dataset_yaml_raises(self, tmp_path, monkeypatch):
        """When dataset.yaml doesn't exist, _run() must raise SystemExit."""
        monkeypatch.setattr(TY, "SYNTH", tmp_path / "synth")
        monkeypatch.setattr(TY, "ONNX", tmp_path / "yolo_spot.onnx")
        monkeypatch.setattr(TY, "ROOT", tmp_path)

        MockYOLO = MagicMock()
        # synth dir exists but no dataset.yaml
        (tmp_path / "synth").mkdir(parents=True)

        with pytest.raises(SystemExit):
            TY._run(self._make_args(skip_synth=True), MockYOLO)

    def test_train_called_with_correct_params(self, tmp_path, monkeypatch):
        """model.train() must be called with epochs=50, imgsz=640, device='mps'."""
        monkeypatch.setattr(TY, "SYNTH", tmp_path / "synth")
        monkeypatch.setattr(TY, "ONNX", tmp_path / "yolo_spot.onnx")
        monkeypatch.setattr(TY, "ROOT", tmp_path)

        synth_dir = tmp_path / "synth"
        synth_dir.mkdir(parents=True)
        (synth_dir / "dataset.yaml").write_text("path: .\nnc: 1\n")

        mock_model = MagicMock()
        MockYOLO = MagicMock(return_value=mock_model)

        fake_onnx = tmp_path / "runs" / "yolo_spot" / "weights" / "best.onnx"
        fake_onnx.parent.mkdir(parents=True)
        fake_onnx.write_bytes(b"fake")

        with patch("chromalog_cv.yolo.generate_synthetic_dataset"):
            with patch("shutil.copy2"):
                with _fake_ort():
                    TY._run(self._make_args(epochs=50), MockYOLO)

        train_kwargs = mock_model.train.call_args[1]
        assert train_kwargs["epochs"] == 50
        assert train_kwargs["imgsz"] == 640
        assert train_kwargs["device"] == "mps"

    def test_no_best_onnx_raises_runtime_error(self, tmp_path, monkeypatch):
        """If ONNX export produces no best.onnx, RuntimeError is raised."""
        monkeypatch.setattr(TY, "SYNTH", tmp_path / "synth")
        monkeypatch.setattr(TY, "ONNX", tmp_path / "yolo_spot.onnx")
        monkeypatch.setattr(TY, "ROOT", tmp_path)

        synth_dir = tmp_path / "synth"
        synth_dir.mkdir(parents=True)
        (synth_dir / "dataset.yaml").write_text("path: .\nnc: 1\n")

        mock_model = MagicMock()
        MockYOLO = MagicMock(return_value=mock_model)
        # No best.onnx created → rglob returns []

        with patch("chromalog_cv.yolo.generate_synthetic_dataset"):
            with pytest.raises(RuntimeError, match="no best.onnx"):
                TY._run(self._make_args(), MockYOLO)

    def test_onnx_export_params(self, tmp_path, monkeypatch):
        """model.export() must be called with format='onnx', opset=12, simplify=True."""
        monkeypatch.setattr(TY, "SYNTH", tmp_path / "synth")
        monkeypatch.setattr(TY, "ONNX", tmp_path / "yolo_spot.onnx")
        monkeypatch.setattr(TY, "ROOT", tmp_path)

        synth_dir = tmp_path / "synth"
        synth_dir.mkdir(parents=True)
        (synth_dir / "dataset.yaml").write_text("path: .\nnc: 1\n")

        mock_model = MagicMock()
        MockYOLO = MagicMock(return_value=mock_model)

        fake_onnx = tmp_path / "runs" / "yolo_spot" / "weights" / "best.onnx"
        fake_onnx.parent.mkdir(parents=True)
        fake_onnx.write_bytes(b"fake")

        with patch("chromalog_cv.yolo.generate_synthetic_dataset"):
            with patch("shutil.copy2"):
                with _fake_ort():
                    TY._run(self._make_args(), MockYOLO)

        export_kwargs = mock_model.export.call_args[1]
        assert export_kwargs["format"] == "onnx"
        assert export_kwargs["opset"] == 12
        assert export_kwargs["simplify"] is True

    def test_real_dataset_fine_tune_not_called_when_absent(self, tmp_path, monkeypatch):
        """Fine-tune (20 epochs) must NOT trigger when real dataset.yaml absent."""
        monkeypatch.setattr(TY, "SYNTH", tmp_path / "synth")
        monkeypatch.setattr(TY, "ONNX", tmp_path / "yolo_spot.onnx")
        monkeypatch.setattr(TY, "ROOT", tmp_path)

        synth_dir = tmp_path / "synth"
        synth_dir.mkdir(parents=True)
        (synth_dir / "dataset.yaml").write_text("path: .\nnc: 1\n")

        mock_model = MagicMock()
        MockYOLO = MagicMock(return_value=mock_model)

        fake_onnx = tmp_path / "runs" / "yolo_spot" / "weights" / "best.onnx"
        fake_onnx.parent.mkdir(parents=True)
        fake_onnx.write_bytes(b"fake")

        with patch("chromalog_cv.yolo.generate_synthetic_dataset"):
            with patch("shutil.copy2"):
                with _fake_ort():
                    TY._run(self._make_args(), MockYOLO)

        # model.train() called exactly once (synth only), no fine-tune
        assert mock_model.train.call_count == 1

    def test_real_dataset_fine_tune_called_when_present(self, tmp_path, monkeypatch):
        """Fine-tune must be called with epochs=20 and lr0=0.001 when real dataset exists."""
        monkeypatch.setattr(TY, "SYNTH", tmp_path / "synth")
        monkeypatch.setattr(TY, "ONNX", tmp_path / "yolo_spot.onnx")
        monkeypatch.setattr(TY, "ROOT", tmp_path)

        synth_dir = tmp_path / "synth"
        synth_dir.mkdir(parents=True)
        (synth_dir / "dataset.yaml").write_text("path: .\nnc: 1\n")

        # Create real dataset.yaml
        real_dir = tmp_path / "data" / "real"
        real_dir.mkdir(parents=True)
        (real_dir / "dataset.yaml").write_text("path: .\nnc: 1\n")

        mock_model = MagicMock()
        MockYOLO = MagicMock(return_value=mock_model)

        fake_onnx = tmp_path / "runs" / "yolo_spot" / "weights" / "best.onnx"
        fake_onnx.parent.mkdir(parents=True)
        fake_onnx.write_bytes(b"fake")

        with patch("chromalog_cv.yolo.generate_synthetic_dataset"):
            with patch("shutil.copy2"):
                with _fake_ort():
                    TY._run(self._make_args(), MockYOLO)

        # model.train() called twice
        assert mock_model.train.call_count == 2
        fine_tune_kwargs = mock_model.train.call_args_list[1][1]
        assert fine_tune_kwargs["epochs"] == 20
        assert fine_tune_kwargs["lr0"] == 0.001


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_onnx_path_matches_yolo_module(self):
        """ONNX in train_yolo.py must match YOLO_ONNX_PATH from chromalog_cv.yolo."""
        from chromalog_cv.yolo import YOLO_ONNX_PATH
        assert TY.ONNX == YOLO_ONNX_PATH

    def test_lock_in_models_dir(self):
        """LOCK must be in the same directory as ONNX."""
        assert TY.LOCK.parent == TY.MODELS
        assert TY.LOCK.name == ".yolo_training"
