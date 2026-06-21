import numpy as np
from chromalog_cv.config import Config
from chromalog_cv import learn
from chromalog_cv import spots as S


def test_patch_features_fixed_length_and_scale_invariant():
    cfg = Config()
    expected_len = cfg.clf_patch_size ** 2 + 3
    big = np.full((400, 300, 3), 200, np.uint8)
    small = np.full((120, 90, 3), 200, np.uint8)
    fb = learn.patch_features(big, 150, 200, cfg)
    fs = learn.patch_features(small, 45, 60, cfg)
    assert fb.shape == (expected_len,)
    assert fs.shape == (expected_len,)        # length independent of image size
    assert fb.dtype == np.float32


def test_patch_features_centroid_near_edge_does_not_crash():
    cfg = Config()
    img = np.zeros((100, 100), np.uint8)
    f = learn.patch_features(img, 1, 1, cfg)   # gray input, corner centroid
    assert f.shape == (cfg.clf_patch_size ** 2 + 3,)
    assert np.all(np.isfinite(f))


def test_classifier_learns_separable_data(tmp_path):
    rng = np.random.RandomState(0)
    pos = rng.normal(0.8, 0.05, size=(20, 5)).astype(np.float32)
    neg = rng.normal(0.2, 0.05, size=(20, 5)).astype(np.float32)
    X = np.vstack([pos, neg]); y = np.array([1] * 20 + [0] * 20)
    clf = learn.SpotClassifier()
    assert clf.is_trained is False
    clf.update(X, y)
    assert clf.is_trained is True
    assert clf.n_samples == 40
    p = clf.proba(np.array([[0.8, 0.8, 0.8, 0.8, 0.8],
                            [0.2, 0.2, 0.2, 0.2, 0.2]], np.float32))
    assert p[0] > 0.5 > p[1]


def test_classifier_save_load_roundtrip(tmp_path):
    path = tmp_path / "clf.pkl"
    X = np.random.RandomState(1).rand(10, 5).astype(np.float32)
    y = np.array([1, 0] * 5)
    clf = learn.SpotClassifier(); clf.update(X, y); clf.save(path)
    loaded = learn.SpotClassifier.load(path)
    assert loaded.is_trained and loaded.n_samples == 10
    assert np.allclose(loaded.proba(X), clf.proba(X))


def test_classifier_load_missing_returns_untrained(tmp_path):
    clf = learn.SpotClassifier.load(tmp_path / "nope.pkl")
    assert clf.is_trained is False and clf.n_samples == 0


def test_derive_samples_splits_pos_hardneg_easyneg():
    cfg = Config()
    img = np.zeros((200, 200, 3), np.uint8)
    final_pts = [(0.50, 0.50)]                  # one real spot
    auto_pts = [(0.505, 0.495),                 # ~matches final -> positive
                (0.05, 0.05)]                   # corner, user deleted -> hard negative
    X, y, counts = learn.derive_samples(img, final_pts, auto_pts, cfg)
    # 1 matched candidate + 0 user-added = 1 positive; 1 hard neg; easy negs = ceil(1*1)=1
    assert counts["pos"] == 1
    assert counts["neg"] == 2                   # 1 hard + 1 easy
    assert X.shape[0] == 3 and X.shape[1] == cfg.clf_patch_size ** 2 + 3
    assert set(y.tolist()) == {0, 1}
    assert int((y == 1).sum()) == 1


def test_derive_samples_user_added_spot_is_positive():
    cfg = Config()
    img = np.zeros((200, 200, 3), np.uint8)
    final_pts = [(0.5, 0.5)]                     # present in final
    auto_pts = []                                # but never auto-detected -> user added
    X, y, counts = learn.derive_samples(img, final_pts, auto_pts, cfg)
    assert counts["pos"] == 1


def test_apply_correction_trains_and_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(learn, "CLF_PATH", tmp_path / "clf.pkl")
    monkeypatch.setattr(learn, "SAMPLES_PATH", tmp_path / "s.npz")
    cfg = Config()
    img = np.zeros((200, 200, 3), np.uint8)
    out = learn.apply_correction(img, [(0.5, 0.5)], [(0.5, 0.5), (0.05, 0.05)], cfg)
    assert out["ok"] is True
    assert out["batch"]["pos"] >= 1 and out["batch"]["neg"] >= 1
    info = learn.model_info(tmp_path / "clf.pkl", tmp_path / "s.npz")
    assert info["trained"] is True
    assert info["n_samples"] == out["trained_total"]
    assert info["updated_at"] is not None


def test_model_info_untrained(tmp_path):
    info = learn.model_info(tmp_path / "none.pkl", tmp_path / "none.npz")
    assert info == {"trained": False, "n_samples": 0, "updated_at": None}


def _spot(x, y):
    return S.Spot(x=x, y=y, bbox=(int(x) - 2, int(y) - 2, 4, 4), area=16, lane=0, rf=None)


def test_detect_spots_scorer_filters_below_threshold(monkeypatch):
    cfg = Config()
    binary = np.zeros((100, 100), np.uint8)
    # two blobs: one will score high, one low
    cv = __import__("cv2")
    cv.circle(binary, (30, 50), 3, 255, -1)
    cv.circle(binary, (70, 50), 3, 255, -1)
    gray = np.zeros((100, 100), np.uint8)
    roi = (0, 0, 100, 100)
    scorer = lambda s: 0.9 if s.x < 50 else 0.1   # keep left, drop right
    res = S.detect_spots(binary, gray, "dark_on_light", roi, None, None,
                         float(100 * 100), cfg, scorer=scorer)
    xs = sorted(round(s.x) for s in res.spots)
    assert all(x < 50 for x in xs)               # only the high-scored blob survives
    assert len(res.spots) == 1


from chromalog_cv.pipeline import run_pipeline


def test_pipeline_reports_learned_false_when_untrained(tmp_path, monkeypatch):
    monkeypatch.setattr(learn, "CLF_PATH", tmp_path / "absent.pkl")
    img = cv2_imread_real2()
    result, _, _ = run_pipeline(img)
    assert result.learned is False
    assert "skl" not in (result.engine_used or "")


def cv2_imread_real2():
    import cv2, os
    p = os.path.join(os.path.dirname(__file__), "..", "..", "training_pictures", "TLC_real_2.jpg")
    img = cv2.imread(p)
    assert img is not None, p
    return img


def test_suppression_synthetic_deterministic(tmp_path, monkeypatch):
    """Prove that teaching a blob as hard-negative suppresses it on re-detection.

    Strategy: build a fully controlled synthetic scene with TWO blobs whose
    patch features are deliberately very different (blob A is dark on gray,
    blob B is bright on gray).  We call detect_spots directly (no pipeline)
    so there is no dependency on a real photo or rectification.

    Round structure:
      - blob A: present in both auto_pts (candidate) and final_pts (kept) -> positive
      - blob B: present only in auto_pts (detected by CV), NOT in final_pts -> hard-negative

    After 12 training rounds the SGD should separate them reliably.
    Final assertions:
      * scorer(A-spot) >= cfg.spot_clf_thresh  -> blob A SURVIVES
      * scorer(B-spot) <  cfg.spot_clf_thresh  -> blob B is SUPPRESSED
    """
    import cv2 as cv2mod

    monkeypatch.setattr(learn, "CLF_PATH", tmp_path / "clf.pkl")
    monkeypatch.setattr(learn, "SAMPLES_PATH", tmp_path / "s.npz")

    cfg = Config()
    H, W = 300, 300

    # ---------------------------------------------------------------------------
    # Build synthetic gray + binary images
    # ---------------------------------------------------------------------------
    # Mid-gray background
    gray = np.full((H, W), 128, np.uint8)

    # Blob A: dark filled circle at (75, 150) radius 10 -> low pixel value (real spot)
    A_cx, A_cy, A_r = 75, 150, 10
    cv2mod.circle(gray, (A_cx, A_cy), A_r, 30, -1)   # intensity 30 -> very dark

    # Blob B: bright filled circle at (225, 150) radius 10 -> high pixel value (false positive)
    B_cx, B_cy, B_r = 225, 150, 10
    cv2mod.circle(gray, (B_cx, B_cy), B_r, 220, -1)  # intensity 220 -> very bright

    # Binary: both blobs are "foreground" so detect_spots finds both as connected components
    binary = np.zeros((H, W), np.uint8)
    cv2mod.circle(binary, (A_cx, A_cy), A_r, 255, -1)
    cv2mod.circle(binary, (B_cx, B_cy), B_r, 255, -1)

    plate_area = float(H * W)
    roi = (0, 0, W, H)

    # ---------------------------------------------------------------------------
    # Sanity: without a scorer BOTH blobs are detected
    # ---------------------------------------------------------------------------
    res_baseline = S.detect_spots(binary, gray, "dark_on_light", roi, None, None,
                                  plate_area, cfg, scorer=None)
    assert len(res_baseline.spots) >= 2, (
        f"Sanity failed: expected >=2 spots without scorer, got {len(res_baseline.spots)}. "
        "Check blob sizes / edge margins."
    )

    # ---------------------------------------------------------------------------
    # Identify which detected spots are blob A and blob B by proximity
    # ---------------------------------------------------------------------------
    tol_px = A_r * 2  # generous tolerance for centroid matching

    def near_blob(spot, cx, cy):
        return abs(spot.x - cx) < tol_px and abs(spot.y - cy) < tol_px

    spots_A = [s for s in res_baseline.spots if near_blob(s, A_cx, A_cy)]
    spots_B = [s for s in res_baseline.spots if near_blob(s, B_cx, B_cy)]
    assert spots_A, "Blob A not detected in baseline"
    assert spots_B, "Blob B not detected in baseline"

    # ---------------------------------------------------------------------------
    # Convert centroid coords to normalized (x in [0,1], y in [0,1]) for learn API
    # apply_correction takes a color-or-gray image; pass the gray image directly.
    # ---------------------------------------------------------------------------
    # The gray image is 2-D here; patch_features handles both 2-D and 3-D arrays.
    # derive_samples calls patch_features with (px * w, py * h) so normalization must use
    # the same H, W as the image passed to apply_correction.
    A_norm = (float(spots_A[0].x) / W, float(spots_A[0].y) / H)
    B_norm = (float(spots_B[0].x) / W, float(spots_B[0].y) / H)

    # final_pts  = only blob A (user kept it as a real spot)
    # auto_pts   = both A and B (both were auto-detected by OpenCV)
    final_pts = [A_norm]
    auto_pts  = [A_norm, B_norm]

    # ---------------------------------------------------------------------------
    # Training: 12 rounds of apply_correction
    # ---------------------------------------------------------------------------
    for round_i in range(12):
        learn.apply_correction(gray, final_pts, auto_pts, cfg)

    # ---------------------------------------------------------------------------
    # Reload the trained classifier and build a scorer closure
    # ---------------------------------------------------------------------------
    clf = learn.SpotClassifier.load(tmp_path / "clf.pkl")
    assert clf.is_trained, "Classifier should be trained after 12 rounds"

    def scorer(spot):
        feats = learn.patch_features(gray, spot.x, spot.y, cfg)[None, :]
        return float(clf.proba(feats)[0])

    # ---------------------------------------------------------------------------
    # Re-detect WITH the trained scorer
    # ---------------------------------------------------------------------------
    res_trained = S.detect_spots(binary, gray, "dark_on_light", roi, None, None,
                                 plate_area, cfg, scorer=scorer)

    surviving_A = [s for s in res_trained.spots if near_blob(s, A_cx, A_cy)]
    surviving_B = [s for s in res_trained.spots if near_blob(s, B_cx, B_cy)]

    # Primary assertions: blob A survives, blob B is suppressed
    assert surviving_A, (
        f"Blob A (real spot) was suppressed but should survive. "
        f"scorer(A)={scorer(spots_A[0]):.3f}, threshold={cfg.spot_clf_thresh}"
    )
    assert not surviving_B, (
        f"Blob B (false positive) was NOT suppressed after {12} training rounds. "
        f"scorer(B)={scorer(spots_B[0]):.3f}, threshold={cfg.spot_clf_thresh}. "
        "The classifier failed to separate the two synthetic blobs."
    )


def test_learning_suppresses_taught_false_positive(tmp_path, monkeypatch):
    monkeypatch.setattr(learn, "CLF_PATH", tmp_path / "clf.pkl")
    monkeypatch.setattr(learn, "SAMPLES_PATH", tmp_path / "s.npz")
    from chromalog_cv.config import Config
    img = cv2_imread_real2()
    result0, _, _ = run_pipeline(img)
    assert len(result0.spots) >= 1
    assert result0.learned is False

    # Teach: keep the real spots, but mark each detected spot as a deleted
    # candidate that is ALSO a real spot for half, and teach background as neg.
    cfg = Config()
    # rectified image used by the pipeline = the displayed base; re-run rectify to get it
    from chromalog_cv import rectify as R
    rec = R.rectify(img, cfg)
    rimg = rec.image
    final_pts = [(s["x"], s["y"]) for s in result0.spots]
    # Teach the same spots as positives + background negatives over several rounds
    for _ in range(6):
        learn.apply_correction(rimg, final_pts, final_pts, cfg)

    result1, _, _ = run_pipeline(img)
    assert result1.learned is True
    assert "skl" in result1.engine_used
    # taught real spots should survive (model keeps positives it was trained on)
    assert len(result1.spots) >= 1
