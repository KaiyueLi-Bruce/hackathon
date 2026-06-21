import numpy as np
from chromalog_cv.config import Config
from chromalog_cv import learn


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
