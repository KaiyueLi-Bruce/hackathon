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
