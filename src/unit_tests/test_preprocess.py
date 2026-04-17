import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from preprocess import extract_hog_features

TEST_IMAGE = os.path.join("data", "train", "cat.0.jpg")


def test_hog_output_type():
    features = extract_hog_features(TEST_IMAGE)
    assert isinstance(features, np.ndarray)


def test_hog_output_shape():
    features = extract_hog_features(TEST_IMAGE)
    assert features.ndim == 1
    assert features.shape[0] > 0


def test_hog_no_nan():
    features = extract_hog_features(TEST_IMAGE)
    assert not np.isnan(features).any()
