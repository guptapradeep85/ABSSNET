"""ablation_raw_vs_gaussian.py  --  Reviewer 1 (Comment 9), Reviewer 2 (Comment 1)

Controlled comparison of RAW (unfiltered APTOS-2019) versus Ben-Graham
Gaussian-filtered inputs under the identical ABSS-Net pipeline, to test whether
the filtering helps or hurts each grade (especially Severe). DR224 is already
filtered, so an unfiltered APTOS-2019 copy (RAW_IMG_DIR, same 5 class subfolders)
is required for the contrast.
"""
import cv2
import numpy as np
from PIL import Image


def ben_graham(img_pil, sigmaX=10):
    """Ben-Graham background-subtraction (the DR224 preprocessing)."""
    a = np.array(img_pil.convert("RGB"))
    a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), sigmaX), -4, 128)
    return Image.fromarray(a)


# Build two datasets from RAW_IMG_DIR:
#   (i)  transform_raw   : standard pipeline WITHOUT ben_graham
#   (ii) transform_filt  : ben_graham THEN the standard pipeline
# Run the SAME run_cross_validation() for each and compare per-grade F1.
# Report especially the Severe / Moderate F1 delta to confirm or refute the
# "Gaussian filtering attenuates discriminative lesions" hypothesis.
