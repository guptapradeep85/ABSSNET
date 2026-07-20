"""reproduce_baselines.py  --  Reviewer 1 (Comments 4 & 6)

Re-trains representative baselines under IDENTICAL DR224 folds, preprocessing and
hyper-parameters so that Table 11 becomes a controlled comparison and paired
significance tests (Wilcoxon / corrected resampled t-test) are statistically
valid. Reuses run_fold()/run_cross_validation() from the notebook by swapping the
backbone.
"""
import timm
import torch
import torch.nn as nn

BASELINES = {
    "efficientnet_b0": "efficientnet_b0",
    "vit_tiny":        "vit_tiny_patch16_224",
    "vim_tiny":        "vim_tiny_patch16_224",   # Vision-Mamba (separate install)
}


class PlainClassifier(nn.Module):
    """Backbone + linear class head + scalar reliability head (same dual-head API)."""

    def __init__(self, backbone_name, num_classes=5):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        feat = self.backbone.num_features
        self.head = nn.Linear(feat, num_classes)
        self.conf = nn.Linear(feat, 1)

    def forward(self, x):
        f = self.backbone(x)
        return self.head(f), torch.sigmoid(self.conf(f))


# For each baseline:
#   1. model = PlainClassifier(BASELINES[name])
#   2. run the SAME run_cross_validation() loop (CV_SEED=42, identical folds,
#      transforms and hyper-parameters) and collect per-fold metrics.
#   3. Paired significance vs ABSS-Net per-fold values, e.g.:
#         from scipy.stats import wilcoxon
#         stat, p = wilcoxon(abss_fold_qwk, baseline_fold_qwk)
#   4. Add a "reproduced under identical conditions" block to Table 11.
