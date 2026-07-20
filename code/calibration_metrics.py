"""calibration_metrics.py  --  Reviewer 1 (Comment 8), Reviewer 2 (Comment 3)

Computes Brier score, Negative Log-Likelihood (NLL) and Maximum Calibration
Error (MCE) in addition to ECE, and benchmarks ABSS-Net's training-time
calibration against post-hoc TEMPERATURE SCALING.

Run AFTER the 5-fold CV cell so that `pooled` (out-of-fold probabilities and
labels) exists. Every number printed is computed at run time -- nothing here is
pre-filled. Insert the printed values into the Calibration subsection.
"""
import numpy as np


def _ece_mce(probs, labels, n_bins=15):
    conf = probs.max(1); pred = probs.argmax(1)
    acc = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); ece = 0.0; mce = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        gap = abs(acc[m].mean() - conf[m].mean())
        ece += m.mean() * gap
        mce = max(mce, gap)
    return ece, mce


def brier_multiclass(probs, labels):
    onehot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def nll(probs, labels, eps=1e-12):
    p = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(p, eps, 1.0))))


def temperature_scale(logits, labels, max_iter=500):
    """Fit a single temperature T on a held-out split by minimising NLL."""
    import torch
    logits = torch.tensor(logits, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter)
    lossf = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad(); loss = lossf(logits / T, labels); loss.backward(); return loss

    opt.step(closure)
    return float(T.detach().item())


def run(probs, labels):
    probs = np.asarray(probs, dtype=np.float64); labels = np.asarray(labels)
    ece, mce = _ece_mce(probs, labels)
    print("=== Calibration metrics (training-time ABSS-Net) ===")
    print(f"Brier : {brier_multiclass(probs, labels):.4f}")
    print(f"NLL   : {nll(probs, labels):.4f}")
    print(f"ECE   : {ece:.4f}")
    print(f"MCE   : {mce:.4f}")
    # Temperature scaling using logits = log(prob) (valid up to an additive
    # constant, which softmax ignores). For an exact result, save raw logits.
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    T = temperature_scale(logits, labels)
    scaled = np.exp(logits / T); scaled /= scaled.sum(1, keepdims=True)
    ece_t, mce_t = _ece_mce(scaled, labels)
    print("\n=== Post-hoc temperature scaling (baseline for comparison) ===")
    print(f"Temperature T*      : {T:.3f}")
    print(f"ECE after T-scaling : {ece_t:.4f}  (training-time ECE was {ece:.4f})")
    print(f"MCE after T-scaling : {mce_t:.4f}  (training-time MCE was {mce:.4f})")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Compute Brier, NLL, ECE, MCE, and temperature scaling from saved OOF arrays."
    )
    parser.add_argument("--npz", type=Path, required=True, help="NPZ containing y_prob and y_true arrays.")
    args = parser.parse_args()

    if not args.npz.exists():
        raise SystemExit(f"Input file not found: {args.npz}")
    data = np.load(args.npz)
    missing = [k for k in ("y_prob", "y_true") if k not in data]
    if missing:
        raise SystemExit(f"Missing required arrays in {args.npz}: {', '.join(missing)}")
    run(data["y_prob"], data["y_true"])
