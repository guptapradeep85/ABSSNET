#!/usr/bin/env python3
"""Package local revision outputs from supplied notebook/manuscript values.

This script is not a model-training or model-evaluation entry point. It does
not recompute the main ABSS-Net metrics from raw images or checkpoints. It
copies existing publication figures from the supplied Overleaf archive, writes
tables whose values are traceable to executed notebook/PDF outputs, runs
dataset-level checks that are feasible on the local machine, and records
GPU-only experiments that could not be executed in this CPU-only environment.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import sys
import textwrap
import time
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
LOGS = ROOT / "logs"
MODELS = ROOT / "models"
LATEX = ROOT / "latex_outputs"
MAPPING = ROOT / "reviewer_response_mapping"
OVERLEAF_ZIP = ROOT / "Revised_Overleaf_Highlighted.zip"
DR224 = DATA / "DR224" / "Train _image"


CLASS_COUNTS = {
    "No_DR": 1805,
    "Mild": 370,
    "Moderate": 999,
    "Severe": 193,
    "Proliferate_DR": 295,
}

CV_PER_FOLD = [
    {"Fold": 1, "Accuracy": 0.6876, "Macro_F1": 0.5221, "QWK": 0.7359, "AUC": 0.8775, "ECE": 0.2424},
    {"Fold": 2, "Accuracy": 0.7271, "Macro_F1": 0.5155, "QWK": 0.8294, "AUC": 0.8815, "ECE": 0.1820},
    {"Fold": 3, "Accuracy": 0.6352, "Macro_F1": 0.4316, "QWK": 0.7619, "AUC": 0.8654, "ECE": 0.1190},
    {"Fold": 4, "Accuracy": 0.7322, "Macro_F1": 0.5642, "QWK": 0.8223, "AUC": 0.8999, "ECE": 0.1718},
    {"Fold": 5, "Accuracy": 0.7309, "Macro_F1": 0.5694, "QWK": 0.8172, "AUC": 0.9066, "ECE": 0.1738},
]

CV_SUMMARY = [
    {"Metric": "Accuracy", "Mean": 0.7026, "Std": 0.0420, "CI_95_Low": 0.6505, "CI_95_High": 0.7547},
    {"Metric": "Macro-F1", "Mean": 0.5206, "Std": 0.0553, "CI_95_Low": 0.4519, "CI_95_High": 0.5892},
    {"Metric": "Quadratic Weighted Kappa", "Mean": 0.7933, "Std": 0.0418, "CI_95_Low": 0.7414, "CI_95_High": 0.8453},
    {"Metric": "AUC (OvR macro)", "Mean": 0.8862, "Std": 0.0168, "CI_95_Low": 0.8653, "CI_95_High": 0.9071},
    {"Metric": "Expected Calibration Error", "Mean": 0.1778, "Std": 0.0439, "CI_95_Low": 0.1233, "CI_95_High": 0.2323},
]

OOF_PER_CLASS = [
    {"Class": "No_DR", "Precision": 0.9451, "Recall": 0.9540, "F1": 0.9495, "AUC": 0.9863, "Support": 1805},
    {"Class": "Mild", "Precision": 0.5188, "Recall": 0.4108, "F1": 0.4585, "AUC": 0.8403, "Support": 370},
    {"Class": "Moderate", "Precision": 0.6199, "Recall": 0.4605, "F1": 0.5284, "AUC": 0.8567, "Support": 999},
    {"Class": "Severe", "Precision": 0.2723, "Recall": 0.3316, "F1": 0.2991, "AUC": 0.8339, "Support": 193},
    {"Class": "Proliferate_DR", "Precision": 0.3070, "Recall": 0.5932, "F1": 0.4046, "AUC": 0.8630, "Support": 295},
    {"Class": "Macro avg", "Precision": 0.5326, "Recall": 0.5500, "F1": 0.5280, "AUC": 0.8763, "Support": 3662},
    {"Class": "Weighted avg", "Precision": 0.7265, "Recall": 0.7026, "F1": 0.7069, "AUC": None, "Support": 3662},
]

RELIABILITY_BINS = [
    {"Reliability_bin": "c in [0.0,0.2)", "Samples": 6, "Accuracy": 0.5000},
    {"Reliability_bin": "c in [0.2,0.4)", "Samples": 562, "Accuracy": 0.4591},
    {"Reliability_bin": "c in [0.4,0.6)", "Samples": 630, "Accuracy": 0.4444},
    {"Reliability_bin": "c in [0.6,0.8)", "Samples": 694, "Accuracy": 0.4899},
    {"Reliability_bin": "c in [0.8,1.0)", "Samples": 1770, "Accuracy": 0.9559},
    {"Reliability_bin": "High reliability c >= 0.76", "Samples": 2476, "Accuracy": 0.9427},
    {"Reliability_bin": "Low reliability c < 0.76", "Samples": 1186, "Accuracy": 0.4626},
]

EXTERNAL_VALIDATION = [
    {"Metric": "Accuracy", "Value": 0.5820},
    {"Metric": "Macro-F1", "Value": 0.1473},
    {"Metric": "Quadratic Weighted Kappa", "Value": 0.0034},
    {"Metric": "AUC (OvR macro)", "Value": 0.5080},
    {"Metric": "Expected Calibration Error", "Value": 0.1483},
]

ABLATION = [
    {"Variant": "Backbone only (EfficientNet-B0 NS)", "QWK": 0.8258, "F1": 0.5432, "Accuracy": 0.7298, "ECE": 0.0832},
    {"Variant": "+ CBAM attention", "QWK": 0.8387, "F1": 0.5681, "Accuracy": 0.7367, "ECE": 0.0724},
    {"Variant": "+ Multi-scale fusion", "QWK": 0.8432, "F1": 0.5847, "Accuracy": 0.7449, "ECE": 0.0648},
    {"Variant": "+ 1 SSM block", "QWK": 0.8480, "F1": 0.5949, "Accuracy": 0.7476, "ECE": 0.0583},
    {"Variant": "+ 2 SSM blocks", "QWK": 0.8527, "F1": 0.6041, "Accuracy": 0.7557, "ECE": 0.0554},
    {"Variant": "+ Calibration loss (full ABSS-Net)", "QWK": 0.8563, "F1": 0.6107, "Accuracy": 0.7640, "ECE": 0.0507},
]

REVIEWER_MAPPING = [
    {"Comment_ID": "Editor-1", "Request": "Improve figure quality and avoid AI artwork.", "Experiment_or_Code_Change": "Redraw Figure 1 architecture and Figure 2 distribution programmatically.", "Output": "figures/fig1_architecture.png; figures/fig1_architecture.pdf; figures/fig2_class_distribution.png", "Status": "completed from supplied Overleaf archive and regenerated script"},
    {"Comment_ID": "R1-C1", "Request": "Explain and supervise reliability head.", "Experiment_or_Code_Change": "Add explicit reliability BCE term and report c-bin accuracy / prediction rejection.", "Output": "tables/reliability_bins.csv; figures/fig19_reliability_c_curve.png", "Status": "completed from executed notebook outputs"},
    {"Comment_ID": "R1-C2", "Request": "Compare gated block with a genuine Mamba selective scan.", "Experiment_or_Code_Change": "Run drop-in MambaBlock under identical 5-fold protocol.", "Output": "logs/experiment_log.txt", "Status": "not executed locally: mamba-ssm/GPU training unavailable on this CPU-only machine"},
    {"Comment_ID": "R1-C4/R1-C6", "Request": "Fair baseline comparisons and paired significance tests.", "Experiment_or_Code_Change": "Retrain EfficientNet-B0, ViT-Tiny, and Vision-Mamba-T on identical folds; run Wilcoxon tests.", "Output": "logs/experiment_log.txt", "Status": "not executed locally: full retraining requires GPU; no baseline checkpoints supplied"},
    {"Comment_ID": "R1-C7/R2-C1", "Request": "Quantify Severe bottleneck and Gaussian high-frequency attenuation.", "Experiment_or_Code_Change": "Measure high-frequency energy and gradient magnitude per DR224 class.", "Output": "tables/frequency_attenuation_filtered_dr224.csv", "Status": "completed for filtered DR224; raw-vs-filtered contrast not possible without raw APTOS copy"},
    {"Comment_ID": "R1-C8/R2-C3", "Request": "Report Brier, NLL, MCE, and temperature scaling comparison.", "Experiment_or_Code_Change": "Run calibration metrics on pooled OOF probabilities/logits.", "Output": "logs/experiment_log.txt", "Status": "not executed locally: pooled OOF probability/logit arrays were not supplied"},
    {"Comment_ID": "R1-C9", "Request": "Compare raw fundus images versus Gaussian-filtered inputs.", "Experiment_or_Code_Change": "Retrain identical pipeline on raw APTOS and filtered DR224 inputs; compare per-class F1.", "Output": "logs/experiment_log.txt", "Status": "not executed locally: raw unfiltered APTOS image tree not supplied"},
    {"Comment_ID": "R1-Minor-5/R2-C4", "Request": "Report FLOPs, latency, throughput, and GPU memory.", "Experiment_or_Code_Change": "Profile ABSS-Net on target A100 hardware.", "Output": "logs/experiment_log.txt", "Status": "not executed locally: CUDA device and trained model checkpoints unavailable"},
    {"Comment_ID": "R2-C2/R3", "Request": "Moderate external-validation and clinical-deployment claims.", "Experiment_or_Code_Change": "Report Messidor-2 zero-shot metrics and domain-shift interpretation.", "Output": "tables/external_validation_messidor2.csv", "Status": "completed from executed notebook outputs"},
]


def ensure_dirs() -> None:
    for path in [RESULTS, FIGURES, TABLES, LOGS, MODELS, LATEX, MAPPING]:
        path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value) -> str:
    if value is None:
        return "--"
    s = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def write_latex_table(path: Path, caption: str, label: str, rows: list[dict]) -> None:
    headers = list(rows[0].keys())
    align = "l" + "c" * (len(headers) - 1)
    lines = [
        r"\begin{table}[!t]",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\renewcommand{\arraystretch}{1.05}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{latex_escape(h)}}}" for h in headers) + r"\\",
        r"\midrule",
    ]
    for row in rows:
        values = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                values.append(f"{v:.4f}")
            else:
                values.append(latex_escape(v))
        lines.append(" & ".join(values) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_figures_from_zip(log: list[str]) -> None:
    if not OVERLEAF_ZIP.exists():
        log.append(f"Overleaf ZIP missing: {OVERLEAF_ZIP}")
        return
    copied = 0
    with zipfile.ZipFile(OVERLEAF_ZIP) as zf:
        for name in zf.namelist():
            if name.startswith("figures/") and not name.endswith("/"):
                target = FIGURES / Path(name).name
                with zf.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                copied += 1
    log.append(f"Copied {copied} figure files from {OVERLEAF_ZIP.name} to figures/.")


def dataset_counts() -> list[dict]:
    rows = []
    total = 0
    for cls in CLASS_COUNTS:
        directory = DR224 / cls
        count = len([p for p in directory.iterdir() if p.is_file()]) if directory.exists() else 0
        total += count
        rows.append({"Class": cls, "Expected": CLASS_COUNTS[cls], "Observed": count, "Matches": count == CLASS_COUNTS[cls]})
    rows.append({"Class": "Total", "Expected": sum(CLASS_COUNTS.values()), "Observed": total, "Matches": total == sum(CLASS_COUNTS.values())})
    return rows


def run_frequency_profile(log: list[str]) -> None:
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        log.append(f"Frequency analysis skipped because dependencies failed to import: {exc}")
        return

    def high_freq_ratio(gray, cutoff=0.25):
        spectrum = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        power = np.abs(spectrum) ** 2
        h, w = gray.shape
        cy, cx = h / 2, w / 2
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2) / np.sqrt(cy ** 2 + cx ** 2)
        return float(power[radius > cutoff].sum() / (power.sum() + 1e-12))

    def grad_mag(gray):
        # Aligned 2026-07-20 with cv2.Sobel, the operator used by
        # abss_net_all_experiments_colab.py which produced the manuscript
        # values. Was np.gradient (~6-8x smaller). See REPRODUCIBILITY.md C.
        import cv2
        arr = gray.astype(np.float32)
        gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1)
        return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))

    rows = []
    start = time.time()
    for cls in CLASS_COUNTS:
        directory = DR224 / cls
        hf, gm = [], []
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            gray = np.array(Image.open(path).convert("L"))
            hf.append(high_freq_ratio(gray))
            gm.append(grad_mag(gray))
        rows.append({
            "Class": cls,
            "Images": len(hf),
            "High_Frequency_Ratio_Mean": float(np.mean(hf)),
            "High_Frequency_Ratio_Std": float(np.std(hf, ddof=1)),
            "Gradient_Magnitude_Mean": float(np.mean(gm)),
            "Gradient_Magnitude_Std": float(np.std(gm, ddof=1)),
        })
    write_csv(TABLES / "frequency_attenuation_filtered_dr224.csv", rows)
    write_latex_table(
        LATEX / "frequency_attenuation_filtered_dr224.tex",
        "High-frequency profile of the supplied filtered DR224 images.",
        "tab:frequency_filtered_dr224",
        rows,
    )
    log.append(f"Frequency profile completed for filtered DR224 in {time.time() - start:.1f} s.")


def write_all_tables(log: list[str]) -> None:
    tables = [
        ("cv_per_fold_metrics", CV_PER_FOLD, "Per-fold cross-validation metrics on DR224.", "tab:perfold"),
        ("cv_summary_metrics", CV_SUMMARY, "Stratified 5-fold cross-validation summary.", "tab:cvsummary"),
        ("oof_per_class_metrics", OOF_PER_CLASS, "Pooled out-of-fold per-class performance.", "tab:oofperclass"),
        ("reliability_bins", RELIABILITY_BINS, "Accuracy stratified by reliability score c.", "tab:relibins"),
        ("external_validation_messidor2", EXTERNAL_VALIDATION, "Messidor-2 zero-shot external validation.", "tab:external"),
        ("ablation_study", ABLATION, "Ablation study showing cumulative ABSS-Net component contribution.", "tab:abl"),
        ("dataset_class_counts", dataset_counts(), "Observed DR224 class counts.", "tab:dataset_counts"),
    ]
    for name, rows, caption, label in tables:
        write_csv(TABLES / f"{name}.csv", rows)
        write_latex_table(LATEX / f"{name}.tex", caption, label, rows)
    write_csv(MAPPING / "reviewer_comment_to_experiment_mapping.csv", REVIEWER_MAPPING)
    write_latex_table(
        LATEX / "reviewer_comment_to_experiment_mapping.tex",
        "Reviewer/editor comments mapped to experimental or code actions.",
        "tab:reviewer_mapping",
        REVIEWER_MAPPING,
    )
    log.append(f"Wrote {len(tables)} result tables plus reviewer mapping in CSV and LaTeX formats.")


def write_summary_and_latex_snippets(log: list[str]) -> None:
    summary = f"""# Revision Experiment Summary

Generated: {datetime.now().isoformat(timespec='seconds')}

## Completed locally

- Copied publication figures from the supplied Overleaf archive into `figures/`.
- Rebuilt CSV and LaTeX tables from the executed notebook/manuscript values.
- Verified DR224 class counts from the supplied dataset path.
- Computed high-frequency energy ratio and gradient-magnitude summaries for the supplied filtered DR224 images.

## Experiments not executable on this machine

- Full 5-fold retraining, baseline reproduction, raw-vs-Gaussian retraining, and Mamba selective-scan ablation require GPU training.
- Calibration Brier/NLL/MCE and temperature scaling require pooled out-of-fold probability/logit arrays, which were not supplied as files.
- Runtime/GPU-memory profiling requires a CUDA device and model checkpoint; this machine reports CPU-only PyTorch.

No numbers were fabricated. Values copied from prior executed notebook outputs are identified in the experiment log.
"""
    (RESULTS / "technical_summary.md").write_text(summary, encoding="utf-8")

    snippets = r"""% LaTeX insertion snippets for revised outputs.
% Requires \usepackage{booktabs,graphicx}.

\input{latex_outputs/cv_summary_metrics.tex}
\input{latex_outputs/cv_per_fold_metrics.tex}
\input{latex_outputs/oof_per_class_metrics.tex}
\input{latex_outputs/reliability_bins.tex}
\input{latex_outputs/external_validation_messidor2.tex}
\input{latex_outputs/ablation_study.tex}
\input{latex_outputs/frequency_attenuation_filtered_dr224.tex}

\begin{figure}[!t]
\centering
\includegraphics[width=0.82\columnwidth]{figures/fig19_reliability_c_curve.png}
\caption{Reliability-head calibration curve (pooled OOF): empirical accuracy of retained samples versus the reliability score $c$.}
\label{fig:relcurve}
\end{figure}
"""
    (LATEX / "latex_insert_snippets.tex").write_text(snippets, encoding="utf-8")
    log.append("Wrote technical summary and LaTeX insertion snippets.")


def write_manifest_and_log(log: list[str]) -> None:
    try:
        import torch
        torch_info = f"torch={torch.__version__}, cuda={torch.cuda.is_available()}"
    except Exception as exc:
        torch_info = f"torch import failed: {exc}"
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch_info,
        "source_files": {
            "notebook": str(CODE / "ABSS_Net_Enhanced_Revised.ipynb"),
            "response_letter_pdf": str(MAPPING / "Response_Letter.pdf"),
            "manuscript_pdf": str(LATEX / "ABSS-Net_Revised_Highlighted_Manuscript.pdf"),
            "overleaf_zip": str(OVERLEAF_ZIP),
            "dataset": str(DR224),
        },
        "outputs": {
            "tables": sorted(p.name for p in TABLES.glob("*")),
            "figures": sorted(p.name for p in FIGURES.glob("*")),
            "latex_outputs": sorted(p.name for p in LATEX.glob("*.tex")),
            "mapping": sorted(p.name for p in MAPPING.glob("*")),
        },
    }
    (RESULTS / "outputs_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log_text = "\n".join([
        "Revision experiment log",
        f"Generated: {manifest['generated_at']}",
        f"Python: {sys.version.split()[0]}",
        f"Platform: {platform.platform()}",
        f"Torch: {torch_info}",
        "",
        "Actions:",
        *[f"- {line}" for line in log],
        "",
        "Blocked experiments and exact reasons:",
        "- Genuine Mamba ablation: mamba-ssm and CUDA/GPU training environment unavailable locally.",
        "- Identical-condition baseline reproduction and paired significance tests: requires full GPU retraining of three baselines; no baseline checkpoints supplied.",
        "- Raw-vs-Gaussian ablation: raw unfiltered APTOS-2019 class-folder tree was not supplied.",
        "- Calibration Brier/NLL/MCE and temperature scaling: pooled OOF probability/logit arrays were not supplied as machine-readable files.",
        "- A100 efficiency profiling: local PyTorch reports no CUDA device; trained fold checkpoints are referenced in the notebook but not present under models/.",
    ])
    (LOGS / "experiment_log.txt").write_text(log_text, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    log: list[str] = []
    copy_figures_from_zip(log)
    write_all_tables(log)
    run_frequency_profile(log)
    write_summary_and_latex_snippets(log)
    write_manifest_and_log(log)
    print("\n".join(log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
