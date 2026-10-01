"""Public reimplementation of the Stage-1 PPG artifact-classification workflow.

This script is reconstructed from the methods reported in the peer-reviewed paper
and prior project notes. It is not claimed to be a byte-identical copy of the
original private analysis script.

Dataset: Pulse Transit Time PPG Dataset v1.1.0 (PhysioNet)
https://physionet.org/content/pulse-transit-time-ppg/1.1.0/
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, periodogram
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut

FS_DEFAULT = 500.0
WINDOW_SECONDS = 8.0
ACTIVITIES = ("sit", "walk", "run")
SUBJECTS_DEFAULT = (1, 2, 3)


def _finite_or_zero(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def bandpass(x: np.ndarray, fs: float, low: float = 0.5, high: float = 3.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=np.nanmedian(x) if np.isfinite(x).any() else 0.0)
    x = x - np.mean(x)
    nyq = fs / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, x)


def spectral_entropy(x: np.ndarray, fs: float, fmin: float = 0.1, fmax: float = 20.0) -> float:
    f, pxx = periodogram(x, fs=fs)
    mask = (f >= fmin) & (f <= fmax)
    p = np.asarray(pxx[mask], dtype=float)
    total = p.sum()
    if p.size == 0 or total <= 0:
        return 0.0
    p = p / total
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    return float(-(p * np.log2(p)).sum() / np.log2(p.size))


def dominant_frequency(x: np.ndarray, fs: float, fmin: float = 0.5, fmax: float = 5.0) -> float:
    f, pxx = periodogram(x, fs=fs)
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return 0.0
    ff, pp = f[mask], pxx[mask]
    return float(ff[int(np.argmax(pp))])


def peak_metrics(x: np.ndarray, fs: float) -> tuple[int, float, float]:
    prominence = max(float(np.std(x)) * 0.20, np.finfo(float).eps)
    peaks, _ = find_peaks(x, distance=max(1, int(0.30 * fs)), prominence=prominence)
    if len(peaks) < 2:
        return int(len(peaks)), 0.0, 0.0
    rr = np.diff(peaks) / fs
    hr = 60.0 / np.mean(rr) if np.mean(rr) > 0 else 0.0
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) >= 2 else 0.0
    return int(len(peaks)), float(rmssd), float(hr)


def ppg_features(raw: np.ndarray, fs: float) -> dict[str, float]:
    raw = np.asarray(raw, dtype=float)
    raw = np.nan_to_num(raw, nan=np.nanmedian(raw) if np.isfinite(raw).any() else 0.0)
    filtered = bandpass(raw, fs)
    centered = raw - np.mean(raw)
    residual = centered - filtered
    signal_power = float(np.var(filtered))
    noise_power = float(np.var(residual))
    snr = 10.0 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12))
    n_peaks, rmssd, hr = peak_metrics(filtered, fs)
    return {
        "ppg_std": float(np.std(filtered)),
        "ppg_skewness": _finite_or_zero(skew(filtered, bias=False)),
        "ppg_kurtosis": _finite_or_zero(kurtosis(filtered, fisher=True, bias=False)),
        "ppg_snr_db": float(snr),
        "ppg_spectral_entropy": spectral_entropy(filtered, fs, 0.5, 8.0),
        "ppg_dominant_frequency_hz": dominant_frequency(filtered, fs),
        "ppg_peak_count": float(n_peaks),
        "ppg_amplitude_range": float(np.percentile(filtered, 95) - np.percentile(filtered, 5)),
        "ppg_hrv_rmssd_s": float(rmssd),
        "ppg_hr_estimate_bpm": float(hr),
    }


def accelerometer_features(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, fs: float) -> dict[str, float]:
    mag = np.sqrt(np.asarray(ax, float) ** 2 + np.asarray(ay, float) ** 2 + np.asarray(az, float) ** 2)
    mag = np.nan_to_num(mag, nan=np.nanmedian(mag) if np.isfinite(mag).any() else 0.0)
    return {
        "acc_magnitude_mean": float(np.mean(mag)),
        "acc_magnitude_std": float(np.std(mag)),
        "acc_magnitude_max": float(np.max(mag)),
        "acc_spectral_entropy": spectral_entropy(mag - np.mean(mag), fs, 0.1, 20.0),
    }


def _select_windows(n_samples: int, window_n: int, windows_per_record: int) -> list[int]:
    n_windows = n_samples // window_n
    if n_windows < windows_per_record:
        raise ValueError(
            f"Record has only {n_windows} complete {WINDOW_SECONDS:g}-s windows; "
            f"{windows_per_record} requested."
        )
    if n_windows == windows_per_record:
        return [i * window_n for i in range(n_windows)]
    indices = np.linspace(0, n_windows - 1, windows_per_record, dtype=int)
    return [int(i * window_n) for i in indices]


def extract_record(
    csv_path: Path,
    subject: int,
    activity: str,
    fs: float,
    ppg_col: str,
    windows_per_record: int,
) -> list[dict[str, float | int | str]]:
    df = pd.read_csv(csv_path)
    required = {ppg_col, "a_x", "a_y", "a_z"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {missing}")
    window_n = int(round(WINDOW_SECONDS * fs))
    starts = _select_windows(len(df), window_n, windows_per_record)
    out = []
    for window_id, start in enumerate(starts):
        stop = start + window_n
        chunk = df.iloc[start:stop]
        row: dict[str, float | int | str] = {
            "subject": subject,
            "activity": activity,
            "window_id": window_id,
            "source_file": csv_path.name,
        }
        row.update(ppg_features(chunk[ppg_col].to_numpy(), fs))
        row.update(
            accelerometer_features(
                chunk["a_x"].to_numpy(),
                chunk["a_y"].to_numpy(),
                chunk["a_z"].to_numpy(),
                fs,
            )
        )
        out.append(row)
    return out


def synthetic_records(
    subjects: tuple[int, ...],
    fs: float,
    windows_per_record: int,
    seed: int = 20260925,
):
    """Create clearly synthetic windows for an offline smoke test only."""
    rng = np.random.default_rng(seed)
    n = int(round(WINDOW_SECONDS * fs))
    t = np.arange(n) / fs
    rows = []
    for subject in subjects:
        for activity in ACTIVITIES:
            for window_id in range(windows_per_record):
                motion = {"sit": 0.02, "walk": 0.15, "run": 0.35}[activity]
                hr_hz = 1.1 + 0.05 * subject + {"sit": 0.0, "walk": 0.35, "run": 0.75}[activity]
                ppg = np.sin(2 * np.pi * hr_hz * t) + 0.25 * np.sin(4 * np.pi * hr_hz * t)
                ppg += motion * rng.normal(size=n)
                ax = motion * rng.normal(size=n)
                ay = motion * rng.normal(size=n)
                az = 1.0 + motion * rng.normal(size=n)
                row = {
                    "subject": subject,
                    "activity": activity,
                    "window_id": window_id,
                    "source_file": "synthetic_demo",
                }
                row.update(ppg_features(ppg, fs))
                row.update(accelerometer_features(ax, ay, az, fs))
                rows.append(row)
    return rows


def loso_metrics(frame: pd.DataFrame, feature_cols: list[str], seed: int = 42):
    X = frame[feature_cols].to_numpy(float)
    y = frame["activity"].to_numpy(str)
    groups = frame["subject"].to_numpy(int)
    logo = LeaveOneGroupOut()
    fold_rows = []
    y_true_all, y_pred_all = [], []
    importances = []
    for fold, (train, test) in enumerate(logo.split(X, y, groups), start=1):
        model = RandomForestClassifier(
            n_estimators=500,
            random_state=seed + fold,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X[train], y[train])
        pred = model.predict(X[test])
        y_true_all.extend(y[test])
        y_pred_all.extend(pred)
        importances.append(model.feature_importances_)
        fold_rows.append(
            {
                "held_out_subject": int(groups[test][0]),
                "accuracy": float(accuracy_score(y[test], pred)),
                "macro_f1": float(f1_score(y[test], pred, average="macro", zero_division=0)),
            }
        )
    labels = list(ACTIVITIES)
    return {
        "accuracy_mean": float(np.mean([r["accuracy"] for r in fold_rows])),
        "accuracy_sd": float(np.std([r["accuracy"] for r in fold_rows], ddof=0)),
        "macro_f1_mean": float(np.mean([r["macro_f1"] for r in fold_rows])),
        "macro_f1_sd": float(np.std([r["macro_f1"] for r in fold_rows], ddof=0)),
        "folds": fold_rows,
        "confusion_matrix": confusion_matrix(y_true_all, y_pred_all, labels=labels).tolist(),
        "labels": labels,
        "feature_importance_mean": {
            name: float(value)
            for name, value in zip(feature_cols, np.mean(np.vstack(importances), axis=0))
        },
    }


def plot_results(ppg_only: dict, combined: dict, out_path: Path) -> None:
    labels = ACTIVITIES
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, result, title in [
        (axes[0], ppg_only, "PPG only"),
        (axes[1], combined, "PPG + accelerometer"),
    ]:
        cm = np.asarray(result["confusion_matrix"])
        ax.imshow(cm)
        ax.set_title(title)
        ax.set_xticks(range(3), labels)
        ax.set_yticks(range(3), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    axes[2].bar(
        ["PPG", "PPG+ACC"],
        [ppg_only["accuracy_mean"], combined["accuracy_mean"]],
        yerr=[ppg_only["accuracy_sd"], combined["accuracy_sd"]],
        capsize=4,
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("LOSO accuracy")
    axes[2].set_title("Model comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, help="PhysioNet CSV directory; alternatively set PPG_DATA_DIR")
    parser.add_argument("--ppg-col", default="pleth_1", help="PPG channel; default pleth_1")
    parser.add_argument("--fs", type=float, default=FS_DEFAULT)
    parser.add_argument("--subjects", nargs="+", type=int, default=list(SUBJECTS_DEFAULT))
    parser.add_argument("--windows-per-record", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--demo-synthetic", action="store_true", help="Run an offline synthetic smoke test; not a paper reproduction")
    args = parser.parse_args()

    subjects = tuple(args.subjects)
    if args.demo_synthetic:
        rows = synthetic_records(subjects, args.fs, args.windows_per_record)
        input_mode = "synthetic_demo"
    else:
        data_dir = args.data_dir or (
            Path(os.environ["PPG_DATA_DIR"]) if os.getenv("PPG_DATA_DIR") else Path("data/csv")
        )
        first_expected = data_dir / f"s{subjects[0]}_sit.csv"
        if not first_expected.is_file():
            raise FileNotFoundError(
                f"Expected {first_expected}. Download the PhysioNet CSV files "
                "or set PPG_DATA_DIR to the directory containing s1_sit.csv, etc."
            )
        rows = []
        for subject in subjects:
            for activity in ACTIVITIES:
                path = data_dir / f"s{subject}_{activity}.csv"
                if not path.is_file():
                    raise FileNotFoundError(path)
                rows.extend(
                    extract_record(path, subject, activity, args.fs, args.ppg_col, args.windows_per_record)
                )
        input_mode = "physionet_csv"

    frame = pd.DataFrame(rows)
    ppg_cols = [c for c in frame.columns if c.startswith("ppg_")]
    acc_cols = [c for c in frame.columns if c.startswith("acc_")]
    ppg_result = loso_metrics(frame, ppg_cols)
    combined_result = loso_metrics(frame, ppg_cols + acc_cols)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "window_features.csv", index=False)
    payload = {
        "provenance": {
            "implementation": "public reimplementation from published methods and prior project history",
            "byte_identical_to_original_private_script": False,
            "input_mode": input_mode,
            "sampling_rate_hz": args.fs,
            "window_seconds": WINDOW_SECONDS,
            "subjects": list(subjects),
            "windows_per_record": args.windows_per_record,
            "ppg_channel": args.ppg_col,
        },
        "ppg_only": ppg_result,
        "ppg_plus_accelerometer": combined_result,
        "published_reference_metrics": {
            "ppg_only_accuracy_mean": 0.361,
            "ppg_only_macro_f1_mean": 0.226,
            "ppg_plus_acc_accuracy_mean": 0.787,
            "ppg_plus_acc_macro_f1_mean": 0.740,
            "note": "Publication reference values; exact replication requires the original private preprocessing/model settings.",
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    plot_results(ppg_result, combined_result, args.output_dir / "results_final.png")
    print(
        json.dumps(
            {
                "windows": len(frame),
                "ppg_only_accuracy": ppg_result["accuracy_mean"],
                "ppg_plus_acc_accuracy": combined_result["accuracy_mean"],
                "results": str(args.output_dir),
                "mode": input_mode,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
