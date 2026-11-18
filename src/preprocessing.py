"""
Phase 2: Dual-Rate Anti-Aliased Preprocessing & TKEO Feature Extraction
Simulates Tier-1 (100 Hz, 45 Hz LPF) and Tier-2 (4000 Hz, 1800 Hz LPF) hardware filters,
computes 3-sample Teager-Kaiser Energy Operator (TKEO), extracts statistical moments,
and caches arrays to disk as binary .npy for zero-latency RL in-memory execution.
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from config import CACHE_DIR, SignalConfig


def design_butterworth_4th_sos(fc: float, fs: float) -> np.ndarray:
    """
    Analytically designs a 4th-order Butterworth low-pass filter using
    two Second-Order Sections (SOS) via the Bilinear Transform.
    Poles Q: 0.5411961 and 1.3065630.
    """
    K = np.tan(np.pi * fc / fs)
    sos = []
    for Q in [0.5411961, 1.3065630]:
        D = 1.0 + (K / Q) + K * K
        b0 = (K * K) / D
        b1 = 2.0 * b0
        b2 = b0
        a1 = 2.0 * (K * K - 1.0) / D
        a2 = (1.0 - (K / Q) + K * K) / D
        sos.append([b0, b1, b2, 1.0, a1, a2])
    return np.array(sos, dtype=np.float64)


def apply_sos_filter(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Applies Second-Order Section (biquad) IIR filtering across a 1D or 2D array.
    """
    if x.ndim == 1:
        y = x.astype(np.float64).copy()
        for s in sos:
            b0, b1, b2, a0, a1, a2 = s
            w1, w2 = 0.0, 0.0
            out = np.zeros_like(y)
            for i in range(len(y)):
                w0 = y[i] - a1 * w1 - a2 * w2
                out[i] = b0 * w0 + b1 * w1 + b2 * w2
                w2 = w1
                w1 = w0
            y = out
        return y.astype(np.float32)

    # 2D batch filtering (N_windows, N_samples)
    N, L = x.shape
    y = x.astype(np.float64).copy()
    for s in sos:
        b0, b1, b2, a0, a1, a2 = s
        w1 = np.zeros(N, dtype=np.float64)
        w2 = np.zeros(N, dtype=np.float64)
        out = np.zeros_like(y)
        for i in range(L):
            w0 = y[:, i] - a1 * w1 - a2 * w2
            out[:, i] = b0 * w0 + b1 * w1 + b2 * w2
            w2 = w1.copy()
            w1 = w0.copy()
        y = out
    return y.astype(np.float32)


def compute_tkeo(x: np.ndarray) -> np.ndarray:
    """
    Computes the 3-sample Teager-Kaiser Energy Operator (TKEO):
    Psi(x_n) = x_n^2 - x_{n-1} * x_{n+1}
    Vectorized across 1D or 2D array.
    """
    if x.ndim == 1:
        psi = np.zeros_like(x)
        psi[1:-1] = x[1:-1]**2 - x[:-2] * x[2:]
        psi[0] = psi[1]
        psi[-1] = psi[-2]
        return psi

    # 2D array: (N_windows, N_samples)
    psi = np.zeros_like(x)
    psi[:, 1:-1] = x[:, 1:-1]**2 - x[:, :-2] * x[:, 2:]
    psi[:, 0] = psi[:, 1]
    psi[:, -1] = psi[:, -2]
    return psi


def extract_tkeo_moments(psi: np.ndarray) -> np.ndarray:
    """
    Extracts four lightweight statistical moments from the TKEO array:
    1. Mean: average instantaneous energy
    2. Variance: energy spread
    3. Kurtosis: impulsiveness (peakedness of defect impacts)
    4. Crest Factor: ratio of peak energy to RMS energy
    Returns array of shape (N_windows, 4).
    """
    # psi shape: (N_windows, 25)
    mean_val = np.mean(psi, axis=1)  # (N,)
    var_val = np.var(psi, axis=1)    # (N,)
    std_val = np.sqrt(var_val) + 1e-9

    # Kurtosis = E[(X - mu)^4] / std^4
    diff = psi - mean_val[:, np.newaxis]
    kurt_val = np.mean(diff**4, axis=1) / (std_val**4)

    # Crest Factor = max(|Psi|) / RMS(Psi)
    peak_val = np.max(np.abs(psi), axis=1)
    rms_val = np.sqrt(np.mean(psi**2, axis=1)) + 1e-9
    crest_val = peak_val / rms_val

    features = np.column_stack([mean_val, var_val, kurt_val, crest_val]).astype(np.float32)
    return features


class DualRatePreprocessor:
    """
    Preprocesses raw 12 kHz windows into Tier-1 (100 Hz) and Tier-2 (4000 Hz) streams
    using anti-aliasing low-pass filters and decimation.
    """

    def __init__(self):
        # Design filters
        self.sos_tier1 = design_butterworth_4th_sos(
            fc=SignalConfig.TIER1_FILTER_CUTOFF_HZ,
            fs=SignalConfig.RAW_FS
        )
        self.sos_tier2 = design_butterworth_4th_sos(
            fc=SignalConfig.TIER2_FILTER_CUTOFF_HZ,
            fs=SignalConfig.RAW_FS
        )

    def preprocess_stream(
        self,
        raw_windows: np.ndarray,
        labels: np.ndarray,
        metadata: Dict,
        cache_dir: Path = CACHE_DIR,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes the stream and saves binary cache to disk.
        Returns:
            t1_windows: (N, 25) float32
            t2_windows: (N, 1000) float32
            tkeo_features: (N, 4) float32
        """
        N = raw_windows.shape[0]
        print(f"[Phase 2] Anti-aliasing filtering {N} raw 12 kHz windows...")

        # 1. Tier-1 Anti-Aliasing (45 Hz cutoff) and Decimation (factor 120 -> 100 Hz, 25 samples)
        print("  -> Filtering Tier-1 (45 Hz LPF)...")
        filtered_t1 = apply_sos_filter(self.sos_tier1, raw_windows)
        t1_windows = filtered_t1[:, ::SignalConfig.TIER1_DOWNSAMPLE_FACTOR].astype(np.float32)
        assert t1_windows.shape[1] == SignalConfig.TIER1_WINDOW_SAMPLES, (
            f"Expected {SignalConfig.TIER1_WINDOW_SAMPLES} samples, got {t1_windows.shape[1]}"
        )

        # 2. Tier-2 Anti-Aliasing (1800 Hz cutoff) and Decimation (factor 3 -> 4000 Hz, 1000 samples)
        print("  -> Filtering Tier-2 (1800 Hz LPF)...")
        filtered_t2 = apply_sos_filter(self.sos_tier2, raw_windows)
        t2_windows = filtered_t2[:, ::SignalConfig.TIER2_DOWNSAMPLE_FACTOR].astype(np.float32)
        assert t2_windows.shape[1] == SignalConfig.TIER2_WINDOW_SAMPLES, (
            f"Expected {SignalConfig.TIER2_WINDOW_SAMPLES} samples, got {t2_windows.shape[1]}"
        )

        # 3. Compute TKEO on Tier-1 (100 Hz) windows
        print("  -> Computing 3-sample TKEO and extracting 4 statistical moments...")
        psi = compute_tkeo(t1_windows)
        tkeo_features = extract_tkeo_moments(psi)

        # 4. Save to Disk Cache
        print(f"  -> Caching preprocessed binary arrays to {cache_dir}...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "stream_raw.npy", raw_windows)
        np.save(cache_dir / "stream_labels.npy", labels)
        np.save(cache_dir / "stream_tier1_100hz.npy", t1_windows)
        np.save(cache_dir / "stream_tier2_4000hz.npy", t2_windows)
        np.save(cache_dir / "stream_tkeo_features.npy", tkeo_features)

        with open(cache_dir / "stream_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print("[Phase 2 Done] Cached preprocessed streams:")
        print(f"  * Tier-1 (100 Hz): {t1_windows.shape} ({(t1_windows.nbytes / 1024):.1f} KB)")
        print(f"  * Tier-2 (4000 Hz): {t2_windows.shape} ({(t2_windows.nbytes / 1024 / 1024):.2f} MB)")
        print(f"  * TKEO Moments: {tkeo_features.shape} (Mean, Var, Kurtosis, Crest Factor)")

        return t1_windows, t2_windows, tkeo_features


def load_cached_stream(cache_dir: Path = CACHE_DIR) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Loads all cached arrays from disk."""
    t1_windows = np.load(cache_dir / "stream_tier1_100hz.npy")
    t2_windows = np.load(cache_dir / "stream_tier2_4000hz.npy")
    tkeo_features = np.load(cache_dir / "stream_tkeo_features.npy")
    labels = np.load(cache_dir / "stream_labels.npy")
    with open(cache_dir / "stream_metadata.json", "r") as f:
        metadata = json.load(f)
    return t1_windows, t2_windows, tkeo_features, labels, metadata
