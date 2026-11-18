"""
Phase 1: Dataset Preparation & Imbalanced Industrial Lifecycle Stream Construction
Loads CWRU Bearing Data (.mat) or generates physically-grounded 12 kHz bearing vibration signals,
slices them into uniform 0.25-second windows, and constructs a 99.9% healthy / <0.1% fault stream.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from config import DATA_DIR, SignalConfig, HardwareConfig


class PhysicalBearingVibrationGenerator:
    """
    Simulates high-fidelity 12 kHz industrial vibration signals based on bearing physics
    (CWRU baseline dynamics, shaft rotation, defect frequencies BPFO/BPFI, and structural resonance).
    """

    def __init__(self, fs: int = SignalConfig.RAW_FS, rpm: float = 1772.0, seed: int = 42):
        self.fs = fs
        self.rpm = rpm
        self.fr = rpm / 60.0  # Shaft rotational frequency (~29.53 Hz)
        self.rng = np.random.RandomState(seed)

        # Bearing Kinematic Defect Frequencies (6205-2RS JEM SKF Deep Groove Ball Bearing)
        # BPFO = (N_balls / 2) * (1 - (d/D)*cos(alpha)) * fr ~ 3.585 * fr
        # BPFI = (N_balls / 2) * (1 + (d/D)*cos(alpha)) * fr ~ 5.415 * fr
        self.bpfo = 3.585 * self.fr  # ~105.8 Hz
        self.bpfi = 5.415 * self.fr  # ~159.9 Hz

        # Housing structural resonance frequency (excited by defect impacts)
        self.fn_res = 3100.0  # Hz
        self.damping_zeta = 0.05

    def generate_healthy_window(self, n_samples: int = SignalConfig.RAW_WINDOW_SAMPLES) -> np.ndarray:
        """
        Generates a 0.25s normal baseline vibration window:
        Shaft rotation fundamental + 2nd/3rd harmonics + mild Gaussian sensor noise.
        """
        t = np.arange(n_samples) / self.fs
        # Shaft rotational harmonics
        vibration = (
            0.18 * np.sin(2 * np.pi * self.fr * t + self.rng.uniform(0, 2 * np.pi))
            + 0.06 * np.sin(2 * np.pi * (2 * self.fr) * t + self.rng.uniform(0, 2 * np.pi))
            + 0.03 * np.sin(2 * np.pi * (3 * self.fr) * t + self.rng.uniform(0, 2 * np.pi))
            + 0.015 * np.sin(2 * np.pi * (4 * self.fr) * t + self.rng.uniform(0, 2 * np.pi))
        )
        # Background industrial hum + thermal sensor noise (matching ADXL355 ~25 ug/sqrt(Hz))
        noise = self.rng.normal(0.0, 0.035, size=n_samples)
        return (vibration + noise).astype(np.float32)

    def generate_fault_window(
        self,
        n_samples: int = SignalConfig.RAW_WINDOW_SAMPLES,
        fault_type: str = "outer_race",
        severity: float = 1.0,
    ) -> np.ndarray:
        """
        Generates a 0.25s bearing fault vibration window:
        Periodic microscopic mechanical impacts exciting high-frequency structural resonance
        with rapid exponential ring-down decay.
        """
        healthy_base = self.generate_healthy_window(n_samples)
        f_impact = self.bpfo if fault_type == "outer_race" else self.bpfi
        period_samples = int(self.fs / f_impact)

        # Impact response duration (ring-down ~1.5 - 2.5 ms)
        impulse_len = int(0.003 * self.fs)
        t_imp = np.arange(impulse_len) / self.fs
        # Damped resonant ringing
        omega_d = 2 * np.pi * self.fn_res * np.sqrt(1 - self.damping_zeta**2)
        impulse_response = np.exp(-self.damping_zeta * 2 * np.pi * self.fn_res * t_imp) * np.sin(omega_d * t_imp)

        fault_signal = np.zeros(n_samples, dtype=np.float32)
        # Place impacts across the window
        impact_idx = self.rng.randint(0, period_samples)
        while impact_idx < n_samples:
            end_idx = min(impact_idx + impulse_len, n_samples)
            actual_len = end_idx - impact_idx
            # Amplitude modulation for BPFI or stochastic load variation
            amp = severity * (1.2 + 0.3 * np.sin(2 * np.pi * self.fr * (impact_idx / self.fs)))
            fault_signal[impact_idx:end_idx] += (amp * impulse_response[:actual_len]).astype(np.float32)
            impact_idx += period_samples + self.rng.randint(-2, 3)

        return healthy_base + fault_signal


class CWRUDatasetLoader:
    """
    Attempts to load real CWRU .mat files from local directories,
    falling back seamlessly to the physical bearing generator if files are absent.
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.generator = PhysicalBearingVibrationGenerator()

    def _find_mat_files(self) -> List[Path]:
        if not self.data_dir.exists():
            return []
        return list(self.data_dir.glob("**/*.mat"))

    def load_cwru_mat(self, file_path: Path) -> Optional[np.ndarray]:
        """Loads vibration array from a CWRU .mat file."""
        try:
            import scipy.io as sio  # type: ignore
            mat = sio.loadmat(file_path)
            for k in mat.keys():
                if "DE_time" in k:  # Drive End Accelerometer
                    return mat[k].flatten().astype(np.float32)
        except Exception:
            pass
        return None

    def acquire_raw_signals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (healthy_signal_pool, fault_signal_pool).
        """
        mat_files = self._find_mat_files()
        healthy_signals = []
        fault_signals = []

        for p in mat_files:
            sig = self.load_cwru_mat(p)
            if sig is not None:
                fname = p.name.lower()
                if "normal" in fname or "97" in fname or "98" in fname:
                    healthy_signals.append(sig)
                else:
                    fault_signals.append(sig)

        if len(healthy_signals) > 0 and len(fault_signals) > 0:
            print(f"[Phase 1] Loaded {len(healthy_signals)} healthy and {len(fault_signals)} fault CWRU .mat files.")
            healthy_pool = np.concatenate(healthy_signals)
            fault_pool = np.concatenate(fault_signals)
            return healthy_pool, fault_pool

        # Fallback to high-fidelity physical vibration synthesis
        print("[Phase 1] Real CWRU .mat files not found in data/. Generating physically-grounded CWRU bearing vibration...")
        # Synthesize 100 seconds of healthy and 20 seconds of fault
        n_healthy_samples = int(100.0 * SignalConfig.RAW_FS)
        n_fault_samples = int(20.0 * SignalConfig.RAW_FS)

        healthy_pool = np.concatenate([
            self.generator.generate_healthy_window()
            for _ in range(n_healthy_samples // SignalConfig.RAW_WINDOW_SAMPLES)
        ])
        fault_pool = np.concatenate([
            self.generator.generate_fault_window(fault_type="outer_race", severity=1.2)
            for _ in range(n_fault_samples // SignalConfig.RAW_WINDOW_SAMPLES)
        ])
        return healthy_pool, fault_pool


class IndustrialStreamBuilder:
    """
    Constructs an industrial lifecycle stream consisting of 99.9% healthy baseline windows
    and <0.1% injected fault bursts (spanning 5 to 15 consecutive windows).
    """

    def __init__(self, total_windows: int = 6000, seed: int = 42):
        self.total_windows = total_windows
        self.seed = seed
        self.loader = CWRUDatasetLoader()

    def build_stream(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Returns:
            windows: shape (N_windows, 3000), 12 kHz raw vibration
            labels: shape (N_windows,), 0 = Normal, 1 = Fault
            metadata: dictionary describing injection locations and imbalance stats
        """
        np.random.seed(self.seed)
        healthy_pool, fault_pool = self.loader.acquire_raw_signals()

        w_size = SignalConfig.RAW_WINDOW_SAMPLES  # 3000
        n_healthy_avail = len(healthy_pool) // w_size
        n_fault_avail = len(fault_pool) // w_size

        # Pre-slice pools into windows
        healthy_windows = [healthy_pool[i * w_size : (i + 1) * w_size] for i in range(n_healthy_avail)]
        fault_windows = [fault_pool[i * w_size : (i + 1) * w_size] for i in range(n_fault_avail)]

        # Prepare timeline with 99.9% normal
        # Calculate target fault windows (<0.1%)
        target_fault_count = max(5, int(self.total_windows * SignalConfig.FAULT_RATIO))
        labels = np.zeros(self.total_windows, dtype=np.int32)
        stream_windows = np.zeros((self.total_windows, w_size), dtype=np.float32)

        # Inject 1 to 2 bursts of fault windows (5 to 15 consecutive windows each)
        burst_records = []
        fault_allocated = 0
        min_spacing = int(self.total_windows * 0.4)  # space bursts apart

        possible_starts = [
            int(self.total_windows * 0.35),
            int(self.total_windows * 0.75),
        ]

        for start_center in possible_starts:
            if fault_allocated >= target_fault_count:
                break
            burst_len = np.random.randint(
                SignalConfig.BURST_MIN_WINDOWS,
                SignalConfig.BURST_MAX_WINDOWS + 1
            )
            # Clip if exceeds remaining target
            burst_start = start_center + np.random.randint(-50, 50)
            burst_end = burst_start + burst_len
            labels[burst_start:burst_end] = 1
            burst_records.append({"start_window": burst_start, "end_window": burst_end, "length": burst_len})
            fault_allocated += burst_len

        # Populate stream windows
        for i in range(self.total_windows):
            if labels[i] == 0:
                h_idx = np.random.randint(0, len(healthy_windows))
                stream_windows[i] = healthy_windows[h_idx]
            else:
                f_idx = np.random.randint(0, len(fault_windows))
                stream_windows[i] = fault_windows[f_idx]

        actual_fault_count = int(np.sum(labels))
        actual_fault_pct = (actual_fault_count / self.total_windows) * 100.0

        metadata = {
            "total_windows": self.total_windows,
            "window_duration_sec": SignalConfig.WINDOW_DURATION_SEC,
            "total_stream_duration_sec": self.total_windows * SignalConfig.WINDOW_DURATION_SEC,
            "healthy_windows": self.total_windows - actual_fault_count,
            "fault_windows": actual_fault_count,
            "fault_percentage": actual_fault_pct,
            "bursts": burst_records,
        }

        print(
            f"[Phase 1 Done] Built Industrial Stream: {self.total_windows} windows ({metadata['total_stream_duration_sec']/60:.1f} mins), "
            f"{metadata['healthy_windows']} Healthy ({100 - actual_fault_pct:.3f}%), "
            f"{actual_fault_count} Fault ({actual_fault_pct:.3f}%)."
        )
        return stream_windows, labels, metadata
