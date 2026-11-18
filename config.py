"""
Configuration and Hardware Parameters for RL-Driven Adaptive Sensing in TinyML
Coupled CMDP Environment based on ADXL355 MEMS Accelerometer and ARM Cortex-M4 MCU.
"""

from dataclasses import dataclass
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"

for d in [DATA_DIR, CACHE_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware energy specifications for ADXL355 and ARM Cortex-M4."""
    # Power Supply & Battery
    VDD: float = 3.3  # Operating voltage (Volts)
    BATTERY_VOLTAGE: float = 3.6  # Li-SOCl2 nominal voltage (Volts)
    BATTERY_CAPACITY_MAH: float = 2600.0  # Li-SOCl2 AA cell capacity (mAh)
    BATTERY_TOTAL_JOULES: float = BATTERY_CAPACITY_MAH * 1e-3 * BATTERY_VOLTAGE * 3600.0  # ~33,696 J

    # ADXL355 MEMS Accelerometer Datasheet Specs
    ADXL355_ACTIVE_CURRENT_UA: float = 200.0  # Microamperes in active sampling
    ADXL355_STANDBY_CURRENT_UA: float = 21.0  # Microamperes in low-power standby
    ADXL355_NOISE_DENSITY: float = 25.0  # ug / sqrt(Hz)

    # STM32L4 Cortex-M4 MCU Specs
    MCU_ACTIVE_CURRENT_MA: float = 10.0  # Active compute @ 80 MHz (mA)
    MCU_DMA_CURRENT_MA: float = 2.0  # DMA receiving without CPU (mA)
    MCU_DEEP_SLEEP_CURRENT_UA: float = 1.0  # Deep sleep with RTC (uA)
    MCU_TOTAL_SRAM_KB: float = 256.0  # 256 KB SRAM
    MCU_TOTAL_FLASH_KB: float = 1024.0  # 1 MB Flash

    # Calibrated Energy Costs per 0.25s Analysis Window (in millijoules, mJ)
    # Tier 1 (100 Hz): 10% sensor duty-cycle + 2ms MCU inference + 248ms deep sleep
    ENERGY_TIER1_MJ: float = 0.098  # ~0.10 mJ
    # Tier 2 (4000 Hz): 100% sensor active + continuous DMA + 50ms 1D-CNN compute
    ENERGY_TIER2_MJ: float = 2.970  # ~3.00 mJ
    ENERGY_MAX_BASELINE_MJ: float = 2.970

    # Transition Penalties (Thrashing Mitigation)
    # Transition 0 -> 1: Sensor wake-up, charge pump stabilization, digital FIR group delay flush
    TRANSITION_COST_0_TO_1_MJ: float = 0.400
    # Transition 1 -> 0: Sensor power-down, clock down-ramp
    TRANSITION_COST_1_TO_0_MJ: float = 0.100

    # Memory budget threshold
    MAX_FLASH_BUDGET_KB: float = 15.0
    MAX_SRAM_BUDGET_KB: float = 15.0


@dataclass(frozen=True)
class SignalConfig:
    """Sampling rates, anti-aliasing filter parameters, and windowing."""
    RAW_FS: int = 12000  # Raw CWRU sampling frequency (Hz)
    WINDOW_DURATION_SEC: float = 0.25  # Duration of each inference window (seconds)
    RAW_WINDOW_SAMPLES: int = int(RAW_FS * WINDOW_DURATION_SEC)  # 3000 samples

    # Tier-1 Low-Power Mode (100 Hz)
    TIER1_FS: int = 100  # 100 Hz
    TIER1_DOWNSAMPLE_FACTOR: int = 120  # 12000 / 120 = 100 Hz
    TIER1_WINDOW_SAMPLES: int = int(TIER1_FS * WINDOW_DURATION_SEC)  # 25 samples
    TIER1_FILTER_CUTOFF_HZ: float = 45.0  # Anti-aliasing LPF cutoff (< 50 Hz Nyquist)
    TIER1_FILTER_ORDER: int = 4

    # Tier-2 High-Fidelity Mode (4000 Hz)
    TIER2_FS: int = 4000  # 4000 Hz
    TIER2_DOWNSAMPLE_FACTOR: int = 3  # 12000 / 3 = 4000 Hz
    TIER2_WINDOW_SAMPLES: int = int(TIER2_FS * WINDOW_DURATION_SEC)  # 1000 samples
    TIER2_FILTER_CUTOFF_HZ: float = 1800.0  # Anti-aliasing LPF cutoff (< 2000 Hz Nyquist)
    TIER2_FILTER_ORDER: int = 4

    # Stream synthesis parameters
    NORMAL_RATIO: float = 0.999  # 99.9% healthy
    FAULT_RATIO: float = 0.001   # <0.1% fault
    BURST_MIN_WINDOWS: int = 5   # 1.25 seconds
    BURST_MAX_WINDOWS: int = 15  # 3.75 seconds


@dataclass(frozen=True)
class ModelConfig:
    """Architectures for TinyML models."""
    # Tier-1 Autoencoder: input 25 -> 8 -> 4 -> 8 -> 25
    AE_INPUT_DIM: int = 25
    AE_HIDDEN_DIM: int = 8
    AE_BOTTLENECK_DIM: int = 4
    AE_EPOCHS: int = 25
    AE_LR: float = 0.003
    AE_BATCH_SIZE: int = 64

    # Tier-2 1D-CNN: input 1000 -> Conv1D(8) -> MaxPool -> Conv1D(16) -> AdaptivePool -> FC(16) -> FC(2)
    CNN_INPUT_LENGTH: int = 1000
    CNN_CHANNELS: int = 1
    CNN_EPOCHS: int = 20
    CNN_LR: float = 0.002
    CNN_BATCH_SIZE: int = 32
    CNN_CLASS_WEIGHT_FAULT: float = 100.0  # Class weighting for high imbalance


@dataclass(frozen=True)
class RLConfig:
    """Constrained MDP and Primal-Dual Double DQN Hyperparameters."""
    STATE_DIM: int = 6  # [mean_tkeo, var_tkeo, kurt_tkeo, crest_tkeo, ae_recon_error, prev_action]
    ACTION_DIM: int = 2  # 0: Tier 1 (100 Hz), 1: Tier 2 (4000 Hz)
    HIDDEN_DIM: int = 32  # 2 hidden layers of 32 units each

    GAMMA: float = 0.95  # Discount factor
    LR: float = 0.001  # Policy learning rate
    BATCH_SIZE: int = 64
    BUFFER_SIZE: int = 10000

    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.02
    EPSILON_DECAY: float = 0.85
    TARGET_UPDATE_FREQ: int = 3  # episodes

    # Dual Lagrangian Parameters
    INITIAL_LAMBDA: float = 5.0  # Initial Lagrange multiplier
    ETA_LAMBDA: float = 15.0  # Dual ascent learning rate
    TARGET_FNR: float = 0.000  # Zero False Negative constraint (< 0.001)

    # Reward Formulation
    WEIGHT_ENERGY: float = 1.0
    HYSTERESIS_PENALTY: float = 0.05  # w_h penalty when switching states
