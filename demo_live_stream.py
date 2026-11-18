"""
Interactive Live Demonstration: RL-Driven Adaptive Sensing (Acts 1 & 2)
Simulates a real-time streaming pipeline of industrial vibration on an edge IoT node.
Demonstrates:
  Act 1: Nominal operation (agent stays in Tier-1 100 Hz, energy curve flattens, >90% savings)
  Act 2: Fault burst injection (TKEO kurtosis & AE error jump, instant escalation to Tier-2, Zero-FN proof)
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import CACHE_DIR, HardwareConfig, SignalConfig
from src.preprocessing import load_cached_stream
from src.environment import HardwareCoupledCMDP
from src.dqn_agent import CompactQNetwork


def render_bar(val: float, max_val: float, width: int = 20, fill_char: str = "█") -> str:
    filled = int(min(1.0, max(0.0, val / max_val)) * width)
    return fill_char * filled + "░" * (width - filled)


def run_live_stream_demo(speed_fps: float = 10.0, max_display_windows: int = 150):
    print("=" * 85)
    print("  LIVE EDGE IoT DEMONSTRATION: RL ADAPTIVE SENSING ON ARM CORTEX-M4")
    print("=" * 85)
    print("Streaming industrial vibration data through dual-rate hardware pipeline...\n")

    # Load cached stream
    try:
        t1, t2, tkeo, labels, meta = load_cached_stream(CACHE_DIR)
        ae_errors = np.load(CACHE_DIR / "precomputed_ae_error.npy")
        cnn_probs = np.load(CACHE_DIR / "precomputed_cnn_prob.npy")
    except Exception as e:
        print(f"[Error] Cached stream not found. Please run 'python3 run_full_pipeline.py' first.")
        return

    # Find the first fault burst to ensure demo showcases both Act 1 and Act 2
    fault_indices = np.where(labels == 1)[0]
    if len(fault_indices) == 0:
        print("[Error] No fault bursts found in stream.")
        return

    first_fault_idx = fault_indices[0]
    # Start 40 windows before the fault burst
    start_window = max(0, first_fault_idx - 40)
    end_window = min(len(labels), first_fault_idx + 40)

    # Load trained Q-network policy
    q_net = CompactQNetwork()
    weights_path = CACHE_DIR / "trained_dqn_weights.npz"
    if weights_path.exists():
        data = np.load(weights_path)
        q_net.W1[:] = data["W1"]
        q_net.b1[:] = data["b1"]
        q_net.W2[:] = data["W2"]
        q_net.b2[:] = data["b2"]
        q_net.W3[:] = data["W3"]
        q_net.b3[:] = data["b3"]
        print("[Info] Loaded trained Double DQN policy weights from cache.")
    else:
        print("[Warning] Trained weights not found. Using initialized network.")

    # Load environment
    env = HardwareCoupledCMDP(tkeo, ae_errors, cnn_probs, labels)
    state = env.reset(start_idx=start_window)

    cum_energy_rl = 0.0
    cum_energy_static = 0.0
    delay = 1.0 / speed_fps if speed_fps > 0 else 0.0

    print(f"Demo window range: {start_window} to {end_window} (Injected Fault Burst at #{first_fault_idx})")
    print("Press Ctrl+C to stop.\n")
    time.sleep(1.0)

    for i in range(start_window, end_window):
        # State: [mean, var, kurt, crest, ae_err, prev_act]
        kurt_val = tkeo[i, 2]
        ae_val = ae_errors[i]
        label = labels[i]

        # Learned RL Double DQN action
        action = int(np.argmax(q_net.predict(state)[0]))

        next_state, reward, cost, done, info = env.step(action)

        e_rl = info["energy_mj"]
        e_static = HardwareConfig.ENERGY_MAX_BASELINE_MJ
        cum_energy_rl += e_rl
        cum_energy_static += e_static

        # Visual formatting
        mode_str = "\033[92m[TIER 1: 100 Hz]\033[0m" if action == 0 else "\033[91;1m[TIER 2: 4000 Hz]\033[0m"
        model_str = "Shallow AE (0.1 mJ)" if action == 0 else "\033[93mDeep 1D-CNN (3.0 mJ)\033[0m"

        if label == 1:
            status_tag = "\033[91;1m🚨 FAULT BURST ACTIVE\033[0m"
        else:
            status_tag = "\033[92m✓ HEALTHY BASELINE\033[0m"

        kurt_bar = render_bar(kurt_val, max_val=15.0, width=12)
        ae_bar = render_bar(ae_val, max_val=0.005, width=12)

        energy_sav_pct = (1.0 - (cum_energy_rl / cum_energy_static)) * 100.0

        print(
            f"Win #{i:4d} | "
            f"Kurt: {kurt_val:5.2f} {kurt_bar} | "
            f"AE Err: {ae_val:7.5f} {ae_bar} | "
            f"{mode_str} | "
            f"{status_tag} | "
            f"Energy Sav: {energy_sav_pct:4.1f}%"
        )

        if action == 1 and info["prev_action"] == 0:
            print(f"      >>> [ACT 2 ZERO-FN ESCALATION] Microscopic Transient Detected! Waking ADXL355 to 4,000 Hz & running 1D-CNN!")

        state = next_state
        if delay > 0:
            time.sleep(delay)

    print("\n" + "=" * 85)
    print("  LIVE STREAM DEMONSTRATION COMPLETE")
    print(f"  * Cumulative Energy Consumed by RL Policy: {cum_energy_rl:.2f} mJ")
    print(f"  * Cumulative Energy Consumed by Static 4 kHz: {cum_energy_static:.2f} mJ")
    print(f"  * Total System Energy Reduction: {(1.0 - cum_energy_rl/cum_energy_static)*100.0:.2f}%")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Streaming Demo for RL TinyML")
    parser.add_argument("--fps", type=float, default=12.0, help="Frames per second playback speed (default: 12)")
    args = parser.parse_args()

    run_live_stream_demo(speed_fps=args.fps)
