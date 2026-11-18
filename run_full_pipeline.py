"""
Master Orchestrator: End-to-End RL-Driven Adaptive Sensing Pipeline
Executes Phase 1 through Phase 7 sequentially:
  Phase 1: Dataset Preparation & Imbalanced Industrial Lifecycle Stream Construction
  Phase 2: Dual-Rate Anti-Aliased Preprocessing & TKEO State Feature Extraction
  Phase 3: Pre-Training Cascaded TinyML Models (Tier-1 AE & Tier-2 1D-CNN)
  Phase 4: Hardware-Coupled CMDP Environment Initialization
  Phase 5: Primal-Dual Double DQN Training with Dual Ascent (Zero-FN Enforcement)
  Phase 6: INT8 Post-Training Quantization & Cortex-M4 Memory Audit (< 15 KB)
  Phase 7: Live Demo Report Generation & Comparative Benchmarking
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Ensure workspace root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import CACHE_DIR, OUTPUT_DIR, HardwareConfig, ModelConfig, RLConfig, SignalConfig
from src.dataset import IndustrialStreamBuilder
from src.preprocessing import DualRatePreprocessor
from src.models import (
    Lightweight1DCNNFast,
    ShallowAutoencoderNumPy,
    precompute_model_responses,
)
from src.environment import HardwareCoupledCMDP
from src.dqn_agent import PrimalDualDoubleDQNAgent
from src.quantization import (
    QuantizedAutoencoder,
    QuantizedDQNPolicy,
    audit_microcontroller_memory,
)
from src.visualization import (
    generate_benchmark_summary,
    generate_html_report,
)


def run_pipeline(total_windows: int = 4000, n_rl_episodes: int = 25, seed: int = 42):
    print("=" * 80)
    print("  RL-DRIVEN ADAPTIVE SENSING FOR TINYML: COMPLETE 7-PHASE PIPELINE")
    print("=" * 80)
    start_total_time = time.time()

    # -----------------------------------------------------------------
    # PHASE 1: Dataset Preparation & Imbalanced Stream Construction
    # -----------------------------------------------------------------
    print("\n>>> [PHASE 1] Dataset Preparation & Imbalanced Stream Construction")
    builder = IndustrialStreamBuilder(total_windows=total_windows, seed=seed)
    raw_windows, labels, metadata = builder.build_stream()

    # -----------------------------------------------------------------
    # PHASE 2: Dual-Rate Anti-Aliased Preprocessing (Offline)
    # -----------------------------------------------------------------
    print("\n>>> [PHASE 2] Dual-Rate Anti-Aliased Preprocessing & TKEO Extraction")
    preprocessor = DualRatePreprocessor()
    t1_windows, t2_windows, tkeo_features = preprocessor.preprocess_stream(
        raw_windows, labels, metadata, cache_dir=CACHE_DIR
    )

    # -----------------------------------------------------------------
    # PHASE 3: Pre-Training Cascaded TinyML Models
    # -----------------------------------------------------------------
    print("\n>>> [PHASE 3] Pre-Training Cascaded TinyML Models")
    # Tier-1 Autoencoder (Unsupervised on healthy 100 Hz windows)
    healthy_indices = np.where(labels == 0)[0]
    print(f"  * Training Tier-1 Autoencoder on {len(healthy_indices)} healthy windows...")
    ae_model = ShallowAutoencoderNumPy(seed=seed)
    ae_model.fit(t1_windows[healthy_indices], epochs=ModelConfig.AE_EPOCHS)

    # Tier-2 1D-CNN (Supervised on 4,000 Hz windows with class-weighted loss)
    print("  * Training Tier-2 1D-CNN on 4,000 Hz windows with class-weighted loss...")
    cnn_model = Lightweight1DCNNFast(seed=seed)
    cnn_model.fit(t2_windows, labels, epochs=ModelConfig.CNN_EPOCHS)

    # Pre-Compute Model Responses Across Stream
    ae_errors, cnn_probs = precompute_model_responses(
        t1_windows, t2_windows, ae_model, cnn_model, cache_dir=CACHE_DIR
    )

    # -----------------------------------------------------------------
    # PHASE 4: Designing Hardware-Coupled CMDP Environment
    # -----------------------------------------------------------------
    print("\n>>> [PHASE 4] Initializing Hardware-Coupled CMDP Environment")
    env = HardwareCoupledCMDP(
        tkeo_features=tkeo_features,
        ae_errors=ae_errors,
        cnn_probs=cnn_probs,
        labels=labels,
        seed=seed,
    )
    print(f"  * Initialized CMDP with {env.N} windows, 6-D state vector, and dual-energy physics.")

    # -----------------------------------------------------------------
    # PHASE 5: Training Primal-Dual Double DQN Agent
    # -----------------------------------------------------------------
    print("\n>>> [PHASE 5] Training Primal-Dual Double DQN Agent (Zero-FN Dual Ascent)")
    agent = PrimalDualDoubleDQNAgent(seed=seed)

    episode_length = min(1500, total_windows)
    for ep in range(n_rl_episodes):
        # Pick random starting slice
        start_idx = np.random.randint(0, total_windows - episode_length) if total_windows > episode_length else 0
        state = env.reset(start_idx=start_idx)

        ep_reward = 0.0
        ep_cost = 0.0
        ep_energy_mj = 0.0
        faults_in_ep = 0
        fn_in_ep = 0

        for step in range(episode_length):
            action = agent.select_action(state, eval_mode=False)
            next_state, reward, cost, done, info = env.step(action)

            agent.replay_buffer.push(state, action, reward, cost, next_state, done)
            loss = agent.update_primal()

            ep_reward += reward
            ep_cost += cost
            ep_energy_mj += info["energy_mj"]

            if info["label"] == 1:
                faults_in_ep += 1
                if info["false_negative"]:
                    fn_in_ep += 1

            state = next_state
            if done:
                break

        # End of episode: Dual Update on Lagrange multiplier
        fnr_ep = (fn_in_ep / max(1, faults_in_ep)) if faults_in_ep > 0 else 0.0
        new_lambda = agent.update_dual(fnr_ep)

        # Decay epsilon and sync target network
        agent.decay_epsilon()
        if (ep + 1) % RLConfig.TARGET_UPDATE_FREQ == 0:
            agent.sync_target_network()

        if (ep + 1) % 5 == 0 or ep == n_rl_episodes - 1:
            avg_e_window = ep_energy_mj / episode_length
            energy_sav_pct = max(0.0, (1.0 - (avg_e_window / HardwareConfig.ENERGY_MAX_BASELINE_MJ)) * 100.0)
            print(
                f"  Episode {ep+1:2d}/{n_rl_episodes:2d} | "
                f"Lambda: {new_lambda:5.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"FNR: {fnr_ep * 100.1:.1f}% ({fn_in_ep}/{faults_in_ep}) | "
                f"Energy Savings: {energy_sav_pct:.1f}%"
            )

    # Save trained model weights to cache
    np.savez(
        CACHE_DIR / "trained_dqn_weights.npz",
        W1=agent.q_online.W1, b1=agent.q_online.b1,
        W2=agent.q_online.W2, b2=agent.q_online.b2,
        W3=agent.q_online.W3, b3=agent.q_online.b3,
        lambda_cost=agent.lambda_cost,
    )
    np.savez(
        CACHE_DIR / "trained_ae_weights.npz",
        W1=ae_model.W1, b1=ae_model.b1,
        W2=ae_model.W2, b2=ae_model.b2,
        W3=ae_model.W3, b3=ae_model.b3,
        W4=ae_model.W4, b4=ae_model.b4,
    )

    # -----------------------------------------------------------------
    # PHASE 6: Edge Quantization & Microcontroller Memory Verification
    # -----------------------------------------------------------------
    print("\n>>> [PHASE 6] INT8 Quantization & MCU Memory Verification")
    quant_ae = QuantizedAutoencoder(ae_model)
    quant_dqn = QuantizedDQNPolicy(agent.q_online)

    # Sample test states for policy agreement audit
    test_env = HardwareCoupledCMDP(
        tkeo_features=tkeo_features,
        ae_errors=ae_errors,
        cnn_probs=cnn_probs,
        labels=labels,
        normalizer=env.normalizer,
    )
    test_states = np.array([test_env._get_state(i, 0) for i in range(min(500, total_windows))])
    mcu_audit = audit_microcontroller_memory(
        fp32_ae=ae_model,
        fp32_dqn=agent.q_online,
        quant_ae=quant_ae,
        quant_dqn=quant_dqn,
        test_states=test_states,
    )

    # -----------------------------------------------------------------
    # PHASE 7: Structuring Live Demo Presentation & Benchmarking
    # -----------------------------------------------------------------
    print("\n>>> [PHASE 7] Running Comparative Simulation on Full Industrial Stream")

    # Evaluate 4 policies across full stream
    policies = {
        "Static 4,000 Hz (Always-On)": lambda s, info: 1,
        "Static 100 Hz (Always-On)": lambda s, info: 0,
        "Fixed Heuristic Threshold (Kurtosis > 3.0)": lambda s, info: 1 if s[2] > 0.8 or s[4] > 1.2 else 0,
        "Proposed RL Adaptive Policy": lambda s, info: agent.select_action(s, eval_mode=True),
    }

    stream_results = {}
    timeline_records = {
        "timestamps": np.arange(total_windows) * SignalConfig.WINDOW_DURATION_SEC,
        "raw_signal": raw_windows[:, 0],  # first sample of each window
        "tkeo_kurtosis": tkeo_features[:, 2],
        "ae_errors": ae_errors,
        "labels": labels,
        "bursts": metadata.get("bursts", []),
    }

    for p_name, policy_fn in policies.items():
        sim_env = HardwareCoupledCMDP(
            tkeo_features=tkeo_features,
            ae_errors=ae_errors,
            cnn_probs=cnn_probs,
            labels=labels,
            normalizer=env.normalizer,
            max_steps=total_windows,
        )
        s = sim_env.reset(start_idx=0)
        e_list, fn_list, fa_list, sw_list, a_list = [], [], [], [], []

        for step in range(total_windows):
            a = policy_fn(s, {})
            s_next, r, c, done, info = sim_env.step(a)
            e_list.append(info["energy_mj"])
            fn_list.append(info["false_negative"])
            fa_list.append(info["false_alarm"])
            sw_list.append(info["switched"])
            a_list.append(a)
            s = s_next

        stream_results[p_name] = {
            "energies_mj": np.array(e_list),
            "false_negatives": np.array(fn_list),
            "false_alarms": np.array(fa_list),
            "switches": np.array(sw_list),
            "actions": np.array(a_list),
            "labels": labels,
        }

    # Save timeline records for Act 1 and Act 2 plotting
    timeline_records["actions_rl"] = stream_results["Proposed RL Adaptive Policy"]["actions"]
    timeline_records["energies_rl"] = stream_results["Proposed RL Adaptive Policy"]["energies_mj"]
    timeline_records["energies_static"] = stream_results["Static 4,000 Hz (Always-On)"]["energies_mj"]

    # Compute benchmarks
    benchmarks = generate_benchmark_summary(stream_results, total_windows)

    # Print clean benchmark table in terminal
    print("\n" + "=" * 90)
    print(f"{'Policy Architecture':<36} | {'Power (mW)':<10} | {'Energy Sav %':<12} | {'FNR %':<10} | {'Switches':<8} | {'Battery Life':<12}")
    print("-" * 90)
    for p_name, b in benchmarks.items():
        print(
            f"{p_name:<36} | "
            f"{b['avg_power_mw']:<10.2f} | "
            f"{b['energy_savings_pct']:<12.1f} | "
            f"{b['fnr_pct']:<10.2f} | "
            f"{b['switches']:<8d} | "
            f"{b['battery_life_years']:<5.2f} years"
        )
    print("=" * 90)

    # Generate publication-grade HTML report
    html_path = generate_html_report(
        timeline_data=timeline_records,
        benchmark_metrics=benchmarks,
        mcu_audit=mcu_audit,
        output_path=OUTPUT_DIR / "rl_tinyml_demo_report.html",
    )

    elapsed = time.time() - start_total_time
    print(f"\n[DONE] Pipeline completed successfully in {elapsed:.1f} seconds!")
    print(f"Interactive Presentation Report: file://{html_path.resolve()}")
    return benchmarks, mcu_audit, html_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-Driven Adaptive Sensing Demo Pipeline")
    parser.add_argument("--windows", type=int, default=3000, help="Total stream windows (default: 3000 = 12.5 mins)")
    parser.add_argument("--episodes", type=int, default=20, help="RL training episodes (default: 20)")
    args = parser.parse_args()

    run_pipeline(total_windows=args.windows, n_rl_episodes=args.episodes)
