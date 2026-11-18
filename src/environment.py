"""
Phase 4: Hardware-Coupled Constrained Markov Decision Process (CMDP) Environment
Couples ADXL355 MEMS accelerometer dynamics, STM32L4 MCU power states,
filter group delays, dual-rate cascaded TinyML models, and dual-energy physics.
"""

from typing import Dict, Optional, Tuple
import numpy as np

from config import HardwareConfig, RLConfig


class StateNormalizer:
    """Standardizes continuous state features (TKEO moments + AE reconstruction error)."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, features: np.ndarray):
        """features: (N, 5) [4 TKEO moments + 1 AE reconstruction error]"""
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0) + self.eps

    def transform(self, feat: np.ndarray) -> np.ndarray:
        if self.mean is None:
            return feat
        return (feat - self.mean) / self.std


class HardwareCoupledCMDP:
    """
    Constrained Markov Decision Process environment for adaptive edge sensing.
    - State: [TKEO_Mean, TKEO_Var, TKEO_Kurt, TKEO_Crest, AE_Recon_Error, a_{t-1}] (6-dim)
    - Action: {0: Tier 1 (100 Hz), 1: Tier 2 (4000 Hz)}
    - Reward: Normalized energy savings minus hysteresis thrashing penalty
    - Safety Cost: 1.0 for False Negative (missed fault when a_t=0), 0.0 otherwise
    """

    def __init__(
        self,
        tkeo_features: np.ndarray,
        ae_errors: np.ndarray,
        cnn_probs: np.ndarray,
        labels: np.ndarray,
        normalizer: Optional[StateNormalizer] = None,
        max_steps: Optional[int] = None,
        seed: int = 42,
    ):
        self.N = len(labels)
        self.max_steps = max_steps or self.N
        self.rng = np.random.RandomState(seed)

        self.tkeo = tkeo_features.astype(np.float32)  # (N, 4)
        self.ae_errors = ae_errors.astype(np.float32)  # (N,)
        self.cnn_probs = cnn_probs.astype(np.float32)  # (N,)
        self.labels = labels.astype(np.int32)          # (N,)

        # Build feature matrix (N, 5) for normalization
        raw_continuous = np.column_stack([self.tkeo, self.ae_errors])
        if normalizer is None:
            self.normalizer = StateNormalizer()
            # Fit on normal windows
            normal_mask = (self.labels == 0)
            if np.sum(normal_mask) > 10:
                self.normalizer.fit(raw_continuous[normal_mask])
            else:
                self.normalizer.fit(raw_continuous)
        else:
            self.normalizer = normalizer

        self.norm_continuous = self.normalizer.transform(raw_continuous)

        # State tracking
        self.current_idx = 0
        self.prev_action = 0
        self.step_count = 0

    def _get_state(self, idx: int, prev_act: int) -> np.ndarray:
        cont_feat = self.norm_continuous[idx]
        # Append prev_action as 6th dimension: shape (6,)
        return np.append(cont_feat, float(prev_act)).astype(np.float32)

    def reset(self, start_idx: int = 0) -> np.ndarray:
        self.current_idx = start_idx % self.N
        self.prev_action = 0
        self.step_count = 0
        return self._get_state(self.current_idx, self.prev_action)

    def step(self, action: int) -> Tuple[np.ndarray, float, float, bool, Dict]:
        """
        Executes one step in the CMDP.
        Returns:
            next_state: (6,) float32
            reward: primary energy reward (scalar)
            cost: safety constraint cost (1.0 if False Negative, else 0.0)
            done: boolean flag
            info: dictionary of diagnostic metrics
        """
        assert action in [0, 1], f"Invalid action {action}"

        label = int(self.labels[self.current_idx])
        cnn_p = float(self.cnn_probs[self.current_idx])
        ae_e = float(self.ae_errors[self.current_idx])

        # 1. Dual-Energy Physics Model
        if action == 0:
            e_base = HardwareConfig.ENERGY_TIER1_MJ
        else:
            e_base = HardwareConfig.ENERGY_TIER2_MJ

        # Transition penalty (filter group delay settling & clock wake-up)
        e_trans = 0.0
        switched = (action != self.prev_action)
        if switched:
            if self.prev_action == 0 and action == 1:
                e_trans = HardwareConfig.TRANSITION_COST_0_TO_1_MJ
            else:
                e_trans = HardwareConfig.TRANSITION_COST_1_TO_0_MJ

        e_total = e_base + e_trans

        # 2. Reward Formulation: Normalized Energy Savings minus Hysteresis Thrashing Penalty
        e_max = HardwareConfig.ENERGY_MAX_BASELINE_MJ
        energy_savings_ratio = max(0.0, (e_max - e_total) / e_max)
        hysteresis_penalty = RLConfig.HYSTERESIS_PENALTY if switched else 0.0

        reward = float(RLConfig.WEIGHT_ENERGY * energy_savings_ratio - hysteresis_penalty)

        # 3. Safety Cost (False Negative: fault present but agent remained in Tier-1 low power)
        is_false_negative = (label == 1 and action == 0)
        is_true_positive = (label == 1 and action == 1)
        is_false_alarm = (label == 0 and action == 1)
        is_true_negative = (label == 0 and action == 0)

        cost = 1.0 if is_false_negative else 0.0

        # Diagnostics info
        info = {
            "window_idx": self.current_idx,
            "action": action,
            "prev_action": self.prev_action,
            "label": label,
            "energy_mj": e_total,
            "switched": switched,
            "false_negative": is_false_negative,
            "true_positive": is_true_positive,
            "false_alarm": is_false_alarm,
            "true_negative": is_true_negative,
            "cnn_prob": cnn_p,
            "ae_error": ae_e,
        }

        # Advance state
        self.prev_action = action
        self.current_idx = (self.current_idx + 1) % self.N
        self.step_count += 1
        done = (self.step_count >= self.max_steps)

        next_state = self._get_state(self.current_idx, self.prev_action)
        return next_state, reward, cost, done, info
