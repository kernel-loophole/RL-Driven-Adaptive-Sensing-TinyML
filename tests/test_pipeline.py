"""
Automated Unit and Integration Tests for RL-Driven Adaptive Sensing TinyML Demo
Tests Phase 1 through Phase 6 components:
- Bearing vibration generation & industrial stream construction
- Anti-aliasing digital filters & downsampling factors
- 3-sample TKEO calculation & moment extraction
- Shallow Autoencoder and Lightweight 1D-CNN architectures
- Hardware-Coupled CMDP environment & dual-energy physics
- INT8 quantization & microcontroller memory budgets
"""

import unittest
from pathlib import Path
import numpy as np

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import HardwareConfig, ModelConfig, RLConfig, SignalConfig
from src.dataset import PhysicalBearingVibrationGenerator
from src.preprocessing import (
    apply_sos_filter,
    compute_tkeo,
    design_butterworth_4th_sos,
    extract_tkeo_moments,
)
from src.models import Lightweight1DCNNFast, ShallowAutoencoderNumPy
from src.environment import HardwareCoupledCMDP
from src.dqn_agent import CompactQNetwork, PrimalDualDoubleDQNAgent
from src.quantization import (
    QuantizedAutoencoder,
    QuantizedDQNPolicy,
    audit_microcontroller_memory,
    dequantize_symmetric_int8,
    quantize_symmetric_int8,
)


class TestRLAdaptiveSensingTinyML(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.generator = PhysicalBearingVibrationGenerator(seed=42)

    def test_01_bearing_physics_generator(self):
        """Test healthy vs fault window properties."""
        w_healthy = self.generator.generate_healthy_window()
        w_fault = self.generator.generate_fault_window(fault_type="outer_race", severity=1.0)

        self.assertEqual(len(w_healthy), SignalConfig.RAW_WINDOW_SAMPLES)
        self.assertEqual(len(w_fault), SignalConfig.RAW_WINDOW_SAMPLES)

        # Fault window should exhibit sharp impulsive impacts -> higher crest factor
        cf_h = np.max(np.abs(w_healthy)) / (np.std(w_healthy) + 1e-9)
        cf_f = np.max(np.abs(w_fault)) / (np.std(w_fault) + 1e-9)
        self.assertGreater(cf_f, cf_h)

    def test_02_anti_aliasing_filter_and_downsampling(self):
        """Test 4th order Butterworth anti-aliasing filter and decimation."""
        sos45 = design_butterworth_4th_sos(fc=45.0, fs=12000.0)
        self.assertEqual(sos45.shape, (2, 6))

        # Check attenuation of 600 Hz component
        t = np.arange(3000) / 12000.0
        sig_hi = np.sin(2 * np.pi * 600.0 * t)
        filt_hi = apply_sos_filter(sos45, sig_hi)
        atten_db = 20 * np.log10(np.std(sig_hi) / (np.std(filt_hi) + 1e-12))
        self.assertGreater(atten_db, 35.0, f"Expected >35 dB attenuation, got {atten_db:.1f} dB")

        # Downsample check
        t1_samples = filt_hi[::SignalConfig.TIER1_DOWNSAMPLE_FACTOR]
        self.assertEqual(len(t1_samples), SignalConfig.TIER1_WINDOW_SAMPLES)

    def test_03_tkeo_moments(self):
        """Test 3-sample Teager-Kaiser Energy Operator and moment extraction."""
        x = np.array([[1.0, 3.0, 2.0, 5.0, 1.0]], dtype=np.float32)
        psi = compute_tkeo(x)
        # psi[1] = x[1]^2 - x[0]*x[2] = 9 - 1*2 = 7
        self.assertAlmostEqual(psi[0, 1], 7.0, places=4)

        moments = extract_tkeo_moments(psi)
        self.assertEqual(moments.shape, (1, 4))
        # Mean, Var, Kurtosis, Crest Factor must all be finite and positive
        self.assertTrue(np.all(np.isfinite(moments)))
        self.assertGreater(moments[0, 3], 0.0)

    def test_04_tinyml_model_topologies(self):
        """Test Tier-1 Autoencoder and Tier-2 1D-CNN dimensions and forward pass."""
        ae = ShallowAutoencoderNumPy()
        x_ae = np.random.randn(8, 25).astype(np.float32)
        recon, _ = ae.forward(x_ae)
        self.assertEqual(recon.shape, (8, 25))
        mse = ae.compute_reconstruction_error(x_ae)
        self.assertEqual(len(mse), 8)

        cnn = Lightweight1DCNNFast()
        x_cnn = np.random.randn(4, 1000).astype(np.float32)
        p_fault = cnn.predict_fault_probability(x_cnn)
        self.assertEqual(len(p_fault), 4)
        self.assertTrue(np.all((p_fault >= 0.0) & (p_fault <= 1.0)))

    def test_05_cmdp_environment_physics(self):
        """Test CMDP state, actions, energy calculations and safety costs."""
        N = 20
        tkeo = np.ones((N, 4), dtype=np.float32)
        ae_err = np.full(N, 0.001, dtype=np.float32)
        cnn_p = np.full(N, 0.1, dtype=np.float32)
        labels = np.zeros(N, dtype=np.int32)
        labels[5] = 1  # Injected fault at index 5

        env = HardwareCoupledCMDP(tkeo, ae_err, cnn_p, labels)
        s0 = env.reset(start_idx=0)
        self.assertEqual(len(s0), RLConfig.STATE_DIM)

        # Step 0: action 0 (Tier-1)
        s1, r0, c0, _, info0 = env.step(0)
        self.assertAlmostEqual(info0["energy_mj"], HardwareConfig.ENERGY_TIER1_MJ, places=3)
        self.assertEqual(c0, 0.0)

        # Step to fault window (index 5) with action 0 -> should trigger FALSE NEGATIVE (cost=1.0)
        env.reset(start_idx=5)
        _, _, c_fn, _, info_fn = env.step(0)
        self.assertEqual(c_fn, 1.0)
        self.assertTrue(info_fn["false_negative"])

        # Step to fault window (index 5) with action 1 -> ESCALATED, cost=0.0 (Zero-FN satisfied)
        env.reset(start_idx=5)
        _, _, c_tp, _, info_tp = env.step(1)
        self.assertEqual(c_tp, 0.0)
        self.assertTrue(info_tp["true_positive"])

    def test_06_int8_quantization_and_mcu_memory(self):
        """Test INT8 post-training quantization and Cortex-M4 memory audit."""
        ae = ShallowAutoencoderNumPy()
        dqn = CompactQNetwork()
        q_ae = QuantizedAutoencoder(ae)
        q_dqn = QuantizedDQNPolicy(dqn)

        # Verify Flash memory is well within 15 KB budget
        total_flash_kb = (q_ae.get_flash_bytes() + q_dqn.get_flash_bytes()) / 1024.0
        self.assertLess(total_flash_kb, HardwareConfig.MAX_FLASH_BUDGET_KB)

        # Verify Peak SRAM is under 15 KB (actually under 1 KB!)
        total_sram_bytes = max(q_ae.get_peak_sram_bytes(), q_dqn.get_peak_sram_bytes()) + 74
        self.assertLess(total_sram_bytes, 1024)

        # Symmetric quantization test
        test_t = np.array([-1.5, 0.0, 0.75, 1.5], dtype=np.float32)
        qt = quantize_symmetric_int8(test_t)
        deq = dequantize_symmetric_int8(qt)
        np.testing.assert_allclose(test_t, deq, atol=0.03)


if __name__ == "__main__":
    unittest.main()
