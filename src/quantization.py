"""
Phase 6: Edge Quantization & Microcontroller Memory Verification
Quantizes Double DQN policy and Tier-1 Autoencoder from FP32 to INT8 precision.
Audits Flash memory, peak SRAM activation buffers, and validates against ARM Cortex-M4 budget.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from config import HardwareConfig
from src.dqn_agent import CompactQNetwork
from src.models import ShallowAutoencoderNumPy


@dataclass
class QuantizedTensor:
    q_data: np.ndarray  # int8 array
    scale: float        # float32 scale
    zero_point: int     # int8 zero point (0 for symmetric)


def quantize_symmetric_int8(tensor: np.ndarray) -> QuantizedTensor:
    """
    Symmetric per-tensor INT8 post-training quantization:
    q = clip(round(x / scale), -127, 127)
    scale = max(|x|) / 127
    """
    max_abs = float(np.max(np.abs(tensor)))
    scale = max(max_abs / 127.0, 1e-9)
    q = np.clip(np.round(tensor / scale), -127, 127).astype(np.int8)
    return QuantizedTensor(q_data=q, scale=scale, zero_point=0)


def dequantize_symmetric_int8(q_tensor: QuantizedTensor) -> np.ndarray:
    """Dequantizes INT8 tensor back to float32 for numerical verification."""
    return (q_tensor.q_data.astype(np.float32) * q_tensor.scale).astype(np.float32)


class QuantizedAutoencoder:
    """INT8 Quantized Tier-1 Shallow Autoencoder."""

    def __init__(self, fp32_ae: ShallowAutoencoderNumPy):
        self.q_W1 = quantize_symmetric_int8(fp32_ae.W1)
        self.q_b1 = quantize_symmetric_int8(fp32_ae.b1)
        self.q_W2 = quantize_symmetric_int8(fp32_ae.W2)
        self.q_b2 = quantize_symmetric_int8(fp32_ae.b2)
        self.q_W3 = quantize_symmetric_int8(fp32_ae.W3)
        self.q_b3 = quantize_symmetric_int8(fp32_ae.b3)
        self.q_W4 = quantize_symmetric_int8(fp32_ae.W4)
        self.q_b4 = quantize_symmetric_int8(fp32_ae.b4)

        self.tensors = [
            self.q_W1, self.q_b1,
            self.q_W2, self.q_b2,
            self.q_W3, self.q_b3,
            self.q_W4, self.q_b4,
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        W1 = dequantize_symmetric_int8(self.q_W1)
        b1 = dequantize_symmetric_int8(self.q_b1)
        W2 = dequantize_symmetric_int8(self.q_W2)
        b2 = dequantize_symmetric_int8(self.q_b2)
        W3 = dequantize_symmetric_int8(self.q_W3)
        b3 = dequantize_symmetric_int8(self.q_b3)
        W4 = dequantize_symmetric_int8(self.q_W4)
        b4 = dequantize_symmetric_int8(self.q_b4)

        h1 = np.maximum(0, x @ W1 + b1)
        z = np.maximum(0, h1 @ W2 + b2)
        h3 = np.maximum(0, z @ W3 + b3)
        out = h3 @ W4 + b4
        return out

    def get_flash_bytes(self) -> int:
        int8_bytes = sum(t.q_data.nbytes for t in self.tensors)
        scale_bytes = len(self.tensors) * 4  # 4 bytes per float32 scale
        return int8_bytes + scale_bytes

    def get_peak_sram_bytes(self) -> int:
        # Ping-pong activation buffer sizing:
        # Layer sizes: input 25, h1 8, z 4, h3 8, out 25 (bytes in int8)
        return max(25 + 8, 8 + 4, 4 + 8, 8 + 25)


class QuantizedDQNPolicy:
    """INT8 Quantized Double DQN Policy Network."""

    def __init__(self, fp32_dqn: CompactQNetwork):
        self.q_W1 = quantize_symmetric_int8(fp32_dqn.W1)
        self.q_b1 = quantize_symmetric_int8(fp32_dqn.b1)
        self.q_W2 = quantize_symmetric_int8(fp32_dqn.W2)
        self.q_b2 = quantize_symmetric_int8(fp32_dqn.b2)
        self.q_W3 = quantize_symmetric_int8(fp32_dqn.W3)
        self.q_b3 = quantize_symmetric_int8(fp32_dqn.b3)

        self.tensors = [
            self.q_W1, self.q_b1,
            self.q_W2, self.q_b2,
            self.q_W3, self.q_b3,
        ]

    def predict(self, s: np.ndarray) -> np.ndarray:
        W1 = dequantize_symmetric_int8(self.q_W1)
        b1 = dequantize_symmetric_int8(self.q_b1)
        W2 = dequantize_symmetric_int8(self.q_W2)
        b2 = dequantize_symmetric_int8(self.q_b2)
        W3 = dequantize_symmetric_int8(self.q_W3)
        b3 = dequantize_symmetric_int8(self.q_b3)

        if s.ndim == 1:
            s_batch = s[np.newaxis, :]
        else:
            s_batch = s

        h1 = np.maximum(0, s_batch @ W1 + b1)
        h2 = np.maximum(0, h1 @ W2 + b2)
        q = h2 @ W3 + b3
        return q

    def get_flash_bytes(self) -> int:
        int8_bytes = sum(t.q_data.nbytes for t in self.tensors)
        scale_bytes = len(self.tensors) * 4
        return int8_bytes + scale_bytes

    def get_peak_sram_bytes(self) -> int:
        # Layer sizes: input 6, h1 32, h2 32, out 2 (bytes in int8)
        return max(6 + 32, 32 + 32, 32 + 2)


def audit_microcontroller_memory(
    fp32_ae: ShallowAutoencoderNumPy,
    fp32_dqn: CompactQNetwork,
    quant_ae: QuantizedAutoencoder,
    quant_dqn: QuantizedDQNPolicy,
    test_states: np.ndarray,
) -> Dict:
    """
    Performs comprehensive edge memory and accuracy audit:
    - Measures FP32 vs INT8 Flash memory consumption
    - Measures peak SRAM activation requirements
    - Validates policy decision fidelity (agreement % between FP32 and INT8)
    - Verifies ARM Cortex-M4 resource limits (< 15 KB Flash/SRAM, > 90% SRAM free)
    """
    print("[Phase 6] Auditing Microcontroller Memory and INT8 Policy Fidelity...")

    # 1. Flash memory sizing
    ae_fp32_flash = sum(p.nbytes for p in fp32_ae.params)
    ae_int8_flash = quant_ae.get_flash_bytes()

    dqn_fp32_flash = sum(p.nbytes for p in fp32_dqn.params)
    dqn_int8_flash = quant_dqn.get_flash_bytes()

    total_int8_flash = ae_int8_flash + dqn_int8_flash
    total_fp32_flash = ae_fp32_flash + dqn_fp32_flash

    # 2. SRAM activation memory sizing
    ae_peak_sram = quant_ae.get_peak_sram_bytes()
    dqn_peak_sram = quant_dqn.get_peak_sram_bytes()
    dma_buffer_sram = 50  # 25 samples * 2 bytes (int16 ADC)
    state_vector_sram = 24  # 6 features * 4 bytes float32
    # Cortex-M4 executes sequentially, so SRAM is max(pipeline stages) + static buffers
    total_pipeline_sram = max(ae_peak_sram, dqn_peak_sram) + dma_buffer_sram + state_vector_sram

    # 3. Policy Fidelity Check
    q_fp32 = fp32_dqn.predict(test_states)
    q_int8 = quant_dqn.predict(test_states)
    actions_fp32 = np.argmax(q_fp32, axis=1)
    actions_int8 = np.argmax(q_int8, axis=1)
    policy_agreement_pct = float(np.mean(actions_fp32 == actions_int8) * 100.0)

    # 4. Cortex-M4 Budget Verification
    mcu_sram_total_kb = HardwareConfig.MCU_TOTAL_SRAM_KB  # 256 KB
    mcu_flash_total_kb = HardwareConfig.MCU_TOTAL_FLASH_KB  # 1024 KB
    sram_used_kb = total_pipeline_sram / 1024.0
    flash_used_kb = total_int8_flash / 1024.0

    sram_free_pct = (1.0 - (sram_used_kb / mcu_sram_total_kb)) * 100.0
    flash_free_pct = (1.0 - (flash_used_kb / mcu_flash_total_kb)) * 100.0

    fits_in_target_budget = (total_int8_flash / 1024.0 < HardwareConfig.MAX_FLASH_BUDGET_KB) and (sram_used_kb < HardwareConfig.MAX_SRAM_BUDGET_KB)

    audit_results = {
        "autoencoder": {
            "params_count": sum(p.size for p in fp32_ae.params),
            "fp32_flash_bytes": ae_fp32_flash,
            "int8_flash_bytes": ae_int8_flash,
            "peak_sram_bytes": ae_peak_sram,
            "compression_ratio": float(ae_fp32_flash / ae_int8_flash),
        },
        "dqn_policy": {
            "params_count": sum(p.size for p in fp32_dqn.params),
            "fp32_flash_bytes": dqn_fp32_flash,
            "int8_flash_bytes": dqn_int8_flash,
            "peak_sram_bytes": dqn_peak_sram,
            "compression_ratio": float(dqn_fp32_flash / dqn_int8_flash),
        },
        "system": {
            "total_int8_flash_bytes": total_int8_flash,
            "total_int8_flash_kb": flash_used_kb,
            "total_runtime_sram_bytes": total_pipeline_sram,
            "total_runtime_sram_kb": sram_used_kb,
            "mcu_total_sram_kb": mcu_sram_total_kb,
            "mcu_sram_free_pct": sram_free_pct,
            "mcu_flash_free_pct": flash_free_pct,
            "fits_15kb_budget": fits_in_target_budget,
            "policy_int8_agreement_pct": policy_agreement_pct,
        },
    }

    print("[Phase 6 Done] MCU Memory Audit Summary:")
    print(f"  * Tier-1 Autoencoder: Flash = {ae_int8_flash} B (INT8) vs {ae_fp32_flash} B (FP32), Peak SRAM = {ae_peak_sram} B")
    print(f"  * Double DQN Policy:  Flash = {dqn_int8_flash} B (INT8) vs {dqn_fp32_flash} B (FP32), Peak SRAM = {dqn_peak_sram} B")
    print(f"  * Combined Pipeline Flash: {flash_used_kb:.2f} KB (< 15 KB Budget: {fits_in_target_budget})")
    print(f"  * Peak Runtime SRAM: {total_pipeline_sram} Bytes ({sram_used_kb:.3f} KB, leaving {sram_free_pct:.2f}% of 256 KB free!)")
    print(f"  * Policy Decision Agreement (FP32 vs INT8): {policy_agreement_pct:.2f}%")

    return audit_results
