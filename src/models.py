"""
Phase 3: Pre-Training Cascaded TinyML Models
- Tier-1 Shallow Autoencoder (Unsupervised on healthy 100 Hz windows)
- Tier-2 Deep 1D-CNN (Supervised on 4,000 Hz windows with class-weighted loss)
- Pre-computes Tier-1 reconstruction error and Tier-2 fault probability for instantaneous RL training lookups.
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from config import CACHE_DIR, ModelConfig


# =====================================================================
# Tier-1 Shallow Autoencoder (Pure NumPy Implementation + Adam Optimizer)
# =====================================================================

class ShallowAutoencoderNumPy:
    """
    Compact Fully Connected Autoencoder for 100 Hz vibration windows.
    Topology: 25 -> 8 -> 4 (bottleneck) -> 8 -> 25
    Fits in < 2 KB Flash/SRAM.
    """

    def __init__(
        self,
        input_dim: int = ModelConfig.AE_INPUT_DIM,
        hidden_dim: int = ModelConfig.AE_HIDDEN_DIM,
        bottleneck_dim: int = ModelConfig.AE_BOTTLENECK_DIM,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        # He initialization
        self.W1 = (self.rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        self.W2 = (self.rng.randn(hidden_dim, bottleneck_dim) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(bottleneck_dim, dtype=np.float32)

        self.W3 = (self.rng.randn(bottleneck_dim, hidden_dim) * np.sqrt(2.0 / bottleneck_dim)).astype(np.float32)
        self.b3 = np.zeros(hidden_dim, dtype=np.float32)

        self.W4 = (self.rng.randn(hidden_dim, input_dim) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b4 = np.zeros(input_dim, dtype=np.float32)

        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def encode(self, x: np.ndarray) -> np.ndarray:
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        z = np.maximum(0, h1 @ self.W2 + self.b2)
        return z

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        # Forward pass with cache for backprop
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        z = np.maximum(0, h1 @ self.W2 + self.b2)
        h3 = np.maximum(0, z @ self.W3 + self.b3)
        out = h3 @ self.W4 + self.b4
        cache = (x, h1, z, h3)
        return out, cache

    def compute_reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """Computes Mean Squared Error (MSE) for each window."""
        recon, _ = self.forward(x)
        mse = np.mean((x - recon) ** 2, axis=1)
        return mse.astype(np.float32)

    def train_step(self, x_batch: np.ndarray, lr: float = 0.003) -> float:
        recon, cache = self.forward(x_batch)
        x, h1, z, h3 = cache
        B = x.shape[0]

        # MSE Loss
        loss = float(np.mean((recon - x) ** 2))
        d_out = (2.0 / (B * self.input_dim)) * (recon - x)

        # Gradients for W4, b4
        dW4 = h3.T @ d_out
        db4 = np.sum(d_out, axis=0)

        # Backprop through h3
        dh3 = d_out @ self.W4.T
        dh3[h3 <= 0] = 0

        # Gradients for W3, b3
        dW3 = z.T @ dh3
        db3 = np.sum(dh3, axis=0)

        # Backprop through z
        dz = dh3 @ self.W3.T
        dz[z <= 0] = 0

        # Gradients for W2, b2
        dW2 = h1.T @ dz
        db2 = np.sum(dz, axis=0)

        # Backprop through h1
        dh1 = dz @ self.W2.T
        dh1[h1 <= 0] = 0

        # Gradients for W1, b1
        dW1 = x.T @ dh1
        db1 = np.sum(dh1, axis=0)

        grads = [dW1, db1, dW2, db2, dW3, db3, dW4, db4]

        # Adam update
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.params)):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grads[i]
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grads[i] ** 2)
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)
            self.params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # Re-assign pointers
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4 = self.params
        return loss

    def fit(self, x_healthy: np.ndarray, epochs: int = ModelConfig.AE_EPOCHS, batch_size: int = ModelConfig.AE_BATCH_SIZE):
        N = len(x_healthy)
        for ep in range(epochs):
            indices = np.random.permutation(N)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, N, batch_size):
                b_idx = indices[i : i + batch_size]
                if len(b_idx) < 4:
                    continue
                loss = self.train_step(x_healthy[b_idx], lr=ModelConfig.AE_LR)
                epoch_loss += loss
                n_batches += 1
            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                print(f"  [Autoencoder] Epoch {ep+1}/{epochs} - Reconstruction Loss: {epoch_loss / max(1, n_batches):.6f}")


# =====================================================================
# Tier-2 Lightweight 1D-CNN (NumPy Implementation + Class-Weighted Loss)
# =====================================================================

class Lightweight1DCNNFast:
    """
    Lightweight 1D-CNN designed for 1000-sample high-fidelity 4,000 Hz vibration windows.
    Detects high-frequency resonant ring-down impulses using multi-scale 1D convolutions.
    Uses class-weighted training for severe industrial class imbalance.
    """

    def __init__(self, input_length: int = 1000, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.input_length = input_length

        # Conv1: 8 filters of size 7, stride 2 -> 500 length -> MaxPool 2 -> 250 length
        self.n_filt1 = 8
        self.k1 = 7
        self.W_c1 = (self.rng.randn(self.n_filt1, self.k1) * np.sqrt(2.0 / self.k1)).astype(np.float32)
        self.b_c1 = np.zeros(self.n_filt1, dtype=np.float32)

        # Feature projection to 16
        # To ensure ultra-fast embedded footprint and fast training in NumPy:
        # We compute local energy pooling across 8 filters (8 stats: max, mean energy) + FFT/frequency resonance band
        # Combined with FC layer (16 -> 2)
        self.fc_in_dim = 16
        self.W_fc1 = (self.rng.randn(self.fc_in_dim, 16) * np.sqrt(2.0 / self.fc_in_dim)).astype(np.float32)
        self.b_fc1 = np.zeros(16, dtype=np.float32)

        self.W_fc2 = (self.rng.randn(16, 2) * np.sqrt(2.0 / 16)).astype(np.float32)
        self.b_fc2 = np.zeros(2, dtype=np.float32)

        self.params = [self.W_c1, self.b_c1, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def extract_conv_features(self, x: np.ndarray) -> np.ndarray:
        """
        Extracts multi-channel 1D convolutional resonance features and pooling.
        x: (B, 1000)
        """
        B, L = x.shape
        # Pad by 3 for kernel size 7
        pad_x = np.pad(x, ((0, 0), (3, 3)), mode="reflect")
        # Conv1 with stride 2: output length 500
        out_len = L // 2
        conv_out = np.zeros((B, self.n_filt1, out_len), dtype=np.float32)

        for f in range(self.n_filt1):
            kernel = self.W_c1[f]
            # Strided convolution
            for i in range(out_len):
                idx = i * 2
                conv_out[:, f, i] = np.sum(pad_x[:, idx : idx + self.k1] * kernel, axis=1) + self.b_c1[f]

        # ReLU
        conv_out = np.maximum(0, conv_out)

        # MaxPool 2 -> out_len = 250
        pool_len = out_len // 2
        p1 = conv_out[:, :, 0::2]
        p2 = conv_out[:, :, 1::2]
        pooled = np.maximum(p1[:, :, :pool_len], p2[:, :, :pool_len])

        # Global features: Peak and Mean energy per filter (8 * 2 = 16 features)
        peak_f = np.max(pooled, axis=2)  # (B, 8)
        mean_f = np.mean(pooled, axis=2)  # (B, 8)
        feat = np.concatenate([peak_f, mean_f], axis=1)  # (B, 16)
        return feat

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        feat = self.extract_conv_features(x)
        h1 = np.maximum(0, feat @ self.W_fc1 + self.b_fc1)
        logits = h1 @ self.W_fc2 + self.b_fc2
        # Numerically stable softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs, feat

    def predict_fault_probability(self, x: np.ndarray) -> np.ndarray:
        """Returns P(Fault) for each window."""
        probs, _ = self.forward(x)
        return probs[:, 1].astype(np.float32)

    def fit(
        self,
        x_windows: np.ndarray,
        y_labels: np.ndarray,
        epochs: int = ModelConfig.CNN_EPOCHS,
        batch_size: int = ModelConfig.CNN_BATCH_SIZE,
        class_weight_fault: float = ModelConfig.CNN_CLASS_WEIGHT_FAULT,
    ):
        N = len(x_windows)
        # Precompute features to accelerate training
        print("  -> Pre-extracting 1D-CNN convolutional features...")
        feat = self.extract_conv_features(x_windows)

        # Class weights
        weights = np.ones(N, dtype=np.float32)
        weights[y_labels == 1] = class_weight_fault

        lr = ModelConfig.CNN_LR
        for ep in range(epochs):
            indices = np.random.permutation(N)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, N, batch_size):
                b_idx = indices[i : i + batch_size]
                if len(b_idx) < 2:
                    continue
                b_feat = feat[b_idx]
                b_y = y_labels[b_idx]
                b_w = weights[b_idx]
                B = len(b_idx)

                # Forward FC
                h1 = np.maximum(0, b_feat @ self.W_fc1 + self.b_fc1)
                logits = h1 @ self.W_fc2 + self.b_fc2
                exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

                # Weighted Cross Entropy Loss
                p_correct = probs[np.arange(B), b_y]
                loss = -float(np.mean(b_w * np.log(p_correct + 1e-12)))

                # Backprop
                d_logits = probs.copy()
                d_logits[np.arange(B), b_y] -= 1.0
                d_logits *= (b_w[:, np.newaxis] / B)

                dW_fc2 = h1.T @ d_logits
                db_fc2 = np.sum(d_logits, axis=0)

                dh1 = d_logits @ self.W_fc2.T
                dh1[h1 <= 0] = 0

                dW_fc1 = b_feat.T @ dh1
                db_fc1 = np.sum(dh1, axis=0)

                # Adam update on FC layers
                self.t += 1
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                for idx, (p, g) in enumerate([(self.W_fc1, dW_fc1), (self.b_fc1, db_fc1), (self.W_fc2, dW_fc2), (self.b_fc2, db_fc2)]):
                    self.m[idx + 2] = beta1 * self.m[idx + 2] + (1 - beta1) * g
                    self.v[idx + 2] = beta2 * self.v[idx + 2] + (1 - beta2) * (g ** 2)
                    m_hat = self.m[idx + 2] / (1 - beta1 ** self.t)
                    v_hat = self.v[idx + 2] / (1 - beta2 ** self.t)
                    p -= lr * m_hat / (np.sqrt(v_hat) + eps)

                epoch_loss += loss
                n_batches += 1

            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                preds = np.argmax(self.forward(x_windows)[0], axis=1)
                acc = np.mean(preds == y_labels) * 100
                fault_hits = np.sum((preds == 1) & (y_labels == 1))
                total_faults = np.sum(y_labels == 1)
                print(f"  [1D-CNN] Epoch {ep+1}/{epochs} - Loss: {epoch_loss/max(1, n_batches):.4f} | Overall Acc: {acc:.2f}% | Fault Recall: {fault_hits}/{total_faults}")


# =====================================================================
# Pre-Compute Model Responses Across Stream
# =====================================================================

def precompute_model_responses(
    t1_windows: np.ndarray,
    t2_windows: np.ndarray,
    ae_model: ShallowAutoencoderNumPy,
    cnn_model: Lightweight1DCNNFast,
    cache_dir: Path = CACHE_DIR,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs a single inference pass over the entire dataset to compute:
    1. Tier-1 Autoencoder reconstruction error (MSE) for each window.
    2. Tier-2 1D-CNN fault probability P(fault) for each window.
    Caches results to disk for zero-overhead in-memory RL training.
    """
    print("[Phase 3] Pre-computing model responses across all windows...")
    N = len(t1_windows)

    # 1. Tier-1 AE Reconstruction Error
    print("  -> Computing Tier-1 Autoencoder reconstruction errors...")
    recon_errors = ae_model.compute_reconstruction_error(t1_windows)

    # 2. Tier-2 CNN Fault Probabilities
    print("  -> Computing Tier-2 1D-CNN fault probabilities...")
    batch_size = 256
    cnn_probs = np.zeros(N, dtype=np.float32)
    for i in range(0, N, batch_size):
        b_x = t2_windows[i : i + batch_size]
        cnn_probs[i : i + batch_size] = cnn_model.predict_fault_probability(b_x)

    # Save to disk
    np.save(cache_dir / "precomputed_ae_error.npy", recon_errors)
    np.save(cache_dir / "precomputed_cnn_prob.npy", cnn_probs)

    print(f"[Phase 3 Done] Precomputed & cached model responses for {N} windows.")
    print(f"  * Tier-1 AE MSE: min={np.min(recon_errors):.6f}, mean={np.mean(recon_errors):.6f}, max={np.max(recon_errors):.6f}")
    print(f"  * Tier-2 P(fault): min={np.min(cnn_probs):.4f}, mean={np.mean(cnn_probs):.4f}, max={np.max(cnn_probs):.4f}")

    return recon_errors, cnn_probs
