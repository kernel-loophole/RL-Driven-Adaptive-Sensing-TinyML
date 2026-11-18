"""
Phase 5: Primal-Dual Double Deep Q-Network (DDQN) Agent for Constrained MDP
Implements compact 2-layer Q-Network (32 units/layer), Double DQN target decoupling,
replay buffer, and Dual Ascent on the Lagrange multiplier (lambda) to strictly enforce Zero-FN.
"""

from typing import List, Tuple
import numpy as np

from config import RLConfig


class CompactQNetwork:
    """
    Compact Two-Layer Multi-Layer Perceptron:
    6 -> Dense(32) + ReLU -> Dense(32) + ReLU -> Dense(2)
    Total parameters: 1,346 weights + biases (~5.4 KB FP32, ~1.35 KB INT8).
    """

    def __init__(self, state_dim: int = RLConfig.STATE_DIM, hidden_dim: int = RLConfig.HIDDEN_DIM, action_dim: int = RLConfig.ACTION_DIM, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # He initialization
        self.W1 = (self.rng.randn(state_dim, hidden_dim) * np.sqrt(2.0 / state_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        self.W2 = (self.rng.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)

        self.W3 = (self.rng.randn(hidden_dim, action_dim) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b3 = np.zeros(action_dim, dtype=np.float32)

        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def forward(self, s: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        # s: (B, 6) or (6,)
        if s.ndim == 1:
            s_batch = s[np.newaxis, :]
        else:
            s_batch = s

        z1 = s_batch @ self.W1 + self.b1
        h1 = np.maximum(0, z1)

        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0, z2)

        q_values = h2 @ self.W3 + self.b3
        cache = (s_batch, h1, h2)
        return q_values, cache

    def predict(self, s: np.ndarray) -> np.ndarray:
        q, _ = self.forward(s)
        return q

    def copy_weights_from(self, source: "CompactQNetwork"):
        for i in range(len(self.params)):
            self.params[i][:] = source.params[i].copy()

    def train_step(self, s: np.ndarray, a: np.ndarray, target: np.ndarray, lr: float = RLConfig.LR) -> float:
        """
        Huber / Smooth-L1 loss training step on batch.
        s: (B, 6), a: (B,), target: (B,)
        """
        B = len(s)
        q_vals, cache = self.forward(s)
        s_batch, h1, h2 = cache

        # Compute TD error
        q_selected = q_vals[np.arange(B), a]
        diff = q_selected - target  # (B,)

        # Smooth L1 / Huber gradient
        huber_delta = 1.0
        abs_diff = np.abs(diff)
        huber_mask = abs_diff <= huber_delta

        d_selected = np.where(huber_mask, diff, huber_delta * np.sign(diff))
        loss = float(np.mean(np.where(huber_mask, 0.5 * (diff**2), huber_delta * (abs_diff - 0.5 * huber_delta))))

        # Gradient for q_vals: (B, 2)
        d_q = np.zeros_like(q_vals)
        d_q[np.arange(B), a] = d_selected / B

        # Backpropagation
        dW3 = h2.T @ d_q
        db3 = np.sum(d_q, axis=0)

        dh2 = d_q @ self.W3.T
        dh2[h2 <= 0] = 0

        dW2 = h1.T @ dh2
        db2 = np.sum(dh2, axis=0)

        dh1 = dh2 @ self.W2.T
        dh1[h1 <= 0] = 0

        dW1 = s_batch.T @ dh1
        db1 = np.sum(dh1, axis=0)

        grads = [dW1, db1, dW2, db2, dW3, db3]

        # Adam optimizer update
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.params)):
            # Gradient clipping to prevent exploding gradients
            g = np.clip(grads[i], -5.0, 5.0)
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (g**2)
            m_hat = self.m[i] / (1 - beta1**self.t)
            v_hat = self.v[i] / (1 - beta2**self.t)
            self.params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = self.params
        return loss


class ReplayBuffer:
    """
    Stratified Experience Replay Buffer for Extreme Industrial Imbalance (<0.1% faults).
    Maintains separate pools for nominal transitions and rare fault events to guarantee
    that safety violations are never drowned out in uniform random batches.
    """

    def __init__(self, capacity: int = RLConfig.BUFFER_SIZE):
        self.capacity = capacity
        self.nominal_buffer: List[Tuple] = []
        self.fault_buffer: List[Tuple] = []
        self.nom_pos = 0
        self.flt_pos = 0

    def push(self, s: np.ndarray, a: int, r_base: float, cost: float, s_next: np.ndarray, done: bool):
        transition = (s, a, r_base, cost, s_next, done)
        if cost > 0.0 or a == 1:  # Escalated or fault event
            if len(self.fault_buffer) < (self.capacity // 4):
                self.fault_buffer.append(transition)
            else:
                self.fault_buffer[self.flt_pos] = transition
                self.flt_pos = (self.flt_pos + 1) % (self.capacity // 4)
        else:
            if len(self.nominal_buffer) < self.capacity:
                self.nominal_buffer.append(transition)
            else:
                self.nominal_buffer[self.nom_pos] = transition
                self.nom_pos = (self.nom_pos + 1) % self.capacity

    def sample(
        self, batch_size: int, current_lambda: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Stratified sampling: 8 fault transitions (if available) + remaining nominal
        batch = []
        n_fault_target = min(12, len(self.fault_buffer))
        if n_fault_target > 0:
            flt_idx = np.random.choice(len(self.fault_buffer), n_fault_target, replace=False)
            batch.extend([self.fault_buffer[i] for i in flt_idx])

        n_nom_needed = batch_size - len(batch)
        if len(self.nominal_buffer) > 0:
            nom_idx = np.random.choice(len(self.nominal_buffer), min(n_nom_needed, len(self.nominal_buffer)), replace=(n_nom_needed > len(self.nominal_buffer)))
            batch.extend([self.nominal_buffer[i] for i in nom_idx])

        s = np.array([t[0] for t in batch], dtype=np.float32)
        a = np.array([t[1] for t in batch], dtype=np.int32)
        # Dynamically compute Lagrangian augmented reward using current lambda!
        r_base = np.array([t[2] for t in batch], dtype=np.float32)
        cost = np.array([t[3] for t in batch], dtype=np.float32)
        r_aug = r_base - current_lambda * cost

        s_next = np.array([t[4] for t in batch], dtype=np.float32)
        done = np.array([t[5] for t in batch], dtype=np.float32)

        return s, a, r_aug, s_next, done

    def __len__(self) -> int:
        return len(self.nominal_buffer) + len(self.fault_buffer)


class PrimalDualDoubleDQNAgent:
    """
    Primal-Dual Double DQN Agent:
    - Primal: Double DQN updates maximizing Lagrangian return (energy savings - lambda * safety cost)
    - Dual: Adaptive ascent on lambda enforcing near-zero False Negative Rate
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.q_online = CompactQNetwork(seed=seed)
        self.q_target = CompactQNetwork(seed=seed + 100)
        self.q_target.copy_weights_from(self.q_online)

        self.replay_buffer = ReplayBuffer(capacity=RLConfig.BUFFER_SIZE)
        self.epsilon = RLConfig.EPSILON_START
        self.lambda_cost = RLConfig.INITIAL_LAMBDA
        self.gamma = RLConfig.GAMMA
        self.batch_size = RLConfig.BATCH_SIZE
        self.train_step_count = 0

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode and self.rng.rand() < self.epsilon:
            return self.rng.randint(0, RLConfig.ACTION_DIM)
        q_vals = self.q_online.predict(state)[0]
        return int(np.argmax(q_vals))

    def update_primal(self) -> float:
        """Double DQN gradient descent step on Lagrangian augmented reward."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        s, a, r_aug, s_next, done = self.replay_buffer.sample(self.batch_size, self.lambda_cost)

        # Double DQN target computation:
        # a* = argmax_a Q_online(s', a)
        q_online_next = self.q_online.predict(s_next)
        best_actions = np.argmax(q_online_next, axis=1)

        # Target = r_aug + gamma * Q_target(s', a*)
        q_target_next = self.q_target.predict(s_next)
        target_q_vals = q_target_next[np.arange(len(s)), best_actions]
        targets = r_aug + self.gamma * (1.0 - done) * target_q_vals

        loss = self.q_online.train_step(s, a, targets)
        self.train_step_count += 1
        return loss

    def sync_target_network(self):
        self.q_target.copy_weights_from(self.q_online)

    def decay_epsilon(self):
        self.epsilon = max(RLConfig.EPSILON_END, self.epsilon * RLConfig.EPSILON_DECAY)

    def update_dual(self, fnr: float) -> float:
        """
        Dual Ascent Update on Lagrange multiplier:
        lambda_{k+1} = max(0, lambda_k + eta_lambda * (FNR - epsilon_target))
        """
        delta = fnr - RLConfig.TARGET_FNR
        # Dual gradient ascent
        self.lambda_cost = max(0.0, float(self.lambda_cost + RLConfig.ETA_LAMBDA * delta))
        return self.lambda_cost
