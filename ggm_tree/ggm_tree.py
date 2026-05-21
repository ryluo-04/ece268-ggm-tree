"""
ggm_tree.py — GGM (Goldreich–Goldwasser–Micali) PRF tree expansion

construction:
    - root seed s ∈ {0,1}^128 is the level-0 node
    - at each level l, every node seed is expanded by PRG G into two children:
      G(s) = (s_left, s_right)
    - after depth d levels we have 2^d leaf seeds

bfs level-by-level layout:
    level 0: 1 node    → shape (1, 16)
    level 1: 2 nodes   → shape (2, 16)
    level d: 2^d nodes → shape (2^d, 16)

only the current level is kept in memory at any time so peak memory is
O(2^(d+1) × 16) bytes (current level + next level simultaneously)

gpu execution: each level is one kernel launch (all nodes in parallel)
data stays on device between levels; only the final leaves transfer to cpu.
the kernel boundary itself acts as the inter-level synchronisation point
(cuda launches in the same stream are ordered by default)
"""

import numpy as np
from typing import Literal

PRF = Literal["aes", "blake2s"]
DEVICE = Literal["cpu", "gpu"]

class GGMTree:
    """
    GGM PRF tree builder

    attributes:
        prf: "aes" or "blake2s"
        device: "cpu" or "gpu"
        depth: number of levels to expand (produces 2^depth leaves)
    """
    def __init__(self, prf: PRF = "aes", device: DEVICE = "cpu", depth: int = 10) -> None:
        if prf not in ("aes", "blake2s"):
            raise ValueError(f"unknown PRF '{prf}'. choose 'aes' or 'blake2s'.")
        if device not in ("cpu", "gpu"):
            raise ValueError(f"unknown device '{device}'. choose 'cpu' or 'gpu'.")
        self.prf    = prf
        self.device = device
        self.depth  = depth

        if prf == "aes":
            from aes_prf import aes_expand_level_cpu, aes_expand_level_gpu
            self._expand_fn = aes_expand_level_cpu if device == "cpu" else aes_expand_level_gpu
        else:
            from blake2s_prf import blake2s_expand_level_cpu, blake2s_expand_level_gpu
            self._expand_fn = blake2s_expand_level_cpu if device == "cpu" else blake2s_expand_level_gpu

    def expand(self, root_seed: bytes) -> np.ndarray:
        """
        expand 16 byte root seed into 2^depth leaf vals

        args:
            root_seed: exactly 16 bytes
        returns:
            np.ndarray of shape (2^depth, 16) dtype uint8 leaves on cpu
        """
        if len(root_seed) != 16:
            raise ValueError("root_seed must be exactly 16 bytes.")
        return self._expand_cpu(root_seed) if self.device == "cpu" else self._expand_gpu(root_seed)

    def _expand_cpu(self, root_seed: bytes) -> np.ndarray:
        """
        bfs expansion on cpu
        """
        level = np.frombuffer(root_seed, dtype=np.uint8).reshape(1, 16).copy()
        for _ in range(self.depth):
            level = self._expand_fn(level)
        return level

    def _expand_gpu(self, root_seed: bytes) -> np.ndarray:
        """
        bfs expansion on gpu
        data stays on device between levels
        """
        import cupy as cp
        level_gpu = cp.asarray(
            np.frombuffer(root_seed, dtype=np.uint8).reshape(1, 16)
        )
        for _ in range(self.depth):
            level_gpu = self._expand_fn(level_gpu)
        return cp.asnumpy(level_gpu)