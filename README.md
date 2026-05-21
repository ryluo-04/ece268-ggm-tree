# GGM PRF Tree Construction on GPU

ECE 268 Final Project - GPU-accelerated GGM (Goldreich-Goldwasser-Micali) PRF tree

---

## What is the GGM Tree?

The GGM construction [Goldreich, Goldwasser, Micali 1986] builds a pseudorandom function (PRF)
from a pseudorandom generator (PRG).

```
Root seed s in {0,1}^128
        |
   G(s) = (s_0, s_1)          <- PRG expands one 128-bit seed into two
      /         \
  G(s_0)      G(s_1)
   / \           / \
 ...  ...      ...  ...
```

After `d` levels the tree has `2^d` leaves. Each leaf is a pseudorandom 128-bit value determined
entirely by the root seed and the leaf's binary index path from the root.

Why GPU? Each level of the tree is a data-parallel operation - every parent node at level `l`
can be expanded independently. A GPU can process all `2^l` nodes simultaneously in a single kernel
launch, making it an ideal accelerator for large trees.

---

## PRG Definitions

| PRF     | G(k) construction |
|---------|-------------------|
| AES-128 | `AES_k(0^128) || AES_k(1*0^120)` - encrypt two fixed plaintexts with key `k` in ECB mode |
| BLAKE2s | `BLAKE2s(key=k, data=0x00, digest=16) || BLAKE2s(key=k, data=0x01, digest=16)` |

ECB mode is safe here because each node key is independent thus no semantic relationship
between siblings so no IV or chaining is needed.

---

## File Structure

```
ggm_tree/
├── aes_cpu.c
├── aes_prf.py
├── blake2s_prf.py
├── ggm_tree.py
├── benchmark.py
├── verify.py
└── reports/
    ├── progress_report.pdf   # progress report 
    └── final_report.pdf      # final report 
```

---

## Dependencies

| Component  | Notes |
|------------|-------|
| Python     | 3.10+ |
| NumPy      | >= 1.22 |
| Matplotlib | >= 3.5 - benchmark plots |
| CuPy       | GPU only; install the wheel matching your CUDA version |
| CUDA       | 11.x or 12.x (GPU path only) |

---

## Installation

```bash
# numpy and matplotlib
pip install numpy matplotlib

# check CUDA version
nvidia-smi

# install the matching cupy wheel (GPU only)
pip install cupy-cuda11x # CUDA 11.x
pip install cupy-cuda12x # CUDA 12.x

# in instances of numpy errors
pip install "numpy<2"
```

---

## Running

### 1. Correctness tests (run this first)

```bash
python verify.py
```

Expected output on a CPU-only machine:

```
[Test 1a] AES - NIST FIPS 197 known-answer tests
  [PASS] vector 0: ...
  [PASS] vector 1: ...
[Test 2] CPU level expansion - AES
  [PASS] node 0 children match
  ...
[Test 3] GGMTree AES CPU determinism, depth=6
  [PASS] leaf[00000] matches path traversal
  ...
[Test 4] GPU vs CPU - SKIPPED (CuPy not available)

[PASS] All tests passed.
```

On a CUDA machine with CuPy installed, Test 4 will run and verify byte exact
equality between CPU and GPU outputs at depths 4, 8, and 12.

### 2. Benchmarks

(not yet implemented - to be added)

---

## Design Notes

### Memory layout
Node seeds at each level are stored as a contiguous `(N, 16)` `uint8` array in row-major order.
This keeps each node's 16 bytes contiguous per thread, enabling coalesced global memory
access - adjacent threads in a warp read adjacent cache lines.

### Inter-level synchronization
No explicit barrier is needed between tree levels. Each level is a separate CUDA kernel
launch in the default stream. CUDA guarantees in-order execution within a stream, so the
output of level `l` is complete before level `l+1` begins.

### No mid-tree CPU/GPU transfers
Data stays on the GPU between level expansions. Only the final leaf array is
transferred to the CPU at the end of `GGMTree.expand()`, keeping the PCIe bus idle
during computation.

---

## Progress

### implemented
- [x] AES-128 from scratch in pure Python (CPU) - verified against NIST FIPS 197 vectors
- [x] AES-128 from scratch in pure C (CPU)
- [x] AES-128 GPU kernel (CuPy RawKernel, CUDA C)
- [x] GGM tree BFS expansion on CPU and GPU (AES)
- [x] Correctness tests: NIST vectors, CPU self-consistency, CPU path-following determinism
- [x] GPU vs CPU byte-exact agreement at depths 4, 8, 12

### to be done
- [ ] BLAKE2s PRG - CPU and GPU (blake2s_prf.py)
- [ ] BLAKE2s correctness tests (to be added to verify.py)
- [ ] Benchmark harness (benchmark.py) - timing, throughput table, plots
- [ ] GPU benchmark results
- [ ] Performance-security tradeoff analysis
- [ ] Final report (reports/final_report.pdf)
- [ ] Pure CUDA implementation (bonus)