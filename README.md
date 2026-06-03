# GGM PRF Tree Construction on GPU

ECE 268 Final Project - GPU-accelerated GGM (Goldreich-Goldwasser-Micali) PRF tree

**Team:** Kevin Liang, Bryce Blair, Ryan Luo
**GitHub:** https://github.com/ryluo-04/ece268-ggm-tree

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
| AES-128 | `AES_k(0^128) \|\| AES_k(1*0^120)` - encrypt two fixed plaintexts with key `k` in ECB mode |
| BLAKE2s | `BLAKE2s(key=k, data=0x00, digest=16) \|\| BLAKE2s(key=k, data=0x01, digest=16)` |

---

## File Structure

```
ggm_tree/
├── aes_cpu.c           # AES-128 from scratch in pure C
├── aes_prf.py          # AES-128 CPU (pure Python) + GPU (CuPy RawKernel)
├── blake2s_prf.py      # BLAKE2s CPU (hashlib) + GPU (CuPy RawKernel, CUDA C from scratch)
├── ggm_tree.py         # GGMTree class: BFS expansion on CPU and GPU
├── benchmark.py        # throughput and speedup benchmarks, saves two PNGs
├── verify.py           # correctness tests
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
| Matplotlib | >= 3.8 — benchmark plots (install `--user` if on DataHub) |
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
pip install cupy-cuda11x   # CUDA 11.x
pip install cupy-cuda12x   # CUDA 12.x

# if matplotlib fails to import after installing cupy (numpy 2.x conflict on DataHub):
pip install --user --upgrade "matplotlib>=3.8" contourpy pillow

# if cupy fails to import due to numpy version mismatch, try reinstalling together:
pip uninstall -y numpy cupy-cuda12x
pip install --user cupy-cuda12x   # pulls in a compatible numpy automatically
```

> **Note on DataHub (dsmlp-jupyter):** `cupy-cuda12x` upgrades numpy to 2.x, which
> causes harmless-but-noisy warnings from preinstalled conda packages (tensorflow,
> ml_dtypes). These do not affect correctness. To suppress them, prefix commands with
> `PYTHONWARNINGS=ignore` or redirect stderr: `python verify.py 2>/dev/null`.
---

## Running

### 1. Correctness tests (run this first)

```bash
python verify.py
```

Expected output on a CPU-only machine:

```
[Test 1a] AES — NIST FIPS 197 known-answer tests
  [PASS]   vector 0: ...
  [PASS]   vector 1: ...
[Test 1b] BLAKE2s — hashlib cross-check
  [PASS]   key[0] left+right match  ...
[Test 2a] CPU level expansion — AES
  [PASS]   node 0 children match  ...
[Test 2b] CPU level expansion — BLAKE2s
  [PASS]   node 0 children match  ...
[Test 3a] GGMTree AES CPU determinism, depth=6
  [PASS]   leaf[00000] matches path traversal  ...
[Test 3b] GGMTree BLAKE2s CPU determinism, depth=6
  [PASS]   leaf[00000] matches path traversal  ...
[Test 4] GPU vs CPU — SKIPPED (CuPy not available)

[PASS] All tests passed.
```

On a CUDA machine with CuPy installed, Test 4 runs and verifies **byte-exact equality**
between CPU and GPU outputs for **both AES and BLAKE2s** at depths 4, 8, and 12.
This has been confirmed on an NVIDIA GTX 1080 Ti (CUDA 12.2).

### 2. Benchmarks

```bash
python benchmark.py
```

Prints a full results table and saves:
- `benchmark_throughput.png` — leaves/s vs depth (log scale) for all four backends
- `benchmark_speedup.png` — GPU/CPU speedup vs depth (only when GPU is available)

By default, CPU AES is measured through depth 16 and CPU BLAKE2s through depth 16;
GPU is measured through depth 20. This avoids the multi-minute pure-Python AES CPU
run at depth 20 while preserving enough overlap for meaningful speedup computation.

---

## Benchmark Results (NVIDIA GTX 1080 Ti, CUDA 12.2)

| PRF | Device | Depth | Leaves | Throughput (leaves/s) | Speedup |
|-----|--------|-------|--------|-----------------------|---------|
| AES | CPU | 4 | 16 | 4,091 | — |
| AES | CPU | 12 | 4,096 | 3,922 | — |
| AES | CPU | 16 | 65,536 | 4,002 | — |
| AES | GPU | 4 | 16 | 69,736 | 17.0× |
| AES | GPU | 8 | 256 | 878,153 | 227.3× |
| AES | GPU | 12 | 4,096 | 7,707,208 | 1,965× |
| AES | GPU | 16 | 65,536 | 43,347,679 | 10,832× |
| AES | GPU | 20 | 1,048,576 | 80,802,924 | — |
| BLAKE2s | CPU | 4 | 16 | 281,261 | — |
| BLAKE2s | CPU | 12 | 4,096 | 274,860 | — |
| BLAKE2s | CPU | 16 | 65,536 | 272,588 | — |
| BLAKE2s | GPU | 4 | 16 | 61,389 | 0.2× |
| BLAKE2s | GPU | 8 | 256 | 694,694 | 2.5× |
| BLAKE2s | GPU | 12 | 4,096 | 7,266,330 | 26.4× |
| BLAKE2s | GPU | 16 | 65,536 | 53,896,218 | 197.7× |
| BLAKE2s | GPU | 20 | 1,048,576 | 118,936,020 | — |

**Key finding:** AES shows a larger speedup ratio (up to 10,832×) because its CPU
baseline is slow pure-Python (~4k leaves/s). In absolute GPU throughput, BLAKE2s is
actually faster (120M vs 81M leaves/s at depth 20), reflecting its ARX design mapping
more efficiently onto GPU registers than AES's table-lookup rounds. BLAKE2s GPU is
also slower than CPU at depth 4, where kernel-launch overhead dominates at only 16 leaves.

---

## Design Notes

### Memory layout
Node seeds at each level are stored as a contiguous `(N, 16)` `uint8` array in row-major
order. This keeps each node's 16 bytes contiguous per thread, enabling coalesced global
memory access — adjacent threads in a warp read adjacent cache lines.

### Inter-level synchronization
No explicit barrier is needed between tree levels. Each level is a separate CUDA kernel
launch in the default stream. CUDA guarantees in-order execution within a stream, so the
output of level `l` is complete before level `l+1` begins.

### No mid-tree CPU/GPU transfers
Data stays on the GPU between level expansions. Only the final leaf array is transferred
to the CPU at the end of `GGMTree.expand()` to keep the PCIe bus idle during computation.

### GPU timing correctness
The benchmark does one untimed warmup run before each timed series to absorb CuPy's
one-time JIT kernel compilation and synchronizes the CUDA stream before
stopping the timer. This ensures measurements reflect true kernel completion rather than
asynchronous launch enqueue time.

### BLAKE2s GPU header note
The CuPy NVRTC compilation environment on some machines (including UCSD DataHub) cannot
locate `<stdint.h>`. The BLAKE2s kernel avoids this by defining the fixed-width integer
types directly (`typedef unsigned char uint8_t; typedef unsigned int uint32_t;`), matching
the header-free style already used by the AES kernel.

---

## Status

### Completed
- [x] AES-128 from scratch in pure Python (CPU) — verified against NIST FIPS 197 vectors
- [x] AES-128 from scratch in pure C (CPU) — independently verified against reference AES
- [x] AES-128 GPU kernel (CuPy RawKernel, CUDA C) — S-box in `__constant__` memory
- [x] BLAKE2s PRG — CPU (hashlib) and GPU (CuPy RawKernel, CUDA C from scratch, RFC 7693)
- [x] GGM tree BFS expansion on CPU and GPU for both PRFs
- [x] Correctness tests: NIST vectors, hashlib cross-check, CPU self-consistency, path-following determinism
- [x] GPU vs CPU byte-exact agreement for both PRFs at depths 4, 8, 12 — confirmed on GTX 1080 Ti
- [x] Benchmark harness — throughput table, `benchmark_throughput.png`, `benchmark_speedup.png`
- [x] Full GPU benchmark results collected (GTX 1080 Ti, CUDA 12.2)
- [x] Performance–security trade-off analysis (AES vs BLAKE2s)
- [x] Final report (`reports/final_report.pdf`)
- [x] Progress report (`reports/progress_report.pdf`)
- [x] 2.5-minute in-class presentation

### To be done
- [ ] 10-minute recorded presentation
- [ ] Pure CUDA implementation without CuPy layer (bonus)