"""
tests for the GGM tree

todo (add for blake2s):
  1b. BLAKE2s: hashlib cross-check
  2.  CPU level expansion for blake2s
  3.  GGMTree CPU determinism for blake2s
  4.  GPU vs CPU for blake2s

tests:
  1a. AES NIST FIPS 197 known answer tests
  2.  AES CPU level expansion self-consistency
  3.  GGMTree AES CPU determinism (path-following)
  4.  AES GPU vs CPU exact agreement (skip if CuPy unavailable)
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from aes_prf  import aes_prf_cpu, aes_expand_level_cpu, aes_encrypt_block
from ggm_tree import GGMTree

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def _check(label: str, condition: bool) -> bool:
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}")
    return condition

def _random_seed(rng: np.random.Generator) -> bytes:
    return bytes(rng.integers(0, 256, size=16, dtype=np.uint8))

# test 1a: AES NIST FIPS 197 known answer tests
# vectors from FIPS 197 appendix B and appendix C.1
_AES_VECTORS = [
    (
        "2b7e151628aed2a6abf7158809cf4f3c",
        "3243f6a8885a308d313198a2e0370734",
        "3925841d02dc09fbdc118597196a0b32",
    ),
    (
        "00000000000000000000000000000000",
        "00000000000000000000000000000000",
        "66e94bd4ef8a2c3b884cfa59ca342b2e",
    ),
]

def test_aes_nist_vectors() -> bool:
    print("\n[Test 1a] AES — NIST FIPS 197 known-answer tests")
    passed = True
    for i, (k_hex, pt_hex, ct_hex) in enumerate(_AES_VECTORS):
        key = bytes.fromhex(k_hex)
        pt  = bytes.fromhex(pt_hex)
        ct  = bytes.fromhex(ct_hex)
        got = aes_encrypt_block(pt, key)
        ok  = got == ct
        passed &= _check(f"  vector {i}: got {got.hex()} expected {ct.hex()}", ok)
    return passed

# test 2: AES cpu level expansion self-consistency
def test_cpu_level_expansion() -> bool:
    """
    expand a batch of nodes and verify each child matches the single-call PRF
    """
    print("\n[Test 2] CPU level expansion — AES")
    rng  = np.random.default_rng(7)
    N    = 16
    keys = rng.integers(0, 256, size=(N, 16), dtype=np.uint8)
    children = aes_expand_level_cpu(keys)

    passed = True
    for i in range(N):
        left_ref, right_ref = aes_prf_cpu(bytes(keys[i]))
        ok = (bytes(children[2*i]) == left_ref) and (bytes(children[2*i+1]) == right_ref)
        passed &= _check(f"  node {i} children match", ok)
    return passed


# test 3: GGMTree AES CPU determinism via path-following
def test_ggm_determinism(depth: int = 6) -> bool:
    """
    reconstruct random leaves by following bit-paths from the root
    compare to expand()
    """
    print(f"\n[Test 3] GGMTree AES CPU determinism, depth={depth}")
    rng       = np.random.default_rng(2024)
    root_seed = _random_seed(rng)
    leaves    = GGMTree(prf="aes", device="cpu", depth=depth).expand(root_seed)

    num_leaves    = 2 ** depth
    check_indices = sorted(
        set([0, 1, num_leaves // 2, num_leaves - 2, num_leaves - 1])
        | set(int(x) for x in rng.integers(0, num_leaves, 20))
    )

    passed = True
    for idx in check_indices:
        node = root_seed
        for bit_pos in range(depth - 1, -1, -1):
            left, right = aes_prf_cpu(node)
            node = right if (idx >> bit_pos) & 1 else left
        ok = np.array_equal(np.frombuffer(node, dtype=np.uint8), leaves[idx])
        passed &= _check(f"  leaf[{idx:05d}] matches path traversal", ok)
    return passed

# test 4: AES GPU vs CPU exact agreement
def test_gpu_vs_cpu(depth: int) -> bool:
    """
    run CPU and GPU expand() with the same seed
    assert byte exact equality
    """
    rng       = np.random.default_rng(depth * 31)
    root_seed = _random_seed(rng)
    leaves_cpu = GGMTree(prf="aes", device="cpu", depth=depth).expand(root_seed)
    leaves_gpu = GGMTree(prf="aes", device="gpu", depth=depth).expand(root_seed)
    ok = np.array_equal(leaves_cpu, leaves_gpu)
    _check(f"  AES depth={depth:2d}: CPU==GPU ({2**depth} leaves)", ok)
    return ok


# main
def main() -> None:
    all_passed = True

    all_passed &= test_aes_nist_vectors()
    all_passed &= test_cpu_level_expansion()
    all_passed &= test_ggm_determinism()

    try:
        import cupy as cp  # noqa: F401
        print("\n[Test 4] GPU vs CPU exact agreement — AES")
        for depth in (4, 8, 12):
            all_passed &= test_gpu_vs_cpu(depth)
    except ImportError:
        print("\n[Test 4] GPU vs CPU — SKIPPED (CuPy not available)")

    print()
    if all_passed:
        print(f"[{PASS}] All tests passed.")
    else:
        print(f"[{FAIL}] Some tests FAILED — see output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
