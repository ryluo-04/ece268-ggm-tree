"""
correctness tests for the GGM tree

tests:
  1a. AES NIST FIPS 197 known-answer tests
  1b. BLAKE2s hashlib cross-check
  2a. AES cpu level expansion self-consistency
  2b. BLAKE2s cpu level expansion self-consistency
  3a. GGMTree AES cpu determinism (path-following)
  3b. GGMTree BLAKE2s cpu determinism (path-following)
  4.  GPU vs CPU exact agreement for both PRFs (skipped if CuPy unavailable)
"""

import sys
import os
import hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from aes_prf     import aes_prf_cpu, aes_expand_level_cpu, aes_encrypt_block
from blake2s_prf import blake2s_prf_cpu, blake2s_expand_level_cpu
from ggm_tree    import GGMTree

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def _check(label: str, condition: bool) -> bool:
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}")
    return condition


def _random_seed(rng: np.random.Generator) -> bytes:
    return bytes(rng.integers(0, 256, size=16, dtype=np.uint8))


# test 1a: AES NIST FIPS 197 known-answer tests
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


# test 1b: BLAKE2s comparison to hashlib

def test_blake2s_hashlib() -> bool:
    """
    compare blake2s_prf_cpu against hashlib reference for 8 random keys
    """
    print("\n[Test 1b] BLAKE2s — comparison to hashlib")
    passed = True
    rng = np.random.default_rng(99)
    for i in range(8):
        key = _random_seed(rng)
        left_ours,  right_ours  = blake2s_prf_cpu(key)
        left_ref  = hashlib.blake2s(b"\x00", key=key, digest_size=16).digest()
        right_ref = hashlib.blake2s(b"\x01", key=key, digest_size=16).digest()
        ok = (left_ours == left_ref) and (right_ours == right_ref)
        passed &= _check(f"  key[{i}] left+right match", ok)
    return passed


# test 2a: AES cpu level expansion self-consistency

def test_cpu_level_expansion_aes() -> bool:
    """
    expand a batch of nodes and verify each child matches the single-call PRF
    """
    print("\n[Test 2a] CPU level expansion — AES")
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


# test 2b: BLAKE2s cpu level expansion self-consistency

def test_cpu_level_expansion_blake2s() -> bool:
    """
    expand a batch of nodes and verify each child matches the single-call PRF
    """
    print("\n[Test 2b] CPU level expansion — BLAKE2s")
    rng  = np.random.default_rng(8)
    N    = 16
    keys = rng.integers(0, 256, size=(N, 16), dtype=np.uint8)
    children = blake2s_expand_level_cpu(keys)

    passed = True
    for i in range(N):
        left_ref, right_ref = blake2s_prf_cpu(bytes(keys[i]))
        ok = (bytes(children[2*i]) == left_ref) and (bytes(children[2*i+1]) == right_ref)
        passed &= _check(f"  node {i} children match", ok)
    return passed


# test 3a: GGMTree AES cpu determinism via path-following

def test_ggm_determinism_aes(depth: int = 6) -> bool:
    """
    reconstruct random leaves by following bit-paths from the root
    compare to expand()
    """
    print(f"\n[Test 3a] GGMTree AES CPU determinism, depth={depth}")
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


# test 3b: GGMTree BLAKE2s cpu determinism via path-following

def test_ggm_determinism_blake2s(depth: int = 6) -> bool:
    """
    reconstruct random leaves by following bit-paths from the root
    compare to expand()
    """
    print(f"\n[Test 3b] GGMTree BLAKE2s CPU determinism, depth={depth}")
    rng       = np.random.default_rng(2025)
    root_seed = _random_seed(rng)
    leaves    = GGMTree(prf="blake2s", device="cpu", depth=depth).expand(root_seed)

    num_leaves    = 2 ** depth
    check_indices = sorted(
        set([0, 1, num_leaves // 2, num_leaves - 2, num_leaves - 1])
        | set(int(x) for x in rng.integers(0, num_leaves, 20))
    )

    passed = True
    for idx in check_indices:
        node = root_seed
        for bit_pos in range(depth - 1, -1, -1):
            left, right = blake2s_prf_cpu(node)
            node = right if (idx >> bit_pos) & 1 else left
        ok = np.array_equal(np.frombuffer(node, dtype=np.uint8), leaves[idx])
        passed &= _check(f"  leaf[{idx:05d}] matches path traversal", ok)
    return passed


# test 4: GPU vs CPU exact agreement for both PRFs

def test_gpu_vs_cpu(prf: str, depth: int) -> bool:
    """
    run CPU and GPU expand() with the same seed
    assert byte-exact equality
    """
    rng       = np.random.default_rng(depth * 31 + (0 if prf == "aes" else 1))
    root_seed = _random_seed(rng)
    leaves_cpu = GGMTree(prf=prf, device="cpu", depth=depth).expand(root_seed)
    leaves_gpu = GGMTree(prf=prf, device="gpu", depth=depth).expand(root_seed)
    ok = np.array_equal(leaves_cpu, leaves_gpu)
    _check(f"  {prf.upper()} depth={depth:2d}: CPU==GPU ({2**depth} leaves)", ok)
    return ok


# main

def main() -> None:
    all_passed = True

    all_passed &= test_aes_nist_vectors()
    all_passed &= test_blake2s_hashlib()
    all_passed &= test_cpu_level_expansion_aes()
    all_passed &= test_cpu_level_expansion_blake2s()
    all_passed &= test_ggm_determinism_aes()
    all_passed &= test_ggm_determinism_blake2s()

    try:
        import cupy as cp  # noqa: F401
        print("\n[Test 4] GPU vs CPU exact agreement")
        for prf in ("aes", "blake2s"):
            for depth in (4, 8, 12):
                all_passed &= test_gpu_vs_cpu(prf, depth)
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
