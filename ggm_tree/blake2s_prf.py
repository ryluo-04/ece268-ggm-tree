"""
BLAKE2s-based PRG for GGM tree construction
"""

import hashlib
import numpy as np

# cpu implementation

def blake2s_prf_cpu(key: bytes) -> tuple:
    """
    evaluate BLAKE2s PRG on a single 16 byte key
    returns (left_child, right_child) each 16 bytes
    G(k) = BLAKE2s(key = k, data = 0x00, digest = 16) || BLAKE2s(key = k, data = 0x01, digest = 16)
    """
    assert len(key) == 16, "BLAKE2s key must be exactly 16 bytes"
    left  = hashlib.blake2s(b"\x00", key=key, digest_size=16).digest()
    right = hashlib.blake2s(b"\x01", key=key, digest_size=16).digest()
    return left, right


def blake2s_expand_level_cpu(keys: np.ndarray) -> np.ndarray:
    """
    expand one GGM tree level on cpu (sequential)
    args: keys — uint8 array (N, 16)
    returns: uint8 array (2N, 16), children [left_0, right_0, left_1, right_1, ...]
    """
    n = keys.shape[0]
    children = np.empty((2 * n, 16), dtype=np.uint8)
    for i in range(n):
        k = bytes(keys[i])
        left, right = blake2s_prf_cpu(k)
        children[2 * i]     = np.frombuffer(left,  dtype=np.uint8)
        children[2 * i + 1] = np.frombuffer(right, dtype=np.uint8)
    return children


# gpu implementation (CuPy RawKernel — BLAKE2s from scratch in CUDA C)
# implements keyed BLAKE2s per RFC 7693:
# 32 bit words, 8 state words, 10 mixing rounds
# keyed mode: 64 byte key block is first message block (counter = 64)
# data block: 1 byte padded to 64 bytes (counter = 65, final flag set)
# digest: first 16 bytes of state in little endian order
# IV and SIGMA in __constant__ memory so each thread's working state lives in registers

_BLAKE2S_KERNEL_SRC = r"""
#include <stdint.h>

// BLAKE2s IV: fractional parts of sqrt of first 8 primes
__constant__ uint32_t BLAKE2S_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

// message schedule permutations (RFC 7693 table 2)
__constant__ uint8_t SIGMA[10][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
    {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4},
    { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8},
    { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13},
    { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9},
    {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11},
    {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10},
    { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0}
};

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

// G mixing function (RFC 7693 section 3.1)
__device__ __forceinline__ void G(uint32_t* v,
                                  int a, int b, int c, int d,
                                  uint32_t mx, uint32_t my) {
    v[a] = v[a] + v[b] + mx;
    v[d] = rotr32(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = rotr32(v[b] ^ v[c], 12);
    v[a] = v[a] + v[b] + my;
    v[d] = rotr32(v[d] ^ v[a],  8);
    v[c] = v[c] + v[d];
    v[b] = rotr32(v[b] ^ v[c],  7);
}

__device__ __forceinline__ uint32_t load_le32(const uint8_t* p) {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] <<  8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

// compress one 64-byte block into state h
// t_lo/t_hi: 64-bit byte counter (low/high 32 bits)
// is_last: set finalization flag when processing final block
__device__ void compress(uint32_t* h, const uint8_t* block,
                         uint32_t t_lo, uint32_t t_hi, int is_last) {
    uint32_t m[16];
    for (int i = 0; i < 16; i++) m[i] = load_le32(block + i * 4);

    uint32_t v[16];
    for (int i = 0; i < 8; i++) v[i]     = h[i];
    for (int i = 0; i < 8; i++) v[i + 8] = BLAKE2S_IV[i];
    v[12] ^= t_lo;
    v[13] ^= t_hi;
    if (is_last) v[14] ^= 0xFFFFFFFFu;

    for (int r = 0; r < 10; r++) {
        const uint8_t* s = SIGMA[r];
        // column step
        G(v, 0, 4,  8, 12, m[s[ 0]], m[s[ 1]]);
        G(v, 1, 5,  9, 13, m[s[ 2]], m[s[ 3]]);
        G(v, 2, 6, 10, 14, m[s[ 4]], m[s[ 5]]);
        G(v, 3, 7, 11, 15, m[s[ 6]], m[s[ 7]]);
        // diagonal step
        G(v, 0, 5, 10, 15, m[s[ 8]], m[s[ 9]]);
        G(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        G(v, 2, 7,  8, 13, m[s[12]], m[s[13]]);
        G(v, 3, 4,  9, 14, m[s[14]], m[s[15]]);
    }
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}

// compute keyed BLAKE2s of one byte with a 16 byte key
// produces 16 bytes of output in digest
__device__ void blake2s_keyed(const uint8_t* key, uint8_t data_byte, uint8_t* digest) {
    // init state: h = IV XOR parameter block
    // param block word 0: digest_len=16, key_len=16, fanout=1, depth=1
    uint32_t h[8];
    for (int i = 0; i < 8; i++) h[i] = BLAKE2S_IV[i];
    h[0] ^= 0x01010000u | (16u << 8) | 16u;

    // block 1: 16-byte key padded to 64 bytes, counter = 64
    uint8_t b1[64];
    for (int i = 0; i < 64; i++) b1[i] = 0;
    for (int i = 0; i < 16; i++) b1[i] = key[i];
    compress(h, b1, 64u, 0u, 0);

    // block 2: data byte padded to 64 bytes, counter = 65, final block
    uint8_t b2[64];
    for (int i = 0; i < 64; i++) b2[i] = 0;
    b2[0] = data_byte;
    compress(h, b2, 65u, 0u, 1);

    // extract first 16 bytes of state as little-endian output
    for (int i = 0; i < 4; i++) {
        uint32_t w = h[i];
        digest[i*4+0] = (uint8_t)( w        & 0xFF);
        digest[i*4+1] = (uint8_t)((w >>  8) & 0xFF);
        digest[i*4+2] = (uint8_t)((w >> 16) & 0xFF);
        digest[i*4+3] = (uint8_t)((w >> 24) & 0xFF);
    }
}

// main kernel expands N parent nodes into 2N children
extern "C" __global__
void blake2s_expand_level(const uint8_t* __restrict__ keys,
                                uint8_t* __restrict__ out,
                          int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    const uint8_t* key = keys + tid * 16;
    blake2s_keyed(key, 0x00, out + (2 * tid    ) * 16);
    blake2s_keyed(key, 0x01, out + (2 * tid + 1) * 16);
}
"""

_blake2s_kernel = None


def _get_blake2s_kernel():
    """
    compile and return the CuPy RawKernel for BLAKE2s expansion
    """
    global _blake2s_kernel
    if _blake2s_kernel is None:
        try:
            import cupy as cp
            _blake2s_kernel = cp.RawKernel(_BLAKE2S_KERNEL_SRC, "blake2s_expand_level")
        except ImportError:
            raise RuntimeError(
                "CuPy is required for GPU BLAKE2s. Install with: pip install cupy-cuda11x"
            )
    return _blake2s_kernel


def blake2s_expand_level_gpu(keys_gpu):
    """
    expand one GGM tree level on the GPU using BLAKE2s PRG
    """
    import cupy as cp

    n = keys_gpu.shape[0]
    out_gpu = cp.empty((2 * n, 16), dtype=cp.uint8)

    kernel = _get_blake2s_kernel()
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    kernel(
        (blocks,), (threads_per_block,),
        (keys_gpu, out_gpu, np.int32(n))
    )
    return out_gpu
