"""
benchmark.py — timing and throughput benchmarks for GGM tree expansion

measures wall-clock time (time.perf_counter) for all combinations of:
    PRF    in {aes, blake2s}
    device in {cpu, gpu}
    depth  in {4, 8, 12, 16, 20}  -> 16 to 1,048,576 leaves

gpu timing: one warm-up run before each timed series; stream sync after each run
so the timer captures actual gpu completion rather than just kernel enqueue time

outputs:
    - formatted results table printed to stdout
    - benchmark_throughput.png  throughput (leaves/s) vs depth, log y scale
    - benchmark_speedup.png     GPU/CPU speedup vs depth (only if GPU available)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ggm_tree import GGMTree

# configuration
DEPTHS    = [4, 8, 12, 16, 20]
PRFS      = ["aes", "blake2s"]
ROOT_SEED = bytes(range(16))
N_REPEATS = 3


def _sync_gpu():
    """flush default CUDA stream so timing captures real completion"""
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    except ImportError:
        pass


def timed_expand(tree: GGMTree, root_seed: bytes, n_repeats: int) -> float:
    """
    run expand() n_repeats times and return average wall-clock seconds
    GPU runs are synchronised before the timer stops
    """
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        tree.expand(root_seed)
        if tree.device == "gpu":
            _sync_gpu()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.mean(times))


def run_benchmarks(gpu_available: bool) -> list:
    """
    run all combinations and return results as a list of dicts
    """
    results = []
    devices = ["cpu", "gpu"] if gpu_available else ["cpu"]

    for prf in PRFS:
        for device in devices:
            for depth in DEPTHS:
                leaves = 2 ** depth
                tree   = GGMTree(prf=prf, device=device, depth=depth)

                # warm-up (amortizes kernel JIT on first GPU launch)
                tree.expand(ROOT_SEED)
                if device == "gpu":
                    _sync_gpu()

                elapsed    = timed_expand(tree, ROOT_SEED, N_REPEATS)
                throughput = leaves / elapsed

                results.append({
                    "prf":        prf.upper(),
                    "device":     device.upper(),
                    "depth":      depth,
                    "leaves":     leaves,
                    "time_s":     elapsed,
                    "throughput": throughput,
                })
                print(
                    f"  {prf.upper():7s} {device.upper():3s} d={depth:2d} "
                    f"({leaves:>8,} leaves)  {elapsed:.4f}s  "
                    f"{throughput:>12,.0f} leaves/s"
                )

    return results


def compute_speedup(results: list) -> list:
    """
    add speedup = cpu_time / gpu_time for each GPU row
    CPU rows get speedup = None
    """
    cpu_times = {
        (r["prf"], r["depth"]): r["time_s"]
        for r in results if r["device"] == "CPU"
    }
    for r in results:
        if r["device"] == "GPU":
            cpu_t = cpu_times.get((r["prf"], r["depth"]))
            r["speedup"] = (cpu_t / r["time_s"]) if cpu_t else None
        else:
            r["speedup"] = None
    return results


def print_table(results: list) -> None:
    header = (
        f"{'PRF':7s} {'Device':6s} {'Depth':>5s} {'Leaves':>10s} "
        f"{'Time (s)':>10s} {'Throughput (leaves/s)':>22s} {'Speedup':>8s}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        su = f"{r['speedup']:.1f}x" if r["speedup"] is not None else "-"
        print(
            f"{r['prf']:7s} {r['device']:6s} {r['depth']:5d} {r['leaves']:10,} "
            f"{r['time_s']:10.4f} {r['throughput']:>22,.0f} {su:>8s}"
        )
    print(sep)


def plot_results(results: list, output_dir: str = ".") -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    styles = {
        ("AES",    "CPU"): dict(color="steelblue",  linestyle="-",  marker="o"),
        ("AES",    "GPU"): dict(color="steelblue",  linestyle="--", marker="s"),
        ("BLAKE2S","CPU"): dict(color="darkorange", linestyle="-",  marker="o"),
        ("BLAKE2S","GPU"): dict(color="darkorange", linestyle="--", marker="s"),
    }

    # plot 1: throughput vs depth
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    seen = set()
    for r in results:
        key = (r["prf"], r["device"])
        if key not in seen:
            xs = [rr["depth"]      for rr in results if (rr["prf"], rr["device"]) == key]
            ys = [rr["throughput"] for rr in results if (rr["prf"], rr["device"]) == key]
            ax1.plot(xs, ys, label=f"{r['prf']} ({r['device']})", **styles.get(key, {}))
            seen.add(key)
    ax1.set_yscale("log")
    ax1.set_xlabel("tree depth")
    ax1.set_ylabel("throughput (leaves/s)")
    ax1.set_title("GGM tree throughput vs depth")
    ax1.set_xticks(DEPTHS)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    fig1.tight_layout()
    out1 = os.path.join(output_dir, "benchmark_throughput.png")
    fig1.savefig(out1, dpi=150)
    print(f"saved: {out1}")
    plt.close(fig1)

    # plot 2: speedup vs depth (GPU only)
    gpu_rows = [r for r in results if r["device"] == "GPU" and r["speedup"] is not None]
    if not gpu_rows:
        print("no GPU results — skipping speedup plot")
        return

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    su_styles = {
        "AES":    dict(color="steelblue",  marker="o"),
        "BLAKE2S": dict(color="darkorange", marker="s"),
    }
    for prf_name in ("AES", "BLAKE2S"):
        rows = [r for r in gpu_rows if r["prf"] == prf_name]
        if rows:
            ax2.plot([r["depth"] for r in rows], [r["speedup"] for r in rows],
                     label=prf_name, **su_styles.get(prf_name, {}))
    ax2.axhline(1.0, color="grey", linestyle=":", linewidth=1)
    ax2.set_xlabel("tree depth")
    ax2.set_ylabel("speedup (GPU / CPU)")
    ax2.set_title("GGM tree GPU speedup vs depth")
    ax2.set_xticks(DEPTHS)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = os.path.join(output_dir, "benchmark_speedup.png")
    fig2.savefig(out2, dpi=150)
    print(f"saved: {out2}")
    plt.close(fig2)


def main() -> None:
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        gpu_available = True
        print("GPU detected — running full benchmark (CPU + GPU)")
    except Exception:
        gpu_available = False
        print("no GPU / CuPy unavailable — running CPU-only benchmark")

    print(f"\ndepths={DEPTHS}  PRFs={PRFS}  repeats={N_REPEATS}\n")
    results = run_benchmarks(gpu_available)
    results = compute_speedup(results)
    print_table(results)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_results(results, output_dir=output_dir)


if __name__ == "__main__":
    main()
