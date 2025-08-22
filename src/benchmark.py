import itertools
from pathlib import Path

import pandas as pd
from analyze import get_roller_hints, MatmulConfig
from arch import CUDA
from roller import Compiler, run_baseline, run_eval


def benchmark(test_shapes: list[tuple[int, int, int]], topk=20):
    results = []

    for m, n, k in test_shapes:
        print(f"--- Benchmarking M={m}, N={n}, K={k} ---")
        config = MatmulConfig(M=m, N=n, K=k, el_bits=32)

        # 1. Get baseline performance from CuPy
        print("Running baseline (CuPy)...")
        try:
            baseline_time, baseline_gflops = run_baseline(config)
            print(
                f"CuPy baseline: {baseline_time:.6f} ms, {baseline_gflops:.6f} GFLOPS"
            )
        except Exception as e:
            print(f"CuPy baseline failed: {e}")
            baseline_time, baseline_gflops = float("inf"), 0

        # 2. Get and evaluate roller hints
        pipeline_depth = 2
        print(f"Generating and testing {topk} roller hints...")
        try:
            hints = get_roller_hints(config, CUDA(), topk=topk)
        except Exception as e:
            print(f"Could not get hints for shape ({m},{n},{k}): {e}")
            hints = []

        best_roller_time = float("inf")
        best_roller_gflops = 0
        best_hint = None

        for i, hint in enumerate(hints):
            print(f"  Testing hint {i+1}/{len(hints)}: {hint}")
            try:
                compiler = Compiler(config, hint, pipeline_depth, dump_path=Path("log"))
                cuda_func = compiler.compile()
                t, gflops = run_eval(
                    cuda_func,
                    pipeline_depth,
                    hint,
                )
                if t < best_roller_time:
                    best_roller_time = t
                    best_roller_gflops = gflops
                    best_hint = hint
            except Exception as e:
                # Some hints might fail during compilation or execution
                print(f"    Hint failed with error: {e}")
                continue

        if best_hint:
            print(
                f"Best roller kernel: {best_roller_time:.6f} ms, {best_roller_gflops:.6f} GFLOPS"
            )
            print(f"Best hint: {best_hint}")
        else:
            print("All roller hints failed.")

        # 3. Store results
        results.append(
            {
                "M": m,
                "N": n,
                "K": k,
                "CuPy Time (ms)": baseline_time,
                "CuPy GFLOPS": baseline_gflops,
                "Roller Time (ms)": best_roller_time,
                "Roller GFLOPS": best_roller_gflops,
                "Speedup vs CuPy": (
                    baseline_time / best_roller_time
                    if best_roller_time > 0 and baseline_time != float("inf")
                    else 0
                ),
                "Best Hint": str(best_hint) if best_hint else "N/A",
            }
        )
        print("-" * 50)

    # 4. Print summary table
    df = pd.DataFrame(results)
    print("\n--- Benchmark Summary ---")
    print(df.to_string())

    # Save results to a file
    df.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to benchmark_results.csv")


if __name__ == "__main__":
    dims = [2048, 4096, 8192]
    test_shapes = list(itertools.product(dims, dims, dims))
    benchmark(test_shapes)
