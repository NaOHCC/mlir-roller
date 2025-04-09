from __future__ import annotations

# !pip install -q cupy-cuda12x==13.0.0
# !pip install -q mlir_python_bindings -f https://makslevental.github.io/wheels/
# !pip install -q git+https://github.com/makslevental/mlir-python-extras
import contextlib
import math

from typing import TypeVar

import cupy as cp
import iree.compiler.extras.types as T
import numpy as np
from cupy.cuda import Module
from iree.compiler.dialects import nvvm
from iree.compiler.extras.ast.canonicalize import canonicalize
from iree.compiler.extras.context import mlir_mod_ctx, MLIRContext
from iree.compiler.extras.dialects.ext import arith, gpu, linalg, memref, scf, vector
from iree.compiler.extras.dialects.ext.gpu import (
    block_dim,
    block_idx,
    get_compile_object_bytes,
    thread_idx,
)
from iree.compiler.extras.dialects.ext.memref import S
from iree.compiler.extras.dialects.ext.scf import range_
from iree.compiler.extras.runtime.passes import Pipeline, run_pipeline
from iree.compiler.extras.util import enable_debug as enable_debug, find_ops

"""# Helpers"""


def build_cuda_func(compiled_module, kernel_name="naive"):
    ptx = get_compile_object_bytes(compiled_module)
    mod = Module()
    mod.load(ptx)
    return mod.get_function(kernel_name)


def print_ptx(compiled_module):
    ptx = get_compile_object_bytes(compiled_module)
    print(ptx.decode())
    with open("log/sgemm.ptx", "w") as f:
        f.write(ptx.decode())


def compile_module(module, enable_ir_printing=False, print_ptx_=False):
    if enable_ir_printing:
        print_ptx_ = True
    print(module)
    with open("log/sgemm.mlir", "w") as f:
        f.write(str(module))
    mod = run_pipeline(
        module,
        # if you're not using vectors you can just uncomment the gpu-lower-to-nvvm-pipeline below
        Pipeline()
        .convert_linalg_to_loops()
        .convert_nvgpu_to_nvvm()
        .gpu_kernel_outlining()
        .convert_vector_to_scf()
        .convert_scf_to_cf()
        .convert_nvvm_to_llvm()
        .convert_func_to_llvm()
        .expand_strided_metadata()
        .add_pass(
            "nvvm-attach-target",
            **{
                "chip": "sm_70",
                "features": "+ptx76",
                "O": "2",
            },
        )
        .lower_affine()
        .convert_arith_to_llvm()
        .convert_index_to_llvm()
        .canonicalize()
        .cse()
        .Gpu(
            Pipeline()
            .strip_debuginfo()
            # TODO(max): upstream this (add to gpu pipeline)
            # vector.transfer
            .convert_vector_to_llvm()
            .convert_gpu_to_nvvm(use_bare_ptr_memref_call_conv=True)
            .canonicalize()
            .cse()
            .reconcile_unrealized_casts()
        )
        .gpu_to_llvm(use_bare_pointers_for_kernels=True)
        .gpu_module_to_binary(format="isa")
        .canonicalize()
        .cse()
        .reconcile_unrealized_casts()
        # .add_pass(
        #     "gpu-lower-to-nvvm-pipeline",
        #     # https://github.com/llvm/llvm-project/blob/ace69e6b942b8fa7e610d70be2a92e801ceea481/mlir/include/mlir/Dialect/GPU/Pipelines/Passes.h#L18
        #     **{
        #         "cubin-chip": "sm_80",
        #         "cubin-features": "+ptx83",
        #         "cubin-format": "isa",
        #         "kernel-bare-ptr-calling-convention": "1",
        #         "opt-level": "2",
        #         # "cubin-format": "fatbin",
        #         # "cubin-format": "bin",
        #     },
        # )
        ,
        enable_ir_printing=True,
    )

    if print_ptx_:
        print_ptx(mod)

    return mod


@contextlib.contextmanager
def time_cuda():
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()

    start_gpu.record()
    yield start_gpu, end_gpu
    end_gpu.record()
    end_gpu.synchronize()


def prepare_non_tiled_kernel(ctx: MLIRContext, kernel, M, K, N, BLOCK_SIZE=32):
    dtype = T.f32()
    npy_dtype = np.float32

    gpu.set_container_module(ctx.module)

    @gpu.module("matmul", ["#nvvm.target"])
    def matmul_mod():
        kernel[M, K, N, dtype, BLOCK_SIZE].emit()

    # print(ctx.module)
    # print(ctx.module.operation.verify())
    # exit()

    kernel_name = kernel.__name__
    compiled_module = compile_module(ctx.module)
    cuda_func = build_cuda_func(compiled_module, kernel_name)
    # print_ptx(compiled_module)

    grid_dims = (math.ceil(M / BLOCK_SIZE), math.ceil(N / BLOCK_SIZE))
    block_dims = (BLOCK_SIZE, BLOCK_SIZE)

    if "shared" in kernel_name:
        shared_mem = 2 * BLOCK_SIZE * BLOCK_SIZE * npy_dtype().nbytes
    else:
        shared_mem = 0

    return (
        cuda_func,
        grid_dims,
        block_dims,
        shared_mem,
        npy_dtype,
        "transpose_B" in kernel_name,
    )


def prepare_tiled_kernel(ctx: MLIRContext, kernel, M, K, N):
    dtype = T.f32()
    npy_dtype = np.float32
    kernel_name = kernel.__name__

    gpu.set_container_module(ctx.module)

    BK = 8
    TM = 8
    TN = 8
    if "2d" in kernel_name and M >= 128 and N >= 128:
        BM = 128
        BN = 128
    else:
        BM = 64
        BN = 64

    @gpu.module("matmul", ["#nvvm.target"])
    def matmul_mod():
        kernel[M, K, N, dtype, BM, BN, BK, TM, TN].emit()

    # print(ctx.module)
    # print(ctx.module.operation.verify())
    # exit()

    compiled_module = compile_module(ctx.module)
    cuda_func = build_cuda_func(compiled_module, kernel_name)
    # print_ptx(compiled_module)

    grid_dims = (math.ceil(N / BN), math.ceil(M / BM))
    if "2d" in kernel_name:
        block_dims = (BM // TM, BN // TN)
    else:
        block_dims = (BM // TM, BN)

    if "shared" in kernel_name:
        shared_mem = ((BM * BK) + (BK * BN)) * npy_dtype().nbytes
    else:
        shared_mem = 0

    return (
        cuda_func,
        grid_dims,
        block_dims,
        shared_mem,
        npy_dtype,
        False,
    )


def prepare_warp_tiled_kernel(ctx: MLIRContext, kernel, M, K, N):

    dtype = T.f32()
    npy_dtype = np.float32
    kernel_name = kernel.__name__
    gpu.set_container_module(ctx.module)

    # Settings for A100 (looks like it works for 3070 too?)
    NUM_THREADS = 128
    BN = 128
    BM = 64
    BK = 16
    WN = 64
    WM = 32
    WNITER = 1
    TN = 4
    TM = 4
    ctx.context.load_all_available_dialects()

    @gpu.module("matmul", ["#nvvm.target"])
    def matmul_mod():
        kernel[M, K, N, dtype, BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS].emit()

    # print(ctx.module)
    # print(ctx.module.operation.verify())
    # exit()

    compiled_module = compile_module(ctx.module, print_ptx_=True)
    cuda_func = build_cuda_func(compiled_module, kernel_name)
    # print_ptx(compiled_module)

    grid_dims = (math.ceil(N / BN), math.ceil(M / BM))
    block_dims = (NUM_THREADS,)
    shared_mem = ((BM * BK) + (BK * BN)) * npy_dtype().nbytes

    return (
        cuda_func,
        grid_dims,
        block_dims,
        shared_mem,
        npy_dtype,
        False,
    )


def run_eval(
    M,
    K,
    N,
    cuda_func,
    grid_dims,
    block_dims,
    shared_mem,
    npy_dtype,
    transpose_B,
    repeat_times=None,
):
    if repeat_times is None:
        repeat_times = 50

    A = np.random.randint(0, 10, (M, K)).astype(npy_dtype)
    B = np.random.randint(0, 10, (K, N)).astype(npy_dtype)
    C = np.zeros((M, N)).astype(npy_dtype)

    dA = cp.asarray(A)
    if transpose_B:
        dB = cp.asarray(np.ascontiguousarray(B.T))
    else:
        dB = cp.asarray(B)
    dC = cp.asarray(C)

    cuda_func(
        grid_dims,
        block_dims,
        (dA.data.ptr, dB.data.ptr, dC.data.ptr),
        shared_mem=shared_mem,
    )
    C = cp.asnumpy(dC)
    if not np.array_equal(C, A @ B + 1):
        print(A @ B + 1)
        print(C)
        assert False
    if repeat_times < 1:
        return

    with time_cuda() as (start_gpu, end_gpu):
        for _ in range(repeat_times):
            cuda_func(
                grid_dims,
                block_dims,
                (dA.data.ptr, dB.data.ptr, dC.data.ptr),
                shared_mem=shared_mem,
            )

    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)

    print(f"t={t_gpu / repeat_times:.6f} ms", end=" ")
    flops = 2 * M * N * K
    print(f"GFLOPS={repeat_times*flops * 1e-9 / (t_gpu/1000):.6f}")


"""# Simple Kernels"""

M, K, N, dtype, BLOCK_SIZE = list(map(TypeVar, ["M", "K", "N", "dtype", "BLOCK_SIZE"]))


@gpu.func(generics=[M, K, N, dtype])
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_naive(
    A: T.memref(M, K, dtype), B: T.memref(K, N, dtype), C: T.memref(M, N, dtype)
):
    one = arith.constant(1.0, type=dtype)
    tmp = arith.constant(0, type=dtype)

    r = block_dim.x * block_idx.x + thread_idx.x
    c = block_dim.y * block_idx.y + thread_idx.y

    for k, tmp, _ in range_(K, iter_args=[tmp]):
        tmp += A[r, k] * B[k, c]
        tmp = yield tmp
    C[r, c] = tmp + one


@gpu.func(generics=[M, K, N, dtype])
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_naive_row_order(
    A: T.memref(M, K, dtype), B: T.memref(K, N, dtype), C: T.memref(M, N, dtype)
):
    one = arith.constant(1.0, type=dtype)
    tmp = arith.constant(0, type=dtype)

    c = block_dim.x * block_idx.x + thread_idx.x
    r = block_dim.y * block_idx.y + thread_idx.y

    for k, tmp, _ in range_(K, iter_args=[tmp]):
        tmp += A[r, k] * B[k, c]
        tmp = yield tmp
    C[r, c] = tmp + one


@gpu.func(generics=[M, K, N, dtype, BLOCK_SIZE])
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_coalesce(
    A: T.memref(M, K, dtype), B: T.memref(K, N, dtype), C: T.memref(M, N, dtype)
):

    tid = gpu.thread_id()
    # this is actually floordiv
    r = block_idx.x * BLOCK_SIZE + (tid / BLOCK_SIZE)
    c = block_idx.y * BLOCK_SIZE + (tid % BLOCK_SIZE)
    # gpu.printf("tid: %ld: (%ld, %ld)\n", tid, r, c)

    one = arith.constant(1.0, type=dtype)
    tmp = arith.constant(0, type=dtype)

    for k, tmp, _ in range_(K, iter_args=[tmp]):
        tmp += A[r, k] * B[k, c]
        tmp = yield tmp
    C[r, c] = tmp + one


@gpu.func(generics=[M, K, N, dtype, BLOCK_SIZE])
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_coalesce_transpose_B(
    A: T.memref(M, K, dtype), B: T.memref(K, N, dtype), C: T.memref(M, N, dtype)
):

    tid = gpu.thread_id()
    r = block_idx.x * BLOCK_SIZE + (tid / BLOCK_SIZE)
    c = block_idx.y * BLOCK_SIZE + (tid % BLOCK_SIZE)

    one = arith.constant(1.0, type=dtype)
    tmp = arith.constant(0, type=dtype)

    for k, tmp, _ in range_(K, iter_args=[tmp]):
        tmp += A[r, k] * B[c, k]
        tmp = yield tmp
    C[r, c] = tmp + one


@gpu.func(generics=[M, K, N, dtype, BLOCK_SIZE])
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_shared_mem_block(
    A: T.memref(M, K, dtype), B: T.memref(K, N, dtype), C: T.memref(M, N, dtype)
):
    base = gpu.dynamic_shared_memory()
    A_shared = memref.view(base, (BLOCK_SIZE, BLOCK_SIZE), dtype=dtype)
    B_shared = memref.view(
        base, (BLOCK_SIZE, BLOCK_SIZE), dtype=dtype, shift=BLOCK_SIZE * BLOCK_SIZE
    )

    tid = gpu.thread_id()
    thread_row = tid / BLOCK_SIZE
    thread_col = tid % BLOCK_SIZE

    c_row = block_idx.x * BLOCK_SIZE
    c_col = block_idx.y * BLOCK_SIZE

    one = arith.constant(1.0, type=dtype)
    tmp = arith.constant(0, type=dtype)

    for bk_idx, tmp, _ in range_(0, K, BLOCK_SIZE, iter_args=[tmp]):
        A_ = A[c_row : c_row + BLOCK_SIZE, bk_idx : bk_idx + BLOCK_SIZE]
        B_ = B[bk_idx : bk_idx + BLOCK_SIZE, c_col : c_col + BLOCK_SIZE]

        A_shared[thread_row, thread_col] = A_[thread_row, thread_col]
        B_shared[thread_row, thread_col] = B_[thread_row, thread_col]

        gpu.barrier()

        for k, tmp, _ in range_(BLOCK_SIZE, iter_args=[tmp]):
            tmp += A_shared[thread_row, k] * B_shared[k, thread_col]
            tmp = yield tmp

        gpu.barrier()

        tmp = yield tmp

    C_ = C[c_row : c_row + BLOCK_SIZE, c_col : c_col + BLOCK_SIZE]
    C_[thread_row, thread_col] = tmp + one


"""# Tiled Kernels"""

BM, BN, BK, TM, TN = list(map(TypeVar, ["BM", "BN", "BK", "TM", "TN"]))


@gpu.func(generics=[M, K, N, dtype, BM, BN, BK, TM])
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_shared_mem_1d_block_tiling(
    A: T.memref(M, K, dtype), B: T.memref(K, N, dtype), C: T.memref(M, N, dtype)
):
    base = gpu.dynamic_shared_memory()
    A_shared = memref.view(base, (BM, BK), dtype=dtype)
    B_shared = memref.view(base, (BK, BN), dtype=dtype, shift=BM * BK)

    c_row = block_idx.y * BM
    c_col = block_idx.x * BN

    tid = gpu.thread_id()
    thread_col = tid % BN
    thread_row = tid / BN

    inner_col_A = tid % BK  # warp-level GMEM coalescing
    inner_row_A = tid / BK
    inner_col_B = tid % BN  # warp-level GMEM coalescing
    inner_row_B = tid / BN

    thread_results = memref.alloca((TM,), dtype)
    linalg.fill(0, thread_results)

    for bk_idx in range_(0, K, BK):
        # Move blocktile to beginning of A's row and B's column
        A_ = A[c_row : c_row + BM, bk_idx : bk_idx + BK]
        B_ = B[bk_idx : bk_idx + BK, c_col : c_col + BN]

        A_shared[inner_row_A, inner_col_A] = A_[inner_row_A, inner_col_A]
        B_shared[inner_row_B, inner_col_B] = B_[inner_row_B, inner_col_B]

        gpu.barrier()

        for dot_idx in range_(BK):
            tmp_B = B_shared[dot_idx, thread_col]
            for res_idx, tmp_B, _ in range_(TM, iter_args=[tmp_B]):
                thread_results[res_idx] += (
                    A_shared[thread_row * TM + res_idx, dot_idx] * tmp_B
                )
                yield tmp_B

        gpu.barrier()

    one = arith.constant(1.0, type=dtype)
    C_ = C[c_row : c_row + BM, c_col : c_col + BN]
    for res_idx in range_(TM):
        C_[thread_row * TM + res_idx, thread_col] = thread_results[res_idx] + one


@gpu.func(generics=[M, K, N, dtype, BM, BN, BK, TM, TN])
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_shared_mem_2d_block_tiling(
    A: T.memref(M, K, dtype), B: T.memref(K, N, dtype), C: T.memref(M, N, dtype)
):
    base = gpu.dynamic_shared_memory()
    A_shared = memref.view(base, (BM, BK), dtype=dtype)
    B_shared = memref.view(base, (BK, BN), dtype=dtype, shift=BM * BK)

    c_row = block_idx.y * BM
    c_col = block_idx.x * BN

    total_results_blocktile = BM * BN
    num_threads_blocktile = total_results_blocktile // (TM * TN)

    tid = gpu.thread_id()
    # BN/TN are the number of threads to span a column
    thread_col = tid % (BN // TN)
    thread_row = tid / (BN // TN)

    inner_col_A = tid % BK  # warp-level GMEM coalescing
    inner_row_A = tid / BK
    stride_A = num_threads_blocktile // BK

    inner_col_B = tid % BN  # warp-level GMEM coalescing
    inner_row_B = tid / BN
    stride_B = num_threads_blocktile // BN

    thread_results = memref.alloca((TM, TN), dtype)
    linalg.fill(0, thread_results)

    reg_M = memref.alloca((TM,), dtype)
    linalg.fill(0, reg_M)

    reg_N = memref.alloca((TN,), dtype)
    linalg.fill(0, reg_N)

    for bk_idx in range_(0, K, BK):
        A_ = A[c_row : c_row + BM, bk_idx : bk_idx + BK]
        B_ = B[bk_idx : bk_idx + BK, c_col : c_col + BN]

        for load_offset in range_(0, BM, stride_A):
            A_shared[inner_row_A + load_offset, inner_col_A] = A_[
                inner_row_A + load_offset, inner_col_A
            ]
        for load_offset in range_(0, BK, stride_B):
            B_shared[inner_row_B + load_offset, inner_col_B] = B_[
                inner_row_B + load_offset, inner_col_B
            ]

        gpu.barrier()

        for dot_idx in range_(BK):
            for i in range_(TM):
                reg_M[i] = A_shared[thread_row * TM + i, dot_idx]
            for i in range_(TN):
                reg_N[i] = B_shared[dot_idx, thread_col * TN + i]

            for res_idx_m in range_(TM):
                for res_idx_n in range_(TN):
                    thread_results[res_idx_m, res_idx_n] += (
                        reg_M[res_idx_m] * reg_N[res_idx_n]
                    )

        gpu.barrier()

    one = arith.constant(1.0, type=dtype)
    C_ = C[c_row : c_row + BM, c_col : c_col + BN]

    for res_idx_m in range_(TM):
        for res_idx_n in range_(TN):
            C_[thread_row * TM + res_idx_m, thread_col * TN + res_idx_n] = (
                thread_results[res_idx_m, res_idx_n] + one
            )


@gpu.func(generics=[M, K, N, dtype, BM, BN, BK, TM, TN])
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_shared_mem_2d_block_tiling_vectorize(
    A: T.memref(M, K, dtype), B: T.memref(K, N, dtype), C: T.memref(M, N, dtype)
):
    VECTOR_WIDTH = 4
    DTYPE_WIDTH = dtype.width // 8

    memref.assume_alignment(A, VECTOR_WIDTH * DTYPE_WIDTH)
    memref.assume_alignment(B, VECTOR_WIDTH * DTYPE_WIDTH)
    memref.assume_alignment(C, VECTOR_WIDTH * DTYPE_WIDTH)

    base = gpu.dynamic_shared_memory()
    base = memref.memory_space_cast(T.memref(S, element_type=T.i8()), base)

    # transpose A
    A_shared = memref.view(base, (BK, BM), dtype=dtype)
    B_shared = memref.view(base, (BK, BN), dtype=dtype, shift=BM * BK)

    c_row = block_idx.y * BM
    c_col = block_idx.x * BN

    tid = gpu.thread_id()
    # BN/TN are the number of threads to span a column
    thread_col = tid % (BN // TN)
    thread_row = tid / (BN // TN)

    # calculating the indices that this thread will load into SMEM
    # we'll load 128bit / 32bit = 4 elements per thread at each step
    inner_col_A = tid % (BK // VECTOR_WIDTH)  # warp-level GMEM coalescing
    inner_row_A = tid / (BK // VECTOR_WIDTH)
    inner_col_B = tid % (BN // VECTOR_WIDTH)  # warp-level GMEM coalescing
    inner_row_B = tid / (BN // VECTOR_WIDTH)

    thread_results = memref.alloca((TM, TN), dtype)
    linalg.fill(0, thread_results)

    reg_M = memref.alloca((TM,), dtype)
    linalg.fill(0, reg_M)

    reg_N = memref.alloca((TN,), dtype)
    linalg.fill(0, reg_N)

    for bk_idx in range_(0, K, BK):
        A_ = A[c_row : c_row + BM, bk_idx : bk_idx + BK]
        B_ = B[bk_idx : bk_idx + BK, c_col : c_col + BN]

        A_vec = vector.load(
            T.vector(VECTOR_WIDTH, dtype), A_, [inner_row_A, inner_col_A * VECTOR_WIDTH]
        )
        for j in range(VECTOR_WIDTH):
            #  transpose A while loading it
            A_shared[inner_col_A * VECTOR_WIDTH + j, inner_row_A] = A_vec[j]

        B_vec = vector.load(
            T.vector(VECTOR_WIDTH, dtype), B_, [inner_row_B, inner_col_B * VECTOR_WIDTH]
        )
        vector.store(B_vec, B_shared, [inner_row_B, inner_col_B * VECTOR_WIDTH])

        gpu.barrier()

        for dot_idx in range_(BK):
            for i in range_(TM):
                reg_M[i] = A_shared[dot_idx, thread_row * TM + i]

            for i in range_(TN):
                reg_N[i] = B_shared[dot_idx, thread_col * TN + i]

            for res_idx_m in range_(TM):
                for res_idx_n in range_(TN):
                    thread_results[res_idx_m, res_idx_n] += (
                        reg_M[res_idx_m] * reg_N[res_idx_n]
                    )

        gpu.barrier()

    one = arith.constant(1.0, type=dtype)
    C_ = C[c_row : c_row + BM, c_col : c_col + BN]

    for res_idx_m in range_(TM):
        for res_idx_n in range_(0, TN, VECTOR_WIDTH):
            tmp = vector.load(
                T.vector(VECTOR_WIDTH, dtype),
                C_,
                [thread_row * TM + res_idx_m, thread_col * TN + res_idx_n],
            )
            for j in range(VECTOR_WIDTH):
                tmp[j] = thread_results[res_idx_m, res_idx_n + j] + one
            vector.store(
                tmp, C_, [thread_row * TM + res_idx_m, thread_col * TN + res_idx_n]
            )


WM, WN, WNITER, NUM_THREADS = list(map(TypeVar, ["WM", "WN", "WNITER", "NUM_THREADS"]))
WARP_SIZE = 32


@gpu.func(generics=[M, K, N, dtype, BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS])
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_warp_tiling(
    A: T.memref(M, K, dtype), B: T.memref(K, N, dtype), C: T.memref(M, N, dtype)
):
    VECTOR_WIDTH = 4
    DTYPE_WIDTH = dtype.width // 8

    tid = gpu.thread_id()

    memref.assume_alignment(A, VECTOR_WIDTH * DTYPE_WIDTH)
    memref.assume_alignment(B, VECTOR_WIDTH * DTYPE_WIDTH)
    memref.assume_alignment(C, VECTOR_WIDTH * DTYPE_WIDTH)

    base = gpu.dynamic_shared_memory()
    base = memref.memory_space_cast(T.memref(S, element_type=T.i8()), base)

    # transpose A
    A_shared = memref.view(base, (BK, BM), dtype=dtype)
    B_shared = memref.view(base, (BK, BN), dtype=dtype, shift=BM * BK)

    c_row = block_idx.y * BM
    c_col = block_idx.x * BN

    # Placement of the warp in the threadblock tile
    warp_idx = tid / WARP_SIZE
    warp_row = warp_idx / (BN // WN)
    warp_col = warp_idx % (BN // WN)

    # size of the warp subtile
    WMITER = (WM * WN) // (WARP_SIZE * TM * TN * WNITER)
    WSUBM = WM // WMITER
    WSUBN = WN // WNITER

    # Placement of the thread in the warp subtile
    thread_idx_in_warp = tid % WARP_SIZE
    thread_col_in_warp = thread_idx_in_warp % (WSUBN // TN)
    thread_row_in_warp = thread_idx_in_warp / (WSUBN // TN)

    # calculating the indices that this thread will load into SMEM
    # we'll load 128bit / 32bit = 4 elements per thread at each step
    inner_row_A = tid / (BK // VECTOR_WIDTH)
    inner_col_A = tid % (BK // VECTOR_WIDTH)
    row_stride_A = (NUM_THREADS * VECTOR_WIDTH) // BK
    inner_row_B = tid / (BN // VECTOR_WIDTH)
    inner_col_B = tid % (BN // VECTOR_WIDTH)
    row_stride_B = NUM_THREADS // (BN // VECTOR_WIDTH)

    # allocate thread-local cache for results in registerfile
    thread_results = memref.alloca((WMITER * TM, WNITER * TN), dtype)
    linalg.fill(0, thread_results)

    reg_M = memref.alloca((WMITER, TM), dtype)
    linalg.fill(0, reg_M)

    reg_N = memref.alloca((WNITER, TN), dtype)
    linalg.fill(0, reg_N)

    for bk_idx in range_(0, K, BK):
        A_ = A[c_row : c_row + BM, bk_idx : bk_idx + BK]
        B_ = B[bk_idx : bk_idx + BK, c_col : c_col + BN]

        for offset in range(0, BM - row_stride_A + 1, row_stride_A):
            A_vec = vector.load(
                T.vector(VECTOR_WIDTH, dtype),
                A_,
                [inner_row_A + offset, inner_col_A * VECTOR_WIDTH],
            )
            for j in range(VECTOR_WIDTH):
                #  transpose A while loading it
                A_shared[inner_col_A * VECTOR_WIDTH + j, inner_row_A + offset] = A_vec[
                    j
                ]

        for offset in range(0, BK - row_stride_B + 1, row_stride_B):
            B_vec = vector.load(
                T.vector(VECTOR_WIDTH, dtype),
                B_,
                [inner_row_B + offset, inner_col_B * VECTOR_WIDTH],
            )
            vector.store(
                B_vec, B_shared, [inner_row_B + offset, inner_col_B * VECTOR_WIDTH]
            )

        gpu.barrier()

        for dot_idx in range_(BK):
            for w_sub_row_idx in range_(WMITER):
                for i in range_(TM):
                    reg_M[w_sub_row_idx, i] = A_shared[
                        dot_idx,
                        warp_row * WM
                        + w_sub_row_idx * WSUBM
                        + thread_row_in_warp * TM
                        + i,
                    ]

            for w_sub_col_idx in range_(WNITER):
                for i in range_(TN):
                    reg_N[w_sub_col_idx, i] = B_shared[
                        dot_idx,
                        warp_col * WN
                        + w_sub_col_idx * WSUBN
                        + thread_col_in_warp * TN
                        + i,
                    ]

            for w_sub_row_idx in range_(WMITER):
                for w_sub_col_idx in range_(WNITER):
                    for res_idx_m in range_(TM):
                        for res_idx_n in range_(TN):
                            thread_results[
                                w_sub_row_idx * TM + res_idx_m,
                                w_sub_col_idx * TN + res_idx_n,
                            ] += (
                                reg_M[w_sub_row_idx, res_idx_m]
                                * reg_N[w_sub_col_idx, res_idx_n]
                            )

        gpu.barrier()

    one = arith.constant(1.0, type=dtype)

    for w_sub_row_idx in range_(WMITER):
        for w_sub_col_idx in range_(WNITER):
            r = c_row + warp_row * WM + w_sub_row_idx * WSUBM
            c = c_col + warp_col * WN + w_sub_col_idx * WSUBN
            C_ = C[r : r + WSUBM, c : c + WSUBN]
            for res_idx_m in range_(TM):
                for res_idx_n in range_(0, TN, VECTOR_WIDTH):
                    tmp = vector.load(
                        T.vector(VECTOR_WIDTH, dtype),
                        C_,
                        [
                            thread_row_in_warp * TM + res_idx_m,
                            thread_col_in_warp * TN + res_idx_n,
                        ],
                    )
                    for j in range(VECTOR_WIDTH):
                        tmp[j] = (
                            thread_results[
                                w_sub_row_idx * TM + res_idx_m,
                                w_sub_col_idx * TN + res_idx_n + j,
                            ]
                            + one
                        )
                    vector.store(
                        tmp,
                        C_,
                        [
                            thread_row_in_warp * TM + res_idx_m,
                            thread_col_in_warp * TN + res_idx_n,
                        ],
                    )


"""# Compile and run and time"""

# sizes = [128, 256, 512, 1024, 2**11, 2**12]
sizes = [1024]
repeats = None

# for k in [
#     sgemm_naive,
#     sgemm_naive_row_order,
#     sgemm_coalesce,
#     sgemm_coalesce_transpose_B,
#     sgemm_shared_mem_block,
# ]:
#     print(f"\n{k.__name__}")
#     for s in sizes:
#         with (
#             mlir_mod_ctx() as ctx,
#             # enable_debug()
#         ):
#             print(f"{s=}", end=" ")
#             cuda_func, grid_dims, block_dims, shared_mem, npy_dtype, transpose_B = (
#                 prepare_non_tiled_kernel(ctx, k, s, s, s)
#             )
#             run_eval(
#                 s,
#                 s,
#                 s,
#                 cuda_func,
#                 grid_dims,
#                 block_dims,
#                 shared_mem,
#                 npy_dtype,
#                 transpose_B,
#             )


for k in [
    # sgemm_shared_mem_1d_block_tiling,
    # sgemm_shared_mem_2d_block_tiling,
    sgemm_shared_mem_2d_block_tiling_vectorize,
]:
    print(f"\n{k.__name__}")
    for s in sizes:
        with (
            mlir_mod_ctx() as ctx,
            # enable_debug()
        ):
            print(f"{s=}", end=" ")
            cuda_func, grid_dims, block_dims, shared_mem, npy_dtype, transpose_B = (
                prepare_tiled_kernel(ctx, k, s, s, s)
            )
            run_eval(
                s,
                s,
                s,
                cuda_func,
                grid_dims,
                block_dims,
                shared_mem,
                npy_dtype,
                transpose_B,
            )

# print(f"\n{sgemm_warp_tiling.__name__}")
# for s in sizes:
#     with (
#         mlir_mod_ctx(allow_unregistered_dialects=True) as ctx,
#         # enable_debug()
#     ):
#         print(f"{s=}", end=" ")
#         ctx.context.enable_multithreading(False)
#         cuda_func, grid_dims, block_dims, shared_mem, npy_dtype, transpose_B = (
#             prepare_warp_tiled_kernel(ctx, sgemm_warp_tiling, s, s, s)
#         )
#         run_eval(
#             s,
#             s,
#             s,
#             cuda_func,
#             grid_dims,
#             block_dims,
#             shared_mem,
#             npy_dtype,
#             transpose_B,
#         )
