import gc
from typing import Optional, TypeVar

from iree.compiler.ir import *
import argparse
import math
from contextlib import contextmanager
from pathlib import Path

import cupy as cp
import iree.compiler.extras.types as T
import numpy as np
from analyze import get_roller_hints, MatmulConfig, TileDict
from arch import CUDA

from iree.compiler.dialects import builtin, iree_codegen, iree_transform, scf, transform
from iree.compiler.dialects.bufferization import LayoutMapOption
from iree.compiler.dialects.transform import (
    any_op_t,
    apply_licm,
    bufferization,
    gpu as gpu_transform,
    loop,
    structured,
    vector,
)
from iree.compiler.dialects.transform.extras import apply_patterns, named_sequence
from iree.compiler.dialects.transform.vector import (
    VectorContractLowering,
    VectorMultiReductionLowering,
    VectorTransferSplit,
    VectorTransposeLowering,
)
from iree.compiler.extras.ast.canonicalize import canonicalize
from iree.compiler.extras.context import (
    ExplicitlyManagedModule,
    mlir_mod_ctx,
    MLIRContext,
    RAIIMLIRContext,
    RAIIMLIRContextModule,
)
from iree.compiler.extras.dialects.ext import (
    arith,
    func,
    gpu,
    linalg,
    memref,
    transform as ext_transform,
)
from iree.compiler.extras.dialects.ext.gpu import (
    block_dim,
    block_idx,
    get_compile_object_bytes,
    thread_idx,
)
from iree.compiler.extras.dialects.ext.memref import S
from iree.compiler.extras.dialects.ext.transform import (
    get_parent_op,
    match,
    tile_to_scf_for,
    transform_any_op_t,
)
from iree.compiler.extras.runtime.passes import Pipeline, run_pipeline
from iree.compiler.extras.runtime.refbackend import LLVMJITBackend
from iree.compiler.extras.util import enable_debug as enable_debug, find_ops
from utils import get_block_dims

parser = argparse.ArgumentParser()
parser.add_argument("--profile", action="store_true", help="profile")


class GEMMBuilder:
    def __init__(self):
        pass

    def build_gemm(self, MM: int, NN: int, KK: int):
        generics = M, N, K, dtype = list(map(TypeVar, ["M", "N", "K", "dtype"]))

        @func.func(generics=generics)
        def gemm(
            A: "T.memref(M, K, dtype)",
            B: "T.memref(K, N, dtype)",
            C: "T.memref(M, N, dtype)",
        ):
            VECTOR_WIDTH = 4
            DTYPE_WIDTH = dtype.width // 8

            memref.assume_alignment(A, VECTOR_WIDTH * DTYPE_WIDTH)
            memref.assume_alignment(B, VECTOR_WIDTH * DTYPE_WIDTH)
            memref.assume_alignment(C, VECTOR_WIDTH * DTYPE_WIDTH)
            linalg.matmul(A, B, C)

        dtype_ = T.f32()

        @builtin.module(attrs={"transform.target_tag": StringAttr.get("payload")})
        def payload():
            gemm[MM, NN, KK, dtype_].emit()

    def get_index_attr(self, val: int) -> IntegerAttr:
        return IntegerAttr.get(IndexType.get(), val)

    def get_i64_attr(self, val: int) -> IntegerAttr:
        return IntegerAttr.get(IntegerType.get_signless(64), val)

    def clean(self, target, op_name="func.func"):
        funcOp = structured.MatchOp.match_op_names(target, op_name)
        with InsertionPoint(transform.ApplyPatternsOp(funcOp).patterns):
            transform.ApplyCanonicalizationPatternsOp()
            structured.ApplyTilingCanonicalizationPatternsOp()

        transform.ApplyCommonSubexpressionEliminationOp(funcOp)

    @contextmanager
    def print_op_context(self, target: Value, hint: str):
        yield
        return
        transform.PrintOp(target=target, name=f"before {hint}")
        try:
            yield
        finally:
            transform.PrintOp(target=target, name=f"after {hint}")

    def build_transform_pipeline(
        self,
        block_tile: list[int],
        thread_tile: list[int],
        rstep: int,
        pipeline_depth: int,
    ):
        block_tile = block_tile + [0]
        thread_tile = thread_tile + [0]
        reduce_tile = [0, 0, rstep]

        @builtin.module(attrs={"transform.with_named_sequence": UnitAttr.get()})
        def mod_transform():
            @named_sequence("main", [any_op_t()], [])
            def main(module_op: any_op_t()):
                smem_space_str = "#gpu.memory_space<workgroup>"  # HACK: what different between memory_space and address_space?
                local_space_str = "#gpu.memory_space<private>"
                # smem_space_str = "#gpu.address_space<workgroup>"
                # reg_space_str = "#gpu.address_space<private>"
                smem_space = ArrayAttr.get([Attribute.parse(smem_space_str)])
                local_space = ArrayAttr.get([Attribute.parse(local_space_str)])
                block_dims = get_block_dims(block_tile, thread_tile)
                CopyToWorkgroupMemoryMarker = {
                    "key": "__internal_linalg_transform__",
                    "value": StringAttr.get("copy_to_workgroup_memory"),
                }
                PipelineMarker = {
                    "key": "__my_pipeline__",
                    "value": StringAttr.get("__my_pipeline__"),
                }

                pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
                    iree_codegen.DispatchLoweringPassPipeline.None_
                )
                TranslationInfo = {
                    "key": "translation_info",
                    "value": iree_codegen.TranslationInfoAttr.get(
                        pipeline_attr,
                        None,
                        block_dims,
                        None,
                        DictAttr.get(
                            {
                                "pipeline": self.get_i64_attr(3),
                                "store_stage": self.get_i64_attr(1),
                            }
                        ),
                    ),
                }

                block_mapping_attr = Attribute.parse("[ #gpu.block<y>, #gpu.block<x> ]")
                thread_mapping_attr = Attribute.parse(
                    "[ #gpu.thread<y>, #gpu.thread<x> ]"
                )

                matmul = match(module_op, ops=["linalg.matmul"])

                with self.print_op_context(module_op, "tile level 2"):
                    level_2_op = structured.TileUsingForallOp(
                        # transform.OperationType.get("linalg.generic"),  # tiled_op_type
                        # transform.OperationType.get("scf.forall"),  # loops_type
                        matmul,
                        tile_sizes=block_tile,
                        mapping=block_mapping_attr,
                    )

                with self.print_op_context(module_op, "map forall to blocks"):
                    gpu_launch_op = gpu_transform.MapForallToBlocks(
                        match(module_op, ops=["func.func"]), generate_gpu_launch=True
                    )
                    self.clean(module_op, "gpu.func")

                with self.print_op_context(module_op, "tile k"):
                    redution_tiled = structured.TileUsingForOp(
                        match(module_op, ["linalg.matmul"]),
                        sizes=reduce_tile,
                    )

                    iree_transform.AddAttrbuiteOp(
                        [redution_tiled.loops[0]], **PipelineMarker
                    )

                with self.print_op_context(module_op, "promote A and B"):
                    input_promoted = structured.PromoteOp(
                        transformed=transform.AnyOpType.get(),
                        target=redution_tiled.tiled_linalg_op,
                        operands_to_promote=[0, 1],
                        mapping=smem_space,
                        use_alloca=True,
                    )

                with self.print_op_context(module_op, "tile level 1"):
                    level_1_op = structured.TileUsingForallOp(
                        match(module_op, ops=["linalg.matmul"]),  # tiled_op_type
                        match(module_op, ops=["scf.forall"]),  # loops_type
                        match(module_op, ops=["linalg.matmul"]),
                        # num_threads=[2, 4, 4],
                        tile_sizes=thread_tile,
                        mapping=thread_mapping_attr,
                    )

                # mark copy to workgroup memory
                memref_copy = match(module_op, ops=["memref.copy"])
                iree_transform.AddAttrbuiteOp(
                    [memref_copy], **CopyToWorkgroupMemoryMarker
                )

                # map gpu hierarchy
                self.clean(module_op, "func.func")
                transform.ApplyRegisteredPassOp(
                    any_op_t(),
                    match(module_op, ["func.func"]),
                    "iree-codegen-gpu-multi-buffering",
                    options=f"num-buffers={pipeline_depth}",
                )

                with self.print_op_context(module_op, "map nested forall to threads"):
                    gpu_launch_op = gpu_transform.MapNestedForallToThreads(
                        match(module_op, ops=["gpu.launch"]), block_dims=block_dims
                    )
                    transform.ApplyRegisteredPassOp(
                        any_op_t(),
                        match(module_op, ops=["func.func"]),
                        "iree-codegen-memrefcopy-to-linalg",
                    )
                    self.clean(module_op, "gpu.func")

                with self.print_op_context(
                    module_op, "gpu-launch-sink-index-computations"
                ):
                    sunk = transform.ApplyRegisteredPassOp(
                        any_op_t(),
                        match(module_op, ["func.func"]),
                        "gpu-launch-sink-index-computations",
                    )

                # gpu-kernel-outlining
                # with self.print_op_context(module_op, "gpu-kernel-outlining"):
                outlined = transform.ApplyRegisteredPassOp(
                    any_op_t(), module_op, "gpu-kernel-outlining"
                )
                module_op = outlined.result
                self.clean(module_op, "gpu.func")

                # DistributeSharedMemoryCopy
                with self.print_op_context(module_op, "distribute shared memory copy"):
                    gpu_module = match(module_op, ["gpu.module"])
                    iree_transform.AddAttrbuiteOp(
                        [match(module_op, ["gpu.func"])], **TranslationInfo
                    )
                    iree_transform.GpuDistributeSharedMemoryCopyOp(
                        match(module_op, ["gpu.func"])
                    )
                    self.clean(module_op, "gpu.func")

                # reduce bank conflicts
                # iree_transform.ReduceSharedMemoryBankConflictsOp(match(module_op, ["gpu.func"]))

                # vectorize children
                # with self.print_op_context(match(module_op, ["gpu.func"]), "vectorize children"):
                structured.VectorizeChildrenAndApplyPatternsOp(
                    match(module_op, ["gpu.func"])
                )

                # fix by https://github.com/iree-org/iree/pull/15192/
                @apply_patterns(match(module_op, ["gpu.func"]))
                def pats():
                    transform.memref.ApplyFoldMemrefAliasOpsPatternsOp()

                transform.ApplyCommonSubexpressionEliminationOp(
                    match(module_op, ["gpu.func"])
                )

                structured.HoistRedundantVectorTransfersOp(
                    any_op_t(), match(module_op, ["gpu.func"])
                )

                with self.print_op_context(module_op, "pipeline"):

                    @apply_patterns(match(module_op, ["gpu.func"]))
                    def pats():
                        vector.ApplyCastAwayVectorLeadingOneDimPatternsOp()

                    iree_transform.CreateAsyncGroupsOp(
                        match(module_op, ["gpu.func"]), use_mma_sync=True
                    )

                    transform.ApplyRegisteredPassOp(
                        any_op_t(),
                        match(module_op, ["gpu.func"]),
                        "iree-codegen-gpu-pipelining",
                        options=f"pipeline-depth={pipeline_depth}",
                    )
                self.clean(module_op, "gpu.func")

                @apply_patterns(match(module_op, ["gpu.func"]))
                def pats():
                    transform.memref.ApplyFoldMemrefAliasOpsPatternsOp()

                transform.ApplyCommonSubexpressionEliminationOp(
                    match(module_op, ["gpu.func"])
                )

                self.clean(module_op, "gpu.func")
                with self.print_op_context(
                    match(module_op, ["gpu.func"]), "lowering contraction"
                ):

                    @apply_patterns(match(module_op, ["gpu.func"]))
                    def pats():
                        vector.ApplyLowerContractionPatternsOp(
                            lowering_strategy=vector.VectorContractLowering.OuterProduct
                        )
                        vector.ApplyTransferPermutationPatternsOp()
                        vector.ApplyLowerMultiReductionPatternsOp(
                            lowering_strategy=VectorMultiReductionLowering.InnerParallel
                        )
                        vector.ApplySplitTransferFullPartialPatternsOp(
                            split_transfer_strategy=VectorTransferSplit.LinalgCopy
                        )
                        # vector.ApplyTransferToScfPatternsOp(
                        #     max_transfer_rank=1,
                        #     # full_unroll=True,
                        # )
                        # vector.ApplyLowerTransferPatternsOp(max_transfer_rank=1)
                        # vector.ApplyLowerShapeCastPatternsOp()
                        vector.ApplyLowerTransposePatternsOp(
                            lowering_strategy=VectorTransposeLowering.Shuffle1D
                        )

                all_loops = match(
                    module_op, interface=structured.MatchInterfaceEnum.LoopLikeInterface
                )
                transform.apply_licm(all_loops)

                self.clean(module_op, "gpu.func")
                # transform.ApplyRegisteredPassOp(
                #     any_op_t(), match(module_op, ["gpu.func"]), "iree-llvmgpu-vector-lowering"
                # )
                # transform.ApplyRegisteredPassOp(
                #     any_op_t(),
                #     match(module_op, ["gpu.func"]),
                #     "nvgpu-optimize-shared-memory",
                # )

                transform.ApplyRegisteredPassOp(
                    any_op_t(),
                    match(module_op, ["gpu.func"]),
                    "iree-llvmgpu-convert-static-shared-memory-alloc",
                )

                transform.ApplyRegisteredPassOp(
                    any_op_t(),
                    match(module_op, ["gpu.func"]),
                    "iree-llvmgpu-assume-argument-alignment",
                    options="alignment=128",
                )

                @apply_patterns(match(module_op, ["gpu.func"]))
                def pats():
                    transform.memref.ApplyFoldMemrefAliasOpsPatternsOp()

                transform.ApplyCommonSubexpressionEliminationOp(
                    match(module_op, ["gpu.func"])
                )

                transform.ApplyRegisteredPassOp(
                    any_op_t(),
                    match(module_op, ["gpu.func"]),
                    "iree-llvmgpu-thread-block-swizzle",
                    options="panel-width=4",
                )
                self.clean(module_op, "gpu.func")

                return


class Lower:
    @staticmethod
    def run_transform(module, enable_ir_printing=True):
        return run_pipeline(
            module,
            pipeline=Pipeline().transform_interpreter(
                entry_point="main", debug_payload_root_tag="payload"
            ),
            enable_ir_printing=enable_ir_printing,
        )

    @staticmethod
    def lower_to_llvm(module, need_find: bool = True, enable_ir_printing: bool = False):
        passes = (
            Pipeline()
            .Gpu(
                Pipeline().Nested(
                    "gpu.func", Pipeline().add_pass("iree-llvmgpu-vector-lowering")
                )
            )
            .convert_linalg_to_loops()
            .convert_nvgpu_to_nvvm()
            # .gpu_kernel_outlining()
            .convert_vector_to_scf()
            .convert_scf_to_cf()
            .convert_nvvm_to_llvm()
            .convert_func_to_llvm()
            .expand_strided_metadata()
            .add_pass(
                "nvvm-attach-target",
                **{
                    "chip": "sm_80",
                    "features": "+ptx80",
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
                # .strip_debuginfo()
                # TODO(max): upstream this (add to gpu pipeline)
                # vector.transfer
                .convert_vector_to_scf()
                .convert_gpu_to_nvvm(use_bare_ptr_memref_call_conv=True)
                .convert_vector_to_llvm()
                .canonicalize()
                .cse()
                .reconcile_unrealized_casts()
            )
            # .add_pass("iree-convert-to-nvvm")
            .gpu_to_llvm(use_bare_pointers_for_kernels=True)
            .gpu_module_to_binary(format="isa")
            .canonicalize()
            .cse()
            .reconcile_unrealized_casts()
        )
        op = (
            find_ops(
                module.operation,
                lambda x: "transform.target_tag" in x.attributes
                and x.attributes["transform.target_tag"].value == "payload",
                single=True,
            )
            if need_find
            else module.operation
        )
        mod = run_pipeline(
            op,
            passes,
            enable_ir_printing=enable_ir_printing,
        )
        return mod

    @staticmethod
    def lower_module(
        module,
        *,
        enable_transform_printing=False,
        enable_llvm_printing=False,
        dump_transformed_path: Optional[Path] = None,
        dump_llvm_path: Optional[Path] = None,
    ):
        """
        Lower the module to LLVM and return the lowered module.
        """
        mod = Lower.run_transform(module, enable_ir_printing=enable_transform_printing)
        if dump_transformed_path:
            dump_transformed_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dump_transformed_path, "w") as f:
                f.write(
                    find_ops(
                        mod.operation,
                        lambda x: "transform.target_tag" in x.attributes
                        and x.attributes["transform.target_tag"].value == "payload",
                        single=True,
                    ).__str__()
                )
        mod = Lower.lower_to_llvm(mod, enable_ir_printing=enable_llvm_printing)
        if dump_llvm_path:
            dump_llvm_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dump_llvm_path, "w") as f:
                f.write(mod.__str__())
        return mod

    @staticmethod
    def prepare_kernel(module, shared_memory_usage, kernel_name="gemm_kernel"):
        def build_cuda_func(compiled_module, kernel_name):
            ptx = get_compile_object_bytes(compiled_module)
            with open("log/kernel.ptx", "w") as f:
                f.write(str(ptx.decode()))
            mod = cp.cuda.Module()
            mod.load(ptx)
            return mod.get_function(kernel_name)

        cuda_func = build_cuda_func(module, kernel_name)
        attr = cp.cuda.driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        cp.cuda.driver.funcSetAttribute(cuda_func.ptr, attr, shared_memory_usage)
        return cuda_func


@contextmanager
def time_cuda():
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()

    start_gpu.record()
    yield start_gpu, end_gpu
    end_gpu.record()
    end_gpu.synchronize()


def run_eval(
    cuda_func,
    pipeline_depth,
    hint: TileDict,
    repeat_times=50,
    profile=False,
) -> float:
    M, N, K = hint.matmul_config.M, hint.matmul_config.N, hint.matmul_config.K
    dA = cp.random.rand(M, K, dtype=cp.float32)
    dB = cp.random.rand(K, N, dtype=cp.float32)
    dC = cp.zeros((M, N), dtype=cp.float32)

    grid_dims = (N // hint.block_tile[1], M // hint.block_tile[0])

    block_dims = (
        hint.block_tile[1] // hint.thread_tile[1],
        hint.block_tile[0] // hint.thread_tile[0],
    )

    shared_memory_usage = hint.smem_cost * pipeline_depth

    cuda_func(
        grid_dims,
        block_dims,
        (dA.data.ptr, dB.data.ptr, dC.data.ptr),
        shared_mem=shared_memory_usage,
    )

    if not cp.allclose(dC, dA @ dB):
        print(dA @ dB)
        print(dC)

    if profile:
        return -1

    for _ in range(10):
        cuda_func(
            grid_dims,
            block_dims,
            (dA.data.ptr, dB.data.ptr, dC.data.ptr),
            shared_mem=shared_memory_usage,
        )
    with time_cuda() as (start_gpu, end_gpu):
        for _ in range(repeat_times):
            cuda_func(
                grid_dims,
                block_dims,
                (dA.data.ptr, dB.data.ptr, dC.data.ptr),
                shared_mem=shared_memory_usage,
            )

    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)

    print(f"t={t_gpu / repeat_times:.6f} ms", end=" ")
    flops = 2 * M * N * K
    print(f"GFLOPS={repeat_times*flops * 1e-9 / (t_gpu/1000):.6f}")
    return t_gpu / repeat_times


def run_baseline(config: MatmulConfig, repeat_times=50):
    M, N, K = config.M, config.N, config.K
    dA = cp.random.rand(M, K, dtype=cp.float32)
    dB = cp.random.rand(K, N, dtype=cp.float32)
    dC = cp.zeros((M, N), dtype=cp.float32)

    for _ in range(10):
        cp.matmul(dA, dB, out=dC)
    with time_cuda() as (start_gpu, end_gpu):
        for _ in range(repeat_times):
            cp.matmul(dA, dB, out=dC)

    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)

    print(f"t={t_gpu / repeat_times:.6f} ms", end=" ")
    flops = 2 * M * N * K
    print(f"GFLOPS={repeat_times*flops * 1e-9 / (t_gpu/1000):.6f}")


if __name__ == "__main__":
    args = parser.parse_args()
    S = 4096
    config = MatmulConfig(M=S, N=S, K=S, el_bits=32)
    pipeline_depth = 2
    hints = get_roller_hints(config, CUDA(), topk=20)
    run_baseline(config)
    result_dict = {}
    for hint in hints:
        with mlir_mod_ctx() as ctx:
            gb = GEMMBuilder()
            gb.build_gemm(config.M, config.N, config.K)
            gb.build_transform_pipeline(
                hint.block_tile, hint.thread_tile, hint.rstep, pipeline_depth
            )
            module = ctx.module
            lowered_module = Lower.lower_module(
                module,
                enable_transform_printing=False,
                enable_llvm_printing=False,
                dump_transformed_path=Path("log/transformed.mlir"),
                dump_llvm_path=Path("log/llvm.mlir"),
            )
            cuda_func = Lower.prepare_kernel(
                lowered_module, shared_memory_usage=hint.smem_cost * pipeline_depth
            )
        print(hint)
        t = run_eval(
            cuda_func,
            pipeline_depth,
            hint,
            profile=args.profile,
        )
        print("---\n")
        result_dict[hint] = t

    best_hint = min(result_dict, key=result_dict.get)
    print(f"Best hint: {best_hint}, time: {result_dict[best_hint]}")
