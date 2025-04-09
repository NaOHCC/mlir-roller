from typing import TypeVar

from iree.compiler.ir import *
import argparse
import math

import sys

import cupy as cp
import iree.compiler.extras.types as T
import iree.compiler.ir as ir
import numpy as np
from cupy.cuda import Module
from iree.compiler.dialects import builtin, iree_codegen, iree_transform, transform
from iree.compiler.dialects.bufferization import LayoutMapOption
from iree.compiler.dialects.transform import (
    any_op_t,
    bufferization,
    gpu as gpu_transform,
    loop,
    structured,
    vector,
)
from iree.compiler.dialects.transform.extras import apply_patterns, named_sequence
from iree.compiler.extras.ast.canonicalize import canonicalize
from iree.compiler.extras.context import (
    ExplicitlyManagedModule,
    mlir_mod_ctx,
    MLIRContext,
    RAIIMLIRContext,
)
from iree.compiler.extras.dialects.ext import (
    arith,
    func,
    gpu,
    linalg,
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

from roller import KK, lower_to_llvm, MM, NN, prepare_kernel, run_eval

# from roller import block_dims, grid_dims, tiling_level_1, tiling_level_2
parser = argparse.ArgumentParser()
parser.add_argument("--profile", action="store_true", help="profile")

if __name__ == "__main__":
    with open("log/debug.log", "w") as f:

        sys.stderr = open("log/debug.log", "w")

        args = parser.parse_args()
        ctx = RAIIMLIRContext()
        backend = LLVMJITBackend()

        with open("log/befor_lower.mlir", "r") as f:
            payload = f.read()

        module = ir.Module.parse(payload)
        # print(module)

        lowered_module = lower_to_llvm(module, False, enable_ir_printing=True)

        with open("log/lower.mlir", "w") as f:
            f.write(lowered_module.__str__())
        cuda_func, grid_dims, block_dims, shared_mem, npy_dtype = prepare_kernel(
            lowered_module, MM, KK, NN
        )
        run_eval(
            MM,
            KK,
            NN,
            cuda_func,
            grid_dims,
            block_dims,
            (128 * 8 + 8 * 64) * npy_dtype().nbytes,
            npy_dtype,
            profile=args.profile,
        )
