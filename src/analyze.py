import functools
import itertools
import math
from copy import deepcopy
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Optional

import humanize

import numpy as np

from arch import CUDA, TileDevice
from utils import coalesced_factor, coalesced_tensor_shape, factorize, get_all_factors


@dataclass
class Shape:
    S: int
    R: int

    def to_list(self) -> list[int]:
        return [self.S, self.R]


@dataclass
class MatmulConfig:
    M: int
    N: int
    K: int
    el_bits: int

    def get_input_shapes(self):
        return Shape(S=self.M, R=self.K), Shape(S=self.N, R=self.K)

    def get_output_shape(self):
        return self.M, self.N

    @property
    def el_bytes(self):
        return (self.el_bits + 7) // 8

    @staticmethod
    def propagate_inputs(tile: list[int], rstep: int):
        return Shape(S=tile[0], R=rstep), Shape(S=tile[1], R=rstep)

    @staticmethod
    def propagate_outputs(tile: list[int]):
        return tile


def read_txn_elems(arch: TileDevice, el_bytes: int) -> int:
    """Returns the number of elements that can be read in a shared memory transaction."""
    return arch.transaction_size[1] // el_bytes


def write_txn_elems(arch: TileDevice, el_bytes: int) -> int:
    """Returns the number of elements that can be written in a global memory transaction."""
    return arch.transaction_size[0] // el_bytes


def compute_memory_traffic(
    output_tile: list[int], matmul_config: MatmulConfig, arch: TileDevice
) -> int:
    """
    计算给定输出 tile 的总读写流量（字节）。
    """
    nbytes = matmul_config.el_bytes
    r_elems = read_txn_elems(arch, nbytes)
    w_elems = write_txn_elems(arch, nbytes)

    inputs = MatmulConfig.propagate_inputs(output_tile, matmul_config.K)
    traffic_elems = 0
    for i, input_shape in enumerate(matmul_config.get_input_shapes()):
        traffic_elems += coalesced_tensor_shape(
            inputs[i].to_list(), input_shape.to_list(), r_elems
        )
    traffic_elems += coalesced_tensor_shape(
        output_tile, list(matmul_config.get_output_shape()), w_elems
    )
    return traffic_elems * nbytes


class TileDict:
    """
    Manages tiling information and configurations for computational tasks.
    """

    def __init__(
        self, block_tile: list[int], matmul_config: MatmulConfig, rstep: int
    ) -> None:
        self.matmul_config = matmul_config

        # schedule config
        self.block_tile = block_tile
        self.thread_tile = []
        self.rstep = rstep

        # analysis
        self.traffic = -1
        self.smem_cost = -1
        self.block_per_SM = -1
        self.num_wave = -1
        self.grid_size = -1
        self.valid = True

    def __hash__(self) -> int:
        return hash(tuple(self.block_tile + self.thread_tile))

    def __repr__(self) -> str:
        total_traffic = self.traffic * self.grid_size
        return (
            f"TileDict(block_tile={self.block_tile}, thread_tile={self.thread_tile}, "
            f"rstep={self.rstep}, block traffic={humanize.naturalsize(self.traffic, binary=True)}, "
            f"total traffic={humanize.naturalsize(total_traffic, binary=True)}, "
            f"smem_cost={humanize.naturalsize(self.smem_cost, binary=True)}, block_per_SM={self.block_per_SM}, "
            f"num_wave={self.num_wave}, grid_size={self.grid_size})"
        )

    def infer_smem_usage(self, rstep: Optional[int] = None) -> int:
        if rstep is None:
            rstep = self.rstep
        return sum([rstep * s for s in self.block_tile]) * self.matmul_config.el_bytes

    @classmethod
    def from_block_tile(
        cls,
        block_tile: list[int],
        rstep: int,
        matmul_config: MatmulConfig,
        arch: TileDevice,
    ) -> "TileDict":
        """
        Builds a TileDict from a block tile and validates resource usage.
        """
        td = cls(block_tile, matmul_config, rstep)

        # smem usage
        td.smem_cost = td.infer_smem_usage()
        if td.smem_cost > arch.smem_cap:
            td.valid = False
            return td

        # reg usage
        reg_usage = int(2 * math.prod(block_tile) * matmul_config.el_bytes / 32)
        if reg_usage > arch.reg_cap:
            td.valid = False
            return td

        # grid size
        output_shape = matmul_config.get_output_shape()
        td.grid_size = int(
            math.prod([(y + x - 1) // x for x, y in zip(block_tile, output_shape)])
        )

        td.traffic = compute_memory_traffic(block_tile, matmul_config, arch)

        # residency and waves
        td.block_per_SM = min(
            arch.max_smem_usage // max(td.smem_cost, 1),
            arch.reg_cap // max(reg_usage, 1),
            arch.sm_partition,
        )
        td.num_wave = int(
            np.ceil(td.grid_size / int(td.block_per_SM * arch.compute_max_core))
        )

        return td


class Policy:
    """
    Default Policy for fastdlight, a heuristic plan that tries to
    minimize memory traffic and maximize parallelism.
    """

    def __init__(self, config: MatmulConfig, arch: TileDevice) -> None:
        self.arch = arch
        self.matmul_config = config

    def _coalesced_score(self, tile: list[int], rstep: int, mode: str = "sim") -> float:
        """
        统一的 coalesced 评分, large is better：
        - mode="sim": 与读事务元素数做相似度评分（用于选规约步长）
        - mode="raw": 直接累加 coalesced 因子（用于扩展规约轴）
        """
        shapes = MatmulConfig.propagate_inputs(tile, rstep)
        score = 0.0
        if mode == "sim":
            rte = read_txn_elems(self.arch, self.matmul_config.el_bytes)

            def sim(a: int, b: int) -> float:
                return (2 * a * b) / (a * a + b * b)

            for i, input_shape in enumerate(self.matmul_config.get_input_shapes()):
                cf = coalesced_factor(shapes[i].to_list(), input_shape.to_list())
                score += sim(cf, rte)
        else:
            for i, input_shape in enumerate(self.matmul_config.get_input_shapes()):
                score += coalesced_factor(shapes[i].to_list(), input_shape.to_list())
        return score

    def get_reduce_step_candidates(self, rstep: int):
        """
        Calculates reduction step candidates for reduction axis.
        General idea : use factor first, since it does not require extra boundary check,
        for large prime number, which is rare case, use power of 2.
        """

        all_factors = get_all_factors(rstep)
        if len(all_factors) == 2 and rstep > 64:  # large prime
            all_factors = [1]
            while all_factors[-1] * 2 < rstep:
                all_factors.append(all_factors[-1] * 2)
        return all_factors

    def _assign_reduce_step(self):
        all_rsteps = self.get_reduce_step_candidates(self.matmul_config.K)
        return max(
            all_rsteps, key=lambda step: self._coalesced_score([1, 1], step, "sim")
        )

    def get_block_tiles_candidates(self, init_tile: list[int], rstep: int):
        output_shape = self.matmul_config.get_output_shape()
        steps = []
        for i, dim_size in enumerate(output_shape):
            # Start with factors greater than or equal to the initial tile size
            initial_factors = [
                f for f in get_all_factors(dim_size) if f >= init_tile[i]
            ]

            # Add powers of 2 that are not already factors and are within the valid range
            powers_of_2_to_add = {
                s for s in [2, 4, 8, 16, 32] if init_tile[i] < s < dim_size
            }

            # Combine and sort the unique step values
            combined_steps = sorted(list(set(initial_factors) | powers_of_2_to_add))
            steps.append(combined_steps)

        def prio(td: TileDict):
            return (td.traffic + 1) * td.num_wave

        results = [
            td
            for m, n in itertools.product(steps[0], steps[1])
            if (
                td := TileDict.from_block_tile(
                    [m, n], rstep, self.matmul_config, self.arch
                )
            ).valid
        ]

        sorted_tiles = sorted(results, key=prio)
        return sorted_tiles

    def _expand_reduce_axis(self, td: TileDict):
        """
        Expands the reduction axis in the TileDict based on shared memory limits.
        """
        smem_limit = min(
            self.arch.max_smem_usage // td.block_per_SM, self.arch.smem_cap
        )
        all_steps = [
            s
            for s in self.get_reduce_step_candidates(self.matmul_config.K)
            if s % td.rstep == 0 and td.infer_smem_usage(s) <= smem_limit
        ]
        if not all_steps:
            all_steps = [td.rstep]
        rstep = max(
            all_steps,
            key=lambda s: self._coalesced_score(td.block_tile, s, "raw"),
        )
        # update rstep
        td.rstep = rstep
        td.smem_cost = td.infer_smem_usage()

    def score_block_size(self, n):
        """
        Scores a block size based on its efficiency and fit relative to the architecture's warp size and SM partition.
        Small is better.

        Parameters
        ----------
        n : int
            The block size to score.

        Returns
        -------
        Tuple[float, float]
            A tuple containing two scores representing efficiency and fit, respectively.
        """
        num_warp = (n + self.arch.warp_size - 1) // self.arch.warp_size
        r1 = max(num_warp / self.arch.sm_partition, self.arch.sm_partition / num_warp)
        r2 = (num_warp * self.arch.warp_size - n) / n
        return (r1, r2)

    def recommend_block_size(self, td: TileDict):
        """
        Recommends optimal block sizes based on the TileDict configuration.
        """
        max_block_size = int(math.prod(td.block_tile))

        possible_block_sizes = get_all_factors(max_block_size)
        possible_block_sizes = list(filter(lambda x: x <= 1024, possible_block_sizes))
        factor_ordered = sorted(possible_block_sizes, key=self.score_block_size)
        return factor_ordered

    def assign_thread_tile(self, td: TileDict, topk=1):
        """
        Assigns thread tile to the TileDict based on the recommended block sizes.
        """
        block_size_ordered = self.recommend_block_size(td)
        for block_size in block_size_ordered:
            failed = False
            result = self._assign_thread_tile(td, block_size)
            if result is None:
                failed = True
                break
            if failed:
                continue
            else:
                yield result
                topk -= 1
                if topk == 0:
                    break

    def _assign_thread_tile(self, td: TileDict, block_size: int):
        """
        Assigns a thread tile to a TileDict configuration.
        """
        block_tile = td.block_tile

        def get_factor_pairs(n: int):
            return [[n // i, i] for i in range(1, int(math.sqrt(n)) + 1) if n % i == 0]

        possible_tile_size = get_factor_pairs(math.prod(block_tile) // block_size)

        def _score(thread_tile):  # small is better
            score = 0
            shapes = MatmulConfig.propagate_inputs(thread_tile, self.matmul_config.K)
            # read
            for shape in shapes:
                score += math.prod(shape.to_list()) / self.arch.bandwidth[1]

            # # write
            # score += (
            #     coalesced_tensor_shape(
            #         thread_tile,
            #         list(self.matmul_config.get_output_shape()),
            #         write_txn_elems(self.arch, self.matmul_config.el_bytes),
            #     )
            #     / self.arch.bandwidth[0]
            # )
            return score

        sorted_possible_tile_size = sorted(possible_tile_size, key=_score)
        best_tile = sorted_possible_tile_size[0]
        reg_usage = int(math.prod(best_tile) * self.matmul_config.el_bytes / 4)
        if reg_usage >= 256:
            return None

        td.thread_tile = best_tile
        return deepcopy(td)

    def emit_config(self, topk=10) -> list[TileDict]:
        base_tile = [1, 1]

        rstep = self._assign_reduce_step()
        block_tile_condidates = self.get_block_tiles_candidates(base_tile, rstep)
        results = []
        for td in block_tile_condidates:
            self._expand_reduce_axis(td)
            for codegen_dicts in self.assign_thread_tile(td, 2):
                results.append(codegen_dicts)
                if len(results) >= topk:
                    break
            if len(results) >= topk:
                break
        return results


def get_roller_hints(matmul_config: MatmulConfig, arch: TileDevice, topk=10):
    policy = Policy(matmul_config, arch)
    roller_hints = policy.emit_config(topk)
    return roller_hints


if __name__ == "__main__":
    S = 8192
    config = MatmulConfig(M=S, N=S, K=S, el_bits=32)
    arch = CUDA()

    get_roller_hints(config, arch, topk=20)
