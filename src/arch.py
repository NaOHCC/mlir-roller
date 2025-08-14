# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from typing import List


@dataclass
class TileDevice:
    """
    Represents the architecture of a computing device, capturing various hardware specifications.
    """

    # Register capacity: The amount of register memory available
    reg_cap: int = 0
    # Shared memory capacity: The amount of shared memory available
    smem_cap: int = 0
    # The maximum number of computing cores
    compute_max_core: int = 0
    # The size of a warp, a group of threads that execute instructions in lockstep
    warp_size: int = 0
    # The number of streaming multiprocessor partitions
    sm_partition: int = 0
    # The maximum shared memory usage allowed
    max_smem_usage: int = 0
    # The platform or manufacturer of the device
    platform: str = "unknown"
    # The compute capability, indicating the feature set and performance level
    compute_capability: str = "unknown"
    l2_cache_size_bytes: int = 0
    # The size of memory transactions, typically in bytes
    transaction_size: List[int] = field(default_factory=lambda: [0, 0])
    # Bandwidth specifications, possibly including peak and sustained rates, in MB/s
    bandwidth: List[int] = field(default_factory=lambda: [0, 0])

    def get_avaliable_tensorintrin_shapes(self):
        raise NotImplementedError()


@dataclass
class CUDA(TileDevice):
    """
    Represents a CUDA-enabled device, inheriting from TileDevice and specifying
    default values for a typical CUDA architecture.
    """

    platform: str = "CUDA"
    smem_cap: int = 64 * 1024
    compute_max_core: int = 108
    warp_size: int = 32
    reg_cap: int = 65536
    sm_partition: int = 4
    transaction_size: List[int] = field(default_factory=lambda: [32, 128])
    bandwidth: List[int] = field(default_factory=lambda: [750, 12080])
    max_smem_usage: int = field(init=False)

    def __post_init__(self):
        """
        Initializes fields that depend on other field values.
        """
        self.max_smem_usage = 2 * self.smem_cap
