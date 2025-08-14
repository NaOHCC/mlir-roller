import math

import numpy as np


def get_all_factors(n: int) -> list[int]:
    # Calculate the square root of n and round it up to the nearest integer
    n0 = int(np.ceil(np.sqrt(n)))

    # Find all divisors of n that are less than n0
    val = np.where(n % np.arange(1, n0) == 0)[0] + 1

    # If n is a perfect square, add the square root to the list of factors
    mid = np.array([], dtype=int) if n0 * n0 != n else [n0]

    # Combine the factors and their corresponding larger pair factors
    return [int(x) for x in np.concatenate([val, mid, n // val[::-1]])]


def factorize(n: int) -> list[int]:
    i = 2  # Start with the smallest prime number
    result = []

    # Iterate through numbers to find factors
    while n > 1:
        if n % i == 0:  # If i is a factor of n
            n //= i  # Divide n by i and keep the integer part
            result.append(i)
        else:
            i += 1  # Try the next number
    return result


def coalesced_factor(subtensor: list[int], tensor: list[int]) -> int:
    # If the last dimension of the subtensor and tensor differ, or subtensor has only one dimension
    if subtensor[-1] != tensor[-1] or len(subtensor) == 1:
        return subtensor[-1]
    else:
        # Recursively calculate the coalesced factor for the remaining dimensions
        return subtensor[-1] * coalesced_factor(subtensor[:-1], tensor[:-1])


# 合并访存需要考虑的元素数量
def coalesced_tensor_shape(
    subtensor: list[int], tensor: list[int], transaction_size: int
) -> int:
    # Calculate the total number of elements in the subtensor
    els = math.prod(subtensor)

    if els == 0:
        return 0

    # Calculate the coalesced factor for the subtensor
    factor = coalesced_factor(subtensor, tensor)

    # Compute the shape of the coalesced tensor
    return transaction_size * els // min(transaction_size, factor)


def get_block_dims(block_tile: list[int], reg_tile: list[int]) -> list[int]:
    block_dims: list[int] = []
    for block_tile_size, reg_tile_size in zip(block_tile, reg_tile):
        if (block_tile_size != 0) and (reg_tile_size != 0):
            block_dims.append(block_tile_size // reg_tile_size)
        else:
            Warning("block_tile_size or reg_tile_size is 0")
            block_dims.append(1)
    block_dims[0], block_dims[1] = block_dims[1], block_dims[0]
    return block_dims
