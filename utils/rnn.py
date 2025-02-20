import math
from typing import Sequence, Tuple

from flax import nnx
from jax import numpy as jnp
from jax.typing import ArrayLike, DTypeLike

from utils.flax_util import Linear


class ResidualRecurrentBlock(nnx.Module):
    def __init__(
        self,
        in_size: int,
        hid_size: int,
        *,
        scale: float = 1.0,
        dtype: DTypeLike = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.norm = nnx.LayerNorm(in_size, dtype=dtype, rngs=rngs)
        self.cell = nnx.GRUCell(in_size, hid_size, dtype=dtype, rngs=rngs)
        scale = scale / math.sqrt(2)
        self.linear1 = Linear(
            hid_size,
            4 * hid_size,
            norm_type="layer",
            act_type="relu",
            scale=scale,
            dtype=dtype,
            rngs=rngs,
        )
        self.linear2 = Linear(
            4 * hid_size,
            hid_size,
            norm_type="none",
            act_type="none",
            scale=scale,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self, carry: ArrayLike, x: ArrayLike, first: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        residual = x
        x = self.norm(x)
        carry = first * self.cell.initialize_carry(x.shape) + (1 - first) * carry
        carry, x = self.cell(carry, x)
        x = x + residual
        residual = x
        x = self.linear2(self.linear1(x))
        x = x + residual
        return carry, x


class ResidualRecurrentBlocks(nnx.Module):
    def __init__(
        self,
        in_size,
        hid_size,
        num_blocks: int,
        *,
        scale: float = 1.0,
        dtype: DTypeLike = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.blocks = [
            ResidualRecurrentBlock(
                in_size, hid_size, scale=scale, dtype=dtype, rngs=rngs
            )
            for _ in range(num_blocks)
        ]

    def __call__(
        self, carry: ArrayLike, x: ArrayLike, first: ArrayLike
    ) -> Tuple[Sequence[ArrayLike], ArrayLike]:
        # TODO: need to use scan
        pass

    def initialize_carry(self, in_shape: Sequence[int]) -> Sequence[ArrayLike]:
        carry = [block.cell.initialize_carry(in_shape) for block in self.blocks]
        return carry
