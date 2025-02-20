import math

import flax.linen as nn
import jax.numpy as jnp
from chex import Array

from utils.flax_util import Linear, Conv2d


class BasicBlock(nn.Module):
    out_size: int
    scale: float = 1.0

    def setup(self):
        super().setup()
        scale = self.scale / jnp.sqrt(2)
        self.conv0 = Conv2d(
            self.out_size,
            act_type="relu",
            norm_type="layer",
            scale=scale,
            kernel_size=3,
            strides=1,
        )
        self.conv1 = Conv2d(
            self.out_size,
            act_type="none",
            norm_type="layer",
            scale=scale,
            kernel_size=3,
            strides=1,
        )

    def __call__(self, x: Array) -> Array:
        x = x + self.conv1(self.conv0(x))
        return x


class DownStack(nn.Module):
    out_size: int
    num_blocks: int
    scale: float = 1.0
    norm_type: str = "none"

    def setup(self):
        super().setup()
        self.first_conv = Conv2d(
            self.out_size,
            act_type="relu",
            norm_type=self.norm_type,
            scale=1.0,
            kernel_size=3,
            strides=1,
        )
        self.norm = nn.LayerNorm(dtype=jnp.bfloat16)
        scale = self.scale / jnp.sqrt(self.num_blocks)
        self.blocks = [
            BasicBlock(self.out_size, scale=scale) for _ in range(self.num_blocks)
        ]

    def __call__(self, x: Array) -> Array:
        x = self.first_conv(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = self.norm(x)
        for block in self.blocks:
            x = block(x)
        return x
