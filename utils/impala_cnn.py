from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp
from chex import Array

from utils.flax_util import Dense, Conv


class BasicBlock(nn.Module):
    out_chan: int
    scale: float = 1.0

    def setup(self):
        super().setup()
        scale = self.scale / jnp.sqrt(2)
        self.conv1 = Conv(
            self.out_chan,
            norm_type="layer",
            act_type="relu",
            scale=scale,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
        )
        self.conv2 = Conv(
            self.out_chan,
            norm_type="layer",
            act_type="none",
            scale=scale,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
        )

    def __call__(self, x: Array) -> Array:
        x = x + self.conv2(self.conv1(x))
        return x


class DownStack(nn.Module):
    out_chan: int
    num_blocks: int
    scale: float = 1.0
    first_conv_norm: bool = False

    def setup(self):
        super().setup()
        self.conv1 = Conv(
            self.out_chan,
            norm_type="layer" if self.first_conv_norm else "none",
            act_type="relu",
            scale=1.0,
            kernel_size=3,
            strides=1,
            padding="SAME",
        )
        self.norm = nn.LayerNorm(dtype=jnp.bfloat16)
        scale = self.scale / jnp.sqrt(self.num_blocks)
        self.blocks = [
            BasicBlock(self.out_chan, scale=scale) for _ in range(self.num_blocks)
        ]

    def __call__(self, x: Array) -> Array:
        x = self.conv1(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = self.norm(x)
        for block in self.blocks:
            x = block(x)
        return x


class ImpalaCNN(nn.Module):
    out_chans: Sequence[int]
    num_blocks: int
    out_size: int
    first_conv_norm: bool = False

    def setup(self):
        super().setup()
        scale = 1.0 / jnp.sqrt(len(self.out_chans))
        self.stacks = [
            DownStack(
                out_chan,
                self.num_blocks,
                scale=scale,
                first_conv_norm=self.first_conv_norm if i == 0 else True,
            )
            for i, out_chan in enumerate(self.out_chans)
        ]
        self.dense = Dense(
            self.out_size, norm_type="layer", act_type="relu", scale=jnp.sqrt(2)
        )

    def __call__(self, x: Array) -> Array:
        for stack in self.stacks:
            x = stack(x)
        x = jnp.reshape(x, (*x.shape[:2], -1))
        x = self.dense(x)
        return x
