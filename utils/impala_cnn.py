import math
from typing import Sequence

import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike

from utils.flax_util import Conv, Linear


class BasicBlock(nnx.Module):
    def __init__(
        self, in_chan: int, out_chan: int, *, scale: float = 1.0, rngs: nnx.Rngs
    ):
        scale = scale / math.sqrt(2)
        self.conv1 = Conv(
            in_chan,
            out_chan,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            norm_type="layer",
            act_type="relu",
            scale=scale,
            dtype=jnp.bfloat16,
            rngs=rngs,
        )
        self.conv2 = Conv(
            in_chan,
            out_chan,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            norm_type="layer",
            act_type="relu",
            scale=scale,
            dtype=jnp.bfloat16,
            rngs=rngs,
        )

    def __call__(self, x: ArrayLike) -> ArrayLike:
        x = x + self.conv2(self.conv1(x))
        return x


class DownStack(nnx.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        num_blocks: int,
        *,
        scale: float = 1.0,
        first_conv_norm: bool = False,
        rngs: nnx.Rngs,
    ):
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.conv1 = Conv(
            in_chan,
            out_chan,
            kernel_size=3,
            strides=1,
            padding="SAME",
            norm_type="layer" if first_conv_norm else "none",
            act_type="relu",
            scale=1.0,
            dtype=jnp.bfloat16,
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(out_chan, dtype=jnp.bfloat16, rngs=rngs)
        scale = scale / math.sqrt(num_blocks)
        self.blocks = [
            BasicBlock(out_chan, out_chan, scale=scale, rngs=rngs)
            for _ in range(num_blocks)
        ]

    def __call__(self, x: ArrayLike) -> ArrayLike:
        x = self.conv1(x)
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = self.norm(x)
        for block in self.blocks:
            x = block(x)
        return x

    def output_shape(self, in_shape: Sequence[int]) -> Sequence[int]:
        h, w, c = in_shape
        assert c == self.in_chan
        return ((h + 1) // 2, (w + 1) // 2, self.out_chan)


class ImpalaCNN(nnx.Module):
    def __init__(
        self,
        in_shape: Sequence[int],
        out_chans: Sequence[int],
        num_blocks: int,
        out_size: int,
        *,
        first_conv_norm: bool = False,
        rngs: nnx.Rngs,
    ):
        scale = 1.0 / math.sqrt(len(out_chans))
        self.stacks = []
        for i, out_chan in enumerate(out_chans):
            stack = DownStack(
                in_shape[-1],
                out_chan,
                num_blocks,
                scale=scale,
                first_conv_norm=first_conv_norm if i == 0 else True,
                rngs=rngs,
            )
            self.stacks.append(stack)
            in_shape = stack.output_shape(in_shape)
        in_size = math.prod(in_shape)
        self.dense = Linear(
            in_size,
            out_size,
            norm_type="layer",
            act_type="relu",
            scale=math.sqrt(2),
            dtype=jnp.bfloat16,
            rngs=rngs,
        )

    def __call__(self, x: ArrayLike) -> ArrayLike:
        for stack in self.stacks:
            x = stack(x)
        x = jnp.reshape(x, (*x.shape[:2], -1))
        x = self.dense(x)
        return x
