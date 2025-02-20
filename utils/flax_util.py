from typing import Sequence

import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike, DTypeLike


class BaseLayer(nnx.Module):
    def __init__(
        self,
        in_size: int,
        *,
        norm_type: str = "layer",
        act_type: str = "relu",
        scale: float = 1.0,
        dtype: DTypeLike = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        if norm_type == "none":
            self.norm = None
        elif norm_type == "layer":
            self.norm = nnx.LayerNorm(in_size, dtype=dtype, rngs=rngs)
        else:
            raise NotImplementedError(norm_type)
        use_bias = self.norm is None
        kernel_init = nnx.nn.initializers.variance_scaling(
            scale, "fan_in", "truncated_normal", dtype=dtype
        )
        self.layer = None
        self.layer_kwargs = {
            "use_bias": use_bias,
            "kernel_init": kernel_init,
            "dtype": dtype,
            "rngs": rngs,
        }
        if act_type == "none":
            self.act = None
        elif act_type == "silu":
            self.act = nnx.silu
        elif act_type == "relu":
            self.act = nnx.relu
        else:
            raise NotImplementedError(act_type)

    def __call__(self, x: ArrayLike) -> ArrayLike:
        if self.norm:
            x = self.norm(x)
        x = self.layer(x)
        if self.act:
            x = self.act(x)
        return x


class Linear(BaseLayer):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        norm_type: str = "layer",
        act_type: str = "relu",
        scale: float = 1.0,
        dtype: DTypeLike = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            in_size,
            norm_type=norm_type,
            act_type=act_type,
            scale=scale,
            dtype=dtype,
            rngs=rngs,
        )
        self.layer = nnx.Linear(in_size, out_size, **self.layer_kwargs)


class Conv(BaseLayer):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        kernel_size: Sequence[int] = (3, 3),
        strides: Sequence[int] = (1, 1),
        padding: str = "SAME",
        norm_type: str = "layer",
        act_type: str = "relu",
        scale: float = 1.0,
        dtype: DTypeLike = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            in_size,
            norm_type=norm_type,
            act_type=act_type,
            scale=scale,
            dtype=dtype,
            rngs=rngs,
        )
        self.layer = nnx.Conv(
            in_size,
            out_size,
            kernel_size,
            strides=strides,
            padding=padding,
            **self.layer_kwargs,
        )
