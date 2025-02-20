import flax.linen as nn
import jax.numpy as jnp
from chex import Array

from flax.linen.initializers import variance_scaling


class BaseLayer(nn.Module):
    out_size: int
    act_type: str = "relu"
    norm_type: str = "layer"
    scale: float = 1.0

    def setup(self):
        if self.norm_type == "none":
            self.norm = None
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(dtype=jnp.bfloat16)
        else:
            raise NotImplementedError(self.norm_type)
        use_bias = self.norm is None
        kernel_init = variance_scaling(
            self.scale,
            mode="fan_in",
            distribution="truncated_normal",
        )
        self.layer = None
        self.layer_kwargs = {
            "use_bias": use_bias,
            "kernel_init": kernel_init,
            "dtype": jnp.bfloat16,
        }
        if self.act_type == "none":
            self.act = None
        elif self.act_type == "silu":
            self.act = nn.silu
        elif self.act_type == "relu":
            self.act = nn.relu
        else:
            raise NotImplementedError(self.act_type)

    def __call__(self, x: Array) -> Array:
        if self.norm:
            x = self.norm(x)
        x = self.layer(x)
        if self.act:
            x = self.act(x)
        return x


class Linear(BaseLayer):
    def setup(self):
        super().setup()
        self.layer = nn.Dense(self.out_size, **self.layer_kwargs)


class Conv2d(BaseLayer):
    kernel_size: int = 3
    strides: int = 1

    def setup(self):
        super().setup()
        self.layer = nn.Conv(
            self.out_size,
            self.kernel_size,
            strides=self.strides,
            **self.layer_kwargs,
        )
