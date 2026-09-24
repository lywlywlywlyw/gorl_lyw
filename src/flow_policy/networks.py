from __future__ import annotations

from typing import NewType

import jax
from jax import Array, nn
from jax import numpy as jnp

from .math_utils import NormalDistribution

# A layer stores (kernel, bias); SERL hidden layers additionally store
# LayerNorm (scale, offset), and its policy output stores two Dense heads.
MlpWeights = NewType("MlpWeights", tuple[tuple[Array, ...], ...])


def mlp_init(
    prng: Array,
    dims: tuple[int, ...],
    init_fn: nn.initializers.Initializer | None = None,
    use_layer_norm: bool = False,
) -> MlpWeights:
    """Initialize an MLP, optionally with SERL LayerNorm parameters."""
    prngs = jax.random.split(prng, len(dims) - 1)
    shapes = zip(dims[:-1], dims[1:])
    if init_fn is None:
        init_fn = (
            nn.initializers.xavier_uniform()
            if use_layer_norm
            else nn.initializers.lecun_uniform()
        )
    layers = []
    for index, (key, shape) in enumerate(zip(prngs, shapes)):
        layer = (init_fn(key, shape), jnp.zeros((shape[1],)))
        if use_layer_norm and index < len(dims) - 2:
            layer = layer + (jnp.ones((shape[1],)), jnp.zeros((shape[1],)))
        layers.append(layer)
    return MlpWeights(tuple(layers))


def gaussian_policy_init(
    prng: Array,
    dims: tuple[int, ...],
) -> MlpWeights:
    """Initialize SERL's shared LayerNorm MLP and separate Gaussian heads.

    dims is (input_dim, *hidden_dims, action_dim).
    """
    if len(dims) < 3:
        raise ValueError("A SERL policy requires at least one hidden layer.")
    keys = jax.random.split(prng, len(dims))
    init_fn = nn.initializers.xavier_uniform()
    layers = []
    for key, shape in zip(keys[:-2], zip(dims[:-2], dims[1:-1])):
        output_dim = shape[1]
        layers.append(
            (
                init_fn(key, shape),
                jnp.zeros((output_dim,)),
                jnp.ones((output_dim,)),
                jnp.zeros((output_dim,)),
            )
        )
    head_shape = (dims[-2], dims[-1])
    mean_key, std_key = keys[-2:]
    layers.append(
        (
            init_fn(mean_key, head_shape),
            jnp.zeros((dims[-1],)),
            init_fn(std_key, head_shape),
            jnp.zeros((dims[-1],)),
        )
    )
    return MlpWeights(tuple(layers))


def _serl_hidden_fwd(weights: MlpWeights, x: Array) -> Array:
    """SERL MLP backbone: Dense -> LayerNorm -> tanh per hidden layer."""
    for layer in weights[:-1]:
        linear, bias, scale, offset = layer
        x = jnp.einsum("...i,ij->...j", x, linear) + bias
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(variance + 1e-6)
        x = x * scale + offset
        x = nn.tanh(x)
    return x


def value_mlp_fwd(weights: MlpWeights, x: Array) -> Array:
    """Apply hidden layers, then output projection.

    Input: (*, obs_dim)
    Output: (*,)
    """
    if len(weights[0]) == 4:
        x = _serl_hidden_fwd(weights, x)
    else:
        for i in range(len(weights) - 1):
            linear, bias = weights[i]
            x = jnp.einsum("...i,ij->...j", x, linear) + bias
            x = nn.silu(x)

    linear, bias = weights[-1]
    x = jnp.einsum("...i,ij->...j", x, linear) + bias
    x = jnp.squeeze(x, axis=-1)
    return x


def flow_mlp_fwd(weights: MlpWeights, *inputs_to_concat: Array) -> Array:
    """Apply hidden layers, then output projection."""
    x = jnp.concatenate(inputs_to_concat, axis=-1)
    for i in range(len(weights) - 1):
        linear, bias = weights[i]
        x = jnp.einsum("...i,ij->...j", x, linear) + bias
        x = nn.silu(x)
    linear, bias = weights[-1]
    x = jnp.einsum("...i,ij->...j", x, linear) + bias
    return x


def gaussian_policy_fwd(
    weights: MlpWeights,
    x: Array,
) -> NormalDistribution:
    """Apply hidden layers, then output projection."""
    uses_serl_backbone = len(weights[0]) == 4
    if uses_serl_backbone:
        x = _serl_hidden_fwd(weights, x)
    else:
        for i in range(len(weights) - 1):
            linear, bias = weights[i]
            x = jnp.einsum("...i,ij->...j", x, linear) + bias
            x = nn.silu(x)

    if uses_serl_backbone and len(weights[-1]) == 4:
        mean_kernel, mean_bias, std_kernel, std_bias = weights[-1]
        mean = jnp.einsum("...i,ij->...j", x, mean_kernel) + mean_bias
        raw_scale = jnp.einsum("...i,ij->...j", x, std_kernel) + std_bias
        # Match the SERL launcher: exp std in [1e-5, 5]. The sampled latent
        # itself remains unsquashed and unbounded.
        scale = jnp.clip(jnp.exp(raw_scale), 1e-5, 5.0)
    else:
        # Legacy GoRL policy checkpoints used one combined output projection.
        linear, bias = weights[-1]
        output = jnp.einsum("...i,ij->...j", x, linear) + bias
        assert output.shape[-1] % 2 == 0
        mean, raw_scale = jnp.split(output, 2, axis=-1)
        scale = jnp.clip(nn.softplus(raw_scale) + 1e-3, 1e-3, 10.0)
    return NormalDistribution(mean, scale)


def q_mlp_fwd(weights: MlpWeights, obs: Array, action: Array) -> Array:
    """Q-function forward pass: Q(s, a) = MLP([s; a]).

    Input: obs (*, obs_dim), action (*, action_dim)
    Output: Q-value (*)
    """
    # Concatenate observation and action
    x = jnp.concatenate([obs, action], axis=-1)

    if len(weights[0]) == 4:
        x = _serl_hidden_fwd(weights, x)
    else:
        for i in range(len(weights) - 1):
            linear, bias = weights[i]
            x = jnp.einsum("...i,ij->...j", x, linear) + bias
            x = nn.silu(x)

    # Final layer (no activation)
    linear, bias = weights[-1]
    x = jnp.einsum("...i,ij->...j", x, linear) + bias
    x = jnp.squeeze(x, axis=-1)
    return x
