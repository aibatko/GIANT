from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm

from omegaconf import OmegaConf
Config = OmegaConf.load("Config.yml")

def _rotate_every_two(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)

def apply_rope(q_or_k, sin, cos):
    return (q_or_k * cos) + (_rotate_every_two(q_or_k) * sin)

class NativeJaxSelfAttention(nn.Module):
    """Multi‑head self‑attention using jax.nn.dot_product_attention (cuDNN)."""

    num_heads: int
    qkv_features: int
    dropout_rate: float = 0.0
    num_kv: int = 1
    dtype: jnp.dtype = Config.compute_dtype

    def setup(self):
        assert (
            self.qkv_features % self.num_heads == 0
        ), "qkv_features must be divisible by num_heads"
        self.head_dim = self.qkv_features // self.num_heads

        self.q_proj = nn.Dense(self.qkv_features, use_bias=False, name="q_proj", dtype=self.dtype, param_dtype=Config.param_dtype)
        self.k_proj = nn.Dense(self.num_kv * self.head_dim, use_bias=False, name="k_proj", dtype=self.dtype, param_dtype=Config.param_dtype)
        self.v_proj = nn.Dense(self.num_kv * self.head_dim, use_bias=False, name="v_proj", dtype=self.dtype, param_dtype=Config.param_dtype)


        self.o_proj = nn.Dense(self.qkv_features, use_bias=False, name="o_proj", dtype=self.dtype, param_dtype=Config.param_dtype)

        self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, *, deterministic: bool, decode: bool = False, cur_index: Optional[int] = None):
        b, l, _ = x.shape
        head_dim = self.qkv_features // self.num_heads

        q = self.q_proj(x).reshape(b, l, self.num_heads, head_dim)

        k = self.k_proj(x).reshape(b, l, self.num_kv, head_dim)
        v = self.v_proj(x).reshape(b, l, self.num_kv, head_dim)

        k = jnp.repeat(k, self.num_heads // self.num_kv, axis=2)
        v = jnp.repeat(v, self.num_heads // self.num_kv, axis=2)

        rot_dim = head_dim
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, rot_dim, 2) / rot_dim))
        seq      = jnp.array([cur_index]) if decode else jnp.arange(l)
        angles   = jnp.einsum('i,j->ij', seq, inv_freq)
        emb      = jnp.repeat(angles, 2, axis=-1)
        sin, cos = jnp.sin(emb).astype(self.dtype), jnp.cos(emb).astype(self.dtype)
        sin, cos = sin[None, :, None, :], cos[None, :, None, :]
        q, k = apply_rope(q, sin, cos), apply_rope(k, sin, cos)


        if decode:
            assert cur_index is not None, "Need cur_index when decode=True"
            cached_k = self.variable( "cache", "k", jnp.zeros, (b, self.num_heads, Config.context_length, head_dim), self.dtype)
            cached_v = self.variable( "cache", "v", jnp.zeros, (b, self.num_heads, Config.context_length, head_dim), self.dtype)


            cached_k.value = cached_k.value.at[:, :, cur_index, :].set(k.squeeze(1))
            cached_v.value = cached_v.value.at[:, :, cur_index, :].set(v.squeeze(1))
            k = jnp.swapaxes(cached_k.value, 1, 2)
            v = jnp.swapaxes(cached_v.value, 1, 2)

            if False:
                q = q / jnp.sqrt(head_dim)

            key_len   = k.shape[1]
            valid     = jnp.arange(key_len) <= cur_index
            attn_bias = jnp.where(valid, 0.0, -1e10).astype(self.dtype)
            attn_bias = attn_bias[None, None, None, :]

            try:
                y = jax.nn.dot_product_attention(
                        q, k, v,
                        bias=attn_bias,
                        is_causal=True,
                        implementation="flash",
                )
            except Exception:
                y = jax.nn.dot_product_attention(
                    q, k, v,
                    bias=attn_bias,
                    is_causal=False,
                    implementation="cudnn",
                )

            y = y.reshape(b, 1, self.qkv_features)

        else:
            if False:
                q = q / jnp.sqrt(head_dim)

            y = jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="cudnn")
            y = y.reshape(b, l, self.qkv_features)

        y = self.o_proj(y)
        y = self.dropout(y, deterministic=deterministic)
        return y


class TinyTransformerBlock(nn.Module):
    """Decoder‑style transformer block (GPT) with checkpointing."""

    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    dtype: jnp.dtype = Config.compute_dtype

    @nn.compact
    def __call__(self, x, *, deterministic: bool, decode: bool = False, cur_index: Optional[int] = None):
        @nn.remat
        def _block(module: "TinyTransformerBlock", h: jnp.ndarray) -> jnp.ndarray:
            residual = h
            h_norm = RMSNorm(name="rms1", dtype=self.dtype)(h)
            h_attn = NativeJaxSelfAttention(
                num_heads=module.n_heads,
                qkv_features=module.d_model,
                dropout_rate=module.dropout_rate,
                dtype=module.dtype,
            )(h_norm, deterministic=deterministic, decode=decode, cur_index=cur_index)
            h = residual + h_attn

            residual = h
            h_norm = RMSNorm(name="rms2", dtype=self.dtype)(h)

            gate_dim = module.d_ff // 3
            proj_dim = gate_dim * 2

            h_proj = nn.Dense(
                proj_dim,
                name="fc1",
                dtype=module.dtype,
                param_dtype=Config.param_dtype,
            )(h_norm)

            u, v = jnp.split(h_proj, 2, axis=-1)
            h_ffn = nn.silu(u) * v

            h_ffn = nn.Dense(module.d_model, name="fc2", dtype=module.dtype, param_dtype=Config.param_dtype)(h_ffn)
            h_ffn = nn.Dropout(rate=module.dropout_rate)(h_ffn, deterministic=deterministic)
            return residual + h_ffn

        return _block(self, x)
