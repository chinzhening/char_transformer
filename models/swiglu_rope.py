import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import attention as attn

# -------------------------
# RoPE utilities
# -------------------------
def apply_rotary_pos_emb(q, k, sin, cos):
    q1, q2 = jnp.split(q, 2, axis=-1)
    k1, k2 = jnp.split(k, 2, axis=-1)
    q_rot = q1 * cos + q2 * sin
    q_pass = -q1 * sin + q2 * cos
    k_rot = k1 * cos + k2 * sin
    k_pass = -k1 * sin + k2 * cos
    q_out = jnp.concatenate([q_rot, q_pass], axis=-1)
    k_out = jnp.concatenate([k_rot, k_pass], axis=-1)
    return q_out, k_out

def build_rope_cache(max_len, head_dim):
    theta = 10000.0 ** (-jnp.arange(0, head_dim, 2) / head_dim)
    pos = jnp.arange(max_len)
    angles = pos[:, None] * theta[None, :]
    sin = jnp.sin(angles)[:, None, :]  # (T, 1, Dh)
    cos = jnp.cos(angles)[:, None, :]
    return sin, cos

# -------------------------
# Feed-forward with SwiGLU
# -------------------------
class SwiGLUMLP(nn.Module):
    d_model: int
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x):
        hidden = int(self.d_model * self.mlp_ratio)
        x_proj = nn.Dense(2 * hidden)(x)
        x1, x2 = jnp.split(x_proj, 2, axis=-1)
        x = x1 * nn.silu(x2)
        x = nn.Dense(self.d_model)(x)
        return x

# -------------------------
# RoPE-based self-attention
# -------------------------
class RoPESelfAttention(nn.Module):
    num_heads: int
    d_model: int
    max_len: int

    def setup(self):
        H = self.num_heads
        Dh = self.d_model // H
        # Precompute RoPE cache
        self.sin, self.cos = build_rope_cache(self.max_len, Dh)
        self.qkv = nn.Dense(3 * self.d_model, use_bias=False)
        self.proj = nn.Dense(self.d_model, use_bias=False)

    @nn.compact
    def __call__(self, x, *, mask=None):
        B, T, D = x.shape
        H = self.num_heads
        Dh = D // H

        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, H, Dh)
        k = k.reshape(B, T, H, Dh)
        v = v.reshape(B, T, H, Dh)

        sin_slice = self.sin[:T]
        cos_slice = self.cos[:T]
        q, k = apply_rotary_pos_emb(q, k, sin_slice, cos_slice)

        scale = 1.0 / jnp.sqrt(Dh)
        att = jnp.einsum("bthd,bThd->bhtT", q, k) * scale

        if mask is not None:
            att = att + (mask * -1e9)

        weights = nn.softmax(att, axis=-1)
        out = jnp.einsum("bhtT,bThd->bthd", weights, v)
        out = out.reshape(B, T, D)
        out = self.proj(out)
        return out

# -------------------------
# Decoder block
# -------------------------
class DecoderBlock(nn.Module):
    d_model: int
    n_heads: int
    mlp_ratio: int = 4
    max_len: int = 2048

    @nn.compact
    def __call__(self, x, *, mask=None):
        h = nn.LayerNorm()(x)
        h = RoPESelfAttention(self.n_heads, self.d_model, self.max_len)(h, mask=mask)
        x = x + h
        h = nn.LayerNorm()(x)
        h = SwiGLUMLP(self.d_model, self.mlp_ratio)(h)
        x = x + h
        return x

# -------------------------
# Decoder-only Transformer
# -------------------------
class DecoderOnlyTransformer(nn.Module):
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    max_len: int
    mlp_ratio: int = 4
    tie_weights: bool = False

    def setup(self):
        self.tok_embed = nn.Embed(self.vocab_size, self.d_model)
        self.blocks = [
            DecoderBlock(self.d_model, self.n_heads, self.mlp_ratio, self.max_len)
            for _ in range(self.n_layers)
        ]
        self.layerNorm_final = nn.LayerNorm()
        self.project_to_vocab = nn.Dense(self.vocab_size, use_bias=False)

    def __call__(self, idx, deterministic: bool = True):
        B, T = idx.shape
        x = self.tok_embed(idx)
        causal = attn.make_causal_mask(jnp.ones((B, T), dtype=bool))
        mask = causal

        for blk in self.blocks:
            x = blk(x, mask=mask)

        x = self.layerNorm_final(x)

        if self.tie_weights:
            logits = jnp.einsum("btd,vd->btv", x, self.tok_embed.embedding)
        else:
            logits = self.project_to_vocab(x)
        return logits

# -------------------------
# Helper to create train state
# -------------------------
def create_train_state(
    model_cls,
    *,
    rng,
    vocab_size=27,
    d_model=64,
    n_layers=6,
    n_heads=8,
    max_len=128,
    tie_weights=True,
):
    model = model_cls(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_len=max_len,
        tie_weights=tie_weights,
    )

    dummy_len = min(16, max_len)
    dummy_input = jnp.zeros((1, dummy_len), dtype=jnp.int32)

    variables = model.init({"params": rng}, dummy_input, deterministic=True)
    params = variables["params"]
    return model, params
