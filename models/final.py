"""
Improved Decoder-only Transformer (Flax)
----------------------------------------

This version fixes all structural errors and improves training stability.

Features:
- Pre-LayerNorm architecture (stable gradients)
- Weight tying (tie_embeddings=True by default)
- Dropout in embedding, attention, and MLP
- SwiGLU MLP option (better parameter efficiency)
- Proper module registration (setup style)
- Explicit LayerNorm eps and safe initializers

Works with your transformer.ipynb directly.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import attention as attn
from typing import Optional

# ============================================================
# 1️⃣ SwiGLU MLP variant
# ============================================================
class SwiGLU(nn.Module):
    """SwiGLU: (Dense -> split -> GELU * linear -> Dense)"""
    d_model: int
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x):
        hidden = int(self.d_model * self.mlp_ratio)
        proj = nn.Dense(2 * hidden, kernel_init=nn.initializers.xavier_uniform())(x)
        a, b = jnp.split(proj, 2, axis=-1)
        return nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform())(nn.gelu(a) * b)


# ============================================================
# 2️⃣ Standard MLP block (GELU or SwiGLU)
# ============================================================
class MLP(nn.Module):
    d_model: int
    mlp_ratio: int = 4
    use_swiglu: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        hidden = int(self.d_model * self.mlp_ratio)
        if self.use_swiglu:
            x = SwiGLU(self.d_model, mlp_ratio=self.mlp_ratio)(x)
        else:
            x = nn.Dense(hidden, kernel_init=nn.initializers.xavier_uniform())(x)
            x = nn.gelu(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
            x = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform())(x)
        return x


# ============================================================
# 3️⃣ Decoder Transformer Block
# ============================================================
class DecoderBlock(nn.Module):
    d_model: int
    n_heads: int
    mlp_ratio: int = 4
    attn_dropout: float = 0.0
    dropout_rate: float = 0.0
    use_swiglu: bool = False
    layernorm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x, *, mask=None, deterministic: bool = True):
        # ---- Self-Attention ----
        h = nn.LayerNorm(epsilon=self.layernorm_eps)(x)
        h = nn.SelfAttention(
            num_heads=self.n_heads,
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dropout_rate=self.attn_dropout,
        )(h, mask=mask)
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
        x = x + h  # residual connection

        # ---- Feed-Forward ----
        h = nn.LayerNorm(epsilon=self.layernorm_eps)(x)
        h = MLP(
            self.d_model,
            mlp_ratio=self.mlp_ratio,
            use_swiglu=self.use_swiglu,
            dropout_rate=self.dropout_rate,
        )(h, deterministic=deterministic)
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
        x = x + h
        return x


# ============================================================
# 4️⃣ Full Decoder-only Transformer
# ============================================================
class DecoderOnlyTransformer(nn.Module):
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    max_len: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    attn_dropout: float = 0.0
    use_swiglu: bool = False
    tie_embeddings: bool = True
    layernorm_eps: float = 1e-5

    # ----------------------------
    # Setup: define submodules
    # ----------------------------
    def setup(self):
        # token embeddings
        self.tok_embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )

        # learned positional embeddings
        self.positional_embed = self.param(
            "positional_embed",
            nn.initializers.normal(stddev=0.01),
            (self.max_len, self.d_model),
        )

        # transformer blocks
        self.blocks = [
            DecoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                attn_dropout=self.attn_dropout,
                dropout_rate=self.dropout_rate,
                use_swiglu=self.use_swiglu,
                layernorm_eps=self.layernorm_eps,
            )
            for _ in range(self.n_layers)
        ]

        self.layernorm_final = nn.LayerNorm(epsilon=self.layernorm_eps)

        # vocab projection if not tying weights
        if not self.tie_embeddings:
            self.project_to_vocab = nn.Dense(
                self.vocab_size,
                use_bias=False,
                kernel_init=nn.initializers.xavier_uniform(),
            )

        # embedding dropout
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    # ----------------------------
    # Forward pass
    # ----------------------------
    def __call__(self, idx, *, deterministic: bool = True):
        B, T = idx.shape

        # token + positional embeddings
        x = self.tok_embed(idx) + self.positional_embed[:T]
        x = self.dropout(x, deterministic=deterministic)

        # causal attention mask
        causal = attn.make_causal_mask(jnp.ones((B, T), dtype=bool))
        mask = causal

        # stacked transformer blocks
        for blk in self.blocks:
            x = blk(x, mask=mask, deterministic=deterministic)

        # final norm + projection
        x = self.layernorm_final(x)

        if self.tie_embeddings:
            embed_matrix = self.variables["params"]["tok_embed"]["embedding"]  # (V, D)
            logits = jnp.einsum("btd,vd->btv", x, embed_matrix)
        else:
            logits = self.project_to_vocab(x)

        return logits
