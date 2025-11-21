"""
Minimal decoder-only Transformer blocks in Flax/JAX, commented for learning.

The model mirrors a GPT-style architecture:
- Token embeddings + learned positional embeddings
- Stack of Pre-LayerNorm decoder blocks with causal self-attention
- Final LayerNorm
- Weight tying between input embeddings and output logits projection

Tensor shape conventions used below:
- B: batch size
- T: sequence length (time/positions)
- D: hidden size / embedding dimension (d_model)
- V: vocabulary size
"""

import jax.numpy as jnp
import flax.linen as nn
from flax.linen import attention as attn

class SwiGLUMLP(nn.Module):
    """Transformer feed-forward network with SwiGLU activation."""

    d_model: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        hidden = int(self.d_model * self.mlp_ratio)
        
        # Project to 2*hidden for SwiGLU
        x_proj = nn.Dense(2 * hidden)(x)
        x1, x2 = jnp.split(x_proj, 2, axis=-1)
        
        # SwiGLU activation: x1 * silu(x2)
        x = x1 * nn.silu(x2)

        # Project back to d_model
        x = nn.Dense(self.d_model)(x)

        # MLP dropout
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        return x


class DecoderBlock(nn.Module):
    """A single decoder block (Pre-LayerNorm + Self-Attn + SwiGLU MLP + residuals)."""
    d_model: int
    n_heads: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, *, mask=None, deterministic: bool = True):
        # Attention sublayer: Pre-LayerNorm -> Self-Attention -> Residual add
        h = nn.LayerNorm()(x)
        h = nn.SelfAttention(
            num_heads=self.n_heads,
            use_bias=False,
            dropout_rate=self.dropout_rate,
        )(h, mask=mask, deterministic=deterministic)
        
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
        x = x + h  # residual connection


        # MLP sublayer: Pre-LayerNorm -> MLP -> Residual add
        h = nn.LayerNorm()(x)
        h = SwiGLUMLP(self.d_model, mlp_ratio=self.mlp_ratio,
                      dropout_rate=self.dropout_rate)(h, deterministic=deterministic)
        x = x + h  # residual connection
        
        return x

class DecoderOnlyTransformer(nn.Module):
    """GPT-style decoder-only Transformer for language modeling.

    Components:
      - Token embeddings: maps token ids to D-dim vectors
      - Learned positional embeddings: adds position information (0..T-1)
      - N stacked decoder blocks with causal self-attention
      - Final LayerNorm
      - Output projection:
          * If tie_weights=True (default), reuse token embedding matrix E to
            compute logits via x @ E^T (implemented via einsum).
          * Else, use a separate linear head to project to V logits.

    Args:
      vocab_size: Vocabulary size V.
      d_model: Hidden size D.
      n_layers: Number of decoder blocks.
      n_heads: Attention heads per block.
      max_len: Maximum supported sequence length for positional embeddings.
    """

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    max_len: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    tie_weights: bool = False

    def setup(self):
        # Token embedding table E with shape (V, D)
        self.tok_embed = nn.Embed(self.vocab_size, self.d_model)

        # Learned positional embeddings P with shape (max_len, D)
        # We'll slice P[:T] each forward pass and add to token embeddings.
        self.positional_embed = self.param(
            "positional_embed",
            nn.initializers.normal(stddev=0.02),
            (self.max_len, self.d_model)
        )

        # Stack of decoder blocks
        self.blocks = [
            DecoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
              )
              for _ in range(self.n_layers)
        ]

        # Final LayerNorm before projecting to logits
        self.layerNorm_final = nn.LayerNorm()

        # Optional separate output head if not weight-tying
        self.project_to_vocab = nn.Dense(self.vocab_size, use_bias=False)

    @nn.compact
    def __call__(self, idx, deterministic: bool = True):
        """Forward pass (causal-only).

        Args:
          idx: Token ids of shape (B, T), dtype int32/int64.

        Returns:
          logits: (B, T, V) unnormalized vocabulary scores for next-token prediction.
        """
        B, T = idx.shape

        # Token + positional embeddings -> (B, T, D)
        x = self.tok_embed(idx) + self.positional_embed[:T]

        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        # Build attention mask: strictly causal (lower-triangular), no padding mask.
        causal = attn.make_causal_mask(jnp.ones((B, T), dtype=bool))

        # Stack of decoder blocks
        for i in range(self.n_layers):
            x = DecoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate
            )(x, mask=causal, deterministic=deterministic)

        # Final LayerNorm before output projection
        x = self.layerNorm_final(x)

        # Weight tying: project using token embedding matrix
        if self.tie_weights:
            logits = jnp.einsum('btd,vd->btv', x, self.tok_embed.embedding)
        else:
            # Output projection to logits over V tokens.
            logits = self.project_to_vocab(x)
        
        return logits