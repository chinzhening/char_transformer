import jax.numpy as jnp
import flax.linen as nn
from flax.linen import attention as attn


class SwiGLUMLP(nn.Module):
    """Transformer feed-forward network with SwiGLU activation."""

    d_model: int
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x):
        hidden = int(self.d_model * self.mlp_ratio)
        
        # Project to 2*hidden for SwiGLU
        x_proj = nn.Dense(2 * hidden)(x)
        
        # Split channels
        x1, x2 = jnp.split(x_proj, 2, axis=-1)
        
        # SwiGLU activation: x1 * silu(x2)
        x = x1 * nn.silu(x2)
        
        # Project back to d_model
        x = nn.Dense(self.d_model)(x)
        return x


class DecoderBlock(nn.Module):
    """A single decoder block (Pre-LayerNorm + Self-Attn + SwiGLU MLP + residuals)."""
    d_model: int
    n_heads: int
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x, *, mask=None):
        # Attention sublayer: Pre-LayerNorm -> Self-Attention -> Residual add
        h = nn.LayerNorm()(x)
        h = nn.SelfAttention(
            num_heads=self.n_heads,
            use_bias=False,
        )(h, mask=mask)
        x = x + h  # residual connection

        # MLP sublayer: Pre-LayerNorm -> MLP -> Residual add
        h = nn.LayerNorm()(x)
        h = SwiGLUMLP(self.d_model, mlp_ratio=self.mlp_ratio)(h)
        x = x + h  # residual connection
        return x


class DecoderOnlyTransformer(nn.Module):
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    max_len: int
    mlp_ratio: int = 4
    tie_weights: bool = False

    def setup(self):
        # Token embedding
        self.tok_embed = nn.Embed(self.vocab_size, self.d_model)

        # Transformer blocks
        self.blocks = [
            DecoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio
            )
            for _ in range(self.n_layers)
        ]

        self.layerNorm_final = nn.LayerNorm()
        self.project_to_vocab = nn.Dense(self.vocab_size, use_bias=False)

    def sinusoidal_positions(self, max_len, d_model):
        position = jnp.arange(max_len)[:, None]              # (T, 1)
        div_term = jnp.exp(
            jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model)
        )                                                    # (D/2,)

        pe = jnp.zeros((max_len, d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        return pe

    def __call__(self, idx, deterministic: bool = True):
        B, T = idx.shape

        # Sinusoidal positional encodings
        pos_embed = self.sinusoidal_positions(T, self.d_model)

        # Token + position embeddings
        x = self.tok_embed(idx) + pos_embed

        # Causal mask
        causal = attn.make_causal_mask(jnp.ones((B, T), dtype=bool))

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x, mask=causal)

        # Final norm
        x = self.layerNorm_final(x)

        # Weight tying vs separate projection
        if self.tie_weights:
            logits = jnp.einsum("btd,vd->btv", x, self.tok_embed.embedding)
        else:
            logits = self.project_to_vocab(x)

        return logits
