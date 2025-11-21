import jax.numpy as jnp
import flax.linen as nn
from flax.linen import attention as attn

class MLP(nn.Module):
    d_model: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # Expand channel dimension (D -> hidden), apply non-linearity, project back to D.
        hidden = int(self.d_model * self.mlp_ratio)
        x = nn.Dense(hidden)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.d_model)(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        return x

class DecoderBlock(nn.Module):
    d_model: int
    n_heads: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, *, mask=None, deterministic: bool = True):
        # Attention sublayer: Pre-LayerNorm -> Self-Attention + dropout -> Dropout -> Residual add
        h = nn.LayerNorm()(x)
        h = nn.SelfAttention(
            num_heads=self.n_heads,
            use_bias=False,
            dropout_rate=self.dropout_rate,
        )(h, mask=mask, deterministic=deterministic)
        
        h = nn.Dropout(self.dropout_rate)(h, deterministic=deterministic) 
        x = x + h  # residual connection

        # MLP sublayer: Pre-LayerNorm -> MLP + dropout -> Residual add
        h = nn.LayerNorm()(x)
        h = MLP(self.d_model, mlp_ratio=self.mlp_ratio, dropout_rate=self.dropout_rate)(h, deterministic=deterministic)
        x = x + h  # residual connection
        return x

class DecoderOnlyTransformer(nn.Module):
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    max_len: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.1

    def setup(self):
        # Token embedding table E with shape (V, D)
        self.tok_embed = nn.Embed(self.vocab_size, self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)

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
            ) for _ in range(self.n_layers)]

        # Final LayerNorm before projecting to logits
        self.layerNorm_final = nn.LayerNorm()

        # Optional separate output head if not weight-tying
        self.project_to_vocab = nn.Dense(self.vocab_size, use_bias=False)

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
        x = self.dropout(x, deterministic=deterministic)

        # Build attention mask: strictly causal (lower-triangular), no padding mask.
        causal = attn.make_causal_mask(jnp.ones((B, T), dtype=bool))
        mask = causal

        # Run the stack of decoder blocks
        for blk in self.blocks:
            x = blk(x, mask=mask, deterministic=deterministic)

        # Final LayerNorm before output projection
        x = self.layerNorm_final(x)

        # Output projection to logits over V tokens.
        logits = self.project_to_vocab(x)
        
        return logits