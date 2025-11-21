import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple

class LSTMCell(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, c_h, x):
        c, h = c_h
        concat = jnp.concatenate([x, h], axis=-1)
        gates = nn.Dense(4 * self.hidden_size)(concat)
        i, f, g, o = jnp.split(gates, 4, axis=-1)

        i = nn.sigmoid(i)
        f = nn.sigmoid(f)
        g = jnp.tanh(g)
        o = nn.sigmoid(o)

        new_c = f * c + i * g
        new_h = o * jnp.tanh(new_c)
        return (new_c, new_h), new_h


class BiLSTM(nn.Module):
    vocab_size: int
    hidden_size: int
    num_layers: int
    max_length: int

    @nn.compact
    def __call__(self, x, initial_states=None):
        B, T = x.shape

        # --- Embeddings ---
        token_emb = nn.Embed(self.vocab_size, self.hidden_size)(x)     # (B, T, D)
        pos_emb = nn.Embed(self.max_length, self.hidden_size)(
            jnp.arange(T)[None, :]
        )                                                               # (1, T, D)

        h = token_emb + pos_emb                                        # (B, T, D)

        # Build default states if none are provided
        if initial_states is None:
            initial_states = []
            for _ in range(self.num_layers):
                forward = (
                    jnp.zeros((B, self.hidden_size)),
                    jnp.zeros((B, self.hidden_size)),
                )
                backward = (
                    jnp.zeros((B, self.hidden_size)),
                    jnp.zeros((B, self.hidden_size)),
                )
                initial_states.append((forward, backward))

        # --- Stacked Bi-LSTM layers ---
        for layer_idx in range(self.num_layers):
            f_state, b_state = initial_states[layer_idx]

            forward_cell = LSTMCell(self.hidden_size)
            backward_cell = LSTMCell(self.hidden_size)

            # Forward pass
            f_outputs = []
            for t in range(T):
                (f_state, h_f) = forward_cell(f_state, h[:, t, :])
                f_outputs.append(h_f)
            f_outputs = jnp.stack(f_outputs, axis=1)                   # (B, T, D)

            # Backward pass
            b_outputs = []
            for t in reversed(range(T)):
                (b_state, h_b) = backward_cell(b_state, h[:, t, :])
                b_outputs.append(h_b)
            b_outputs = jnp.stack(b_outputs[::-1], axis=1)             # (B, T, D)

            # Concatenate forward + backward
            h = jnp.concatenate([f_outputs, b_outputs], axis=-1)       # (B, T, 2D)

        # --- Output projection ---
        logits = nn.Dense(self.vocab_size)(h)                          # (B, T, vocab)

        return logits
