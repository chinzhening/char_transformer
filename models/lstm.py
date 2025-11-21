# write a LSTM cell using jax/flax
# then implement a multi-layer LSTM model with token embeddings,
# positional embeddings, stacked LSTM layers, and an output projection layer.

import jax.numpy as jnp 
import flax.linen as nn
from typing import Any, Callable, Optional, Tuple

class LSTMCell(nn.Module):
    """A single LSTM cell.

    Args:
        hidden_size: The number of features in the hidden state h.
    
    Input shape:  (B, input_size)
    Output shape: (B, hidden_size)
    """

    hidden_size: int

    @nn.compact
    def __call__(self, c_h, x):
        c, h = c_h  # Unpack hidden state and cell state
        input_size = x.shape[-1]

        # Concatenate input and hidden state
        concat = jnp.concatenate([x, h], axis=-1)

        # Compute gates
        gates = nn.Dense(4 * self.hidden_size)(concat)
        i, f, g, o = jnp.split(gates, 4, axis=-1)

        # Apply activations
        i = nn.sigmoid(i)  # input gate
        f = nn.sigmoid(f)  # forget gate
        g = jnp.tanh(g)    # cell candidate
        o = nn.sigmoid(o)  # output gate

        # Update cell state and hidden state
        new_c = f * c + i * g
        new_h = o * jnp.tanh(new_c)

        return (new_c, new_h), new_h  # Return new hidden state and (h, c) tuple
    
class LSTM(nn.Module):
    """A multi-layer LSTM module.

    Components:
      - Token embeddings: map token ids to D-dim vectors.
      - Learned positional embeddings: adds position information (0..T-1).
      - Stacked LSTM layers
      - Output projection layer: maps hidden states to output logits.
    """

    vocab_size: int
    hidden_size: int
    num_layers: int

    @nn.compact
    def __call__(self, x, initial_states=None):
        B, T = x.shape  # Batch size and sequence length

        # Token embeddings
        h = nn.Embed(self.vocab_size, self.hidden_size)(x)  # (B, T, D)

        # Initialize LSTM states if not provided
        if initial_states is None:
            initial_states = [
                (jnp.zeros((B, self.hidden_size)), jnp.zeros((B, self.hidden_size)))
                for _ in range(self.num_layers)
            ]

        # Stacked LSTM layers
        states = initial_states
        for layer_idx in range(self.num_layers):
            lstm_cell = LSTMCell(self.hidden_size)
            new_states = []
            outputs = []
            for t in range(T):
                (states[layer_idx], h_t) = lstm_cell(states[layer_idx], h[:, t, :])
                outputs.append(h_t)
            h = jnp.stack(outputs, axis=1)  # (B, T, D)
            states[layer_idx] = states[layer_idx]

        # Output projection layer
        logits = nn.Dense(self.vocab_size)(h)  # (B, T, vocab_size)

        return logits  # Return logits and final states
