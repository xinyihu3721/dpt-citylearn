"""
Causal (GPT-style) Decision-Pretrained Transformer for CityLearn.

- Transition token: linear embedding of concat(obs[52], action[9], reward[1], next_obs[52]) = 114-d
  -> d_model.
- Query token: SEPARATE linear embedding of query obs[52] -> d_model, placed FIRST in the sequence
  (DPT-paper convention -- corrected 2026-07-01, was query-LAST; see CLAUDE.md). Causal masking
  means every context token can attend back to the query, so query information propagates forward
  through the whole sequence rather than only being read out at a single dedicated position.
- No positional encoding on context tokens (context is treated as an order-agnostic set); the query
  is distinguished from context tokens by using its own embedding matrix, not by a position index.
- Causal self-attention (GPT-style): combined with a key-padding mask so no token attends to padded
  context slots.
- Output is read at the LAST VALID position of the sequence (query position 0 if context is empty,
  else the last non-padded context token) via a per-example gather on context_mask.sum(dim=1) --
  NOT a fixed index, since query-first + right-padded context means the physically-last sequence
  slot is a padding slot whenever h < H_max. A per-action-dim Linear(d_model -> n_bins) head then
  produces bin logits for each of the 9 action dims from that gathered representation.
"""
import math

import torch
import torch.nn as nn

OBS_DIM = 52
ACTION_DIM = 9
REWARD_DIM = 1
TRANSITION_DIM = OBS_DIM + ACTION_DIM + REWARD_DIM + OBS_DIM  # 114


class DPT(nn.Module):
    def __init__(self, d_model=256, n_layers=4, n_heads=8, dropout=0.1, n_bins=21,
                 action_dim=ACTION_DIM, obs_dim=OBS_DIM, transition_dim=TRANSITION_DIM):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.n_bins = n_bins

        self.transition_embed = nn.Linear(transition_dim, d_model)
        self.query_embed = nn.Linear(obs_dim, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)

        self.action_heads = nn.ModuleList([nn.Linear(d_model, n_bins) for _ in range(action_dim)])

    def forward(self, context_obs, context_action, context_reward, context_next_obs,
                context_mask, query_obs):
        """
        context_obs:      (B, H, obs_dim)
        context_action:   (B, H, action_dim)
        context_reward:   (B, H)
        context_next_obs: (B, H, obs_dim)
        context_mask:     (B, H) bool, True = valid (non-padded) context position
        query_obs:        (B, obs_dim)

        Returns: list of action_dim tensors, each (B, n_bins) bin logits.
        """
        bsz, h_len, _ = context_obs.shape
        device = context_obs.device

        transition = torch.cat(
            [context_obs, context_action, context_reward.unsqueeze(-1), context_next_obs], dim=-1
        )  # (B, H, transition_dim)
        context_tok = self.transition_embed(transition)  # (B, H, d_model)
        query_tok = self.query_embed(query_obs).unsqueeze(1)  # (B, 1, d_model)

        tokens = torch.cat([query_tok, context_tok], dim=1)  # (B, 1+H, d_model) -- query FIRST
        tokens = self.embed_dropout(tokens)
        seq_len = h_len + 1

        # causal mask: position i cannot attend to position j > i
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1
        )

        # key padding mask: True = ignore. The query slot (now position 0) is always valid.
        query_valid = torch.ones(bsz, 1, dtype=torch.bool, device=device)
        valid = torch.cat([query_valid, context_mask], dim=1)  # (B, 1+H)
        key_padding_mask = ~valid  # True = ignore

        out = self.transformer(tokens, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        out = self.ln_f(out)

        # Gather the last VALID position per example: query is at index 0, valid context occupies
        # indices [1, h], so index h is the last valid slot (h=0 -> gather the query itself).
        valid_lengths = context_mask.sum(dim=1)  # (B,), = h per example
        readout = out[torch.arange(bsz, device=device), valid_lengths, :]  # (B, d_model)

        bin_logits = [head(readout) for head in self.action_heads]  # each (B, n_bins)
        return bin_logits
