"""
Shared DPT batch collator. Pads a batch of variable-length (context, query, label) examples
(as produced by data/dpt_dataset_hard.py's HardTaskDPTDataset) to a common context length with a
boolean validity mask.
"""
import torch


def collate_dpt(batch, h_max=None):
    """Pads variable-length context to a common length with a boolean validity mask."""
    obs_dim = batch[0]['context_obs'].shape[1]
    action_dim = batch[0]['context_action'].shape[1]
    pad_len = h_max if h_max is not None else max(ex['h'] for ex in batch)
    bsz = len(batch)

    context_obs = torch.zeros(bsz, pad_len, obs_dim, dtype=torch.float32)
    context_action = torch.zeros(bsz, pad_len, action_dim, dtype=torch.float32)
    context_reward = torch.zeros(bsz, pad_len, dtype=torch.float32)
    context_next_obs = torch.zeros(bsz, pad_len, obs_dim, dtype=torch.float32)
    context_mask = torch.zeros(bsz, pad_len, dtype=torch.bool)

    query_obs = torch.stack([ex['query_obs'] for ex in batch])
    action_label = torch.stack([ex['action_label'] for ex in batch])
    seeds = torch.tensor([ex['seed'] for ex in batch], dtype=torch.long)
    hs = torch.tensor([ex['h'] for ex in batch], dtype=torch.long)

    for i, ex in enumerate(batch):
        h = ex['h']
        context_obs[i, :h] = ex['context_obs']
        context_action[i, :h] = ex['context_action']
        context_reward[i, :h] = ex['context_reward']
        context_next_obs[i, :h] = ex['context_next_obs']
        context_mask[i, :h] = True

    return {
        'seed': seeds,
        'h': hs,
        'context_obs': context_obs,
        'context_action': context_action,
        'context_reward': context_reward,
        'context_next_obs': context_next_obs,
        'context_mask': context_mask,
        'query_obs': query_obs,
        'action_label': action_label,
    }
