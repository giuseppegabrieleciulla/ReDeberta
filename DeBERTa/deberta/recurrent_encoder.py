import torch
from torch import nn
from collections.abc import Sequence

from .bert import BertLayer, LayerNorm, ConvLayer
from .da_utils import build_relative_position

class RecurrentBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = BertLayer(config)
        self.max_steps = getattr(config, 'num_hidden_layers', 24)
        
        self.relative_attention = getattr(config, 'relative_attention', False)
        if self.relative_attention:
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.position_buckets = getattr(config, 'position_buckets', -1)
            pos_ebd_size = self.max_relative_positions * 2
            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2
            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

        self.norm_rel_ebd = [x.strip() for x in getattr(config, 'norm_rel_ebd', 'none').lower().split('|')]
        if 'layer_norm' in self.norm_rel_ebd:
            self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)
            
        kernel_size = getattr(config, 'conv_kernel_size', 0)
        self.with_conv = False
        if kernel_size > 0:
            self.with_conv = True
            self.conv = ConvLayer(config)

        # ACT Halting head
        self.halting_head = nn.Linear(config.hidden_size, 1)
        # Initialize bias to negative value to encourage more steps at start
        nn.init.constant_(self.halting_head.bias, -1.0)
        
        self.halting_threshold = getattr(config, 'halting_threshold', 0.99)

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and ('layer_norm' in self.norm_rel_ebd):
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
            attention_mask = attention_mask.byte()
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(q, hidden_states.size(-2), bucket_size=self.position_buckets,
                                                   max_position=self.max_relative_positions, device=hidden_states.device)
        return relative_pos

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, return_att=False, query_states=None, relative_pos=None):
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = (attention_mask.sum(-2) > 0).byte()
            
        attention_mask_ext = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)
        rel_embeddings = self.get_rel_embedding()

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states

        batch_size = next_kv.size(0)
        device = next_kv.device

        # ACT loop state
        accumulated_states = torch.zeros_like(next_kv)
        cumulative_halting_probs = torch.zeros(batch_size, 1, 1, device=device)
        updates = torch.zeros(batch_size, 1, 1, dtype=torch.bool, device=device)

        # Expected steps (ponder cost)
        expected_steps = torch.zeros(batch_size, 1, 1, device=device)
        total_steps_taken = torch.ones(batch_size, 1, 1, device=device)

        all_encoder_layers = []
        att_matrices = []

        is_training = self.training

        for i in range(self.max_steps):
            output_states = self.layer(next_kv, attention_mask_ext, return_att, query_states=query_states, 
                                       relative_pos=relative_pos, rel_embeddings=rel_embeddings)
            if return_att:
                output_states, att_m = output_states

            if i == 0 and self.with_conv:
                prenorm = output_states
                output_states = self.conv(hidden_states, prenorm, input_mask)

            # Halting probability based on CLS token of output_states
            cls_state = output_states[:, 0:1, :] # (batch, 1, dim)
            halt_logits = self.halting_head(cls_state) # (batch, 1, 1)
            c_t = torch.sigmoid(halt_logits)

            is_last_step = (i == self.max_steps - 1)
            still_running = ~updates
            remainder = 1.0 - cumulative_halting_probs
            
            if is_last_step:
                p_t = remainder
            else:
                p_t = torch.where(c_t < remainder, c_t, remainder)
            
            p_t = p_t * still_running.float()
            
            accumulated_states = accumulated_states + p_t * output_states
            cumulative_halting_probs = cumulative_halting_probs + p_t
            
            expected_steps = expected_steps + p_t * (i + 1)
            
            new_updates = cumulative_halting_probs >= self.halting_threshold
            total_steps_taken = total_steps_taken + (still_running & ~new_updates).float()
            updates = updates | new_updates
            
            next_kv = output_states

            if output_all_encoded_layers:
                all_encoder_layers.append(accumulated_states.clone())
                if return_att:
                    att_matrices.append(att_m)

            if not is_training and updates.all():
                break

        if not output_all_encoded_layers:
            all_encoder_layers.append(accumulated_states)
            if return_att:
                att_matrices.append(att_m)

        ponder_cost = expected_steps.squeeze(-1).squeeze(-1) # (batch,)

        res = {
            'hidden_states': all_encoder_layers,
            'ponder_cost': ponder_cost
        }
        if return_att:
            res['attention_matrices'] = att_matrices
        return res
