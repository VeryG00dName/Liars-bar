# src/model/ppo_reactive_model_base.py

"""Shared PPO reactive model components."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple, TypedDict

import torch
import torch.nn as nn


__all__ = [
    "PPOReactiveModelBase",
    "MoETransformerEncoderLayer",
    "AttentionCacheEntry",
]


AttentionCacheEntry = Dict[str, torch.Tensor]


class SelfAttentionWithCache(nn.Module):
    """Multi-head self-attention that supports key/value caching."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._norm_factor = float(self.head_dim) ** -0.5

    def _shape(self, tensor: torch.Tensor) -> torch.Tensor:
        B, T, _ = tensor.shape
        return tensor.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _combine(self, tensor: torch.Tensor) -> torch.Tensor:
        B, H, T, D = tensor.shape
        return tensor.transpose(1, 2).reshape(B, T, H * D)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        cache: Optional[AttentionCacheEntry] = None,
    ) -> Tuple[torch.Tensor, AttentionCacheEntry]:
        """Run attention, optionally using a key/value cache."""

        q = self._shape(self.q_proj(x))
        new_k = self._shape(self.k_proj(x))
        new_v = self._shape(self.v_proj(x))

        # --- THIS IS THE FIX ---
        # Initialize variables to None to ensure they are always defined
        # for the JIT compiler, regardless of the 'cache' branch.
        past_k: Optional[torch.Tensor] = None
        past_v: Optional[torch.Tensor] = None
        past_lengths: Optional[torch.Tensor] = None

        if cache is not None:
            past_k = cache["key"]
            past_v = cache["value"]
            past_lengths = cache["lengths"]
        # The 'else' part is now handled by the initialization above.

        if past_k is None:
            combined_k = new_k
            combined_v = new_v
            past_len = 0
        else:
            assert past_v is not None
            past_len = int(past_k.size(2))
            combined_k = torch.cat([past_k, new_k], dim=2)
            combined_v = torch.cat([past_v, new_v], dim=2)

        total_len = combined_k.size(2)
        B, H, T, _ = q.shape

        attn_weights = torch.matmul(q, combined_k.transpose(-2, -1)) * self._norm_factor

        if past_lengths is not None and past_len > 0:
            past_positions = torch.arange(past_len, device=attn_weights.device)
            mask = past_positions.view(1, 1, 1, past_len) >= past_lengths.view(B, 1, 1, 1)
            attn_weights[..., :past_len] = attn_weights[..., :past_len].masked_fill(
                mask, float("-inf")
            )

        if key_padding_mask is not None:
            # key_padding_mask has shape [B, T_chunk]; apply it to the last
            # T_chunk columns (new chunk) using broadcast without changing
            # element count: [B, 1, 1, T_chunk].
            chunk_mask = key_padding_mask.view(B, 1, 1, key_padding_mask.size(-1))
            attn_weights[..., past_len:] = attn_weights[..., past_len:].masked_fill(
                chunk_mask, float("-inf")
            )

        if attn_mask is not None:
            if attn_mask.dim() != 2 or attn_mask.size(0) != T:
                raise ValueError("attn_mask must be of shape [T, T]")
            causal_mask = attn_mask
            if causal_mask.dtype == torch.bool:
                mask_tensor = causal_mask.unsqueeze(0).unsqueeze(0)
                attn_weights[..., past_len:] = attn_weights[..., past_len:].masked_fill(
                    mask_tensor, float("-inf")
                )
            else:
                attn_weights[..., past_len:] = attn_weights[..., past_len:] + causal_mask.unsqueeze(0).unsqueeze(0)

        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        context = torch.matmul(attn_probs, combined_v)

        output = self.out_proj(self._combine(context))

        if past_lengths is not None:
            base_lengths = past_lengths.to(torch.long)
        else:
            base_lengths = torch.full((B,), past_len, dtype=torch.long, device=output.device)

        if key_padding_mask is not None:
            chunk_lengths = (~key_padding_mask).sum(dim=1).to(torch.long)
        else:
            chunk_lengths = torch.full((B,), T, dtype=torch.long, device=output.device)

        updated_lengths = base_lengths + chunk_lengths

        updated_cache: AttentionCacheEntry = {
            "key": combined_k.detach(),
            "value": combined_v.detach(),
            "lengths": updated_lengths.detach(),
        }

        return output, updated_cache


def _make_moe_ffn(hidden_dim: int, expert_ffn_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, expert_ffn_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(expert_ffn_dim, hidden_dim),
        nn.Dropout(dropout),
    )


class TopKMoELayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        expert_ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList(
            [_make_moe_ffn(hidden_dim, expert_ffn_dim, dropout) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        gate_logits = self.gate(x)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        topk_scores, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        expert_outs: List[torch.Tensor] = []
        for expert in self.experts:
            expert_outs.append(expert(x))
        expert_outputs = torch.stack(expert_outs, dim=2)

        bsz, tsz, ksz = topk_indices.shape
        gather_index = topk_indices.unsqueeze(-1).expand(bsz, tsz, ksz, self.hidden_dim)
        
        topk_outputs = torch.gather(expert_outputs, 2, gather_index)
        combined = (topk_outputs * topk_weights.unsqueeze(-1)).sum(dim=2)

        routing_info = {
            "gate_logits": gate_logits,
            "topk_indices": topk_indices,
            "topk_scores": topk_weights,
        }
        return combined, routing_info


class MoETransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        num_experts: int,
        top_k: int,
        expert_ffn_dim: int,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.self_attn = SelfAttentionWithCache(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.moe = TopKMoELayer(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            expert_ffn_dim=expert_ffn_dim,
            dropout=dropout,
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[AttentionCacheEntry] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], AttentionCacheEntry]:
        attn_output, updated_cache = self.self_attn(
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            cache=kv_cache,
        )
        src = self.norm1(src + self.dropout1(attn_output))
        moe_output, routing = self.moe(src)
        src = self.norm2(src + self.dropout2(moe_output))
        return src, routing, updated_cache


class MoETransformerEncoder(nn.Module):
    def __init__(self, layer: MoETransformerEncoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(layer.self_attn.embed_dim)

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[AttentionCacheEntry]] = None,
    ) -> Tuple[
        torch.Tensor,
        List[torch.Tensor],
        Dict[str, torch.Tensor],
        List[AttentionCacheEntry],
    ]:
        gate_logits: List[torch.Tensor] = []
        routing: Dict[str, torch.Tensor] = {}
        new_caches: List[AttentionCacheEntry] = []
        output = src
        for idx, layer in enumerate(self.layers):
            layer_cache = torch.jit.annotate(Optional[AttentionCacheEntry], None)
            if kv_cache is not None and idx < len(kv_cache):
                layer_cache = kv_cache[idx]
            output, routing, updated_cache = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                kv_cache=layer_cache,
            )
            gate_logits.append(routing["gate_logits"])
            new_caches.append(updated_cache)
        output = self.norm(output)
        return output, gate_logits, routing, new_caches


class PPOReactiveModelBase(nn.Module):
    """Shared architecture for PPO reactive models."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        max_seq_length: int = 480,
        num_agent_types: int = 4,
        *,
        num_experts: int = 8,
        top_k: int = 2,
        expert_ffn_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_ffn_dim = expert_ffn_dim or hidden_dim * 2
        self.count_pad = 4
        self.tflag_pad = 3

        self.register_buffer(
            "causal_bool_mask_full",
            torch.triu(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool), 1),
        )

        # === Input Encoders ===
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.act_kind_embedding = nn.Embedding(3, hidden_dim, padding_idx=0)
        self.count_embedding = nn.Embedding(5, hidden_dim, padding_idx=self.count_pad)
        self.table_flag_embedding = nn.Embedding(4, hidden_dim, padding_idx=self.tflag_pad)
        self.agent_embedding = nn.Embedding(num_agent_types, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_dim)

        # === Gating Layers (Independent) ===
        def make_gate_net(h_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.Tanh(),
                nn.Linear(h_dim, h_dim),
                nn.Sigmoid(),
            )

        self.gate_obs = make_gate_net(hidden_dim)
        self.gate_action = make_gate_net(hidden_dim)
        self.gate_agent = make_gate_net(hidden_dim)
        self.gate_position = make_gate_net(hidden_dim)

        # === Factorization Look-up Tables ===
        self.register_buffer(
            "lut_act_kind", torch.tensor([1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0], dtype=torch.long)
        )
        self.register_buffer(
            "lut_count", torch.tensor([1, 2, 3, 1, 2, 3, 0, 1, 2, 3, 4], dtype=torch.long)
        )
        self.register_buffer(
            "lut_table_flag", torch.tensor([1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 3], dtype=torch.long)
        )

        # === Transformer Backbone ===
        base_layer = MoETransformerEncoderLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            num_experts=self.num_experts,
            top_k=self.top_k,
            expert_ffn_dim=self.expert_ffn_dim,
        )
        self.transformer = MoETransformerEncoder(base_layer, num_layers=num_layers)

        def make_head(out_dim: int) -> nn.ModuleList:
            return nn.ModuleList([nn.Linear(hidden_dim, out_dim) for _ in range(self.num_experts)])

        self.action_heads = make_head(action_dim)
        self.reward_stream_heads = make_head(1)
        self.win_prob_heads = make_head(1)
        self.opp_action_heads = make_head(action_dim)

    # -------------------------- utils --------------------------
    # NOTE: Avoid @torch.no_grad() on scripted methods; TorchScript can’t script it.
    def _decompose_actions(
        self,
        action_sequence: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = action_sequence.long()
        act_kind, count, tflag = self.lut_act_kind[a], self.lut_count[a], self.lut_table_flag[a]
        if padding_mask is not None:
            act_pad = torch.zeros_like(act_kind)
            count_pad = torch.full_like(count, self.count_pad, dtype=torch.long)
            tflag_pad = torch.full_like(tflag, self.tflag_pad, dtype=torch.long)
            act_kind = torch.where(padding_mask, act_pad, act_kind)
            count = torch.where(padding_mask, count_pad, count)
            tflag = torch.where(padding_mask, tflag_pad, tflag)
        return act_kind, count, tflag

    def _encode_inputs(
        self,
        obs_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        agent_types: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        obs_embed = self.obs_encoder(obs_sequence)
        act_kind_ids, count_ids, table_flag_ids = self._decompose_actions(action_sequence, padding_mask)
        action_embed = (
            self.act_kind_embedding(act_kind_ids)
            + self.count_embedding(count_ids)
            + self.table_flag_embedding(table_flag_ids)
        )
        agent_embed = self.agent_embedding(agent_types)
        position_embed = self.position_embedding(positions)

        g_obs = self.gate_obs(obs_embed)
        g_action = self.gate_action(action_embed)
        g_agent = self.gate_agent(agent_embed)
        g_position = self.gate_position(position_embed)

        fused = (
            g_obs * obs_embed
            + g_action * action_embed
            + g_agent * agent_embed
            + g_position * position_embed
        )
        combined = nn.functional.layer_norm(fused, (self.hidden_dim,))
        return combined

    def _prepare_masks(
        self, encoded_inputs: torch.Tensor, padding_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        T = encoded_inputs.size(1)
        causal_mask = self.causal_bool_mask_full[:T, :T].to(encoded_inputs.device)
        key_padding = torch.jit.annotate(Optional[torch.Tensor], None)
        if padding_mask is not None:
            key_padding = padding_mask.to(dtype=torch.bool).contiguous()
        return causal_mask, key_padding

    def _run_transformer(
        self,
        encoded_inputs: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding: Optional[torch.Tensor],
        kv_cache: Optional[List[AttentionCacheEntry]] = None,
    ) -> Tuple[
        torch.Tensor,
        List[torch.Tensor],
        Dict[str, torch.Tensor],
        List[AttentionCacheEntry],
    ]:
        return self.transformer(
            encoded_inputs,
            mask=causal_mask,
            src_key_padding_mask=key_padding,
            kv_cache=kv_cache,
        )

    # --------- TorchScript-safe helpers that read ModuleLists via self ---------
    def _stack_action_heads(self, transformer_output: torch.Tensor) -> torch.Tensor:
        outs = torch.jit.annotate(List[torch.Tensor], [])
        for _, head in enumerate(self.action_heads):  # iterate attribute, not index with i
            outs.append(head(transformer_output))
        return torch.stack(outs, dim=2)

    def _stack_reward_heads(self, transformer_output: torch.Tensor) -> torch.Tensor:
        outs = torch.jit.annotate(List[torch.Tensor], [])
        for _, head in enumerate(self.reward_stream_heads):
            outs.append(head(transformer_output))
        return torch.stack(outs, dim=2)

    def _stack_win_heads(self, transformer_output: torch.Tensor) -> torch.Tensor:
        outs = torch.jit.annotate(List[torch.Tensor], [])
        for _, head in enumerate(self.win_prob_heads):
            outs.append(head(transformer_output))
        return torch.stack(outs, dim=2)

    def _stack_opp_heads(self, transformer_output: torch.Tensor) -> torch.Tensor:
        outs = torch.jit.annotate(List[torch.Tensor], [])
        for _, head in enumerate(self.opp_action_heads):
            outs.append(head(transformer_output))
        return torch.stack(outs, dim=2)

    def _reduce_heads(
        self,
        stacked: torch.Tensor,                      # [B, T, H, D]
        final_indices: Optional[torch.Tensor],      # [B, T, K] or None
        final_scores: Optional[torch.Tensor],       # [B, T, K] or None
    ) -> torch.Tensor:
        if final_indices is None or final_scores is None:
            return stacked.mean(dim=2)              # [B, T, D]
        bsz, tsz, ksz = final_indices.shape
        out_dim = stacked.size(-1)
        gather_idx = final_indices.unsqueeze(-1).expand(bsz, tsz, ksz, out_dim)  # [B,T,K,D]
        top_outputs = torch.gather(stacked, 2, gather_idx)                        # [B,T,K,D]
        return (top_outputs * final_scores.unsqueeze(-1)).sum(dim=2)              # [B,T,D]

    def _combine_action_heads(
        self,
        transformer_output: torch.Tensor,
        final_indices: Optional[torch.Tensor],
        final_scores: Optional[torch.Tensor],
    ) -> torch.Tensor:
        stacked = self._stack_action_heads(transformer_output)
        return self._reduce_heads(stacked, final_indices, final_scores)

    def _combine_reward_heads(
        self,
        transformer_output: torch.Tensor,
        final_indices: Optional[torch.Tensor],
        final_scores: Optional[torch.Tensor],
    ) -> torch.Tensor:
        stacked = self._stack_reward_heads(transformer_output)
        return self._reduce_heads(stacked, final_indices, final_scores)

    def _combine_win_heads(
        self,
        transformer_output: torch.Tensor,
        final_indices: Optional[torch.Tensor],
        final_scores: Optional[torch.Tensor],
    ) -> torch.Tensor:
        stacked = self._stack_win_heads(transformer_output)
        return self._reduce_heads(stacked, final_indices, final_scores)

    def _combine_opp_heads(
        self,
        transformer_output: torch.Tensor,
        final_indices: Optional[torch.Tensor],
        final_scores: Optional[torch.Tensor],
    ) -> torch.Tensor:
        stacked = self._stack_opp_heads(transformer_output)
        return self._reduce_heads(stacked, final_indices, final_scores)

    def _head_outputs(self, transformer_output: torch.Tensor, routing: Dict[str, torch.Tensor]):
        final_indices = torch.jit.annotate(Optional[torch.Tensor], routing.get("topk_indices"))
        final_scores = torch.jit.annotate(Optional[torch.Tensor], routing.get("topk_scores"))

        action_logits = self._combine_action_heads(transformer_output, final_indices, final_scores)
        state_values  = self._combine_reward_heads(transformer_output, final_indices, final_scores)
        win_logits    = self._combine_win_heads(transformer_output, final_indices, final_scores)
        opp_logits    = self._combine_opp_heads(transformer_output, final_indices, final_scores)
        return action_logits, opp_logits, state_values, win_logits


    # -------------------------- forward --------------------------
    def forward(
        self,
        obs_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        agent_types: torch.Tensor,
        positions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        valid_lengths: Optional[torch.Tensor] = None
    ):
        # obs_sequence: [B, T, obs_dim]
        # action_sequence: [B, T]
        # agent_types: [B, T]
        # positions: [B, T]
        encoded_inputs = self._encode_inputs(
            obs_sequence=obs_sequence,
            action_sequence=action_sequence,
            agent_types=agent_types,
            positions=positions,
            padding_mask=padding_mask,
        )

        causal_mask, key_padding = self._prepare_masks(encoded_inputs, padding_mask)

        transformer_output, _, routing, _ = self._run_transformer(
            encoded_inputs,
            causal_mask=causal_mask,
            key_padding=key_padding,
        )

        action_logits, opp_logits, state_values, win_logits = self._head_outputs(
            transformer_output,
            routing,
        )

        return action_logits, opp_logits, state_values, win_logits

    def _slice_new_tokens(
        self,
        tensor: torch.Tensor,
        prev_lengths: torch.Tensor,
        new_lengths: torch.Tensor,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        batch = int(tensor.size(0))
        total_len = int(tensor.size(1))

        chunk_lengths = (new_lengths - prev_lengths).clamp_min(0)
        max_chunk = int(chunk_lengths.max().item())

        # Build output shape as a List[int] (TorchScript-friendly)
        out_shape: List[int] = [batch, max_chunk]
        dim = int(tensor.dim())
        if dim > 2:
            for d in range(2, dim):
                out_shape.append(int(tensor.size(d)))

        chunk = tensor.new_full(out_shape, fill_value)

        for i in range(batch):
            start = int(prev_lengths[i].item())
            end = int(new_lengths[i].item())
            if start >= end or start >= total_len:
                continue
            end = min(end, total_len)
            length = min(max_chunk, end - start)
            if length <= 0:
                continue

            # Slice and copy the new tokens for this batch element
            if dim == 2:
                chunk[i, :length] = tensor[i, start:start + length]
            else:
                chunk[i, :length, ...] = tensor[i, start:start + length, ...]

        return chunk

    def _build_chunk_padding(
        self,
        prev_lengths: torch.Tensor,
        new_lengths: torch.Tensor,
    ) -> torch.Tensor:
        chunk_lengths = (new_lengths - prev_lengths).clamp_min(0)
        max_chunk = int(chunk_lengths.max().item())
        padding = torch.ones(
            (prev_lengths.size(0), max_chunk), dtype=torch.bool, device=prev_lengths.device
        )
        for i in range(prev_lengths.size(0)):
            length = int(chunk_lengths[i].item())
            if length <= 0:
                continue
            padding[i, :length] = False
        return padding
    
    @torch.jit.export
    def forward_with_kv_cache(
        self,
        obs_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        agent_types: torch.Tensor,
        positions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        valid_lengths: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[AttentionCacheEntry]] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[AttentionCacheEntry],
    ]:
        if kv_cache is None or len(kv_cache) == 0:
            encoded_inputs = self._encode_inputs(
                obs_sequence=obs_sequence,
                action_sequence=action_sequence,
                agent_types=agent_types,
                positions=positions,
                padding_mask=padding_mask,
            )

            causal_mask, key_padding = self._prepare_masks(encoded_inputs, padding_mask)

            transformer_output, _, routing, new_cache = self._run_transformer(
                encoded_inputs,
                causal_mask=causal_mask,
                key_padding=key_padding,
            )

            action_logits, opp_logits, state_values, win_logits = self._head_outputs(
                transformer_output,
                routing,
            )

            return action_logits, opp_logits, state_values, win_logits, new_cache

        if valid_lengths is None:
            raise ValueError("valid_lengths is required when using kv_cache")

        new_lengths = valid_lengths.to(torch.long).contiguous()
        prev_lengths = kv_cache[0]["lengths"].to(new_lengths.device)

        if torch.all(new_lengths <= prev_lengths):
            batch_size = obs_sequence.size(0)
            action_dim = self.action_heads[0].out_features
            opp_dim = self.opp_action_heads[0].out_features
            value_dim = self.reward_stream_heads[0].out_features
            win_dim = self.win_prob_heads[0].out_features
            zeros_actions = obs_sequence.new_zeros((batch_size, 0, action_dim))
            zeros_opp = obs_sequence.new_zeros((batch_size, 0, opp_dim))
            zeros_values = obs_sequence.new_zeros((batch_size, 0, value_dim))
            zeros_win = obs_sequence.new_zeros((batch_size, 0, win_dim))
            return zeros_actions, zeros_opp, zeros_values, zeros_win, kv_cache

        obs_chunk = self._slice_new_tokens(obs_sequence, prev_lengths, new_lengths)
        action_chunk   = self._slice_new_tokens(action_sequence, prev_lengths, new_lengths, fill_value=0.0)
        agent_chunk    = self._slice_new_tokens(agent_types,     prev_lengths, new_lengths, fill_value=0.0)
        position_chunk = self._slice_new_tokens(positions,       prev_lengths, new_lengths, fill_value=0.0)
        if padding_mask is not None:
            padding_chunk = self._slice_new_tokens(
                padding_mask,
                prev_lengths,
                new_lengths,
                fill_value=True,
            )
        else:
            padding_chunk = None

        chunk_padding_mask = self._build_chunk_padding(prev_lengths, new_lengths)

        encoded_inputs = self._encode_inputs(
            obs_sequence=obs_chunk,
            action_sequence=action_chunk,
            agent_types=agent_chunk,
            positions=position_chunk,
            padding_mask=chunk_padding_mask,
        )

        causal_mask, key_padding = self._prepare_masks(encoded_inputs, chunk_padding_mask)

        transformer_output, _, routing, new_cache = self._run_transformer(
            encoded_inputs,
            causal_mask=causal_mask,
            key_padding=key_padding,
            kv_cache=kv_cache,
        )

        action_logits, opp_logits, state_values, win_logits = self._head_outputs(
            transformer_output,
            routing,
        )

        return action_logits, opp_logits, state_values, win_logits, new_cache
