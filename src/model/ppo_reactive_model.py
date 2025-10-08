# src/model/ppo_reactive_model.py
from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.model.ppo_reactive_model_base import (
    PPOReactiveModelBase,
    MoETransformerEncoderLayer,
    AttentionCacheEntry,
)


class PPOReactiveModel(PPOReactiveModelBase):
    """Training variant of the PPO reactive model."""

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
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            max_seq_length=max_seq_length,
            num_agent_types=num_agent_types,
            num_experts=num_experts,
            top_k=top_k,
            expert_ffn_dim=expert_ffn_dim,
        )
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

    @staticmethod
    def _apply_action_mask(
        action_logits: torch.Tensor,
        agent_types: torch.Tensor,
        action_masks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if action_masks is None:
            return action_logits
        neg = torch.tensor(
            torch.finfo(action_logits.dtype).min / 4.0,
            dtype=action_logits.dtype,
            device=action_logits.device,
        )
        our_turns = (agent_types == 0).unsqueeze(-1)
        invalid = (~action_masks.bool()) & our_turns
        return torch.where(invalid, neg, action_logits)

    def _forward_with_gradient_checkpointing(
        self,
        encoded_inputs: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        
        hidden = encoded_inputs
        gate_logits_list = []
        routing: Dict[str, torch.Tensor] = {}

        def create_custom_forward(module: MoETransformerEncoderLayer):
            def custom_forward(*inputs):
                inp = inputs[0]
                output, layer_routing, _ = module(
                    inp,
                    src_mask=causal_mask,
                    src_key_padding_mask=key_padding,
                )
                return (
                    output,
                    layer_routing["gate_logits"],
                    layer_routing.get("topk_indices"),
                    layer_routing.get("topk_scores"),
                )
            return custom_forward

        for layer in self.transformer.layers:
            hidden, gate_logit, topk_indices, topk_scores = checkpoint(
                create_custom_forward(layer), hidden, use_reentrant=False
            )

            gate_logits_list.append(gate_logit)
            routing = {
                "gate_logits": gate_logit,
                "topk_indices": topk_indices,
                "topk_scores": topk_scores,
            }

        transformer_output = self.transformer.norm(hidden) if self.transformer.norm is not None else hidden
        
        gate_logits_tensor = self._stack_gate_logits(gate_logits_list, transformer_output)
        
        return transformer_output, gate_logits_tensor, routing
        
    def _stack_gate_logits(self, gate_logits_list: List[torch.Tensor], ref_tensor: torch.Tensor) -> torch.Tensor:
         return (
            torch.stack(gate_logits_list, dim=0)
            if gate_logits_list
            else ref_tensor.new_zeros(
                0, ref_tensor.size(0), ref_tensor.size(1), self.num_experts
            )
        )

    def forward(
        self,
        obs_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        agent_types: torch.Tensor,
        positions: torch.Tensor,
        action_masks: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
        valid_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        # 1. Prepare inputs by calling methods from the base class
        encoded_inputs = self._encode_inputs(
            obs_sequence, action_sequence, agent_types, positions, padding_mask
        )
        causal_mask, key_padding = self._prepare_masks(encoded_inputs, padding_mask)

        # 2. Run transformer (with or without checkpointing)
        if self.training and self.use_gradient_checkpointing:
            transformer_output, gate_logits_tensor, routing = self._forward_with_gradient_checkpointing(
                encoded_inputs, causal_mask, key_padding
            )
        else:
            # Call the base class's run_transformer method
            transformer_output, gate_logits_list, routing = self._run_transformer(
                encoded_inputs,
                causal_mask=causal_mask,
                key_padding=key_padding,
            )
            gate_logits_tensor = self._stack_gate_logits(gate_logits_list, transformer_output)
        
        # 3. Get head outputs by calling the base class's method
        action_logits, opp_logits, state_values, win_logits = self._head_outputs(
            transformer_output, routing
        )
        
        # 4. Apply training-specific masking
        action_logits = self._apply_action_mask(action_logits, agent_types, action_masks)
        
        return (
            action_logits,
            opp_logits,
            state_values,
            win_logits,
            gate_logits_tensor,
            routing,
        )

    # ---------------------- rollout cache utils ----------------------
    def _stack_kv_cache(
        self,
        caches: List[List[AttentionCacheEntry]],
        device: torch.device,
    ) -> List[AttentionCacheEntry]:
        if not caches:
            return []

        num_layers = len(caches[0])
        stacked: List[AttentionCacheEntry] = []
        for layer_idx in range(num_layers):
            layer_keys: List[torch.Tensor] = []
            layer_values: List[torch.Tensor] = []
            layer_lengths: List[torch.Tensor] = []
            max_len = 0
            for cache in caches:
                entry = cache[layer_idx]
                key_tensor = entry["key"].to(device)
                value_tensor = entry["value"].to(device)
                length_tensor = entry["lengths"].to(device)
                max_len = max(max_len, key_tensor.size(2))
                layer_keys.append(key_tensor)
                layer_values.append(value_tensor)
                layer_lengths.append(length_tensor)

            padded_keys: List[torch.Tensor] = []
            padded_values: List[torch.Tensor] = []
            for key_tensor, value_tensor in zip(layer_keys, layer_values):
                pad_len = max_len - key_tensor.size(2)
                if pad_len > 0:
                    key_tensor = F.pad(key_tensor, (0, 0, 0, pad_len))
                    value_tensor = F.pad(value_tensor, (0, 0, 0, pad_len))
                padded_keys.append(key_tensor)
                padded_values.append(value_tensor)

            stacked.append(
                {
                    "key": torch.cat(padded_keys, dim=0),
                    "value": torch.cat(padded_values, dim=0),
                    "lengths": torch.cat(layer_lengths, dim=0),
                }
            )

        return stacked

    def _prepare_kv_updates(
        self,
        keys: List[Optional[Tuple[int, int]]],
        caches: List[AttentionCacheEntry],
    ) -> Dict[Tuple[int, int], List[AttentionCacheEntry]]:
        updates: Dict[Tuple[int, int], List[AttentionCacheEntry]] = {}
        if not keys or not caches:
            return updates

        num_layers = len(caches)
        batch_size = caches[0]["key"].size(0)
        for batch_idx in range(batch_size):
            key = keys[batch_idx] if batch_idx < len(keys) else None
            if key is None:
                continue
            per_cache: List[AttentionCacheEntry] = []
            for layer_idx in range(num_layers):
                layer_entry = caches[layer_idx]
                key_tensor = layer_entry["key"][batch_idx : batch_idx + 1].detach().cpu()
                value_tensor = layer_entry["value"][batch_idx : batch_idx + 1].detach().cpu()
                length_tensor = layer_entry["lengths"][batch_idx : batch_idx + 1].detach().cpu()

                seq_len = int(length_tensor[0].item())
                max_allowed = self.max_seq_length or seq_len
                if seq_len > max_allowed:
                    start = seq_len - max_allowed
                    key_tensor = key_tensor[..., start:, :].contiguous()
                    value_tensor = value_tensor[..., start:, :].contiguous()
                    length_tensor = torch.clamp(length_tensor - start, max=max_allowed)
                    seq_len = max_allowed
                else:
                    length_tensor = torch.clamp(length_tensor, max=max_allowed)

                key_tensor = key_tensor[..., :seq_len, :].contiguous()
                value_tensor = value_tensor[..., :seq_len, :].contiguous()
                per_cache.append(
                    {
                        "key": key_tensor,
                        "value": value_tensor,
                        "lengths": length_tensor,
                    }
                )
            updates[key] = per_cache
        return updates

    def forward_rollout(
        self,
        filtered_inputs: Dict[str, torch.Tensor],
        full_inputs: Dict[str, torch.Tensor],
        kv_cache_map: Optional[Dict[Tuple[int, int], List[AttentionCacheEntry]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[Tuple[int, int], List[AttentionCacheEntry]]]:
        """
        Cache-aware rollout step that selects per-sample cached vs. non-cached
        paths and returns the last-step logits/mask/value along with cache updates.

        This does not modify training forward(); it's a separate inference helper.
        """
        obs_sequence = filtered_inputs["obs_sequence"]
        action_sequence = filtered_inputs["action_sequence"]
        agent_types = filtered_inputs["agent_types"]
        positions = filtered_inputs["positions"]
        action_masks = filtered_inputs["action_masks"]
        padding_mask = filtered_inputs["padding_mask"]

        device = obs_sequence.device
        valid_lengths = full_inputs.get("valid_lengths")
        if not isinstance(valid_lengths, torch.Tensor):
            valid_lengths = torch.as_tensor(valid_lengths, device=device)
        valid_lengths = valid_lengths.to(device).long()

        env_indices = full_inputs.get("env_indices")
        seat_indices = full_inputs.get("seat_indices")

        batch_size = obs_sequence.size(0)
        action_dim = action_masks.size(-1)

        logits_last = torch.empty((batch_size, action_dim), device=device)
        mask_last = torch.empty((batch_size, action_dim), dtype=torch.bool, device=device)
        values_last = torch.empty((batch_size,), device=device)

        caches_available = (
            isinstance(env_indices, torch.Tensor)
            and isinstance(seat_indices, torch.Tensor)
            and env_indices.numel() == batch_size
            and seat_indices.numel() == batch_size
        )

        cache_map = kv_cache_map or {}
        cache_keys: List[Optional[Tuple[int, int]]] = [None] * batch_size
        with_cache_indices: List[int] = []
        without_cache_indices: List[int] = []
        cache_entries: List[List[AttentionCacheEntry]] = []

        if caches_available:
            env_cpu = env_indices.detach().cpu().to(torch.long)
            seat_cpu = seat_indices.detach().cpu().to(torch.long)
        else:
            env_cpu = seat_cpu = None

        for idx in range(batch_size):
            cache = None
            if caches_available and env_cpu is not None and seat_cpu is not None:
                key = (int(env_cpu[idx].item()), int(seat_cpu[idx].item()))
                cache = cache_map.get(key)
                cache_keys[idx] = key
            else:
                key = None
                cache_keys[idx] = None
            if cache is None:
                without_cache_indices.append(idx)
            else:
                with_cache_indices.append(idx)
                cache_entries.append(cache)

        if without_cache_indices:
            idx_tensor = torch.tensor(without_cache_indices, device=device, dtype=torch.long)
            subset_inputs = {k: t.index_select(0, idx_tensor) for k, t in filtered_inputs.items()}
            subset_lengths = valid_lengths.index_select(0, idx_tensor)
            action_logits, _, state_values, _, new_cache = self.forward_with_kv_cache(
                obs_sequence=subset_inputs["obs_sequence"],
                action_sequence=subset_inputs["action_sequence"],
                agent_types=subset_inputs["agent_types"],
                positions=subset_inputs["positions"],
                action_masks=subset_inputs["action_masks"],
                padding_mask=subset_inputs["padding_mask"],
                valid_lengths=subset_lengths,
                kv_cache=None,
            )
            mask_subset = action_masks.index_select(0, idx_tensor)
            last_indices = (subset_lengths - 1).clamp_min(0)
            for local_idx, batch_idx in enumerate(without_cache_indices):
                last_pos = int(last_indices[local_idx].item())
                logits_last[batch_idx] = action_logits[local_idx, last_pos]
                mask_last[batch_idx] = mask_subset[local_idx, last_pos]
                values_last[batch_idx] = state_values[local_idx, last_pos, 0]

            updates = self._prepare_kv_updates(
                [cache_keys[idx] for idx in without_cache_indices],
                new_cache,
            )
            cache_map.update(updates)

        if with_cache_indices:
            idx_tensor = torch.tensor(with_cache_indices, device=device, dtype=torch.long)
            subset_inputs = {k: t.index_select(0, idx_tensor) for k, t in filtered_inputs.items()}
            subset_lengths = valid_lengths.index_select(0, idx_tensor)
            stacked_cache = self._stack_kv_cache(cache_entries, device)
            action_logits, _, state_values, _, new_cache = self.forward_with_kv_cache(
                obs_sequence=subset_inputs["obs_sequence"],
                action_sequence=subset_inputs["action_sequence"],
                agent_types=subset_inputs["agent_types"],
                positions=subset_inputs["positions"],
                action_masks=subset_inputs["action_masks"],
                padding_mask=subset_inputs["padding_mask"],
                valid_lengths=subset_lengths,
                kv_cache=stacked_cache,
            )
            if action_logits.size(1) > 0:
                mask_subset = action_masks.index_select(0, idx_tensor)
                prev_lengths = stacked_cache[0]["lengths"].to(device)
                chunk_lengths = (subset_lengths - prev_lengths).clamp_min(1)
                last_chunk = (chunk_lengths - 1).clamp_min(0)
                final_positions = (subset_lengths - 1).clamp_min(0)
                for local_idx, batch_idx in enumerate(with_cache_indices):
                    chunk_pos = int(last_chunk[local_idx].item())
                    final_pos = int(final_positions[local_idx].item())
                    logits_last[batch_idx] = action_logits[local_idx, chunk_pos]
                    mask_last[batch_idx] = mask_subset[local_idx, final_pos]
                    values_last[batch_idx] = state_values[local_idx, chunk_pos, 0]

            updates = self._prepare_kv_updates(
                [cache_keys[idx] for idx in with_cache_indices],
                new_cache,
            )
            cache_map.update(updates)

        return logits_last, mask_last, values_last, cache_map
