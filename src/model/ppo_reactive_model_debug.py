"""Debug variant of PPO reactive model that returns intermediate layer outputs."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.model.ppo_reactive_model_base import (
    PPOReactiveModelBase,
    MoETransformerEncoderLayer,
)


class PPOReactiveModelDebug(PPOReactiveModelBase):
    """Debug variant that can return all intermediate layer outputs for validation."""

    def forward(
        self,
        obs_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        agent_types: torch.Tensor,
        positions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        valid_lengths: Optional[torch.Tensor] = None,
        return_debug_outputs: bool = False,
    ):
        """
        Forward pass with optional debug output collection.

        Args:
            return_debug_outputs: If True, returns a dict with all intermediate outputs
                                  instead of the normal tuple output.

        Returns:
            If return_debug_outputs=False (normal mode):
                (action_logits, opp_logits, state_values, win_logits)

            If return_debug_outputs=True:
                dict with keys like:
                    'input/obs_sequence', 'input/action_sequence', etc.
                    'embeddings/obs_embed', 'embeddings/action_embed', etc.
                    'gating/g_obs', 'gating/g_action', etc.
                    'fusion/combined', 'transformer/layer_0/input', etc.
        """
        debug_outputs = {} if return_debug_outputs else None

        # === Step 1: Decompose actions ===
        if return_debug_outputs:
            debug_outputs['input/obs_sequence'] = obs_sequence.detach().cpu()
            debug_outputs['input/action_sequence'] = action_sequence.detach().cpu()
            debug_outputs['input/agent_types'] = agent_types.detach().cpu()
            debug_outputs['input/positions'] = positions.detach().cpu()
            if padding_mask is not None:
                debug_outputs['input/padding_mask'] = padding_mask.detach().cpu()

        act_kind_ids, count_ids, table_flag_ids = self._decompose_actions(
            action_sequence, padding_mask
        )

        if return_debug_outputs:
            debug_outputs['decompose/act_kind_ids'] = act_kind_ids.detach().cpu()
            debug_outputs['decompose/count_ids'] = count_ids.detach().cpu()
            debug_outputs['decompose/table_flag_ids'] = table_flag_ids.detach().cpu()

        # === Step 2: Embeddings ===
        # Break down obs_encoder into individual steps: Linear -> LayerNorm -> GELU
        obs_linear = self.obs_encoder[0](obs_sequence)  # Linear layer
        if return_debug_outputs:
            debug_outputs['embeddings/obs_linear'] = obs_linear.detach().cpu()

        obs_layernorm = self.obs_encoder[1](obs_linear)  # LayerNorm
        if return_debug_outputs:
            debug_outputs['embeddings/obs_layernorm'] = obs_layernorm.detach().cpu()

        obs_embed = self.obs_encoder[2](obs_layernorm)  # GELU
        if return_debug_outputs:
            debug_outputs['embeddings/obs_embed'] = obs_embed.detach().cpu()

        act_kind_embed = self.act_kind_embedding(act_kind_ids)
        count_embed = self.count_embedding(count_ids)
        table_flag_embed = self.table_flag_embedding(table_flag_ids)
        action_embed = act_kind_embed + count_embed + table_flag_embed

        if return_debug_outputs:
            debug_outputs['embeddings/act_kind_embed'] = act_kind_embed.detach().cpu()
            debug_outputs['embeddings/count_embed'] = count_embed.detach().cpu()
            debug_outputs['embeddings/table_flag_embed'] = table_flag_embed.detach().cpu()
            debug_outputs['embeddings/action_embed'] = action_embed.detach().cpu()

        agent_embed = self.agent_embedding(agent_types)
        position_embed = self.position_embedding(positions)

        if return_debug_outputs:
            debug_outputs['embeddings/agent_embed'] = agent_embed.detach().cpu()
            debug_outputs['embeddings/position_embed'] = position_embed.detach().cpu()

        # === Step 3: Gating ===
        g_obs = self.gate_obs(obs_embed)
        g_action = self.gate_action(action_embed)
        g_agent = self.gate_agent(agent_embed)
        g_position = self.gate_position(position_embed)

        if return_debug_outputs:
            debug_outputs['gating/g_obs'] = g_obs.detach().cpu()
            debug_outputs['gating/g_action'] = g_action.detach().cpu()
            debug_outputs['gating/g_agent'] = g_agent.detach().cpu()
            debug_outputs['gating/g_position'] = g_position.detach().cpu()

        # === Step 4: Fusion ===
        fused = (
            g_obs * obs_embed
            + g_action * action_embed
            + g_agent * agent_embed
            + g_position * position_embed
        )

        if return_debug_outputs:
            debug_outputs['fusion/fused_raw'] = fused.detach().cpu()

        combined = nn.functional.layer_norm(fused, (self.hidden_dim,))

        if return_debug_outputs:
            debug_outputs['fusion/combined'] = combined.detach().cpu()

        # === Step 5: Prepare masks ===
        causal_mask, key_padding = self._prepare_masks(combined, padding_mask)

        if return_debug_outputs:
            debug_outputs['masks/causal_mask'] = causal_mask.detach().cpu()
            if key_padding is not None:
                debug_outputs['masks/key_padding'] = key_padding.detach().cpu()

        # === Step 6: Transformer layers ===
        hidden = combined
        gate_logits_list: List[torch.Tensor] = []

        for layer_idx, layer in enumerate(self.transformer.layers):
            if return_debug_outputs:
                debug_outputs[f'transformer/layer_{layer_idx}/input'] = hidden.detach().cpu()

            # Attention
            attn_output, _ = layer.self_attn(
                hidden, hidden, hidden,
                attn_mask=causal_mask,
                key_padding_mask=key_padding,
                need_weights=False,
            )

            if return_debug_outputs:
                debug_outputs[f'transformer/layer_{layer_idx}/attn_output'] = attn_output.detach().cpu()

            # Post-attention residual + norm
            hidden_post_attn = layer.norm1(hidden + layer.dropout1(attn_output))

            if return_debug_outputs:
                debug_outputs[f'transformer/layer_{layer_idx}/post_attn'] = hidden_post_attn.detach().cpu()

            # MoE
            moe_output, routing = layer.moe(hidden_post_attn)
            gate_logits_list.append(routing['gate_logits'])

            if return_debug_outputs:
                debug_outputs[f'transformer/layer_{layer_idx}/moe_output'] = moe_output.detach().cpu()
                debug_outputs[f'transformer/layer_{layer_idx}/moe_gate_logits'] = routing['gate_logits'].detach().cpu()
                debug_outputs[f'transformer/layer_{layer_idx}/moe_topk_indices'] = routing['topk_indices'].detach().cpu()
                debug_outputs[f'transformer/layer_{layer_idx}/moe_topk_scores'] = routing['topk_scores'].detach().cpu()

            # Post-MoE residual + norm
            hidden = layer.norm2(hidden_post_attn + layer.dropout2(moe_output))

            if return_debug_outputs:
                debug_outputs[f'transformer/layer_{layer_idx}/output'] = hidden.detach().cpu()

        # Final norm
        transformer_output = self.transformer.norm(hidden)

        if return_debug_outputs:
            debug_outputs['transformer/final_output'] = transformer_output.detach().cpu()

        # Use last layer routing for heads
        final_routing = {
            'topk_indices': routing.get('topk_indices'),
            'topk_scores': routing.get('topk_scores'),
        }

        # === Step 7: Head outputs ===
        # Stack all expert outputs
        action_heads_stacked = self._stack_action_heads(transformer_output)
        reward_heads_stacked = self._stack_reward_heads(transformer_output)
        win_heads_stacked = self._stack_win_heads(transformer_output)
        opp_heads_stacked = self._stack_opp_heads(transformer_output)

        if return_debug_outputs:
            debug_outputs['heads/action_heads_stacked'] = action_heads_stacked.detach().cpu()
            debug_outputs['heads/reward_heads_stacked'] = reward_heads_stacked.detach().cpu()
            debug_outputs['heads/win_heads_stacked'] = win_heads_stacked.detach().cpu()
            debug_outputs['heads/opp_heads_stacked'] = opp_heads_stacked.detach().cpu()

        # Combine with routing
        action_logits = self._reduce_heads(
            action_heads_stacked,
            final_routing['topk_indices'],
            final_routing['topk_scores']
        )
        state_values = self._reduce_heads(
            reward_heads_stacked,
            final_routing['topk_indices'],
            final_routing['topk_scores']
        )
        win_logits = self._reduce_heads(
            win_heads_stacked,
            final_routing['topk_indices'],
            final_routing['topk_scores']
        )
        opp_logits = self._reduce_heads(
            opp_heads_stacked,
            final_routing['topk_indices'],
            final_routing['topk_scores']
        )

        if return_debug_outputs:
            debug_outputs['heads/action_logits'] = action_logits.detach().cpu()
            debug_outputs['heads/state_values'] = state_values.detach().cpu()
            debug_outputs['heads/win_logits'] = win_logits.detach().cpu()
            debug_outputs['heads/opp_logits'] = opp_logits.detach().cpu()

            # Return debug outputs dict
            return debug_outputs

        # Normal mode - return standard outputs
        return action_logits, opp_logits, state_values, win_logits
