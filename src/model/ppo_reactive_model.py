# src/model/ppo_reactive_model.py
from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import torch
from torch.utils.checkpoint import checkpoint

from src.model.ppo_reactive_model_base import PPOReactiveModelBase, MoETransformerEncoderLayer
from src.model.moe_autograd import grouped_moe_ffn


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

        def moe_layer_grouped(module: MoETransformerEncoderLayer, inp: torch.Tensor):
            attn_output, _ = module.self_attn(
                inp, inp, inp,
                attn_mask=causal_mask,
                key_padding_mask=key_padding,
                need_weights=False,
            )
            x1 = module.norm1(inp + module.dropout1(attn_output))

            # Gate logits (logits per expert)
            B, T, H = x1.shape
            gate_logits = module.moe.gate(x1.view(-1, H)).view(B, T, self.num_experts)
            if gate_logits.dtype != x1.dtype:
                gate_logits = gate_logits.to(dtype=x1.dtype)

            # Pack expert weights to expected shapes for grouped MoE
            w1_list, b1_list, w2_list, b2_list = [], [], [], []
            for expert in module.moe.experts:
                lin1 = expert[0]
                lin2 = expert[3]
                w1_list.append(lin1.weight)
                b1_list.append(lin1.bias)
                w2_list.append(lin2.weight)
                b2_list.append(lin2.bias)
            w1 = torch.stack(w1_list, dim=0).unsqueeze(0).to(dtype=torch.float16, device=x1.device).contiguous()
            w2 = torch.stack(w2_list, dim=0).unsqueeze(0).to(dtype=torch.float16, device=x1.device).contiguous()
            b1 = torch.stack(b1_list, dim=0).unsqueeze(0).to(dtype=torch.float16, device=x1.device).contiguous()
            b2 = torch.stack(b2_list, dim=0).unsqueeze(0).to(dtype=torch.float16, device=x1.device).contiguous()

            policy_indices = torch.zeros((B, T), dtype=torch.long, device=x1.device)
            y = grouped_moe_ffn(
                x1,
                w1, w2, b1, b2,
                routing_scores=gate_logits,
                policy_indices=policy_indices,
                top_k=module.moe.top_k,
                compute_routing_grads=True,
            )

            # Routing info from logits
            probs = torch.softmax(gate_logits, dim=-1)
            topk = torch.topk(probs, module.moe.top_k, dim=-1)
            rout = {"gate_logits": gate_logits, "topk_indices": topk.indices, "topk_scores": topk.values}

            x2 = module.norm2(x1 + module.dropout2(y))
            return x2, rout

        for layer in self.transformer.layers:
            hidden, layer_routing, = checkpoint(
                lambda inp: moe_layer_grouped(layer, inp), hidden, use_reentrant=False
            )
            gate_logits_list.append(layer_routing["gate_logits"])  # type: ignore[index]
            routing = layer_routing

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
            # Run transformer with grouped MoE autograd per layer
            gate_logits_list: list[torch.Tensor] = []
            routing: Dict[str, torch.Tensor] = {}

            x = encoded_inputs
            for layer in self.transformer.layers:
                # Reuse the same per-layer logic without checkpointing
                attn_output, _ = layer.self_attn(
                    x, x, x,
                    attn_mask=causal_mask,
                    key_padding_mask=key_padding,
                    need_weights=False,
                )
                x1 = layer.norm1(x + layer.dropout1(attn_output))

                B, T, H = x1.shape
                gate_logits = layer.moe.gate(x1.view(-1, H)).view(B, T, self.num_experts)
                if gate_logits.dtype != x1.dtype:
                    gate_logits = gate_logits.to(dtype=x1.dtype)

                w1_list, b1_list, w2_list, b2_list = [], [], [], []
                for expert in layer.moe.experts:
                    lin1 = expert[0]
                    lin2 = expert[3]
                    w1_list.append(lin1.weight)
                    b1_list.append(lin1.bias)
                    w2_list.append(lin2.weight)
                    b2_list.append(lin2.bias)
                w1 = torch.stack(w1_list, dim=0).unsqueeze(0).to(dtype=torch.float16, device=x1.device).contiguous()
                w2 = torch.stack(w2_list, dim=0).unsqueeze(0).to(dtype=torch.float16, device=x1.device).contiguous()
                b1 = torch.stack(b1_list, dim=0).unsqueeze(0).to(dtype=torch.float16, device=x1.device).contiguous()
                b2 = torch.stack(b2_list, dim=0).unsqueeze(0).to(dtype=torch.float16, device=x1.device).contiguous()

                policy_indices = torch.zeros((B, T), dtype=torch.long, device=x1.device)
                y = grouped_moe_ffn(
                    x1,
                    w1, w2, b1, b2,
                    routing_scores=gate_logits,
                    policy_indices=policy_indices,
                    top_k=layer.moe.top_k,
                    compute_routing_grads=True,
                )

                probs = torch.softmax(gate_logits, dim=-1)
                topk = torch.topk(probs, layer.moe.top_k, dim=-1)
                routing = {"gate_logits": gate_logits, "topk_indices": topk.indices, "topk_scores": topk.values}

                x = layer.norm2(x1 + layer.dropout2(y))
                gate_logits_list.append(gate_logits)

            transformer_output = self.transformer.norm(x) if self.transformer.norm is not None else x
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
