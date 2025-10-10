# src/training/tracing_utils.py

import logging
from pathlib import Path

import torch

from src import config
from src.agents.learner_ar_agent import LearnerAutoregressiveAgent
from src.model.ppo_reactive_model import PPOReactiveModel
from src.model.ppo_reactive_model_script import PPOReactiveModelScript

# This is a simplified version of the logic from trace_models.py,
# designed to be called as a library function.

def build_script_model_from_training_model(model: torch.nn.Module) -> PPOReactiveModelScript:
    base = getattr(model, "_orig_mod", model)

    if not isinstance(base, PPOReactiveModel):
        raise TypeError(
            "PPOReactiveModelScript export currently expects a PPOReactiveModel training module"
        )

    obs_linear = base.obs_encoder[0]
    action_head = base.action_heads[0]
    transformer_layer = base.transformer.layers[0]

    script_model = PPOReactiveModelScript(
        obs_dim=getattr(obs_linear, "in_features"),
        action_dim=getattr(action_head, "out_features"),
        hidden_dim=base.hidden_dim,
        num_heads=transformer_layer.self_attn.num_heads,
        num_layers=len(base.transformer.layers),
        dropout_rate=transformer_layer.dropout1.p,
        max_seq_length=base.max_seq_length,
        num_agent_types=base.agent_embedding.num_embeddings,
        num_experts=base.num_experts,
        top_k=base.top_k,
        expert_ffn_dim=base.expert_ffn_dim,
    )

    script_model.load_state_dict(base.state_dict(), strict=False)
    script_model.eval()
    return script_model


def trace_model_from_checkpoint(checkpoint_path: str, output_path: str, device: torch.device):
    """Loads a .pth checkpoint, traces it, and saves it as a .pt file."""
    metadata_path = Path(str(output_path) + ".max_seq_length")
    try:
        agent = LearnerAutoregressiveAgent(device=device, player_id="tracer", compile=True)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        agent.load_models_from_checkpoint({"policy_nets": {"agent_model": state_dict}}, "agent_model")
        
        training_model = getattr(agent.model, "_orig_mod", agent.model)
        script_model = build_script_model_from_training_model(training_model)

        with torch.no_grad():
            scripted_model = torch.jit.script(script_model)
            traced_model = torch.jit.freeze(
                scripted_model, preserved_attrs=["forward_packed"]
            )
        # Save to CPU for portability
        traced_model_cpu = traced_model.to("cpu")
        torch.jit.save(traced_model_cpu, output_path)

        max_seq_length = int(getattr(training_model, "max_seq_length", 0) or 0)
        try:
            if max_seq_length > 0:
                metadata_path.write_text(f"{max_seq_length}\n")
            else:
                metadata_path.write_text("")
        except Exception:
            logging.warning(
                "Unable to persist max_seq_length metadata alongside %s", output_path, exc_info=True
            )
        logging.info(f"Successfully traced model from {checkpoint_path} to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to trace model from {checkpoint_path}: {e}", exc_info=True)
        # Remove any partially written artifact so downstream consumers do not
        # attempt to load an invalid TorchScript file on subsequent runs.
        try:
            Path(output_path).unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
        except Exception:
            logging.debug(
                "Unable to clean up incomplete TorchScript artifact at %s", output_path,
                exc_info=True,
            )
        return False
