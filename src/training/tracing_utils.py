# src/training/tracing_utils.py

import logging
from pathlib import Path
import torch
from src import config
from src.agents.learner_ar_agent import LearnerAutoregressiveAgent

# This is a simplified version of the logic from trace_models.py,
# designed to be called as a library function.

def _build_example_inputs(model: torch.nn.Module, device: torch.device):
    """Constructs deterministic example tensors for tracing."""
    max_len = int(getattr(model, "max_seq_length"))
    seq_len = min(max_len, 256) # Use a longer sequence for better generalization
    obs_dim = int(getattr(model, "obs_dim"))
    action_dim = int(getattr(model, "action_dim", 7))

    return {
        "obs_sequence": torch.zeros(1, seq_len, obs_dim, dtype=torch.float32, device=device),
        "action_sequence": torch.zeros(1, seq_len, dtype=torch.long, device=device),
        "agent_types": torch.zeros(1, seq_len, dtype=torch.long, device=device),
        "positions": torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0),
        "action_masks": torch.zeros(1, seq_len, action_dim, dtype=torch.bool, device=device),
        "padding_mask": torch.zeros(1, seq_len, dtype=torch.bool, device=device),
    }

def trace_model_from_checkpoint(checkpoint_path: str, output_path: str, device: torch.device):
    """Loads a .pth checkpoint, traces it, and saves it as a .pt file."""
    try:
        agent = LearnerAutoregressiveAgent(device=device, player_id="tracer", compile=True)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        agent.load_models_from_checkpoint({"policy_nets": {"agent_model": state_dict}}, "agent_model")
        
        model_to_trace = getattr(agent.model, "_orig_mod", agent.model)
        model_to_trace.eval()

        example_inputs = _build_example_inputs(model_to_trace, device)

        with torch.no_grad():
            scripted_model = torch.jit.script(model_to_trace)
            traced_model = torch.jit.freeze(scripted_model)
        # Save to CPU for portability
        traced_model_cpu = traced_model.to("cpu")
        torch.jit.save(traced_model_cpu, output_path)
        logging.info(f"Successfully traced model from {checkpoint_path} to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to trace model from {checkpoint_path}: {e}", exc_info=True)
        # Remove any partially written artifact so downstream consumers do not
        # attempt to load an invalid TorchScript file on subsequent runs.
        try:
            Path(output_path).unlink(missing_ok=True)
        except Exception:
            logging.debug(
                "Unable to clean up incomplete TorchScript artifact at %s", output_path,
                exc_info=True,
            )
        return False
