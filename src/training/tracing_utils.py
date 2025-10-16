# src/training/tracing_utils.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.agents.learner_ar_agent import LearnerAutoregressiveAgent


def trace_model_from_checkpoint(
    checkpoint_path: str, output_path: str, device: torch.device
) -> Optional[Dict[str, Any]]:
    """
    Load a ``.pth`` checkpoint, script it, and persist a TorchScript artifact.

    Returns a metadata dictionary when successful. The dictionary contains the
    TorchScript path, the inferred ``max_seq_length`` (if available), and the
    location of the sidecar metadata file. ``None`` is returned on failure.
    """

    metadata_path = Path(str(output_path) + ".max_seq_length")
    try:
        agent = LearnerAutoregressiveAgent(device=device, player_id="tracer", compile=True)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        agent.load_models_from_checkpoint({"policy_nets": {"agent_model": state_dict}}, "agent_model")

        model_to_trace = getattr(agent.model, "_orig_mod", agent.model)
        model_to_trace.eval()

        with torch.no_grad():
            scripted_model = torch.jit.script(model_to_trace)
            scripted_model.eval()

        traced_model_cpu = scripted_model.to("cpu")
        torch.jit.save(traced_model_cpu, output_path)

        max_seq_length = int(getattr(model_to_trace, "max_seq_length", 0) or 0)
        try:
            if max_seq_length > 0:
                metadata_path.write_text(f"{max_seq_length}\n")
            else:
                metadata_path.write_text("")
        except Exception:
            logging.warning(
                "Unable to persist max_seq_length metadata alongside %s", output_path, exc_info=True
            )

        logging.info("Successfully traced model from %s to %s", checkpoint_path, output_path)
        result: Dict[str, Any] = {"path": output_path}
        if max_seq_length > 0:
            result["max_seq_length"] = max_seq_length
        result["metadata_path"] = str(metadata_path)
        return result
    except Exception as exc:
        logging.error("Failed to trace model from %s: %s", checkpoint_path, exc, exc_info=True)
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
        return None
