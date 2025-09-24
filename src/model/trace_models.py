#!/usr/bin/env python3
"""Trace PPO policy checkpoints into TorchScript modules.

Loads a PPO checkpoint for a specified run/generation, recreates the policy
network, and exports a TorchScript trace alongside metadata for later lookup.
"""

from __future__ import annotations
import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Repository bootstrap so we can import from ``src`` when executed as a script
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2] # Change this from 1 to 2
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config  # noqa: E402
from src.agents.batch_autoregressive_ppo_agent import BatchPPOAutoregressiveAgent  # noqa: E402


@dataclass
class TraceMetadata:
    run_name: str
    generation: str
    source_checkpoint: str
    traced_module: str
    traced_on: str
    device: str
    example_sequence_length: int
    obs_dim: int
    action_dim: int
    notes: Optional[str] = None


def _resolve_device(requested: str) -> torch.device:
    """Resolve a torch.device, gracefully falling back to CPU when needed."""
    requested_device = torch.device(requested)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return requested_device


def _candidate_checkpoints(run_dir: Path, generation: str) -> List[Path]:
    base_dir = run_dir / f"gen_{generation}"
    candidates: List[Path] = []
    if not base_dir.exists():
        raise FileNotFoundError(f"Generation directory not found: {base_dir}")

    preferred_names = [
        "compiled_final.pth",
        "final.pth",
        "autoreg_model_final.pth",
        "autoreg_model_best.pth",
    ]
    candidates.extend([base_dir / name for name in preferred_names if (base_dir / name).exists()])

    if candidates:
        return candidates

    candidates = sorted(base_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No .pth checkpoint files found under {base_dir}")
    return candidates


def _normalize_checkpoint_payload(raw: object) -> Tuple[Dict[str, Dict[str, dict]], str]:
    """Convert heterogeneous checkpoint payloads into the agent loader format."""
    if isinstance(raw, dict):
        policy_nets = raw.get("policy_nets")
        if isinstance(policy_nets, dict) and policy_nets:
            for key, value in policy_nets.items():
                if isinstance(value, dict):
                    return {"policy_nets": policy_nets}, str(key)

        for key in ("model_state_dict", "state_dict"):
            state_dict = raw.get(key)
            if isinstance(state_dict, dict):
                return {"policy_nets": {"agent_model": state_dict}}, "agent_model"

    if isinstance(raw, dict) and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in raw.items()):
        return {"policy_nets": {"agent_model": raw}}, "agent_model"

    raise ValueError("Unsupported checkpoint format; expected 'policy_nets' or a raw state_dict.")


def _load_agent(checkpoint_path: Path, device: torch.device) -> BatchPPOAutoregressiveAgent:
    checkpoint_raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint, agent_key = _normalize_checkpoint_payload(checkpoint_raw)

    agent = BatchPPOAutoregressiveAgent(device=device, player_id=f"trace::{checkpoint_path.stem}")
    agent.load_models_from_checkpoint(checkpoint, agent_key)
    if agent.model is None:
        raise RuntimeError("Agent failed to load model from checkpoint.")
    agent.model.eval()
    return agent


def _build_example_inputs(model: torch.nn.Module, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
    """Construct deterministic example tensors that respect embedding bounds."""
    if not hasattr(model, "max_seq_length") or not hasattr(model, "obs_dim"):
        raise ValueError("Model missing expected attributes required for tracing.")

    max_len = int(getattr(model, "max_seq_length"))
    seq_len = int(seq_len or min(max_len, 256))

    obs_dim = int(getattr(model, "obs_dim"))
    action_dim = int(getattr(model, "action_dim", 7))

    obs_sequence = torch.zeros(1, seq_len, obs_dim, dtype=torch.float32, device=model.device)
    action_sequence = torch.zeros(1, seq_len, dtype=torch.long, device=model.device)
    agent_types = torch.zeros(1, seq_len, dtype=torch.long, device=model.device)
    positions = torch.arange(seq_len, dtype=torch.long, device=model.device).unsqueeze(0)
    action_masks = torch.zeros(1, seq_len, action_dim, dtype=torch.bool, device=model.device)
    padding_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=model.device)
    valid_lengths = torch.full((1,), seq_len, dtype=torch.long, device=model.device)

    return (
        obs_sequence,
        action_sequence,
        agent_types,
        positions,
        action_masks,
        padding_mask,
        valid_lengths,
    )


def _save_metadata(metadata_path: Path, entry: TraceMetadata) -> None:
    existing: List[Dict[str, object]] = []
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            print(f"[WARN] Could not parse existing metadata at {metadata_path}; regenerating file.")
    existing = [item for item in existing if item.get("traced_module") != entry.traced_module]
    existing.append(asdict(entry))
    metadata_path.write_text(json.dumps(existing, indent=2))


def trace_model(run_name: str, generation: str, device: torch.device, checkpoint_name: Optional[str], notes: Optional[str]) -> Path:
    run_dir = Path(config.CHECKPOINT_DIR) / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    if checkpoint_name:
        candidate_path = Path(checkpoint_name)
        if not candidate_path.is_absolute():
            candidate_path = run_dir / f"gen_{generation}" / checkpoint_name
        checkpoint_path = candidate_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Specified checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_path = _candidate_checkpoints(run_dir, generation)[0]

    print(f"[trace] Loading checkpoint: {checkpoint_path}")
    agent = _load_agent(checkpoint_path, device=device)
    model = agent.model
    assert model is not None

    example_inputs = _build_example_inputs(model)
    with torch.no_grad():
        traced = torch.jit.trace(model, example_inputs, strict=False)
        traced = torch.jit.freeze(traced)

    traced_filename = f"{checkpoint_path.stem}_traced.pt"
    traced_path = checkpoint_path.with_name(traced_filename)
    traced_cpu = traced.to("cpu")
    torch.jit.save(traced_cpu, traced_path)
    print(f"[trace] Saved TorchScript module to {traced_path}")

    metadata_entry = TraceMetadata(
        run_name=run_name,
        generation=generation,
        source_checkpoint=str(checkpoint_path.resolve()),
        traced_module=traced_filename,
        traced_on=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        device=str(device),
        example_sequence_length=example_inputs[0].shape[1],
        obs_dim=int(getattr(model, "obs_dim", -1)),
        action_dim=int(getattr(model, "action_dim", -1)),
        notes=notes,
    )

    metadata_path = traced_path.with_suffix(".json")
    _save_metadata(metadata_path, metadata_entry)
    print(f"[trace] Updated metadata at {metadata_path}")

    index_path = traced_path.parent / "traced_index.json"
    _save_metadata(index_path, metadata_entry)
    print(f"[trace] Recorded trace entry in {index_path}")

    return traced_path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Trace PPO policy checkpoints into TorchScript modules.")
    parser.add_argument("--run-name", required=True, help="Training run directory under checkpoints/.")
    parser.add_argument("--gen", required=True, help="Generation identifier (e.g. 10 or final).")
    parser.add_argument("--checkpoint-name", help="Specific checkpoint filename to load (defaults to common names).")
    parser.add_argument("--device", default="cpu", help="Torch device to load the model on (default: cpu).")
    parser.add_argument("--notes", help="Optional free-form notes to store in the trace metadata entry.")

    args = parser.parse_args(argv)

    device = _resolve_device(args.device)
    traced_path = trace_model(
        run_name=args.run_name,
        generation=args.gen,
        device=device,
        checkpoint_name=args.checkpoint_name,
        notes=args.notes,
    )
    print(f"[trace] Completed tracing: {traced_path}")


if __name__ == "__main__":
    main()
