from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Compute a deterministic SHA-256 digest over sorted names, shapes, dtypes, and raw bytes."""
    h = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        # Clone guarantees zero storage offset and exact nbytes for raw byte hashing (no numpy).
        tensor = state_dict[name].detach().cpu().contiguous().clone()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(bytes(tensor.untyped_storage()))
    return h.hexdigest()


def assert_weights_synced(expected_sha256: str, actual_sha256: str, policy_version: int) -> None:
    """Fail loud if rollout weights diverge from the learner's published checkpoint."""
    if expected_sha256 != actual_sha256:
        raise RuntimeError(
            f"Weight staleness detected at policy_version={policy_version}: "
            f"learner={expected_sha256[:12]} != rollout={actual_sha256[:12]}"
        )
