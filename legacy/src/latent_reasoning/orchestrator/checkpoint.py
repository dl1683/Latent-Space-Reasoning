"""Checkpointing for fault tolerance and restart capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
from torch import Tensor

from latent_reasoning.core.chain import ChainState


@dataclass
class Checkpoint:
    """A checkpoint of the evolution state."""

    latents: List[Tensor]
    scores: List[float]
    generation: int
    best_latent: Tensor | None = None
    best_score: float = float("-inf")
    metadata: dict = field(default_factory=dict)

    def to_chain_states(self) -> List[ChainState]:
        """Convert checkpoint to chain states."""
        return [
            ChainState(latent=latent, score=score, generation=self.generation)
            for latent, score in zip(self.latents, self.scores)
        ]


class CheckpointManager:
    """Manages checkpoints for the orchestrator."""

    def __init__(self, checkpoint_dir: str | Path | None = None):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints (optional)
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.best_checkpoint: Checkpoint | None = None
        self.checkpoints: List[Checkpoint] = []
        self.max_checkpoints = 5  # Keep only last N checkpoints

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        chains: List[ChainState],
        generation: int,
        best_latent: Tensor | None = None,
        best_score: float = float("-inf"),
        metadata: dict | None = None,
    ) -> Checkpoint:
        """
        Save a checkpoint.

        Args:
            chains: Current chain states
            generation: Current generation
            best_latent: Best latent found so far
            best_score: Best score found so far
            metadata: Optional metadata

        Returns:
            The created checkpoint
        """
        checkpoint = Checkpoint(
            latents=[c.latent.clone() for c in chains],
            scores=[c.score for c in chains],
            generation=generation,
            best_latent=best_latent.clone() if best_latent is not None else None,
            best_score=best_score,
            metadata=metadata or {},
        )

        self.checkpoints.append(checkpoint)

        # Update best checkpoint
        if self.best_checkpoint is None or best_score > self.best_checkpoint.best_score:
            self.best_checkpoint = checkpoint

        # Prune old checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints = self.checkpoints[-self.max_checkpoints:]

        # Save to disk if configured
        if self.checkpoint_dir:
            self._save_to_disk(checkpoint, generation)

        return checkpoint

    def get_best_checkpoint(self) -> Checkpoint | None:
        """Get the best checkpoint seen so far."""
        return self.best_checkpoint

    def get_latest_checkpoint(self) -> Checkpoint | None:
        """Get the most recent checkpoint."""
        return self.checkpoints[-1] if self.checkpoints else None

    def restore_from_best(self) -> List[ChainState]:
        """
        Restore chain states from the best checkpoint.

        Returns:
            List of chain states

        Raises:
            RuntimeError: If no checkpoint exists
        """
        if self.best_checkpoint is None:
            raise RuntimeError("No checkpoint available for restoration")

        return self.best_checkpoint.to_chain_states()

    def restore_from_latest(self) -> List[ChainState]:
        """
        Restore chain states from the latest checkpoint.

        Returns:
            List of chain states

        Raises:
            RuntimeError: If no checkpoint exists
        """
        checkpoint = self.get_latest_checkpoint()
        if checkpoint is None:
            raise RuntimeError("No checkpoint available for restoration")

        return checkpoint.to_chain_states()

    def _save_to_disk(self, checkpoint: Checkpoint, generation: int) -> None:
        """Save checkpoint to disk."""
        if self.checkpoint_dir is None:
            return

        filename = f"checkpoint_gen{generation:04d}.pt"
        filepath = self.checkpoint_dir / filename

        state = {
            "latents": [l.cpu() for l in checkpoint.latents],
            "scores": checkpoint.scores,
            "generation": checkpoint.generation,
            "best_latent": checkpoint.best_latent.cpu() if checkpoint.best_latent is not None else None,
            "best_score": checkpoint.best_score,
            "metadata": checkpoint.metadata,
        }

        torch.save(state, filepath)

    def load_from_disk(self, filepath: str | Path) -> Checkpoint:
        """
        Load a checkpoint from disk.

        Args:
            filepath: Path to the checkpoint file

        Returns:
            Loaded checkpoint
        """
        state = torch.load(filepath)

        return Checkpoint(
            latents=state["latents"],
            scores=state["scores"],
            generation=state["generation"],
            best_latent=state["best_latent"],
            best_score=state["best_score"],
            metadata=state.get("metadata", {}),
        )

    def clear(self) -> None:
        """Clear all checkpoints."""
        self.checkpoints.clear()
        self.best_checkpoint = None
