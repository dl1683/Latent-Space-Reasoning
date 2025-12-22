"""Judge implementations for scoring and modification of latent vectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, AutoConfig

from latent_reasoning.utils.device import get_device, ensure_tensor_device
from latent_reasoning.utils.quantization import get_default_dtype, get_quantization_kwargs

if TYPE_CHECKING:
    from latent_reasoning.core.chain import ModificationContext


class Judge(ABC):
    """Abstract base class for judges."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device the judge is on."""
        pass

    def to(self, device: torch.device | str) -> "Judge":
        """Move judge to a different device."""
        pass


class ScorerJudge(Judge):
    """
    Judge that scores latent vectors using semantic similarity.

    Uses a BERT-style model to evaluate latent representations by comparing
    them to a reference (query) embedding. Can also use a trained regression head.
    """

    def __init__(
        self,
        model_name: str,
        extraction_layer: int = -1,
        head_weights: str | None = None,
        canonical_dim: int | None = None,
        device_preference: str = "auto",
        use_semantic_similarity: bool = True,  # Default to similarity-based scoring
        quantization: str = "none",
    ):
        """
        Initialize the scorer judge.

        Args:
            model_name: HuggingFace model ID or local path
            extraction_layer: Which layer to use for scoring
            head_weights: Path to trained regression head weights
            canonical_dim: Dimension of canonical latent space (for projection)
            device_preference: Device to use
            use_semantic_similarity: If True, use cosine similarity to reference
        """
        self.model_name = model_name
        self.extraction_layer = extraction_layer
        self._device = get_device(device_preference)
        self.use_semantic_similarity = use_semantic_similarity
        self.quantization = quantization

        # Load model
        self._load_model()

        # Set up projection if needed
        self.canonical_dim = canonical_dim or self.hidden_size
        if self.canonical_dim != self.hidden_size:
            self.projection_in = nn.Linear(self.canonical_dim, self.hidden_size)
            self.projection_in.to(self._device)
        else:
            self.projection_in = None

        # Reference embedding for similarity-based scoring
        self._reference_embedding: Tensor | None = None

        # Regression head (optional, for trained scoring)
        self.regression_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 4, 1),
            nn.Tanh(),
        )
        self.regression_head.to(self._device)

        # Load trained weights if provided
        if head_weights:
            self._load_head_weights(head_weights)
            self.use_semantic_similarity = False  # Use trained head if weights provided

        # Calibration parameters
        self.calibration_mean = 0.0
        self.calibration_std = 1.0

    def set_reference(self, text: str | None = None, embedding: Tensor | None = None) -> None:
        """
        Set the reference for similarity-based scoring.

        IMPORTANT: When using semantic similarity mode with an external encoder,
        pass the encoder's latent as `embedding` (not text), since the scorer
        and encoder use different models/latent spaces.

        Args:
            text: Text to encode as reference (uses scorer's model)
            embedding: Pre-computed embedding to use as reference (from encoder)
        """
        if embedding is not None:
            self._reference_embedding = ensure_tensor_device(embedding, self._device)
            if self._reference_embedding.dim() == 1:
                self._reference_embedding = self._reference_embedding.unsqueeze(0)
            # Normalize for stable cosine similarity
            self._reference_embedding = torch.nn.functional.normalize(
                self._reference_embedding.float(), dim=-1
            )
        elif text is not None:
            self._reference_embedding = self._encode_text(text)
            self._reference_embedding = torch.nn.functional.normalize(
                self._reference_embedding.float(), dim=-1
            )
        else:
            self._reference_embedding = None

    def _encode_text(self, text: str) -> Tensor:
        """Encode text using the scorer's model."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Mean pooling over sequence
            hidden = outputs.hidden_states[self.extraction_layer]
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        return pooled

    def _load_model(self) -> None:
        """Load the model."""
        model_path = self.model_name
        if Path(self.model_name).exists():
            model_path = str(Path(self.model_name).resolve())

        # Get config first to determine hidden size
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.hidden_size = config.hidden_size

        model_kwargs = {
            "trust_remote_code": True,
            "output_hidden_states": True,
        }
        quant_kwargs, is_quantized, reason = get_quantization_kwargs(
            self.quantization,
            self._device,
        )
        if reason and self.quantization != "auto":
            print(f"Quantization disabled: {reason}")
        if is_quantized:
            model_kwargs.update(quant_kwargs)
        else:
            model_kwargs["torch_dtype"] = get_default_dtype(self._device)

        try:
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
        except Exception as exc:
            if not is_quantized:
                raise
            print(f"Quantized load failed: {exc}. Falling back to default dtype.")
            model_kwargs.pop("quantization_config", None)
            model_kwargs.pop("device_map", None)
            model_kwargs["torch_dtype"] = get_default_dtype(self._device)
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
            is_quantized = False

        if not is_quantized:
            self.model.to(self._device)
        self.model.eval()

        # Also load tokenizer for potential text input
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_head_weights(self, path: str) -> None:
        """Load regression head weights."""
        state_dict = torch.load(path, map_location=self._device)
        self.regression_head.load_state_dict(state_dict)

    def score(self, latent: Tensor) -> float:
        """
        Score a latent vector.

        Args:
            latent: Latent vector to score

        Returns:
            Score between -1 and 1
        """
        latent = ensure_tensor_device(latent, self._device)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        with torch.no_grad():
            if self.use_semantic_similarity and self._reference_embedding is not None:
                # For semantic similarity, compare DIRECTLY in canonical space
                # Do NOT project - both reference and latent are in same space
                latent_norm = torch.nn.functional.normalize(latent.float(), dim=-1)
                ref_norm = torch.nn.functional.normalize(self._reference_embedding.float(), dim=-1)
                similarity = torch.nn.functional.cosine_similarity(latent_norm, ref_norm)
                return similarity.item()  # Already in [-1, 1]
            else:
                # For regression head scoring, project if needed
                if self.projection_in is not None:
                    latent = self.projection_in(latent.float())
                latent = latent.to(torch.float16)
                score = self.regression_head(latent.float())
                raw_score = score.item()

                # Apply calibration
                calibrated = (raw_score - self.calibration_mean) / max(self.calibration_std, 1e-6)
                return max(-1.0, min(1.0, calibrated))

    def score_batch(self, latents: Tensor) -> list[float]:
        """
        Score multiple latent vectors.

        Args:
            latents: Batch of latent vectors (batch_size, latent_dim)

        Returns:
            List of scores
        """
        latents = ensure_tensor_device(latents, self._device)

        if self.projection_in is not None:
            latents = self.projection_in(latents.float())

        latents = latents.to(torch.float16)

        with torch.no_grad():
            scores = self.regression_head(latents.float())

        raw_scores = scores.squeeze(-1).tolist()

        # Apply calibration
        calibrated = [
            max(-1.0, min(1.0, (s - self.calibration_mean) / max(self.calibration_std, 1e-6)))
            for s in raw_scores
        ]

        return calibrated

    def calibrate(self, latents: Tensor) -> None:
        """
        Calibrate the scorer on a set of latents.

        Args:
            latents: Batch of latent vectors for calibration
        """
        latents = ensure_tensor_device(latents, self._device)

        if self.projection_in is not None:
            latents = self.projection_in(latents.float())

        latents = latents.to(torch.float16)

        with torch.no_grad():
            scores = self.regression_head(latents.float())

        scores = scores.squeeze(-1)
        self.calibration_mean = scores.mean().item()
        self.calibration_std = scores.std().item()

    def save_head(self, path: str) -> None:
        """Save the regression head weights."""
        torch.save(self.regression_head.state_dict(), path)

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "ScorerJudge":
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        self.model.to(device)
        self.regression_head.to(device)
        if self.projection_in is not None:
            self.projection_in.to(device)
        return self


class ModifierJudge(Judge):
    """
    Judge that suggests modifications to latent vectors using a decoder model.

    Uses a "gutted" decoder model to process latent vectors and output
    modification suggestions.
    """

    def __init__(
        self,
        model_name: str,
        layers: tuple[int, int] = (-8, -4),
        canonical_dim: int | None = None,
        device_preference: str = "auto",
        quantization: str = "auto",
    ):
        """
        Initialize the modifier judge.

        Args:
            model_name: HuggingFace model ID or local path
            layers: Which layers to use (start, end)
            canonical_dim: Dimension of canonical latent space
            device_preference: Device to use
        """
        self.model_name = model_name
        self.layers = layers
        self._device = get_device(device_preference)
        self.quantization = quantization

        # Load model
        self._load_model()

        # Set up projections
        self.canonical_dim = canonical_dim or self.hidden_size
        self.projection_in = nn.Linear(self.canonical_dim, self.hidden_size)
        self.projection_out = nn.Linear(self.hidden_size, self.canonical_dim)

        # Context projection (for trajectory context)
        # Input: current + momentum + recent_mods (flattened) + stuck_signal + cross_chain_summary
        context_input_dim = self.canonical_dim * 5 + 3  # Approximate
        self.context_projection = nn.Linear(context_input_dim, self.canonical_dim)

        # Move to device
        self.projection_in.to(self._device)
        self.projection_out.to(self._device)
        self.context_projection.to(self._device)

    def _load_model(self) -> None:
        """Load the model."""
        model_path = self.model_name
        if Path(self.model_name).exists():
            model_path = str(Path(self.model_name).resolve())

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.hidden_size = config.hidden_size

        model_kwargs = {
            "trust_remote_code": True,
            "output_hidden_states": True,
        }
        quant_kwargs, is_quantized, reason = get_quantization_kwargs(
            self.quantization,
            self._device,
        )
        if reason and self.quantization != "auto":
            print(f"Quantization disabled: {reason}")
        if is_quantized:
            model_kwargs.update(quant_kwargs)
        else:
            model_kwargs["torch_dtype"] = get_default_dtype(self._device)

        try:
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
        except Exception as exc:
            if not is_quantized:
                raise
            print(f"Quantized load failed: {exc}. Falling back to default dtype.")
            model_kwargs.pop("quantization_config", None)
            model_kwargs.pop("device_map", None)
            model_kwargs["torch_dtype"] = get_default_dtype(self._device)
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
            is_quantized = False

        if not is_quantized:
            self.model.to(self._device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def suggest_modification(
        self,
        latent: Tensor,
        context: "ModificationContext | None" = None,
    ) -> Tensor:
        """
        Suggest a modification vector for a latent.

        Args:
            latent: Current latent vector
            context: Optional trajectory context

        Returns:
            Modification vector (delta to add to latent)
        """
        latent = ensure_tensor_device(latent, self._device)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        # Build input with context if available
        if context is not None:
            input_tensor = self._build_context_input(latent, context)
        else:
            input_tensor = latent

        # Project into model's space
        projected = self.projection_in(input_tensor.float())

        # We need to create a proper input for the model
        # For decoder models, we'll use a dummy input and inject our latent
        dummy_input = self.tokenizer("<|start|>", return_tensors="pt")
        dummy_input = {k: v.to(self._device) for k, v in dummy_input.items()}

        with torch.no_grad():
            # Get model hidden states
            outputs = self.model(**dummy_input, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Get representation from specified layer range
            # Average across the layer range
            layer_outputs = []
            for layer_idx in range(self.layers[0], self.layers[1] + 1):
                if layer_idx < 0:
                    layer_idx = len(hidden_states) + layer_idx
                layer_outputs.append(hidden_states[layer_idx])

            # Average layers and sequence positions
            combined = torch.stack(layer_outputs).mean(dim=0)
            model_output = combined.mean(dim=1)  # Average over sequence

        # Blend with our projected input
        blended = model_output + projected.to(model_output.dtype)

        # Project back to canonical space
        modification = self.projection_out(blended.float())

        # The modification is the delta from current
        delta = modification - latent.float()

        return delta.squeeze(0)

    def _build_context_input(
        self,
        latent: Tensor,
        context: "ModificationContext",
    ) -> Tensor:
        """Build input tensor with context information."""
        components = [latent.float()]

        # Add momentum
        if context.momentum is not None:
            momentum = ensure_tensor_device(context.momentum, self._device)
            if momentum.dim() == 1:
                momentum = momentum.unsqueeze(0)
            components.append(momentum.float())

        # Add recent modifications (flatten if needed)
        if context.recent_modifications is not None:
            recent = ensure_tensor_device(context.recent_modifications, self._device)
            if recent.dim() == 3:  # (n_recent, 1, dim)
                recent = recent.view(1, -1)  # Flatten
            elif recent.dim() == 2:  # (n_recent, dim)
                recent = recent.view(1, -1)
            elif recent.dim() == 1:
                recent = recent.unsqueeze(0)
            components.append(recent.float())

        # Add stuck signal
        stuck = torch.tensor([[context.stuck_signal]], device=self._device, dtype=torch.float32)
        components.append(stuck)

        # Add cross-chain summary if available
        if context.cross_chain is not None:
            cc = context.cross_chain
            cc_tensor = torch.tensor(
                [[cc.num_active_chains, cc.best_global_score, cc.spread]],
                device=self._device,
                dtype=torch.float32,
            )
            components.append(cc_tensor)

            if cc.centroid is not None:
                centroid = ensure_tensor_device(cc.centroid, self._device)
                if centroid.dim() == 1:
                    centroid = centroid.unsqueeze(0)
                components.append(centroid.float())

        # Concatenate and project
        try:
            combined = torch.cat(components, dim=-1)
            # Use context projection to reduce to canonical dim
            if combined.size(-1) > self.canonical_dim:
                # Pad context projection if needed
                if self.context_projection.in_features != combined.size(-1):
                    self.context_projection = nn.Linear(combined.size(-1), self.canonical_dim)
                    self.context_projection.to(self._device)
                return self.context_projection(combined)
            else:
                # Pad with zeros
                padding = torch.zeros(
                    combined.size(0),
                    self.canonical_dim - combined.size(-1),
                    device=self._device,
                    dtype=torch.float32,
                )
                return torch.cat([combined, padding], dim=-1)
        except Exception:
            # Fallback to just using the latent
            return latent.float()

    def save_projections(self, path: str) -> None:
        """Save projection layer weights."""
        state = {
            "projection_in": self.projection_in.state_dict(),
            "projection_out": self.projection_out.state_dict(),
            "context_projection": self.context_projection.state_dict(),
        }
        torch.save(state, path)

    def load_projections(self, path: str) -> None:
        """Load projection layer weights."""
        state = torch.load(path, map_location=self._device)
        self.projection_in.load_state_dict(state["projection_in"])
        self.projection_out.load_state_dict(state["projection_out"])
        if "context_projection" in state:
            self.context_projection.load_state_dict(state["context_projection"])

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "ModifierJudge":
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        self.model.to(device)
        self.projection_in.to(device)
        self.projection_out.to(device)
        self.context_projection.to(device)
        return self


class TrainedLatentJudge(Judge):
    """
    Judge that uses a trained LatentScorer to score latent vectors.

    This scorer works directly in the encoder's latent space,
    making it fast and well-calibrated for evolution.

    Supports projection from different encoder dimensions to the scorer's
    expected dimension, enabling multi-model benchmarking with a single scorer.

    Load via config:
        judges:
          scorers:
            - type: trained_latent
              checkpoint: checkpoints/latent_scorer/final_model.pt
              latent_dim: 1024
    """

    def __init__(
        self,
        checkpoint_path: str,
        latent_dim: int = 1024,
        device_preference: str = "auto",
    ):
        """
        Initialize the trained latent judge.

        Args:
            checkpoint_path: Path to the trained LatentScorer checkpoint
            latent_dim: Dimension of the encoder's latent space (may differ from scorer)
            device_preference: Device to use
        """
        from latent_reasoning.core.latent_scorer import LatentScorer

        self._device = get_device(device_preference)
        self.input_latent_dim = latent_dim  # The encoder's actual dimension

        # Load the trained model
        checkpoint = torch.load(checkpoint_path, map_location=self._device)

        # Get model config from checkpoint if available
        self.scorer_latent_dim = checkpoint.get("latent_dim", 1024)  # Scorer's expected dim
        use_query_conditioning = checkpoint.get("use_query_conditioning", True)

        self.model = LatentScorer(
            latent_dim=self.scorer_latent_dim,
            use_query_conditioning=use_query_conditioning,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self._device)
        self.model.eval()

        # Create projection layer if dimensions don't match
        self._projection: torch.nn.Linear | None = None
        if latent_dim != self.scorer_latent_dim:
            self._projection = torch.nn.Linear(latent_dim, self.scorer_latent_dim)
            # Initialize with small weights for stable projection
            torch.nn.init.xavier_uniform_(self._projection.weight, gain=0.1)
            torch.nn.init.zeros_(self._projection.bias)
            self._projection.to(self._device)
            print(f"Created projection layer: {latent_dim} -> {self.scorer_latent_dim}")

        # Reference latent for query conditioning
        self._reference_latent: Tensor | None = None

        print(f"Loaded TrainedLatentJudge from {checkpoint_path} (scorer_dim={self.scorer_latent_dim}, input_dim={latent_dim})")

    def _project(self, latent: Tensor) -> Tensor:
        """Project latent to scorer's expected dimension if needed."""
        if self._projection is not None:
            with torch.no_grad():
                return self._projection(latent)
        return latent

    def set_reference(self, text: str | None = None, embedding: Tensor | None = None) -> None:
        """
        Set the reference latent for query conditioning.

        Args:
            text: Ignored (use embedding instead)
            embedding: Query latent from the encoder
        """
        if embedding is not None:
            ref_latent = ensure_tensor_device(embedding, self._device)
            if ref_latent.dim() == 1:
                ref_latent = ref_latent.unsqueeze(0)
            ref_latent = ref_latent.float()
            # Project to scorer's dimension if needed
            self._reference_latent = self._project(ref_latent)

    def score(self, latent: Tensor) -> float:
        """
        Score a latent vector.

        Args:
            latent: Latent vector to score

        Returns:
            Score between 0 and 1
        """
        latent = ensure_tensor_device(latent, self._device).float()
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        # Project to scorer's dimension if needed
        latent = self._project(latent)

        with torch.no_grad():
            score = self.model(latent, self._reference_latent)

        # Convert 0-1 range to -1 to 1 for compatibility with other judges
        return (score.item() * 2) - 1

    def score_batch(self, latents: Tensor) -> list[float]:
        """
        Score multiple latent vectors.

        Args:
            latents: Batch of latent vectors (batch_size, latent_dim)

        Returns:
            List of scores
        """
        latents = ensure_tensor_device(latents, self._device).float()

        # Project to scorer's dimension if needed
        latents = self._project(latents)

        with torch.no_grad():
            scores = self.model(latents, self._reference_latent)

        # Convert 0-1 range to -1 to 1
        return [(s.item() * 2) - 1 for s in scores]

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "TrainedLatentJudge":
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        self.model.to(device)
        if self._projection is not None:
            self._projection.to(device)
        if self._reference_latent is not None:
            self._reference_latent = self._reference_latent.to(device)
        return self


class DecodeScoreJudge(Judge):
    """
    Judge that scores latent vectors by decoding them and scoring the text.

    This is the approach that actually works: decode → score text with heuristics.
    More expensive than direct latent scoring, but actually produces meaningful results.

    Usage:
        judge = DecodeScoreJudge(encoder)
        judge.set_reference(embedding=query_latent, query="your query text")
        score = judge.score(mutated_latent)
    """

    def __init__(
        self,
        encoder,  # LLMEncoder instance
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        device_preference: str = "auto",
    ):
        """
        Initialize the decode-score judge.

        Args:
            encoder: LLMEncoder instance for decoding
            max_new_tokens: Max tokens for decoded text
            temperature: Sampling temperature for decode
            device_preference: Device to use
        """
        from latent_reasoning.core.heuristic_scorer import HeuristicScorer

        self.encoder = encoder
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._device = get_device(device_preference)

        self.heuristic_scorer = HeuristicScorer()

        # Query text for decoding context
        self._query_text: str | None = None
        self._reference_latent: Tensor | None = None

        # Cache to avoid re-decoding the same latent
        self._decode_cache: dict[str, float] = {}
        self._cache_max_size = 100

    def set_reference(self, text: str | None = None, embedding: Tensor | None = None, query: str | None = None) -> None:
        """
        Set the reference for scoring.

        Args:
            text: Ignored (for compatibility)
            embedding: Reference latent (for cache key generation)
            query: Query text for decoding context (IMPORTANT!)
        """
        if embedding is not None:
            self._reference_latent = ensure_tensor_device(embedding, self._device)
        if query is not None:
            self._query_text = query
        elif text is not None:
            self._query_text = text

        # Clear cache when reference changes
        self._decode_cache.clear()

    def _latent_to_cache_key(self, latent: Tensor) -> str:
        """Generate a cache key from a latent vector."""
        # Use first/last few values + norm as a cheap hash
        flat = latent.flatten()
        key_vals = [flat[0].item(), flat[-1].item(), flat[len(flat)//2].item(), latent.norm().item()]
        return f"{key_vals[0]:.4f}_{key_vals[1]:.4f}_{key_vals[2]:.4f}_{key_vals[3]:.4f}"

    def score(self, latent: Tensor) -> float:
        """
        Score a latent by decoding it and scoring the text.

        Args:
            latent: Latent vector to score

        Returns:
            Score between -1 and 1 (mapped from heuristic's 0-1 scale)
        """
        latent = ensure_tensor_device(latent, self._device)

        # Check cache
        cache_key = self._latent_to_cache_key(latent)
        if cache_key in self._decode_cache:
            return self._decode_cache[cache_key]

        # Decode the latent to text
        try:
            decoded_text = self.encoder.decode(
                latent,
                query=self._query_text,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            print(f"Decode error: {e}")
            return -1.0  # Failed decode gets worst score

        # Score with heuristics
        result = self.heuristic_scorer.score(decoded_text)
        raw_score = result.overall_score  # 0-1 range

        # Map to -1 to 1 for compatibility
        score = (raw_score * 2) - 1

        # Cache (with size limit)
        if len(self._decode_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._decode_cache))
            del self._decode_cache[oldest_key]
        self._decode_cache[cache_key] = score

        return score

    def score_batch(self, latents: Tensor) -> list[float]:
        """
        Score multiple latent vectors.

        Args:
            latents: Batch of latent vectors (batch_size, latent_dim)

        Returns:
            List of scores
        """
        return [self.score(latent) for latent in latents]

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "DecodeScoreJudge":
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        return self


def create_scorer_from_config(
    config: "ScorerConfig",
    device: str = "auto",
    encoder_latent_dim: int | None = None,
) -> Judge:
    """
    Factory function to create a scorer judge from config.

    Args:
        config: ScorerConfig instance
        device: Device preference
        encoder_latent_dim: Actual latent dimension from the encoder (for projection)

    Returns:
        Judge instance (ScorerJudge or TrainedLatentJudge)
    """
    from latent_reasoning.config import ScorerConfig

    if config.type == "trained_latent":
        if not config.checkpoint:
            raise ValueError("trained_latent scorer requires 'checkpoint' path in config")
        # Use encoder's latent_dim if provided, otherwise fall back to config
        latent_dim = encoder_latent_dim if encoder_latent_dim is not None else config.latent_dim
        return TrainedLatentJudge(
            checkpoint_path=config.checkpoint,
            latent_dim=latent_dim,
            device_preference=device,
        )
    else:  # semantic
        if not config.model:
            raise ValueError("semantic scorer requires 'model' in config")
        return ScorerJudge(
            model_name=config.model,
            extraction_layer=config.layer,
            head_weights=config.head,
            canonical_dim=encoder_latent_dim,
            device_preference=device,
            quantization=config.quantization,
        )
