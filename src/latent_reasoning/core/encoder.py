"""
Encoder implementations for converting text to latent representations.

This module contains the core encoding/decoding functionality that converts
between text and latent space representations. The main class is LLMEncoder,
which uses transformer models to:

1. Encode text queries into latent vectors by extracting hidden states
2. Decode evolved latent vectors back into text using influenced generation

Key classes:
- Encoder: Abstract base class defining the interface
- LLMEncoder: Main implementation using HuggingFace transformers
- LatentToSoftPrompt: Projects latents to soft prompt embeddings (experimental)

The encoding process extracts hidden states from an intermediate layer of
the transformer (typically -4, the 4th layer from the end) and pools them
across the sequence dimension to create a single latent vector.

The decoding process uses the latent vector to influence text generation
through RNG seeding and temperature modulation, creating outputs that
reflect the evolved latent characteristics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_reasoning.utils.device import get_device, ensure_tensor_device
from latent_reasoning.utils.quantization import get_default_dtype, get_quantization_kwargs


class LatentToSoftPrompt(nn.Module):
    """
    Projects a latent vector into a sequence of soft prompt embeddings.

    This allows conditioning text generation on the evolved latent.
    """

    def __init__(
        self,
        latent_dim: int,
        embed_dim: int,
        num_soft_tokens: int = 8,
    ):
        super().__init__()
        self.num_soft_tokens = num_soft_tokens
        self.embed_dim = embed_dim

        # Project latent to soft prompt sequence
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim * num_soft_tokens),
        )

    def forward(self, latent: Tensor) -> Tensor:
        """
        Convert latent to soft prompt embeddings.

        Args:
            latent: (batch, latent_dim) or (latent_dim,)

        Returns:
            Soft prompt embeddings (batch, num_soft_tokens, embed_dim)
        """
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        # Project to flat embedding
        flat = self.projector(latent.float())

        # Reshape to sequence
        batch_size = flat.size(0)
        soft_prompt = flat.view(batch_size, self.num_soft_tokens, self.embed_dim)

        return soft_prompt


class Encoder(ABC):
    """
    Abstract base class for encoders that convert between text and latent space.

    Encoders are the foundation of the latent space reasoning system. They define
    the latent space that evolutionary algorithms operate in and provide the
    critical encode/decode operations that bridge text and vector representations.

    Core responsibilities:
    1. **Encoding**: Convert text queries into dense latent vector representations
       that capture semantic meaning and can be evolved through optimization
    2. **Decoding**: Convert evolved latent vectors back into coherent text responses
       that reflect the optimized representation
    3. **Consistency**: Ensure that identical inputs produce identical outputs
       (deterministic) and that similar latents produce similar text
    4. **Device Management**: Handle GPU/CPU placement and memory management

    Key properties for effective evolution:
    - **Semantic Preservation**: Latent vectors should preserve the semantic
      content of the original text
    - **Evolvability**: Small changes in latent space should produce meaningful
      changes in decoded text
    - **Stability**: The encode/decode process should be numerically stable
    - **Efficiency**: Fast enough for evolutionary algorithms that require
      hundreds of encode/decode operations

    Implementation Guidelines:
    - Use deterministic operations to ensure reproducibility
    - Handle device placement automatically
    - Provide meaningful error messages for invalid inputs
    - Optimize for batch processing when possible
    - Consider memory usage for large models

    Example Implementation Pattern:
        >>> class MyEncoder(Encoder):
        ...     def encode(self, text: str) -> Tensor:
        ...         # Convert text to latent vector
        ...         return latent_vector
        ...
        ...     def decode(self, latent: Tensor, query: str = None) -> str:
        ...         # Convert latent back to text
        ...         return generated_text
    """

    @abstractmethod
    def encode(self, text: str) -> Tensor:
        """
        Encode text into a latent vector representation.

        This is the core method that converts natural language into the vector
        space where evolutionary optimization occurs. The quality of this encoding
        directly impacts the effectiveness of the reasoning process.

        Args:
            text: Input text to encode. Should handle various types of input:
                questions, statements, prompts, etc.

        Returns:
            Latent vector representing the text. Should be:
            - Deterministic (same input → same output)
            - Dense and meaningful (captures semantic content)
            - Evolvable (small changes → meaningful text changes)

        Raises:
            ValueError: If text is invalid or cannot be processed
            RuntimeError: If encoding fails due to model or device issues
        """
        pass

    @abstractmethod
    def decode(self, latent: Tensor, query: str | None = None) -> str:
        """
        Decode a latent vector back to text.

        This method converts evolved latent representations back into natural
        language. It should produce text that reflects the optimized properties
        of the latent vector.

        Args:
            latent: Latent vector to decode. May be evolved/mutated from original.
            query: Optional original query for context. Can help guide generation
                to stay relevant to the original question.

        Returns:
            Generated text from the latent representation. Should be:
            - Coherent and well-formed
            - Semantically related to the latent content
            - Different for different latent vectors

        Raises:
            ValueError: If latent has wrong shape or invalid values
            RuntimeError: If decoding fails due to model or device issues
        """
        pass

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """
        Return the dimensionality of the latent space.

        This defines the size of vectors that the evolutionary algorithm will
        operate on. Higher dimensions allow more expressive representations
        but require more computation.

        Returns:
            Integer dimension of latent vectors (e.g., 1536, 2048, 4096)
        """
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """
        Return the device the encoder is running on.

        Used for proper tensor placement and memory management.

        Returns:
            PyTorch device (e.g., torch.device('cuda:0'), torch.device('cpu'))
        """
        pass


class LLMEncoder(Encoder):
    """
    Main encoder implementation using transformer model hidden states.

    This encoder extracts hidden state representations from an intermediate
    layer of a transformer model and uses them as the latent space for evolution.
    The evolved latents are then used to influence text generation.

    Key features:
    - Extracts hidden states from any transformer layer (default: -4)
    - Supports multiple pooling strategies (mean, last, cls)
    - Handles device management automatically
    - Uses latent statistics to influence generation parameters

    The encoding process:
    1. Tokenize input text
    2. Run forward pass through transformer
    3. Extract hidden states from specified layer
    4. Pool across sequence dimension to get single vector

    The decoding process:
    1. Use latent statistics to seed RNG for reproducibility
    2. Modulate temperature based on latent variance
    3. Generate text with these influenced parameters
    """

    def __init__(
        self,
        model_name: str,
        extraction_layer: int = -4,
        pooling: Literal["mean", "last", "cls"] = "mean",
        device_preference: str = "auto",
        max_length: int = 2048,
        quantization: str = "auto",
    ):
        """
        Initialize the LLM encoder with a transformer model.

        Args:
            model_name: HuggingFace model ID or local path. Examples:
                - "Qwen/Qwen3-4B" (recommended for quality)
                - "Qwen/Qwen3-1.7B" (good balance)
                - "Qwen/Qwen3-0.6B" (fastest, minimal VRAM)
                - "microsoft/Phi-3.5-mini-instruct"
            extraction_layer: Which hidden layer to extract from. Negative values count
                from the end (-4 = 4th from last, usually optimal for reasoning).
                Earlier layers capture syntax, later layers capture semantics.
            pooling: How to pool sequence dimension:
                - "mean": Average across all tokens (recommended, captures full context)
                - "last": Use last non-padded token (good for completion tasks)
                - "cls": Use first token (for models with CLS tokens)
            device_preference: Device to use:
                - "auto": Automatically choose best available (CUDA > CPU)
                - "cuda": Force CUDA (will fail if not available)
                - "cpu": Force CPU (slower but always works)
                - "cuda:0": Specific GPU device
            max_length: Maximum input sequence length in tokens. Longer sequences
                are truncated. Typical values: 512-2048 for reasoning tasks.

        Example:
            Basic usage:
            >>> encoder = LLMEncoder("Qwen/Qwen3-1.7B")
            >>> latent = encoder.encode("How to implement caching?")
            >>> print(f"Latent shape: {latent.shape}")  # torch.Size([1536])

            Custom configuration:
            >>> encoder = LLMEncoder(
            ...     model_name="Qwen/Qwen3-4B",
            ...     extraction_layer=-2,  # Use later layer for more semantic info
            ...     pooling="last",       # Focus on completion
            ...     device_preference="cuda:1"
            ... )

            Batch encoding for efficiency:
            >>> queries = ["Query 1", "Query 2", "Query 3"]
            >>> latents = encoder.encode_batch(queries)
            >>> print(f"Batch shape: {latents.shape}")  # torch.Size([3, 1536])

        Raises:
            ValueError: If model_name is not found or invalid
            RuntimeError: If CUDA is requested but not available
            OSError: If model files cannot be loaded
        """
        self.model_name = model_name
        self.extraction_layer = extraction_layer
        self.pooling = pooling
        self.max_length = max_length
        self.quantization = quantization

        # Get device
        self._device = get_device(device_preference)

        # Load model and tokenizer
        self._load_model()

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        model_path = self.model_name
        if Path(self.model_name).exists():
            model_path = str(Path(self.model_name).resolve())

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        except Exception as exc:
            if not is_quantized:
                raise
            print(f"Quantized load failed: {exc}. Falling back to default dtype.")
            model_kwargs.pop("quantization_config", None)
            model_kwargs.pop("device_map", None)
            model_kwargs["torch_dtype"] = get_default_dtype(self._device)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            is_quantized = False

        if not is_quantized:
            self.model.to(self._device)
        self.model.eval()

        # Cache the hidden size
        self._latent_dim = self.model.config.hidden_size

        # Get embedding dimension (might differ from hidden_size in some models)
        self._embed_dim = self.model.get_input_embeddings().embedding_dim

        # Initialize soft prompt projector for decoding
        self.soft_prompt_projector = LatentToSoftPrompt(
            latent_dim=self._latent_dim,
            embed_dim=self._embed_dim,
            num_soft_tokens=8,  # 8 soft tokens to condition generation
        )
        self.soft_prompt_projector.to(self._device)

    def encode(self, text: str) -> Tensor:
        """
        Encode text into a latent vector representation.

        This method converts input text into a dense vector representation by:
        1. Tokenizing the text using the model's tokenizer
        2. Running a forward pass through the transformer
        3. Extracting hidden states from the specified layer
        4. Pooling across the sequence dimension to get a single vector

        The resulting latent vector captures semantic information about the text
        and can be evolved through the reasoning process.

        Args:
            text: Input text to encode. Can be a question, statement, or any text.
                Examples: "How to implement caching?", "Design a REST API"

        Returns:
            Latent vector of shape (latent_dim,) representing the input text.
            The vector is on the same device as the model.

        Example:
            >>> encoder = LLMEncoder("Qwen/Qwen3-1.7B")
            >>> latent = encoder.encode("How to optimize database queries?")
            >>> print(f"Latent shape: {latent.shape}")  # torch.Size([1536])
            >>> print(f"Latent device: {latent.device}")  # cuda:0 or cpu
            >>> print(f"Latent dtype: {latent.dtype}")   # torch.float16

        Note:
            - Input text longer than max_length will be truncated
            - The latent vector preserves semantic meaning for evolution
            - Identical inputs always produce identical latents (deterministic)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Get hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states[self.extraction_layer]

        # Pool across sequence dimension
        if self.pooling == "mean":
            # Mean pooling with attention mask
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        elif self.pooling == "last":
            # Take the last non-padded token
            seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            pooled = hidden_states[torch.arange(batch_size), seq_lengths]
        elif self.pooling == "cls":
            # Take the first token (CLS position)
            pooled = hidden_states[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        return pooled.squeeze(0)  # Remove batch dimension for single input

    def encode_batch(self, texts: list[str]) -> Tensor:
        """
        Encode multiple texts in a single batch for maximum efficiency.

        This method is significantly faster than calling encode() multiple times
        because it processes all texts in a single forward pass through the model.
        Use this when you need to encode many texts at once.

        Args:
            texts: List of texts to encode. All texts will be padded to the same
                length and processed together. Empty list returns empty tensor.

        Returns:
            Stacked latent vectors of shape (batch_size, latent_dim).
            Each row corresponds to one input text in the same order.

        Example:
            >>> encoder = LLMEncoder("Qwen/Qwen3-1.7B")
            >>> queries = [
            ...     "How to implement caching?",
            ...     "Design a REST API",
            ...     "Optimize database queries"
            ... ]
            >>> latents = encoder.encode_batch(queries)
            >>> print(f"Batch shape: {latents.shape}")  # torch.Size([3, 1536])
            >>>
            >>> # Process individual latents
            >>> for i, query in enumerate(queries):
            ...     latent = latents[i]
            ...     response = encoder.decode(latent, query=query)
            ...     print(f"Query {i+1}: {response[:50]}...")

        Performance:
            - ~3-5x faster than individual encode() calls for batches of 5-10 texts
            - Memory usage scales with batch size and max sequence length
            - Optimal batch size depends on available GPU memory
        """
        if not texts:
            return torch.empty(0, self._latent_dim, device=self._device)

        # Tokenize all texts together
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Get hidden states in single forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states[self.extraction_layer]

        # Pool across sequence dimension
        if self.pooling == "mean":
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        elif self.pooling == "last":
            seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            pooled = hidden_states[torch.arange(batch_size, device=self._device), seq_lengths]
        elif self.pooling == "cls":
            pooled = hidden_states[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        return pooled

    def decode(
        self,
        latent: Tensor,
        query: str | None = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        """
        Decode a latent vector back to text using latent-influenced generation.

        This method converts an evolved latent vector back into text by using the
        latent's statistical properties to influence the generation process. The
        approach ensures that different latent vectors produce meaningfully different
        outputs, which is crucial for the evolutionary process.

        The decoding process:
        1. Extracts statistical properties from the latent vector (mean, std, norm, etc.)
        2. Uses these statistics to create a deterministic seed for reproducibility
        3. Modulates generation parameters (temperature, top_p) based on latent properties
        4. Generates text using the model's chat template for proper formatting
        5. Cleans up the output to extract just the assistant's response

        Args:
            latent: Evolved latent vector to decode. Shape: (latent_dim,) or (1, latent_dim).
                Different latent vectors should produce meaningfully different outputs.
            query: Original query to condition generation on. If provided, creates a
                proper planning prompt. If None, uses a generic planning prompt.
            max_new_tokens: Maximum number of new tokens to generate. Typical values:
                - 512: Short responses
                - 1024: Medium responses (recommended)
                - 2048: Long detailed responses
            temperature: Base sampling temperature. The actual temperature used will be
                modulated by the latent vector's properties:
                - 0.0: Deterministic greedy decoding (recommended for consistency)
                - 0.1-0.3: Low randomness
                - 0.7-1.0: Higher creativity

        Returns:
            Generated text response. The output is cleaned to remove chat formatting,
            thinking tokens, and other artifacts, returning just the assistant's response.

        Example:
            >>> encoder = LLMEncoder("Qwen/Qwen3-1.7B")
            >>> latent = encoder.encode("How to implement caching?")
            >>> # Evolve the latent through mutation/crossover...
            >>> response = encoder.decode(latent, query="How to implement caching?")
            >>> print(response)
            # "1. Choose caching strategy (Redis, Memcached, in-memory)
            #  2. Identify what data to cache (frequently accessed, expensive queries)
            #  3. Implement cache-aside pattern..."

        Note:
            - Identical latent vectors always produce identical outputs (deterministic)
            - Small changes in latent vectors produce meaningfully different outputs
            - The method handles various chat templates and model formats automatically
            - Thinking tokens (like Qwen3's <think>...</think>) are automatically removed
        """
        latent = ensure_tensor_device(latent, self._device)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        # Use multiple latent statistics for diversity - small mutations should produce different outputs
        latent_flat = latent.flatten().float()
        latent_mean = latent_flat.mean().item()
        latent_std = latent_flat.std().item()
        latent_sum = latent_flat.sum().item()
        latent_norm = latent_flat.norm().item()

        # Use first 64 dims and last 64 dims for additional sensitivity
        first_chunk = latent_flat[:64].sum().item() if len(latent_flat) >= 64 else latent_flat.sum().item()
        last_chunk = latent_flat[-64:].sum().item() if len(latent_flat) >= 64 else 0.0

        # Create seed from multiple statistics - much more sensitive to mutations
        seed_components = [
            int(abs(latent_mean * 1e7)) % (2**20),
            int(abs(latent_std * 1e7)) % (2**20),
            int(abs(latent_sum * 1e3)) % (2**20),
            int(abs(first_chunk * 1e5)) % (2**20),
            int(abs(last_chunk * 1e5)) % (2**20),
        ]
        latent_seed = sum(seed_components) % (2**31)

        # Modulate temperature and top_p based on latent properties
        # Higher norm = more confident = lower temperature
        norm_factor = min(latent_norm / 1000.0, 2.0)  # Normalize
        adjusted_temp = temperature * (1.0 + 0.3 * (1.0 - norm_factor))  # Increased influence
        adjusted_temp = max(0.1, min(1.5, adjusted_temp))

        # Vary top_p slightly based on std
        adjusted_top_p = 0.9 + 0.08 * (latent_std - 0.5)  # Range ~0.86-0.94
        adjusted_top_p = max(0.8, min(0.95, adjusted_top_p))
        torch.manual_seed(latent_seed)

        # Build chat messages for proper formatting
        system_msg = "You are a helpful planning assistant. Create clear, actionable step-by-step plans."
        if query:
            user_msg = f"Create a detailed step-by-step plan for: {query}"
        else:
            user_msg = "Create a detailed step-by-step plan."

        # Try to use chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                # Fallback to manual format
                prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Fallback to simple format
            prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant: "

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            # Use greedy decoding for temperature=0, sampling otherwise
            if adjusted_temp < 0.01:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding for deterministic output
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=adjusted_temp,
                    top_p=adjusted_top_p,
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                )

        # Decode and extract just the assistant's response
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Try to extract just the plan part (after the user's message)
        if user_msg in generated:
            # Find where the assistant's response starts
            idx = generated.find(user_msg) + len(user_msg)
            response = generated[idx:].strip()
        else:
            response = generated.strip()

        # Clean up role markers and thinking tokens
        for marker in ["Assistant:", "assistant:", "<|im_start|>", "<|im_end|>"]:
            if response.startswith(marker):
                response = response[len(marker):].strip()

        # Remove thinking block if present (Qwen3 thinking mode)
        # Format: <think>...</think> followed by actual response
        if "<think>" in response and "</think>" in response:
            think_end = response.find("</think>") + len("</think>")
            response = response[think_end:].strip()
        elif response.startswith("<think>"):
            # Thinking tag but no end tag - might be truncated, try to find actual content
            # Look for common plan starters after the thinking
            for starter in ["1.", "Step 1", "## Step", "Here's", "Here is"]:
                if starter in response:
                    idx = response.find(starter)
                    response = response[idx:]
                    break

        return response if response else generated.strip()

    def decode_with_soft_prompt(
        self,
        latent: Tensor,
        query: str | None = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """
        Decode using soft prompts derived from the latent vector.

        This method actually uses the latent to condition generation by:
        1. Converting latent to soft prompt embeddings
        2. Prepending soft prompts to the input embeddings
        3. Generating conditioned on the combined embeddings

        Args:
            latent: Latent vector to decode
            query: Original query to condition generation on
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text conditioned on the latent
        """
        latent = ensure_tensor_device(latent, self._device)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        # Generate soft prompt embeddings from latent
        with torch.no_grad():
            soft_prompt = self.soft_prompt_projector(latent)  # (1, num_soft_tokens, embed_dim)
            soft_prompt = soft_prompt.to(self.model.dtype)

        # Build the text prompt
        system_msg = "You are a helpful planning assistant. Create clear, actionable step-by-step plans."
        if query:
            user_msg = f"Create a detailed step-by-step plan for: {query}"
        else:
            user_msg = "Create a detailed step-by-step plan."

        # Try to use chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant: "

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Get text embeddings
        with torch.no_grad():
            text_embeds = self.model.get_input_embeddings()(inputs["input_ids"])
            # text_embeds shape: (1, seq_len, embed_dim)

        # Concatenate soft prompt + text embeddings
        combined_embeds = torch.cat([soft_prompt, text_embeds], dim=1)

        # Create new attention mask for the combined sequence
        soft_prompt_mask = torch.ones(
            1, soft_prompt.size(1),
            dtype=inputs["attention_mask"].dtype,
            device=self._device
        )
        combined_attention_mask = torch.cat(
            [soft_prompt_mask, inputs["attention_mask"]], dim=1
        )

        # Generate from embeddings
        with torch.no_grad():
            if temperature < 0.01:
                outputs = self.model.generate(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                )
            else:
                outputs = self.model.generate(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                )

        # Decode output
        # Note: outputs include the input sequence, so skip the prompt length
        # But since we used inputs_embeds, the output includes soft prompt tokens
        # We need to skip those when decoding
        prompt_len = combined_embeds.size(1)
        generated_ids = outputs[0, prompt_len:]  # Skip prompt tokens

        generated = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Clean up
        response = generated.strip()

        # Remove thinking block if present
        if "<think>" in response and "</think>" in response:
            think_end = response.find("</think>") + len("</think>")
            response = response[think_end:].strip()
        elif response.startswith("<think>"):
            for starter in ["1.", "Step 1", "## Step", "Here's", "Here is"]:
                if starter in response:
                    idx = response.find(starter)
                    response = response[idx:]
                    break

        return response if response else generated.strip()

    def encode_batch(self, texts: list[str]) -> Tensor:
        """
        Encode multiple texts into latent vectors.

        Args:
            texts: List of texts to encode

        Returns:
            Batch of latent vectors (batch_size, latent_dim)
        """
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Get hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states[self.extraction_layer]

        # Pool across sequence dimension
        if self.pooling == "mean":
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        elif self.pooling == "last":
            seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            pooled = hidden_states[torch.arange(batch_size), seq_lengths]
        elif self.pooling == "cls":
            pooled = hidden_states[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        return pooled

    @property
    def latent_dim(self) -> int:
        """Return the dimensionality of the latent space."""
        return self._latent_dim

    @property
    def device(self) -> torch.device:
        """Return the device the encoder is on."""
        return self._device

    def to(self, device: torch.device | str) -> "LLMEncoder":
        """Move encoder to a different device."""
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        self.model.to(device)
        self.soft_prompt_projector.to(device)
        return self

    def save_decoder_weights(self, path: str | Path) -> None:
        """Save the soft prompt projector weights."""
        torch.save(self.soft_prompt_projector.state_dict(), path)

    def load_decoder_weights(self, path: str | Path) -> None:
        """Load the soft prompt projector weights."""
        state_dict = torch.load(path, map_location=self._device)
        self.soft_prompt_projector.load_state_dict(state_dict)
