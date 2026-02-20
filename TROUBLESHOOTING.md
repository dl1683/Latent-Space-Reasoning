# Troubleshooting Guide

This guide helps you resolve common issues when using the Latent Space Reasoning Engine.

Important: scorer numbers (`latent_score`, `confidence`) are internal search guidance only. Evaluate decoded outputs directly for quality decisions.

## Quick Diagnostics

First, run the system check:
```bash
latent-reason check-gpu
```

This will show your Python version, PyTorch installation, GPU availability, and memory.

## Common Issues

### 1. Installation Problems

**Error: `ModuleNotFoundError: No module named 'latent_reasoning'`**
```bash
# Solution: Install the package
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

**Error: `No module named 'torch'`**
```bash
# Solution: Install PyTorch
pip install torch torchvision torchaudio

# For CUDA support (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. GPU/Memory Issues

**Error: `CUDA out of memory`**
```bash
# Solution 1: Use smaller model
latent-reason run "your query" --encoder Qwen/Qwen3-0.6B

# Solution 2: Reduce parameters
latent-reason run "your query" --chains 3 --max-tokens 1024

# Solution 3: Use CPU (slower but works)
# Edit config.yaml: device: "cpu"
```

**Error: `RuntimeError: No CUDA GPUs are available`**
```bash
# Check GPU status
latent-reason check-gpu

# Use CPU fallback
latent-reason run "your query" --encoder Qwen/Qwen3-0.6B
# The system will automatically fall back to CPU
```

### 3. Model Loading Issues

**Error: `FileNotFoundError: checkpoints/latent_scorer/final_model.pt`**
```bash
# Solution: Make sure you have the complete repository
git clone https://github.com/dl1683/Latent-Space-Reasoning.git
cd Latent-Space-Reasoning

# The checkpoints/ directory should exist with trained models
ls checkpoints/latent_scorer/
```

**Error: `OSError: Can't load tokenizer for 'Qwen/Qwen3-4B'`**
```bash
# Solution: Check internet connection and try smaller model
latent-reason run "your query" --encoder Qwen/Qwen3-0.6B

# Or use a local model path if you have models downloaded
latent-reason run "your query" --encoder /path/to/local/model
```

### 4. Performance Issues

**Problem: Very slow execution**
```bash
# Solution 1: Use smaller model
latent-reason run "your query" --encoder Qwen/Qwen3-0.6B

# Solution 2: Reduce evolution parameters
latent-reason run "your query" --chains 3 --generations 5

# Solution 3: Use minimal verbosity
latent-reason run "your query" --verbosity minimal
```

**Problem: Low quality results**
```bash
# Solution 1: Use larger model
latent-reason run "your query" --encoder Qwen/Qwen3-4B

# Solution 2: Increase evolution parameters
latent-reason run "your query" --chains 8 --generations 15

# Solution 3: Compare with baseline to see improvement
latent-reason compare "your query"
```

### 5. Output Issues

**Problem: Garbled or missing characters in output**
```bash
# This is usually a console encoding issue
# The actual results are fine, just display problems

# Solution 1: Use JSON output
latent-reason run "your query" --format json

# Solution 2: Save to file
latent-reason run "your query" --output results.json

# Solution 3: Use markdown format
latent-reason run "your query" --format markdown
```

**Problem: Results saved but not displayed**
```bash
# Check if results were saved to file
ls -la *.json

# View saved results
cat results.json | jq .plan
```

## Model Recommendations by Hardware

### Limited GPU (2-4GB VRAM)
```bash
latent-reason run "query" --encoder Qwen/Qwen3-0.6B --chains 3
```

### Mid-range GPU (4-8GB VRAM)  
```bash
latent-reason run "query" --encoder Qwen/Qwen3-1.7B --chains 5
```

### High-end GPU (8GB+ VRAM)
```bash
latent-reason run "query" --encoder Qwen/Qwen3-4B --chains 8 --generations 15
```

### CPU Only
```bash
latent-reason run "query" --encoder Qwen/Qwen3-0.6B --chains 2 --generations 5
```

## Getting Help

1. **Check the logs**: Use `--verbose` or `--debug` for detailed output
2. **Try the demo**: Run `python demo.py` to test basic functionality  
3. **Compare methods**: Use `latent-reason compare "query"` to see if evolution helps
4. **Check examples**: Look at `examples/quick_start.py` for working code
5. **Read the README**: Full documentation with examples and configuration options

## Reporting Issues

If you encounter a bug, please include:
- Output of `latent-reason check-gpu`
- The exact command you ran
- The full error message
- Your operating system and Python version

This helps us diagnose and fix issues quickly!
