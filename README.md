# Distributed Training Framework for Large Models

This project is a toy implementation of a distributed training framework for large models, focusing on model parallelism and pipeline parallelism. 

## Setup

It is recommended to use a virtual environment. This project uses `uv` for dependency management.

**Requirements:**
- For AMD GPUs: ROCm support is required. See [AMD ROCm installation guide](https://rocm.docs.amd.com/)
- For NVIDIA GPUs: CUDA support is required. See [PyTorch website](https://pytorch.org/) for CUDA installation instructions.

1.  **Create a virtual environment:**
    ```bash
    uv venv
    ```

2.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    uv sync
    ```
    
    The `pyproject.toml` file is configured for ROCm support by default. The project uses PyTorch nightly builds with ROCm 6.2 support.
    
    **For NVIDIA GPUs:** You may need to modify the `extra-index-url` in `pyproject.toml` to use the CUDA index URL:
    ```toml
    [tool.uv]
    extra-index-url = ["https://download.pytorch.org/whl/nightly/cu121"]
    prerelease = "allow"
    ```

4.  **AMD GPU Setup (if applicable):**
    If you have an AMD GPU, you may need to set environment variables. Add these to your `~/.profile` or `~/.bashrc`:
    ```bash
    # For RDNA and RDNA 2 cards (RX 6000 series)
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    
    # For RDNA 3 cards (RX 7000 series)  
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    
    # If you have both integrated and dedicated GPU
    export HIP_VISIBLE_DEVICES=0
    ```
    
    To find your GPU architecture:
    ```bash
    rocminfo | grep gfx
    ```

## Running the Model Parallelism Example

The `src/main.py` script demonstrates a simple model parallelism setup where a model is split across two processes.

To run the example with 2 processes on the same machine:

```bash
uv run torchrun --nproc_per_node=2 src/main.py
```

This will launch two processes. The environment variables `RANK` and `WORLD_SIZE` will be automatically set by `torchrun`. You should see output from both ranks in your terminal. 

> **Note**: `uv run` executes the command within the project's virtual environment. If you have already activated the virtual environment (using `source .venv/bin/activate`), you can omit `uv run` and just use `torchrun`.

## Running the Pipeline Parallelism Example

The `src/pipeline.py` script demonstrates a naive pipeline parallelism setup. It splits a mini-batch into micro-batches and processes them in a pipelined fashion (forward passes, then backward passes). This example also uses gradient checkpointing to reduce memory usage.

To run the example with 4 processes on the same machine:

```bash
uv run torchrun --nproc_per_node=4 src/pipeline.py
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:
- `docs/project_documentation.md`: Overview of the project setup and implementation
- `docs/theoretical_deep_dive.md`: In-depth theoretical background and code analysis 