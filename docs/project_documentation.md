# Project Documentation: Distributed Training Framework

This document details the setup, architecture, and implementation of the Distributed Training Framework project.

## 1. Project Overview and Theory

The goal of this project is to explore and implement fundamental distributed training techniques for large-scale deep learning models. As models grow in size, they often exceed the memory capacity of a single GPU. Distributed training allows us to overcome this limitation by parallelizing the training process across multiple devices or machines.

This framework focuses on two primary parallelism strategies:

### 1.1. Model Parallelism

**Concept:** Model parallelism involves splitting a single large model across multiple devices. Each device holds a different part (a subset of layers) of the model.

**How it works:**
1.  The model is partitioned vertically (layer-wise).
2.  Each partition is placed on a separate device (e.g., a GPU).
3.  During the forward pass, the input data flows sequentially through the devices. The output (activation) of a layer on one device is passed as input to the next layer on the subsequent device.
4.  The backward pass works in reverse. The gradient is passed from the last device back to the first, and each device computes the gradients for its local portion of the model.

This approach is essential when a model is too large to fit into the memory of a single accelerator.

### 1.2. Pipeline Parallelism

**Concept:** Pipeline parallelism is an enhancement of model parallelism that aims to improve device utilization. In naive model parallelism, only one device is active at a time, leading to significant idle periods.

**How it works:**
1.  The training mini-batch is split into smaller chunks called **micro-batches**.
2.  The model partitions are processed in a pipelined fashion. As soon as one device finishes processing a micro-batch, it passes the result to the next device and immediately starts working on the next micro-batch.
3.  This creates a "wave" of forward and backward passes, keeping multiple devices active simultaneously and reducing the "bubble" of idle time.

### 1.3. Gradient Checkpointing

**Concept:** A memory-saving technique used to trade compute for memory. Instead of storing all intermediate activations for the backward pass, gradient checkpointing recomputes them during the backward pass. This significantly reduces the memory footprint of the model at the cost of some additional computation, making it possible to train larger models or use larger batch sizes.

## 2. Project Setup

The project uses `uv` for fast Python environment and package management.

### 2.1. Requirements

-   **AMD GPUs:** ROCm support is required.
-   **NVIDIA GPUs:** CUDA support is required.
-   `uv` for environment management.

### 2.2. Installation Steps

1.  **Create a virtual environment:**
    ```bash
    uv venv
    ```

2.  **Activate the environment:**
    ```bash
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    The `requirements.txt` file is configured for ROCm by default.
    ```bash
    uv pip install -r requirements.txt
    ```
    For **NVIDIA GPUs**, you might need to modify `requirements.txt` to point to the CUDA-specific PyTorch wheels:
    ```
    --index-url https://download.pytorch.org/whl/cu121
    ```

4.  **AMD GPU Specific Setup:**
    For certain AMD GPUs (like RDNA series), you may need to override the graphics version. Add the appropriate line to your shell profile (`~/.bashrc` or `~/.profile`):
    ```bash
    # For RDNA 2 (e.g., RX 6000 series)
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    
    # For RDNA 3 (e.g., RX 7000 series)
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    ```

## 3. Code Implementation

The core logic is implemented in the `src/` directory.

### 3.1. `src/main.py`: Model Parallelism Example

This script provides a simple implementation of model parallelism.

-   **Initialization:** It uses `torch.distributed.init_process_group` to set up the distributed environment. The `RANK` and `WORLD_SIZE` are automatically provided by `torchrun`.
-   **Model Splitting:** A `torch.nn.Sequential` model is defined. The script then automatically splits the layers of this model as evenly as possible across the available processes (`WORLD_SIZE`).
-   **Forward Pass:**
    -   Rank 0 creates the initial input tensor.
    -   Each rank executes its part of the model.
    -   The output tensor is sent to the next rank using `dist.send()`.
    -   Subsequent ranks receive the input tensor from the previous rank using `dist.recv()`.
-   **Backward Pass:**
    -   The process is reversed. Rank `N-1` (the last rank) calculates the loss and initiates the backward pass.
    -   It sends the gradient of its input back to the previous rank (`N-2`).
    -   Each rank receives the gradient for its output, performs its backward pass, and then sends the gradient of its input to the previous rank.

### 3.2. `src/pipeline.py`: Pipeline Parallelism Example

This script demonstrates a naive implementation of pipeline parallelism.

-   **Micro-Batching:** The script processes a mini-batch by splitting it into several `num_micro_batches`.
-   **Model Partitioning:** The model is manually split into stages (partitions). In the example, it's hardcoded for 4 ranks.
-   **Pipelined Execution:**
    1.  **Forward Pass:** The script loops through all micro-batches. Each rank processes one micro-batch and immediately sends the output to the next rank before waiting for the whole mini-batch to complete. Intermediate activations are stored.
    2.  **Backward Pass:** The backward passes are also pipelined, executed in reverse order of the micro-batches.
-   **Gradient Checkpointing:** It uses `torch.utils.checkpoint.checkpoint_sequential` to wrap the model part. This avoids storing intermediate activations for the inner layers of the model part, recomputing them during the backward pass to save memory.

## 4. Tools and Linters

-   **`uv`**: A fast package installer and resolver for Python. Used for managing the project's virtual environment and dependencies.
-   **`torchrun`**: A tool from the PyTorch team for launching distributed training jobs. It simplifies the process by managing environment variables for each process.
-   **Linters**:
    -   `ruff`: An extremely fast Python linter and code formatter.
    -   `mypy`: A static type checker for Python.

## 5. How to Run

### 5.1. Model Parallelism

To run the model parallelism example with 2 processes on the same machine:

```bash
uv run torchrun --nproc_per_node=2 src/main.py
```

### 5.2. Pipeline Parallelism

To run the pipeline parallelism example with 4 processes on the same machine:

```bash
uv run torchrun --nproc_per_node=4 src/pipeline.py
```

> **Note**: `uv run` executes the command within the project's virtual environment. If you have already activated the virtual environment, you can omit `uv run`. 