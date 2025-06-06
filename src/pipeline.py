import torch
import torch.distributed as dist
import os
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint_sequential

def init_process_group():
    """Initializes the distributed process group."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",
        rank=rank,
        world_size=world_size
    )
    print(f"Initialized process group for rank {rank} of {world_size}")

def cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def run(rank, world_size):
    """Main function for each process."""
    print(f"Hello from pipeline parallelism rank {rank} of {world_size}!")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Pipeline parameters
    micro_batch_size = 4
    num_micro_batches = 8
    
    # Define the full model
    full_model = nn.Sequential(
        nn.Linear(10, 20), nn.ReLU(),
        nn.Linear(20, 30), nn.ReLU(),
        nn.Linear(30, 40), nn.ReLU(),
        nn.Linear(40, 50), nn.ReLU(),
        nn.Linear(50, 60), nn.ReLU(),
        nn.Linear(60, 70), nn.ReLU(),
        nn.Linear(70, 80), nn.ReLU(),
        nn.Linear(80, 5),
    )

    # Manually define model parts for 4 ranks
    model_parts = [
        nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 30), nn.ReLU()),
        nn.Sequential(nn.Linear(30, 40), nn.ReLU(), nn.Linear(40, 50), nn.ReLU()),
        nn.Sequential(nn.Linear(50, 60), nn.ReLU(), nn.Linear(60, 70), nn.ReLU()),
        nn.Sequential(nn.Linear(70, 80), nn.ReLU(), nn.Linear(80, 5))
    ]

    if rank < len(model_parts):
        model_part = model_parts[rank].to(device)
    else:
        # Handle cases where world_size is not 4
        model_part = nn.Sequential().to(device)
    
    optimizer = optim.SGD(model_part.parameters(), lr=0.01)

    # Training loop
    for epoch in range(3):
        # Store activations and outputs for backward pass
        activations = []
        outputs = []
        
        # === FORWARD PASS ===
        for i in range(num_micro_batches):
            if rank == 0:
                input_tensor = torch.randn(micro_batch_size, 10, device=device)
            else:
                in_features = model_part[0].in_features
                input_tensor = torch.empty(micro_batch_size, in_features, device=device)
                dist.recv(input_tensor, src=rank - 1)
                input_tensor.requires_grad = True
            
            # Use gradient checkpointing for the forward pass
            output = checkpoint_sequential(model_part, 2, input_tensor, use_reentrant=False)
            
            if rank < world_size - 1:
                dist.send(output, dst=rank + 1)
            
            activations.append(input_tensor)
            outputs.append(output)

        # === BACKWARD PASS ===
        optimizer.zero_grad()
        for i in range(num_micro_batches - 1, -1, -1):
            if rank == world_size - 1:
                output = outputs[i]
                act = activations[i]
                loss = output.sum()
                loss.backward()
                if world_size > 1:
                    dist.send(act.grad, dst=rank-1)
            else:
                output = outputs[i]
                act = activations[i]
                
                out_features = model_part[-1].out_features
                grad_output = torch.empty(micro_batch_size, out_features, device=device)
                dist.recv(grad_output, src=rank + 1)
                
                output.backward(grad_output)

                if rank > 0:
                    dist.send(act.grad, dst=rank - 1)

        optimizer.step()
        print(f"Epoch {epoch}, Rank {rank}: step complete")

def main():
    """Main entry point for the script."""
    init_process_group()
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    run(rank, world_size)
    
    cleanup()

if __name__ == "__main__":
    main() 