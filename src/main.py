import torch
import torch.distributed as dist
import os
import sys
import torch.nn as nn
import torch.optim as optim

def init_process_group():
    """Initializes the distributed process group."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group(
        backend="gloo",
        init_method="file:///tmp/dist-train",
        rank=rank,
        world_size=world_size
    )
    print(f"Initialized process group for rank {rank} of {world_size}")

def cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)
        self.layer3 = nn.Linear(30, 40)
        self.layer4 = nn.Linear(40, 5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def run(rank, world_size):
    """Main function for each process."""
    print(f"Hello from rank {rank} of {world_size}!")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Define the full model
    full_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 30),
        nn.ReLU(),
        nn.Linear(30, 40),
        nn.ReLU(),
        nn.Linear(40, 50),
        nn.ReLU(),
        nn.Linear(50, 60),
        nn.ReLU(),
        nn.Linear(60, 5),
    )

    # Split the model into parts for each process
    num_layers = len(full_model)
    layers_per_rank = (num_layers + world_size - 1) // world_size
    start = rank * layers_per_rank
    end = min(start + layers_per_rank, num_layers)
    
    model_part = nn.Sequential(*list(full_model.children())[start:end]).to(device)
    
    optimizer = optim.SGD(model_part.parameters(), lr=0.01)

    # Dummy training loop
    for i in range(5):
        # === FORWARD PASS ===
        if rank == 0:
            # First rank creates the input
            input_tensor = torch.randn(1, 10, device=device)
            output = model_part(input_tensor)
        else:
            # Receive from previous rank
            in_features = model_part[0].in_features
            input_tensor = torch.empty(1, in_features, device=device)
            dist.recv(tensor=input_tensor, src=rank - 1)
            print(f"Epoch {i}, Rank {rank}: received input from rank {rank-1}")
            input_tensor.requires_grad = True
            output = model_part(input_tensor)

        if rank < world_size - 1:
            dist.send(tensor=output, dst=rank + 1)
            print(f"Epoch {i}, Rank {rank}: sent output to rank {rank+1}")
        
        # === BACKWARD PASS ===
        optimizer.zero_grad()
        if rank == world_size - 1:
            # Last rank computes loss and starts backward pass
            loss = output.sum()
            print(f"Epoch {i}, Rank {rank}: final output sum: {loss.item()}")
            loss.backward()
        else:
            # Receive gradients from the next rank
            grad_shape = output.shape
            grad = torch.empty(grad_shape, device=device)
            dist.recv(tensor=grad, src=rank + 1)
            print(f"Epoch {i}, Rank {rank}: received grad from rank {rank+1}")
            output.backward(grad)

        if rank > 0:
            # Send gradients to the previous rank
            dist.send(tensor=input_tensor.grad, dst=rank - 1)
            print(f"Epoch {i}, Rank {rank}: sent grad to rank {rank-1}")
        
        optimizer.step()
        print(f"Epoch {i}, Rank {rank}: step complete")


def main():
    """Main entry point for the script."""
    init_process_group()
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Redirect stdout to a file for each rank - This can be problematic for debugging.
    # I will comment it out for now to see print statements in the console.
    # sys.stdout = open(f"rank_{rank}.log", "w")
    
    run(rank, world_size)
    
    cleanup()

if __name__ == "__main__":
    main() 