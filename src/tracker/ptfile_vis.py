import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def visualize_pt_file(pt_file_path):
    """
    Visualize the content of a PyTorch .pt file
    
    Args:
        pt_file_path (str): Path to the .pt file
    """
    print(f"Loading PyTorch file: {pt_file_path}")
    
    # Load the PyTorch file
    try:
        data = torch.load(pt_file_path, map_location=torch.device('cpu'))
        print(f"Successfully loaded {pt_file_path}")
    except Exception as e:
        print(f"Error loading {pt_file_path}: {e}")
        return
    
    # Display basic information about the loaded data
    print(f"\nData type: {type(data)}")
    
    # Handle different types of PyTorch objects
    if isinstance(data, torch.Tensor):
        visualize_tensor(data, pt_file_path)
    elif isinstance(data, dict):
        visualize_dict(data, pt_file_path)
    elif isinstance(data, list):
        visualize_list(data, pt_file_path)
    else:
        print(f"Unsupported data type for visualization: {type(data)}")

def visualize_tensor(tensor, file_path):
    """Visualize a tensor object"""
    print(f"\nTensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Tensor device: {tensor.device}")
    print(f"Tensor statistics:")
    print(f"  - Min: {tensor.min().item()}")
    print(f"  - Max: {tensor.max().item()}")
    print(f"  - Mean: {tensor.mean().item()}")
    print(f"  - Std: {tensor.std().item()}")
    
    # For 2D tensors, create a heatmap
    if len(tensor.shape) == 2:
        plt.figure(figsize=(10, 8))
        plt.imshow(tensor.numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f"Heatmap of tensor from {os.path.basename(file_path)}")
        plt.savefig(f"{os.path.splitext(file_path)[0]}_heatmap.png")
        print(f"Heatmap saved as {os.path.splitext(file_path)[0]}_heatmap.png")
    
    # For 1D tensors, create a line plot
    elif len(tensor.shape) == 1:
        plt.figure(figsize=(10, 6))
        plt.plot(tensor.numpy())
        plt.title(f"Line plot of tensor from {os.path.basename(file_path)}")
        plt.grid(True)
        plt.savefig(f"{os.path.splitext(file_path)[0]}_lineplot.png")
        print(f"Line plot saved as {os.path.splitext(file_path)[0]}_lineplot.png")
    
    # For higher dimensional tensors, show first few values
    else:
        print("\nFirst few values:")
        print(tensor.flatten()[:20])
        
        # Try to visualize the first 2D slice if tensor has more dimensions
        if len(tensor.shape) > 2:
            first_slice = tensor[0] if len(tensor.shape) == 3 else tensor[0, 0]
            if isinstance(first_slice, torch.Tensor) and len(first_slice.shape) == 2:
                plt.figure(figsize=(10, 8))
                plt.imshow(first_slice.numpy(), cmap='viridis')
                plt.colorbar()
                plt.title(f"First slice from tensor in {os.path.basename(file_path)}")
                plt.savefig(f"{os.path.splitext(file_path)[0]}_firstslice.png")
                print(f"First slice visualization saved as {os.path.splitext(file_path)[0]}_firstslice.png")

def visualize_dict(data_dict, file_path):
    """Visualize a dictionary of tensors or other PyTorch objects"""
    print(f"\nDictionary with {len(data_dict)} keys:")
    for i, (key, value) in enumerate(data_dict.items()):
        print(f"\nKey {i+1}: {key}")
        print(f"  Type: {type(value)}")
        
        if isinstance(value, torch.Tensor):
            print(f"  Shape: {value.shape}")
            if len(value.shape) <= 2:  # Only visualize 1D or 2D tensors automatically
                out_filename = f"{os.path.splitext(file_path)[0]}_{key.replace('.', '_')}"
                
                if len(value.shape) == 1:
                    plt.figure(figsize=(10, 6))
                    plt.plot(value.detach().cpu().numpy())
                    plt.title(f"{key} (1D tensor)")
                    plt.grid(True)
                    plt.savefig(f"{out_filename}_lineplot.png")
                    print(f"  Plot saved as {out_filename}_lineplot.png")
                
                elif len(value.shape) == 2:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(value.detach().cpu().numpy(), cmap='viridis')
                    plt.colorbar()
                    plt.title(f"{key} (2D tensor)")
                    plt.savefig(f"{out_filename}_heatmap.png")
                    print(f"  Heatmap saved as {out_filename}_heatmap.png")

def visualize_list(data_list, file_path):
    """Visualize a list of PyTorch objects"""
    print(f"\nList with {len(data_list)} items:")
    for i, item in enumerate(data_list[:5]):  # Show info for first 5 items
        print(f"\nItem {i+1}:")
        print(f"  Type: {type(item)}")
        
        if isinstance(item, torch.Tensor):
            print(f"  Shape: {item.shape}")
            print(f"  Data type: {item.dtype}")
            if i == 0:  # Visualize only the first tensor in the list
                out_filename = f"{os.path.splitext(file_path)[0]}_item{i}"
                if len(item.shape) == 1:
                    plt.figure(figsize=(10, 6))
                    plt.plot(item.detach().cpu().numpy())
                    plt.title(f"Item {i} (1D tensor)")
                    plt.grid(True)
                    plt.savefig(f"{out_filename}_lineplot.png")
                    print(f"  Plot saved as {out_filename}_lineplot.png")
                elif len(item.shape) == 2:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(item.detach().cpu().numpy(), cmap='viridis')
                    plt.colorbar()
                    plt.title(f"Item {i} (2D tensor)")
                    plt.savefig(f"{out_filename}_heatmap.png")
                    print(f"  Heatmap saved as {out_filename}_heatmap.png")
    
    if len(data_list) > 5:
        print(f"\n... and {len(data_list) - 5} more items")

if __name__ == "__main__":
    # Use the file provided as a command-line argument if available
    if len(sys.argv) > 1:
        pt_file = sys.argv[1]
    else:
        # Default to subject_features.pt in the current directory
        pt_file = "subject_features.pt"
        
    if not os.path.exists(pt_file):
        # Try to find it in the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pt_file = os.path.join(current_dir, "subject_features.pt")
        
    if not os.path.exists(pt_file):
        print(f"Error: Cannot find PyTorch file {pt_file}")
        print("Usage: python ptfile_vis.py [path/to/file.pt]")
        sys.exit(1)
        
    visualize_pt_file(pt_file)
