"""Entry point. Verifies PyTorch + CUDA, numpy, and basic imports."""

import numpy as np
import torch


def main():
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        x = torch.randn(3, 3, device="cuda")
        print(f"GPU tensor:\n{x}")
    else:
        print("Running on CPU")
        x = torch.randn(3, 3)
        print(f"CPU tensor:\n{x}")


if __name__ == "__main__":
    main()
