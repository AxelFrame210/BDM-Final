import torch
import main

# Force CPU usage
torch.cuda.is_available = lambda: False

# Run the main script with CPU
if __name__ == "__main__":
    main.main() 