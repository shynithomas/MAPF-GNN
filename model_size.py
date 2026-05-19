import torch
import numpy as np
import networkx as nx
import math
import os
import time
import matplotlib.pyplot as plt
import glob
#from torchsummary import summary

# Import your model and XCBS solver
from behavioural_cloning_3 import GeneralizedGraphActionGCN

if __name__ == "__main__":
    # Specify your paths here
    MODEL_PATH = 'BC_config1_1CommRadius.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialize Model
    model = GeneralizedGraphActionGCN(emb_dim=16, hidden_dim=32).to(device)
    
    # Load correctly using state_dict extraction
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint) # Fallback if saved as raw weights
    else:
        print(f"Warning: Model file {MODEL_PATH} not found. Running with random weights.")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {total_params}")
    
    #print(summary(model, ()))