import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence

class VariableLengthPosnEncoder(nn.Module):
    def __init__(self, input_dim=2, emb_dim=16, hidden_dim=32):
        """
        input_dim: 2 for (x, y) coordinates.
        """
        super(VariableLengthPosnEncoder, self).__init__()
        
        # Coordinate Encoders
        self.node_encoder = nn.Linear(input_dim, emb_dim)
        self.edge_encoder = nn.Linear(input_dim, emb_dim) 
        
        # LSTM
        self.lstm = nn.LSTM(input_size=emb_dim, 
                            hidden_size=hidden_dim,
                            batch_first=True)

    def forward(self, node_coords, edge_vecs, edge_lengths):
        """
        node_coords: [Batch, Max_Seq, 2] -> Normalized (x, y)
        edge_vecs:   [Batch, Max_Seq, 2] -> Normalized (dx, dy)
        edge_lengths: [Batch] -> Number of edges per path
        """
        batch_size = node_coords.size(0)
        device = node_coords.device 
        #print(f"posn_encoder_generalised: batch_size {batch_size}")

        # Encode Features
        n_vec = F.relu(self.node_encoder(node_coords)) 
        e_vec = F.relu(self.edge_encoder(edge_vecs))
        
        # 
        # Calculate true lengths (Nodes + Edges)
        # Each edge adds 1 edge token + 1 node token. 
        # Total Length = (Num_Edges * 2) + 1 (for start node)
        total_lengths = (edge_lengths * 2) + 1
        total_lengths = total_lengths.view(-1).cpu() # Force 1D Vector

        # Determine required tensor size based on actual data
        required_len = int(total_lengths.max().item())
        
        # Create Interleaved Tensor
        # Shape: [Batch, required_len, emb_dim]
        combined_input = torch.zeros(batch_size, required_len, n_vec.size(2), device=device)

        # Fill Nodes (Even indices: 0, 2, 4...)
        seq_len_n = min(required_len, n_vec.size(1) * 2)
        node_indices = torch.arange(0, seq_len_n, 2, device=device)
        # Slice input to match
        n_input_slice = n_vec[:, :len(node_indices), :]
        combined_input[:, node_indices, :] = n_input_slice

        # Fill Edges (Odd indices: 1, 3, 5...)
        seq_len_e = min(required_len, e_vec.size(1) * 2 + 1)
        edge_indices = torch.arange(1, seq_len_e, 2, device=device)
        if len(edge_indices) > 0:
            e_input_slice = e_vec[:, :len(edge_indices), :]
            combined_input[:, edge_indices, :] = e_input_slice

        # 4. Pack
        packed_input = pack_padded_sequence(
            combined_input, 
            total_lengths, 
            batch_first=True, 
            enforce_sorted=False
        )

        # 5. LSTM
        _, (hidden_last, _) = self.lstm(packed_input)
        
        # Return the final hidden state of the path
        return hidden_last[-1]