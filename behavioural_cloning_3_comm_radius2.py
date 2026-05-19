import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from torch_geometric.nn import GCNConv

from posn_encoder_generalised import VariableLengthPosnEncoder
from real_env_general import RealTrafficEnvGeneral

#Updated to propagate the loss only after completion of one full plan
# Changed to alternated netween 8-8 and 16-16
# ==========================================
# 1. Generalized GCN Model
# ==========================================
class GeneralizedGraphActionGCN(nn.Module):
    def __init__(self, emb_dim=16, hidden_dim=32):
        super().__init__()

        # 1. Body Encoder (Processes Head-to-Tail spatial sequence)
        self.position_encoder = VariableLengthPosnEncoder(input_dim=2, emb_dim=emb_dim, hidden_dim=hidden_dim)

        # 2. Static Feature Encoder (Map Topology + Head Status)
        # Input: [Norm_Degree, Centrality, Is_Moving_Head]
        self.static_encoder = nn.Sequential(
                                nn.Linear(3, emb_dim),
                                nn.ReLU(),
                                nn.Linear(emb_dim, emb_dim))

        # 3. Goal Encoder
        self.goal_encoder = nn.Linear(2, emb_dim)

        # 4. Fusion
        self.fusion = nn.Sequential(
                      nn.Linear(hidden_dim + emb_dim + emb_dim, hidden_dim),
                      nn.ReLU(),
                      nn.Linear(hidden_dim, hidden_dim))

        # 5. Social GCN (Interaction between agents)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # 6. Action Head
        # Neighbor Feats: 5 dims (dx, dy, is_goal, static_deg, is_target)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim + 5, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1)
        )

    def forward(self, v_seq, e_seq, s_feat, pos, edge_index, n_feats, mask):
        #print(f"   forward called of GeneralisedGraphAction")
        batch_size = v_seq.size(0)
        #print(f"batch size {batch_size}")
        # In spatial logic, a body of N nodes has N-1 edge segments
        body_segments = torch.full((batch_size,), v_seq.size(1) - 1, dtype=torch.long).to(v_seq.device)
        
        # Encode physical body structure
        pos_emb = self.position_encoder(v_seq, e_seq, body_segments)
        
        # Encode static/goal features
        static_emb = F.relu(self.static_encoder(s_feat))
        #print(f"static embedding size {static_emb}")
        goal_emb = F.relu(self.goal_encoder(pos[:, 2:])) 
        
        # Fuse and process through GCN
        agent_state = F.relu(self.fusion(torch.cat([pos_emb, static_emb, goal_emb], dim=1)))
        
        if edge_index.size(1) > 0:
            agent_state = F.relu(self.conv1(agent_state, edge_index))
            agent_state = F.relu(self.conv2(agent_state, edge_index))
        
        # Map agent state to available actions (neighbors)
        agent_state_expanded = agent_state.unsqueeze(1).expand(-1, n_feats.size(1), -1)
        combined = torch.cat([agent_state_expanded, n_feats], dim=2)
        
        logits = self.action_head(combined).squeeze(-1)
        logits = logits.masked_fill(~mask, float('-inf'))
        #print(f"   logits {logits}") #softmax {F.softmax(logits, dim=-1)}")
        return logits
        #return F.softmax(logits, dim=-1)

# ==========================================
# 2. Training Wrapper (PPO Logic / IL)
# ==========================================
class BehaviourCloning_Agent:
    def __init__(self, lr=1e-3):
        self.policy = GeneralizedGraphActionGCN()
        #print(f"policy params {self.policy.parameters()}")
        #self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=0.001, weight_decay=1e-4)
        
        # --- NEW: Add this scheduler ---
        # If the loss doesn't drop for 5 epochs (patience=5), cut the LR in half (factor=0.5)
        #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)

    def calculate_loss(self, v_seq, e_seq, s_feat, pos, edge_idx, n_feats, mask, target_action):
        #print(f"calculate_loss called")
        #print(f"n_feat {n_feats}")
        #print(f"target action {target_action}")
        probs = self.policy(v_seq, e_seq, s_feat, pos, edge_idx, n_feats, mask)
        #print(f"probs {probs}")
        loss = self.loss_fn(probs, target_action)
        #if(math.isinf(loss)):
            #print("=====INFINITY===")

        return loss

    
# ==========================================
# 3. Main Training Execution
# ==========================================
if __name__ == "__main__":

    # Define Multiple Training Tasks
    TRAINING_TASKS = [
        {"map": "../maps/empty-8-8.map", "scen": "../scenarios/empty-8-8-even/empty-8-8-even-"},
        {"map": "../maps/empty-16-16.map", "scen": "../scenarios/empty-16-16-even/empty-16-16-even-"},
        {"map": "../maps/empty-32-32.map", "scen": "../scenarios/empty-32-32-even/empty-32-32-even-"},
        {"map": "../maps/empty-48-48.map", "scen": "../scenarios/empty-48-48-even/empty-48-48-even-"},
    ]

    EPOCHS_PER_TASK = 20
    env = RealTrafficEnvGeneral(body_len=1, comm_radius=2)
    agent = BehaviourCloning_Agent(lr=0.005)
   #print(f"agent {agent.policy.parameters}")
   
    global_losses = []
    lr_history = []

    for scenario_idx in range(1,12):
        for agent_idx in range(2,20):
            if scenario_idx == 12:
                continue
            for task_idx, task in enumerate(TRAINING_TASKS):
    
                scenario_file = task['scen']+str(scenario_idx)+".scen"
                print(f"\n>>> Switching to Task {task_idx+1}: {task['map']}, scenario {scenario_file} with {agent_idx} agents")
                
                for start_agent_idx in range(2, 10-agent_idx):
                    success = env.load_config(task['map'], scenario_file, start_agent_idx, agent_idx)
                    if not success:
                        print(f"Skipping task {task_idx+1}, solver failed.")
                        continue

                    for epoch in range(EPOCHS_PER_TASK):
                        obs = env.reset()
                        #print(f"========epoch {epoch}=====")
                        #print(f"obs after env reset {obs}")
                        done = False
                        total_loss, steps = 0, 0
                        agent.optimizer.zero_grad() # 1. Zero grads at the START of the episode

                        while not done:
                            v_seq, e_seq, s_feat, pos, edge_idx, n_feats, mask = obs
                            _, expert_actions, done, _ = env.step()
                            #print(f"env.step()")
                            loss = agent.calculate_loss(v_seq, e_seq, s_feat, pos, edge_idx, n_feats, mask, expert_actions)
                            #print(f"1. loss {loss} ")
                            loss.backward() # 2. Accumulate gradients (do not zero them yet)
                            #print(f"2. loss {loss} ")
                            
                            total_loss += loss
                            #print(f"3. total loss {total_loss}")
                            steps += 1
                            obs = env._get_obs()

                        #print(f"4. total loss {total_loss} steps {steps}")    
                        avg_loss = total_loss / steps if steps > 0 else 0
                        global_losses.append(avg_loss)
                        current_lr = agent.optimizer.param_groups[0]['lr']
                        lr_history.append(current_lr)
                        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_norm=0.5)
                        
                        agent.optimizer.step()
                        #print(f"calling scheduler step")

                        agent.scheduler.step()

                        #if epoch % 10 == 0:
                            #print(f"Task {task_idx+1} | Epoch {epoch:02d} | Avg Loss: {avg_loss:.6f}")

    
    # ==========================================
    # 4. Plotting & Saving
    # ==========================================
    # Plot Loss Curve
    #plt.figure(figsize=(10, 5))
    #plt.plot(range(EPOCHS_PER_TASK), epoch_losses, label='Training Loss (BC)')
    #plt.xlabel('Epochs')
    #plt.ylabel('Cross Entropy Loss')
    #plt.title('Imitation Learning Training Progress')
    #plt.legend()
    #plt.grid(True)
    #plt.savefig('training_loss_plot.png')
    #print("Loss plot saved as training_loss_plot.png")
    #plt.show()

    # Save trained model
    MODEL_SAVE_PATH="BC_config1_2CommRadius.pth"
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    torch.save({
        'epoch': EPOCHS_PER_TASK,
        'model_state_dict': agent.policy.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'loss': global_losses[-1],
    }, MODEL_SAVE_PATH)
    
    print("Training Complete.")

    import matplotlib.pyplot as plt

    # Plot Learning Rate Curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(lr_history)), lr_history, label='Learning Rate', color='orange', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule during Training')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('LR_Config1_CommRadius2.png')
    print("Learning rate plot saved as 'LR_Config1_CommRadius2.png'")
    
    # Show the plot (optional, will pop up a window)
    plt.show()

    # ==========================================
    # 4. Plotting & Saving
    # ==========================================
    
    # Plot Loss Curve
    plt.figure(figsize=(10, 5))
    # We use global_losses, which you are already populating in your training loop!
    plt.plot(range(len(global_losses)), global_losses, label='Training Loss', color='blue', linewidth=2)
    
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Progress')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot to a file
    plt.savefig('TL_Config1_CommRadius2.png')
    print("Loss plot saved as 'TL_Config1_CommRadius2.png'")
    
    # Show the plot on screen
    plt.show()
  