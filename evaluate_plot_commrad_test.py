import torch
import numpy as np
import networkx as nx
import math
import os
import time
import matplotlib.pyplot as plt
import glob

# Import your model and XCBS solver
from behavioural_cloning_3 import GeneralizedGraphActionGCN
from xcbs_5 import build_graph_from_map, load_agents_from_scen, xcbs_solve

MAX_STEPS = 90  # Timeout threshold to prevent infinite loops For 16x16 =20, 32x32 =70

class MapfInferenceEnv:
    def __init__(self, map_file, scen_file, num_agents, body_len=1, max_degree=5):
        self.max_degree = max_degree
        self.body_len = body_len
        self.G = build_graph_from_map(map_file, connectivity_mode=2)
        nodes = list(self.G.nodes())
        self.height = max(n[0] for n in nodes) + 1
        self.width = max(n[1] for n in nodes) + 1
        
        # Static Features
        self.static_features = {}
        degrees = dict(self.G.degree())
        betw = nx.betweenness_centrality(self.G, k=min(len(nodes), 20))
        max_deg = max(degrees.values()) if degrees else 1
        for n in nodes: 
            self.static_features[n] = [degrees[n]/max_deg, betw.get(n, 0.0)]
            
        self.agents_objs = load_agents_from_scen(scen_file, 0, num_agents)
        if len(self.agents_objs) == 0:
            raise ValueError(f"CRITICAL ERROR: No agents loaded! Check scenario file path: {scen_file}")
        self.num_agents = len(self.agents_objs)
        #for agent in self.agents_objs:
            #print(f"agent src {agent.start} goal {agent.goal}")

    def reset(self):
        self.global_time = 0.0
        self.collisions = 0
        
        # Track the exact float time each agent arrives at its next node
        self.timers = {i: 0.0 for i in range(self.num_agents)}
        self.current_nodes = {i: a.start for i, a in enumerate(self.agents_objs)}
        self.target_nodes = {i: a.start for i, a in enumerate(self.agents_objs)}
        self.finished = {i: False for i in range(self.num_agents)}
        
        # Store paths for SOC and Makespan calculation
        self.paths = {i: [(a.start, 0.0, a.length/a.speed)] for i, a in enumerate(self.agents_objs)}
        
        return self._get_obs()

    def _get_occupied_nodes(self, agent_id):
        """
        Dynamically calculates the trail of nodes currently occupied by the agent's body 
        based on continuous time, speed, and length.
        """
        occupied = []
        
        # self.paths stores: (node_location, time_head_arrived, time_tail_clears)
        for node, t_in, t_out in self.paths[agent_id]:
            # If the current time is less than or equal to the time the tail clears the node,
            # it means a portion of the agent's body is still occupying this node.
            if self.global_time <= t_out:
                
                # Prevent consecutive duplicate nodes if the agent took a "Wait" action
                if not occupied or occupied[-1] != node:
                    occupied.append(node)
                    
        # Failsafe: The agent's head must always be the final node in the sequence.
        # (If global time advanced but the tail just barely cleared, ensure head is present)
        current_head = self.current_nodes[agent_id]
        if not occupied:
            occupied.append(current_head)
        elif occupied[-1] != current_head:
            occupied.append(current_head)
            
        return occupied

    def _get_head_status(self, agent_id):
        # INFERENCE LOGIC: Agent is moving if its timer is active
        is_moving = 1.0 if self.timers[agent_id] > 0.001 else 0.0
        return self.current_nodes[agent_id], is_moving

    def _get_obs(self):
        body_coords, edge_vecs, static_feats = [], [], []
        for i in range(self.num_agents):
            body = self._get_occupied_nodes(i)
            coords = [[r/self.height, c/self.width] for r, c in body]
            #print(f"coords {coords}")
            edges = [[0.0, 0.0]] 
            for k in range(1, len(coords)):
                dx = coords[k][0] - coords[k-1][0]
                dy = coords[k][1] - coords[k-1][1]
                edges.append([dx, dy])

            pad = self.body_len - len(coords)
            if pad > 0:
                coords = ([[0.0, 0.0]] * pad) + coords
                edges = ([[0.0, 0.0]] * pad) + edges
            
            body_coords.append(coords[-self.body_len:])
            edge_vecs.append(edges[-self.body_len:])
            
            head, is_moving = self._get_head_status(i)
            static_feats.append(self.static_features[head] + [is_moving])

        n_feats = torch.zeros(self.num_agents, self.max_degree, 5)
        mask = torch.zeros(self.num_agents, self.max_degree, dtype=torch.bool)
        for i in range(self.num_agents):
            occupied_nodes = self._get_occupied_nodes(i)
            u = occupied_nodes[-1] # The agent's head

            # Prepend 'u' so Action 0  maps to 'Wait'
            candidate_nodes = [u] + sorted(list(self.G.neighbors(u))) 
            #print(f"candidate_nodes {candidate_nodes}")
            target, is_moving = self._get_head_status(i)
            #print(f" target, is_moving {target}, {is_moving}")
            for k, nbr in enumerate(candidate_nodes[:self.max_degree]):
                #print(f"nbr {nbr}")
                if nbr in occupied_nodes and nbr != u:
                    mask[i, k] = False # Invalid move
                else:
                    mask[i, k] = True  # Valid move
                n_feats[i, k, :2] = torch.tensor([(nbr[0]-u[0])/self.height, (nbr[1]-u[1])/self.width])
                #n_feats[i, k, 2] = 1.0 if nbr == self.agents_objs[i].goal else 0.0 
                #dist = (abs(nbr[0] - self.agents_objs[i].goal[0]) + abs(nbr[1] - self.agents_objs[i].goal[1]))/(self.height * self.width)
                try:
                    # Using weight='weight' ensures diagonal edges are accounted for correctly
                    dist = nx.shortest_path_length(self.G, source=nbr, target=self.agents_objs[i].goal, weight='weight')
                except nx.NetworkXNoPath:
                    dist = inf
                n_feats[i, k, 2] = dist#torch.tensor([(nbr[0]-u[0])/self.height, (nbr[1]-u[1])/self.width])
                #print(f"dist of nbr from goal {dist} {abs(nbr[0] - self.agents_objs[i].goal[0])} {abs(nbr[1] - self.agents_objs[i].goal[1])}")
                #print(f"static feats {self.static_features[nbr]}")
                n_feats[i, k, 3] = self.static_features[nbr][0]
                n_feats[i, k, 4] = self.static_features[nbr][1]

        goal_pos = torch.tensor([[self._get_occupied_nodes(i)[-1][0]/self.height, 
                                 self._get_occupied_nodes(i)[-1][1]/self.width, 
                                 self.agents_objs[i].goal[0]/self.height, 
                                 self.agents_objs[i].goal[1]/self.width] for i in range(self.num_agents)], dtype=torch.float)
        
        # Radius-based communication
        COMM_RADIUS = 1.0
        edge_list = []
        for i in range(self.num_agents):
            loc_i = self._get_occupied_nodes(i)[-1]
            for j in range(self.num_agents):
                if i != j:
                    loc_j = self._get_occupied_nodes(j)[-1]
                    dist = abs(loc_i[0] - loc_j[0]) + abs(loc_i[1] - loc_j[1])
                    if dist <= COMM_RADIUS:
                    #if abs(loc_i[0] - loc_j[0]) <= COMM_RADIUS or abs(loc_i[1] - loc_j[1]) <= COMM_RADIUS :
                        edge_list.append([i, j])

        if len(edge_list) > 0:
            edge_idx = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_idx = torch.empty((2, 0), dtype=torch.long)

        return (torch.tensor(body_coords, dtype=torch.float), torch.tensor(edge_vecs, dtype=torch.float), 
                torch.tensor(static_feats, dtype=torch.float), goal_pos, edge_idx, n_feats, mask)

    def step(self, actions, steps):
        #print(f" Applying prediction to the environment on the next step")
        # 1. Apply Actions only to free agents
        for i, act in enumerate(actions):
            if self.finished[i] or self.timers[i] > 0.001:
                continue  # Agent is already moving or done
                
            u = self.current_nodes[i]
            spd = self.agents_objs[i].speed
            #print(f"evaluate.py step agent {i} act {act}")
            if act == 0:
                self.target_nodes[i] = u
                self.timers[i] = 1.0 / spd 
            else:
                neighbors = sorted(list(self.G.neighbors(u)))
                #print(f"evaluate.py step neighbours {neighbors}")
                if act - 1 < len(neighbors):
                    nxt = neighbors[act - 1]
                    dist = self.G[u][nxt].get('weight', 1.0)
                    self.target_nodes[i] = nxt
                    self.timers[i] = dist / spd
                    #print(f"next {nxt}")
                else:
                    self.target_nodes[i] = u # Invalid, force wait
                    self.timers[i] = 1.0 / spd

        # 2. Collision Checking! 
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if self.finished[i] and self.finished[j]: continue
                
                # Vertex Collision
                if self.target_nodes[i] == self.target_nodes[j]:
                    self.collisions += 1
                # Edge Collision (Head-to-head swap)
                elif (self.current_nodes[i] == self.target_nodes[j] and 
                      self.target_nodes[i] == self.current_nodes[j] and 
                      self.current_nodes[i] != self.target_nodes[i]):
                    self.collisions += 1

        # 3. Advance Global Time
        active_timers = [self.timers[i] for i in range(self.num_agents) if not self.finished[i] and self.timers[i] > 0]
        if not active_timers:
            return self._get_obs(), True 
            
        delta_t = min(active_timers)
        self.global_time += delta_t

        # 4. Update Agents state
        for i in range(self.num_agents):
            if self.finished[i]: continue
            
            self.timers[i] -= delta_t
            if self.timers[i] <= 0.001: 
                self.timers[i] = 0.0
                self.current_nodes[i] = self.target_nodes[i]
                
                a_len, a_spd = self.agents_objs[i].length, self.agents_objs[i].speed
                self.paths[i].append((self.current_nodes[i], self.global_time, self.global_time + (a_len/a_spd)))
                
                if self.current_nodes[i] == self.agents_objs[i].goal:
                    self.finished[i] = True
                    #print(f" agent {i} finished step {steps}")

        done = all(self.finished.values())
        #print(f" steps {steps} done {done} all {all(self.finished.values())} finished {self.finished.values()}")
        return self._get_obs(), done


def evaluate_scenario(model_path, map_file, scen_file, num_agents):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialize Model
    model = GeneralizedGraphActionGCN(emb_dim=16, hidden_dim=32).to(device)
    
    # Load correctly using state_dict extraction
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint) # Fallback if saved as raw weights
    else:
        print(f"Warning: Model file {model_path} not found. Running with random weights.")
    
    model.eval()
    
    # 2. Initialize Environment
    env = MapfInferenceEnv(map_file, scen_file, num_agents=num_agents)
    start_marl = time.time()
    # --- Part 1: Run Model ---
    obs = env.reset()
    done = False
    steps = 0
    timeout = 5
    #print("Running Neural Network Inference...")
    while not done and steps < MAX_STEPS:
        v_seq, e_seq, s_feat, pos, edge_idx, n_feats, mask = obs
        #print(f" ==evaluate step {steps}==")
        # Move inputs to device
        v_seq, e_seq, s_feat = v_seq.to(device), e_seq.to(device), s_feat.to(device)
        pos, edge_idx, n_feats, mask = pos.to(device), edge_idx.to(device), n_feats.to(device), mask.to(device)
        #print(f"  goal pos {pos}")
       # print(f" comm agents {edge_idx}")
       # print(f" n_feats{n_feats}")
        #print(f"  mask {mask}")
        with torch.no_grad():
            logits = model(v_seq, e_seq, s_feat, pos, edge_idx, n_feats, mask)
            if torch.isnan(logits).any():
                print("CRITICAL WARNING: The model is outputting NaNs!")
            # Mask out illegal moves securely
            #logits = logits.masked_fill(~mask, float('-inf'))
            actions = logits.argmax(dim=-1).cpu().tolist() # Using tolist() to avoid Numpy 2.0 errors
            #print(f"  predicted action scores {logits}")
            #print(f"step {steps} actions {actions}")
        obs, done = env.step(actions, steps)
        steps += 1
        #print(f"done {done}")
        
    m_makespan = env.global_time if done else float('inf')
    m_soc = sum([env.paths[i][-1][1] for i in range(env.num_agents)]) if done else float('inf')
    m_collisions = env.collisions
    time_marl = time.time()-start_marl
    
    if time_marl > timeout:
        m_success = False
    else:
        m_success = done 
    print(f"time_marl {time_marl} m_succ {m_success} done {done}")    
    
    # --- Part 2: Run Expert (XCBS) ---
    print("Running XCBS Expert for comparison...")
    start_expert = time.time()
    e_succ = False
    expert_solution = xcbs_solve(env.G, env.agents_objs)
    if expert_solution:
        e_makespan = max([path[-1]['t_out'] for path in expert_solution.values()])
        e_soc = sum([path[-1]['t_out'] for path in expert_solution.values()])
        time_expert = time.time()-start_expert
        if time_expert > timeout:
            e_succ = False
        else:
            e_succ = True
        print(f"time_expert {time_expert} e_succ {e_succ}")    
    
    else:
        e_makespan, e_soc, time_expert = 0, 0, -1

    # --- Part 3: Calculate Metrics Table ---
    gap = ((m_makespan - e_makespan) / e_makespan * 100) if e_makespan > 0 and m_makespan != float('inf') else 0

    #print("\n" + "="*60)
    #print(f"{'METRIC':<20} | {'MODEL':<15} | {'EXPERT (XCBS)':<15}")
    #print("-" * 60)
    #print(f"{'Status':<20} | {'Success' if m_success else 'Timeout/Fail':<15} | {'Success' if expert_solution else 'Fail':<15}")
    #print(f"{'Collisions':<20} | {m_collisions:<15} | {0:<15}")
    #print(f"{'Makespan':<20} | {m_makespan:<15.2f} | {e_makespan:<15.2f}")
    #print(f"{'Sum of Costs (SOC)':<20} | {m_soc:<15.2f} | {e_soc:<15.2f}")
    #print(f"{'Optimality Gap':<20} | {gap:<14.2f} | {'0.00%':<15}")
    #print(f"{'Time Taken':<20} | {time_marl:<14.3f} | {time_expert:<14.3f}")
    #print("="*60)
    return m_success, time_marl, e_succ, time_expert, m_collisions

if __name__ == "__main__":
    # Specify your paths here
    MODEL_PATH = 'BC_config1_1CommRadius.pth'
    MAP_FILE = '../maps/maze-16.map'
    SCEN_FILE = '../scenarios/maze-16.scen'
    NUM_AGENTS = 45
    #timeout = 10
    result_text="No of agents | Model Success Rate | Model Avg Time | Model Avg Collisions | Expert Success Rate | Expert Avg Time \n"
    # Dictionaries to store aggregated results
    summary_results = {
        'model_success_rate': [],
        'expert_success_rate': [],
        'model_avg_time': [],
        'expert_avg_time': [],
        'model_collisions':[]
    }
    start_marl = time.time()
    
    for no_agents in range(5,50,5):
        m_successes = 0
        e_successes = 0
        m_times = []
        e_times = []
        m_collisions= []
        num_valid_scenarios = 0

        for scenario_idx in range(1,3,2):
            file = SCEN_FILE#+str(scenario_idx)+".scen"

            if not os.path.exists(file):
                continue # Skip if file doesn't exist
                
            num_valid_scenarios += 1

            print(f"===file {file}===")
            m_succ, m_time, e_succ, e_time, m_collide = evaluate_scenario(MODEL_PATH, MAP_FILE, file, no_agents)
            print(f"time.time {time.time()} start_time {start_marl} diff{time.time()- start_marl}")
           
            if m_succ: m_successes += 1
            if e_succ: e_successes += 1
            
            m_times.append(m_time)
            e_times.append(e_time)
            m_collisions.append(m_collide)
            
            '''
            if (time.time()- start_marl > timeout):
              time_out = True
              break;  
            '''
        if num_valid_scenarios == 0:
            print(f"Warning: No valid scenario files found for {no_agents} agents.")
            continue
        # Calculate averages for  agent count
        m_success_rate = (m_successes / num_valid_scenarios) * 100
        e_success_rate = (e_successes / num_valid_scenarios) * 100
        m_avg_time = sum(m_times) / len(m_times)
        e_avg_time = sum(e_times) / len(e_times)
        m_avg_collisions = sum(m_collisions)/len(m_collisions)

        # Store for plotting
        summary_results['model_success_rate'].append(m_success_rate)
        summary_results['expert_success_rate'].append(e_success_rate)
        summary_results['model_avg_time'].append(m_avg_time)
        summary_results['expert_avg_time'].append(e_avg_time)
        summary_results['model_collisions'].append(m_avg_collisions)

        # Print Tabular Summary Row
        
        print(f"{no_agents:<10} | {f'{m_successes}/{num_valid_scenarios} ({m_success_rate:.1f}%)':<15} | {m_avg_collisions:<15.4f} | "
              f"{f'{e_successes}/{num_valid_scenarios} ({e_success_rate:.1f}%)':<15} | "
              f"{m_avg_time:<15.4f} | {e_avg_time:<15.4f} \n")
        result_text += str(no_agents)+ "|"+ str(m_successes/num_valid_scenarios)+"|"+ str(m_success_rate)+"| "+str(m_avg_collisions)+" | "+str(e_successes/num_valid_scenarios)+"|"+ str(e_success_rate)+"|"+str(m_avg_time)+"|"+str(e_avg_time)+"\n"

        '''
        if time_out:
            break;     
        '''
    # Writing to the file.
   
    with open("Result_1CommRadius_test.txt", "w") as file:
        file.write(result_text)

    print("File written successfully.")
    # ==========================================
    # Plotting the Graphs
    # ==========================================
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))

    # Plot 1: Success Rate
    ax1.plot(range(5,100,5), summary_results['model_success_rate'], marker='o', color='blue', label='Model (GCN)')
    ax1.plot(range(5,100,5), summary_results['expert_success_rate'], marker='s', color='green', label='Expert (XCBS)')
    ax1.set_title('Success Rate vs Number of Agents')
    ax1.set_xlabel('Number of Agents')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Plot 2: Computation Time
    ax2.plot(range(5,100,5), summary_results['model_avg_time'], marker='o', color='blue', label='Model (GCN)')
    ax2.plot(range(5,100,5), summary_results['expert_avg_time'], marker='s', color='green', label='Expert (XCBS)')
    ax2.set_title('Average Computation Time vs Number of Agents')
    ax2.set_xlabel('Number of Agents')
    ax2.set_ylabel('Time (Seconds)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    ax2.set_yscale('log') # Log scale is highly recommended since XCBS scales exponentially

    ax3.plot(range(5, 100, 5), summary_results['model_collisions'], marker='o', color='red', label='Model (GCN)')
    ax3.plot(range(5, 100, 5), [0]*len(range(5, 100, 5)), marker='s', color='green', label='Expert (XCBS)')
    ax3.set_title('Average Collisions vs Number of Agents')
    ax3.set_xlabel('Number of Agents')
    ax3.set_ylabel('Average Number of Collisions')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()

    plt.tight_layout()
    plt.title("Test")
    plt.savefig('evaluation_results_1CommRadius_test.png')
    print("\nEvaluation complete. Plots saved as 'evaluation_results_1CommRadius_test.png'.")
    plt.show()    
