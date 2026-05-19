import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np

class GraphThreadVisualizer:
    def __init__(self, grid_map, paths, agents_list=None, speed_factor=1.0):
        self.grid_map = grid_map
        self.paths = paths
        print(f"self.paths {self.paths}")
        self.height = len(grid_map)
        self.width = len(grid_map[0])
        
        # 1. Configuration (Length/Speed)
        self.agents_config = {}
        if agents_list:
            for a in agents_list:
                self.agents_config[a.id] = {'len': a.length, 'spd': a.speed}
        else:
            for aid in paths.keys():
                self.agents_config[aid] = {'len': 1.0, 'spd': 1.0}

        # 2. Build NetworkX Graph from Grid
        self.G = nx.Graph()
        self.pos = {} # Mapping node -> (x, y) for plotting
        
        for r in range(self.height):
            for c in range(self.width):
                if grid_map[r][c] == '.':
                    node = (r, c)
                    self.G.add_node(node)
                    # Note: We plot x=col, y=row. 
                    # We will invert Y axis later to match matrix coordinates
                    self.pos[node] = (c, r) 
                    
                    # Add edges (Look Up and Left to avoid duplicates)
                    # Check Up
                    if r > 0 and grid_map[r-1][c] == '.':
                        self.G.add_edge(node, (r-1, c))
                    # Check Left
                    if c > 0 and grid_map[r][c-1] == '.':
                        self.G.add_edge(node, (r, c-1))

        # 3. Timing
        self.max_time = 0
        for aid, path in paths.items():
            if path:
                self.max_time = max(self.max_time, path[-1]['t_out'])
                
        self.dt = 0.05 * speed_factor
        self.total_frames = int(self.max_time / self.dt) + 20
        self.colors = plt.cm.get_cmap('tab10', len(paths))

    def _get_exact_position(self, agent_id, t):
        """Calculates (row, col) at exact time t."""
        path = self.paths[agent_id]
        if t <= path[0]['t_out']: return path[0]['loc']
        if t >= path[-1]['t_out']: return path[-1]['loc']
        
        for i in range(0,len(path) - 2,2):
            #u, t_arr_u, t_dep_u, type = path[i]
            #v, t_arr_v, t_dep_v, type = path[i+2]
            u = path[i]['loc']
            t_arr_u = float(path[i]['t_in'])
            t_dep_u = float(path[i]['t_out'])
            
            v = path[i+2]['loc']
            t_arr_v = float(path[i+2]['t_in'])
            t_dep_v = float(path[i+2]['t_out'])
            
            if t_arr_u <= t <= t_dep_u: return u # Waiting
            
            if t_dep_u < t < t_arr_v: # Moving
                ratio = (t - t_dep_u) / (t_arr_v) - (t_dep_u)
                r = u[0] + (v[0] - u[0]) * ratio
                c = u[1] + (v[1] - u[1]) * ratio
                return (r, c)
        return path[-1]['loc']

    def _get_body_trace(self, agent_id, current_time):
        """Returns X and Y coordinates for the thread trace."""
        cfg = self.agents_config.get(agent_id, {'len': 0.5, 'spd': 1.0})
        t_head = current_time
        t_tail = t_head - (cfg['len'] / cfg['spd'])
        
        path = self.paths[agent_id]
        start_time = path[0]['t_in']
        
        if t_head < start_time: return [], []
        if t_tail < start_time: t_tail = start_time
        
        head_pos = self._get_exact_position(agent_id, t_head)
        tail_pos = self._get_exact_position(agent_id, t_tail)
        
        corners_r = [head_pos[0]]
        corners_c = [head_pos[1]]
        
        # Identify intermediate nodes
        for i in range(len(path)-1, -1, -2):
            node = path[i]['loc']
            t_arr = float(path[i]['t_in'])
            t_dep = float(path[i]['t_out'])
            type = path[i]['type']

            if t_tail < t_arr < t_head:
                corners_r.append(node[0])
                corners_c.append(node[1])
        
        corners_r.append(tail_pos[0])
        corners_c.append(tail_pos[1])
        
        return corners_c, corners_r # Return x, y

    def animate(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # --- 1. Draw Static Graph Structure ---
        # Draw Nodes
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax, 
                               node_size=30, node_color='black')
        # Draw Edges
        nx.draw_networkx_edges(self.G, self.pos, ax=ax, 
                               edge_color='grey', width=1, style='dashed')
        
        # Formatting to match Matrix orientation
        ax.invert_yaxis() # Puts row 0 at the top
        ax.set_aspect('equal')
        
        # --- 2. Initialize Dynamic Agents ---
        threads = []
        heads = []
        texts = []
        
        for i, aid in enumerate(self.paths.keys()):
            color = self.colors(i)
            # Thread
            line, = ax.plot([], [], '-', linewidth=5, color=color, alpha=0.7, solid_capstyle='round')
            threads.append(line)
            # Head
            head, = ax.plot([], [], 'o', markersize=8, color=color, markeredgecolor='white')
            heads.append(head)
            # Label
            txt = ax.text(0, 0, str(aid), color='black', fontsize=9, fontweight='bold', zorder=10)
            texts.append(txt)

        time_label = ax.text(0.05, 0.95, '', transform=ax.transAxes, 
                            bbox=dict(boxstyle="round", fc="white"))

        def init():
            for line in threads: line.set_data([], [])
            for head in heads: head.set_data([], [])
            for txt in texts: txt.set_position((-10, -10))
            return threads + heads + texts

        def update(frame):
            t = frame * self.dt
            time_label.set_text(f"Time: {t:.2f}")
            
            for i, aid in enumerate(self.paths.keys()):
                xs, ys = self._get_body_trace(aid, t)
                if xs:
                    threads[i].set_data(xs, ys)
                    heads[i].set_data([xs[0]], [ys[0]])
                    texts[i].set_position((xs[0], ys[0]))
            
            return threads + heads + texts + [time_label]

        ani = animation.FuncAnimation(fig, update, frames=self.total_frames, 
                                      init_func=init, blit=True, interval=30)
        
        plt.title("MAPF Graph Execution")
        plt.axis('off') # Hide axes for cleaner graph look
        plt.show()

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # 1. Define Map
    # '.' = Node, '@' = No Node
    grid = [
        "....@.....",
        ".@@.@.@@@.",
        ".......@..",
        ".@@@@@.@@.",
        ".........."
    ]
    
    # 2. Define Dummy Paths
    # Agent 1: Top-Left to Bottom-Right
    p1 = [((0,0),0,1), ((0,1),1,2), ((0,2),2,3), ((1,2),3,4), ((2,2),4,5), 
          ((2,3),5,6), ((2,4),6,7), ((3,4),7,8), ((4,4),8,9)]
          
    # Agent 2: Bottom-Left to Top-Right
    p2 = [((4,0),0,1), ((4,1),1,2), ((4,2),2,3), ((3,2),3,4), ((2,2),5,6), # Wait at (2,2)
          ((2,1),6,7), ((1,1),7,8), ((0,1),8,9)]

    p3 = [{'loc': (0, 0), 't_in': 0.0, 't_out': 1.0, 'type': 'vertex'}, 
          {'loc': ((0, 0), (0, 1)), 't_in': 0.0, 't_out': 2.0, 'type': 'edge'}, 
          {'loc': (0, 1), 't_in': 1.0, 't_out': 2.0, 'type': 'vertex'},
          {'loc': ((0, 1), (0, 2)), 't_in': 1.0, 't_out': 3.0, 'type': 'edge'},
          {'loc': (0, 2), 't_in': 2.0, 't_out': 3.0, 'type': 'vertex'}]
    p4 = [{'loc': (0, 2), 't_in': 2.0, 't_out': 3.0, 'type': 'vertex'},
          {'loc': ((0, 2), (0, 3)), 't_in': 2.0, 't_out': 4.0, 'type': 'edge'}, 
          {'loc': (0, 3), 't_in': 3.0, 't_out': 4.0, 'type': 'vertex'}, 
          {'loc': ((0, 3), (0, 4)), 't_in': 3.0, 't_out': 5.0, 'type': 'edge'},
          {'loc': (0, 4), 't_in': 4.0, 't_out': 5.0, 'type': 'vertex'}, 
          {'loc': ((0, 4), (0, 5)), 't_in': 4.0, 't_out': 6.0, 'type': 'edge'},
          {'loc': (0, 5), 't_in': 5.0, 't_out': 6.0, 'type': 'vertex'}]
    paths = {1: p3, 2: p4}

    # 3. Visualize
    
    vis = GraphThreadVisualizer(grid, paths)
    #print(f"paths t_out {paths[1][-1]['t_out']}")
    vis.animate()