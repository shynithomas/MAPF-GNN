import networkx as nx
import heapq
import math
import copy
import traceback
from mapf_visualiser import GraphThreadVisualizer

class Agent:
    def __init__(self, agent_id, start, goal, length, speed):
        self.id = agent_id
        self.start = start
        self.goal = goal
        self.length = float(length)
        self.speed = float(speed)

    def __repr__(self):
        return f"Agent{self.id}"

class HighLevelNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.constraints = []
        self.paths = {}  # Map: agent_id -> [(Node, Head_In, Tail_Out)]
        self.cost = 0.0
        self.conflict = None
        self.parent_node_id= 0

    def __lt__(self, other):
        return self.cost < other.cost

# --- Interval Helper (Explicit Node & Edge Generation) ---
def get_occupancy_intervals(path, agent_len, speed):
    """
    Parses path of [(Node, Head_In, Tail_Out), ...] into occupancy intervals.
    Generates specific intervals for Vertices and Edges.
    """
    intervals = []
    if not path: return intervals

    for i in range(len(path) - 1):
        curr_node, t_head_curr, t_tail_curr = path[i]
        next_node, t_head_next, t_tail_next = path[i+1]
        
        # --- Case 1: WAIT Action (curr == next) ---
        if curr_node == next_node:
            # The agent is occupying the vertex 'curr' continuously during the wait.
            # We create an interval spanning from the start of this node to the end of the wait state.
            intervals.append({
                'loc': curr_node,
                't_in': t_head_curr,
                't_out': t_tail_next, # Continues until the end of the next tuple (end of wait)
                'type': 'vertex'
            })

        # --- Case 2: MOVE Action (curr -> next) ---
        else:
            # 1. Occupy Source Vertex (curr)
            # Head leaves 'curr' immediately at t_head_curr to enter edge.
            # Tail leaves 'curr' at t_head_curr + (L/S).
            
            intervals.append({
                'loc': curr_node,
                't_in': t_head_curr,
                't_out': t_head_curr + (agent_len/speed),
                'type': 'vertex'
            })

            # 2. Occupy Edge (curr, next)
            # Head enters edge at t_head_curr.
            # Tail leaves edge (arrives at next) at t_tail_next.
            intervals.append({
                'loc': (curr_node, next_node),
                't_in': t_head_curr,
                't_out': t_tail_next,
                'type': 'edge'
            })

    # Add Final Node Occupancy (Goal)
    if path:
        last_node, t_h, t_t = path[-1]
        intervals.append({
            'loc': last_node,
            't_in': t_h,
            't_out': t_t, #float('inf'), # Stays at goal
            'type': 'vertex'
        })
 
    return intervals

# --- Conflict Detection (Earliest Conflict First) ---
def find_earliest_conflict(agents, paths):
    """
    Scans all agent pairs for conflicts on both Nodes and Edges.
    Returns the earliest occurrence.
    """
    agent_intervals = {} #dictionary maintaining agent_id as key and node and edge information traversed by it.
    for a in agents:
        if a.id in paths:
            agent_intervals[a.id] = get_occupancy_intervals(paths[a.id], a.length, a.speed)

    min_conflict = None
    min_time = float('inf')

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a1, a2 = agents[i], agents[j]
            
            ints1 = agent_intervals.get(a1.id, [])
            ints2 = agent_intervals.get(a2.id, [])

            for int1 in ints1:
                for int2 in ints2:
                    is_same_loc = False
                    
                    # Check if Locations match (Edge Tuple)
                    if int1['loc'] == int2['loc'] and int1['type']== 'edge':
                        is_same_loc = True
                        #print(f"conflict location {int1['loc']} edge {int1['type']== 'vertex'}")
                    if is_same_loc:
                        # Check Temporal Overlap
                        # Overlap exists if max(start) < min(end)
                        start_overlap = max(int1['t_in'], int2['t_in'])
                        end_overlap = min(int1['t_out'], int2['t_out'])
                        if start_overlap < end_overlap:
                            # We found a conflict. Check if it's the earliest one.
                            #check if one can overtake the other.
                            if int1['t_in'] <= int2['t_in']:
                                lead_agent, follow_agent = a1, a2
                                lead_agent_interval, follow_agent_interval = int1, int2
                            else:
                                lead_agent, follow_agent = a2, a1
                                lead_agent_interval, follow_agent_interval = int2, int1 

                            if follow_agent.speed > lead_agent.speed:
                                #compute the time for collision
                                t_collision = (follow_agent.speed * follow_agent_interval['t_in'] -
                                              lead_agent.speed * lead_agent_interval['t_in'] - lead_agent.length) / (follow_agent.speed - lead_agent.speed)        

                                if t_collision < lead_agent_interval['t_out']:                                              
                                    if start_overlap < min_time:
                                        
                                        min_time = start_overlap
                                        #print(f"agent 1 {a1.id} in {int1['t_in']} out {int1['t_out']} agent2 {a2.id} in {int2['t_in']} out {int2['t_out']}")
                                
                                        min_conflict = {
                                            'agent1': a1.id, 
                                            'agent2': a2.id,
                                            'loc': int1['loc'], # Can be Node ID or (u, v) tuple
                                            't_min': start_overlap,
                                            't_max': end_overlap,
                                            'agent1_start':int1['t_in'],
                                            'agent1_end': int1['t_out'],
                                            'agent2_start':int2['t_in'],
                                            'agent2_end': int2['t_out']
                                        }

    return min_conflict

# --- Constraint Logic ---
def get_blocking_constraint(curr_loc, next_loc, current_time, travel_time, agent, constraints):
    # Calculate Occupancy for the proposed move
    # Head enters edge at 'current_time'
    edge_in = current_time
    edge_out = current_time + travel_time + (agent.length / agent.speed)
    
    # Vertex arrival
    vertex_in = current_time + travel_time
    vertex_out = vertex_in + (agent.length / agent.speed)

    max_block = -1.0
    blocked = False

    for c in constraints:
        if c['agent_id'] != agent.id: continue
        
        is_conflict = False
        
        # Check Edge Constraint
        if isinstance(c['loc'], tuple) and c['loc'] == (curr_loc, next_loc):
            if edge_in < c['t_max'] and c['t_min'] < edge_out:
                is_conflict = True
                
        # Check Vertex Constraint (Target Node)
        if c['loc'] == next_loc:
             if vertex_in < c['t_max'] and c['t_min'] < vertex_out:
                is_conflict = True
        
        if is_conflict:
            blocked = True
            if c['t_max'] > max_block:
                max_block = c['t_max']

    if blocked: return max_block
    return None
'''
# --- Low Level A* (with Node IDs and Heuristic) ---
def low_level_a_star(graph, agent, constraints):
    if graph.is_directed():
        search_graph = graph.reverse()
    else:
        search_graph = graph
    print(f"low_level_a_star for agent {agent.id} with constraints {constraints}")
    try:
        heuristic_map = nx.shortest_path_length(search_graph, source=agent.goal, target=agent.start, weight='weight')
        shortest_path = nx.shortest_path(search_graph, source = agent.goal, target=agent.start)
        print(f"heuristic_map{heuristic_map}")
        print(f"shortest path {shortest_path}")
    except nx.NetworkXNoPath:
        heuristic_map = {agent.goal: 0.0}
    
    if agent.start not in heuristic_map: 
        return None, float('inf')

    pq = []
    h_start = heuristic_map[agent.start] / agent.speed
    t_start_tail = 0.0 + (agent.length / agent.speed)
    initial_path = [(agent.start, 0.0, t_start_tail)]
    
    # Priority Queue: (f, g_head_time, node, path)
    heapq.heappush(pq, (h_start, 0.0, agent.start, initial_path))
    min_arrival_times = {agent.start: 0.0}

    while pq:
        f, curr_head_t, u, path = heapq.heappop(pq)
        
        if u == agent.goal:
            return path, curr_head_t

        if u in min_arrival_times and curr_head_t > min_arrival_times[u] + 20.0:
            continue
        min_arrival_times[u] = curr_head_t

        for v in list(graph.neighbors(u)):
            dist = graph[u][v].get('weight', 1.0)
            travel_time = dist / agent.speed
            
            block_time = get_blocking_constraint(u, v, curr_head_t, travel_time, agent, constraints)
            
            if block_time is None:
                # Valid Move
                new_head_t = curr_head_t + travel_time
                new_tail_t = new_head_t + (agent.length / agent.speed)
                
                if v in heuristic_map:
                    new_f = new_head_t + (heuristic_map[v] / agent.speed)
                    new_path = path + [(v, new_head_t, new_tail_t)]
                    
                    if v not in min_arrival_times or new_head_t < min_arrival_times[v] + 5.0:
                        heapq.heappush(pq, (new_f, new_head_t, v, new_path))
                        if new_head_t < min_arrival_times.get(v, float('inf')):
                            min_arrival_times[v] = new_head_t
            else:
                # Blocked -> Wait
                wait_until = block_time + 0.01
                if wait_until > curr_head_t:
                    waited_tail_t = wait_until + (agent.length / agent.speed)
                    waited_path = path + [(u, wait_until, waited_tail_t)]
                    new_f = wait_until + (heuristic_map[u] / agent.speed)
                    heapq.heappush(pq, (new_f, wait_until, u, waited_path))

    return None, float('inf')
'''
def evaluate_spatial_path(graph, spatial_path, agent, constraints):
    """
    Helper function for steps 2, 3, and 4.
    Takes a fixed sequence of nodes, calculates the timeline, and injects wait states 
    if it hits a time constraint.
    """
    path_schedule = []
    curr_time = 0.0
    
    # Initial state
    t_tail = curr_time + (agent.length / agent.speed)
    path_schedule.append((spatial_path[0], curr_time, t_tail))

    first_constrained_node = None
    first_constrained_edge = None

    for i in range(len(spatial_path) - 1):
        curr_node = spatial_path[i]
        next_node = spatial_path[i+1]
        
        # Get edge weight (distance)
        dist = graph[curr_node][next_node].get('weight', 1.0)
        travel_time = dist / agent.speed

        waited = False
        
        # Steps 3 & 4: Check constraints and generate wait schedule
        while True:
            # We reuse your existing get_blocking_constraint logic
            block_time = get_blocking_constraint(curr_node, next_node, curr_time, travel_time, agent, constraints)
            if block_time is None:
                break # Safe to move
            
            # Constraint found: Wait at current node until the block clears
            waited = True
            curr_time = block_time + 0.01
            t_tail = curr_time + (agent.length / agent.speed)
            path_schedule.append((curr_node, curr_time, t_tail))

        # Record the first obstacle encountered to help find an alternate path later
        if waited and not first_constrained_node:
            first_constrained_node = next_node
            first_constrained_edge = (curr_node, next_node)

        # Execute move
        curr_time += travel_time
        t_tail = curr_time + (agent.length / agent.speed)
        path_schedule.append((next_node, curr_time, t_tail))

    return path_schedule, curr_time, first_constrained_node, first_constrained_edge

def get_initial_plan(graph, agent):
    #print(f"Generating initial plan for agent {agent.id}")
    
    # Get shortest path from source to target using networkx api
    try:
        # Using weight='weight' ensures diagonal edges are accounted for correctly
        shortest_path = nx.shortest_path(graph, source=agent.start, target=agent.goal, weight='weight')
    except nx.NetworkXNoPath:
        return None, float('inf')

    # Generate schedule
    path_schedule = []
    curr_time = 0.0
    
    # Initial state
    #print(f"shortest path agent{agent.id} path:{shortest_path}")
    t_tail = curr_time + (agent.length / agent.speed)
    path_schedule.append((shortest_path[0], curr_time, t_tail))
    #print(f"path schedule{path_schedule}")
   
    for i in range(1, len(shortest_path) - 1):
        curr_node = shortest_path[i]
        next_node = shortest_path[i+1]
        
        # Get edge weight (distance)
        dist = graph[curr_node][next_node].get('weight', 1.0)
        travel_time = dist / agent.speed

        curr_time += travel_time
        t_tail = curr_time + (agent.length / agent.speed)
        path_schedule.append((curr_node, curr_time, t_tail))
        #print(f"path schedule{path_schedule}")
   
    #Add entry for the goal node also
    dist = graph[shortest_path[-2]][shortest_path[-1]].get('weight', 1.0)
    travel_time = dist / agent.speed
    curr_time += travel_time
        
    t_tail = curr_time + (agent.length / agent.speed)
    path_schedule.append((shortest_path[-1], curr_time, t_tail))
    #print(f"path schedule{path_schedule}")
   
    return path_schedule, curr_time

"""
    Recomputes the schedule for agent1 by injecting a wait time to avoid 
    overtaking agent2's tail at the specified conflict location.
    
    Args:
        agent1: The agent whose plan is being modified.
        agent2: The conflicting agent (provided for reference/logging if needed).
        agent1_plan: List of tuples [(node, t_head, t_tail), ...]
        conflict_loc: The location of the conflict (Node ID or Edge Tuple).
        constraint: Dictionary containing the conflict window {'t_min': float, 't_max': float}.
        agent_constraints: existing constraints on the agent
    """
def generate_plan_with_wait(agent1, agent2, agent1_plan,constraint, agent_constraints):
    #print(f" generate plan with wait a1: {agent1.id} a2:{agent2.id} agent1 plan {agent1_plan} constraint {constraint} agentconstraints {agent_constraints}")
    new_plan = []
    conflict_idx = -1
    conflict_loc = constraint['loc']
    is_edge_conflict = isinstance(constraint['loc'], tuple)

    # 1. Locate the conflict and the node where the agent must wait
    for i in range(len(agent1_plan)):
        node, t_head, t_tail = agent1_plan[i]
        
        if is_edge_conflict:
            # Edge conflict: (u, v)
            if i < len(agent1_plan) - 1:
                next_node = agent1_plan[i+1][0]
                if conflict_loc == (node, next_node):
                    wait_node_idx = i         # Wait at 'u' before entering edge
                    conflict_idx = i + 1      # The target node 'v'
                    break
        else:
            # Vertex conflict: node 'v'
            if node == conflict_loc:
                conflict_idx = i
                wait_node_idx = max(0, i - 1) # Wait at the previous node
                break

    if conflict_idx == -1:
        # Conflict location not found in the current plan
        return agent1_plan, agent1_plan[-1][2]

    # 2. Compute the exact original entry time into the conflict zone
    if is_edge_conflict:
        # Head enters the edge exactly when it arrives at 'u' (assuming no prior waits)
        original_entry_t = agent1_plan[wait_node_idx][1] 
    else:
        # Head enters the vertex exactly when it arrives at 'v'
        original_entry_t = agent1_plan[conflict_idx][1]

    # 3. Compute wait duration
    # To ensure the head of agent1 never overtakes the tail of agent2, 
    # agent1 must enter ONLY AFTER agent2's tail has left (constraint['t_max']).
    safe_entry_t = constraint['t_max'] + 0.01
    wait_duration = safe_entry_t - original_entry_t

    if wait_duration <= 0:
        return agent1_plan, agent1_plan[-1][2] 

    # 4. Build the new schedule
    # Copy all steps up to and including arrival at the wait node
    for i in range(wait_node_idx + 1):
        new_plan.append(agent1_plan[i])

    # 5. Inject the Wait State
    wait_node = agent1_plan[wait_node_idx][0]
    wait_start_t = agent1_plan[wait_node_idx][1]
    wait_end_t = wait_start_t + wait_duration
    
    # Calculate when the tail catches up after the wait
    waited_tail_t = wait_end_t + (agent1.length / agent1.speed)
    
    # Append the explicit wait action
    new_plan.append((wait_node, wait_end_t, waited_tail_t))

    # 6. Shift the remaining plan forward by the wait_duration
    for i in range(wait_node_idx + 1, len(agent1_plan)):
        node, orig_head, orig_tail = agent1_plan[i]
        shifted_head = orig_head + wait_duration
        shifted_tail = orig_tail + wait_duration
        new_plan.append((node, shifted_head, shifted_tail))
    
    is_safe = verify_plan_against_constraints(new_plan, agent1, agent_constraints)
    
    if not is_safe:
        # The shifted plan violates a pre-existing constraint!
        # This wait strategy is invalid.
        return None, None
    #print(f"returning plan {new_plan} {new_plan[-1][2]}")
    return new_plan, new_plan[-1][2]
    
def verify_plan_against_constraints(plan, agent, constraints):
    """
    Checks if a generated plan violates any existing constraints.
    Returns True if the plan is safe, False if it violates a constraint.
    """
    if not plan or not constraints:
        return True

    # Generate the explicit time intervals for every node and edge in the new plan
    intervals = get_occupancy_intervals(plan, agent.length, agent.speed)

    for interval in intervals:
        loc = interval['loc']
        t_in = interval['t_in']
        t_out = interval['t_out']

        # Check this interval against all constraints
        for cons in constraints:
            if cons['loc'] == loc:
                # Check for time overlap
                # Two intervals [a_start, a_end] and [b_start, b_end] overlap if:
                # a_start < b_end AND a_end > b_start
                if (t_in < cons['t_max']) and (t_out > cons['t_min']):
                    return False  # Violation found!

    return True  # Plan is safe

"""
    Computes an alternative path from start to goal that strictly avoids the conflict_loc.
    Generates a new schedule for this path and verifies it against existing constraints.
    
    Args:
        graph: NetworkX graph of the map.
        agent1: The agent object (needs start, goal, length, speed).
        conflict_loc: The node or edge tuple to avoid.
        conflict_constraint: The constraint that caused the conflict (for reference).
        existing_constraints: List of all constraints this agent must obey.
        
    Returns:
        new_plan (list of tuples): The verified alternative schedule, or None if impossible.
"""
def generate_alt_plan(graph, agent1,  conflict_constraint, existing_constraints):

    #print(f" generate aletnate plan a1: {agent1.id} conflict constraint {conflict_constraint} agentconstraints {existing_constraints}")
    
    # 1. Create a temporary graph to modify
    temp_graph = graph.copy()
    conflict_loc = conflict_constraint['loc']
    # 2. Identify if the conflict is an edge or a vertex and sever the graph
    # In a grid graph, nodes are usually tuples like (x,y), and edges are ((x1,y1), (x2,y2))
    is_edge_conflict = isinstance(conflict_loc, tuple) and len(conflict_loc) == 2 and isinstance(conflict_loc[0], tuple)
    
    if is_edge_conflict:
        u, v = conflict_loc
        if temp_graph.has_edge(u, v):
            temp_graph.remove_edge(u, v)
    else:
        # Vertex conflict
        # If the conflict is exactly on the start or goal node, a spatial detour is impossible
        if conflict_loc == agent1.start or conflict_loc == agent1.goal:
            return None, None
            
        if temp_graph.has_node(conflict_loc):
            temp_graph.remove_node(conflict_loc)

    # 3. Find the alternative shortest path avoiding the severed location
    try:
        alt_path = nx.shortest_path(temp_graph, source=agent1.start, target=agent1.goal, weight='weight')
    except nx.NetworkXNoPath:
        # No physical alternative path exists (e.g., it's a dead-end corridor)
        return None, None

    # 4. Generate the continuous movement schedule for this new path
    new_plan = []
    curr_time = 0.0
    
    # Initial state at start node
    t_tail = curr_time + (agent1.length / agent1.speed)
    new_plan.append((alt_path[0], curr_time, t_tail))
    
    for i in range(len(alt_path) - 1):
        u = alt_path[i]
        v = alt_path[i+1]
        
        # Get actual edge distance (accounts for diagonals if weight is set)
        dist = graph[u][v].get('weight', 1.0)
        travel_time = dist / agent1.speed
        
        curr_time += travel_time
        t_tail = curr_time + (agent1.length / agent1.speed)
        new_plan.append((v, curr_time, t_tail))

    # 5. Verify the newly generated schedule against all existing constraints
    # (Relies on the verify_plan_against_constraints function defined previously)
    is_safe = verify_plan_against_constraints(new_plan, agent1, existing_constraints)
    
    if is_safe:
        print(f"Returning plan {new_plan} ")
        return new_plan, curr_time
    else:
        # The alternate path physically exists, but traveling it at full speed 
        # violates another pre-existing time constraint. 
        return None, None

# --- CBS Solver (High Level) ---
def xcbs_solve(graph, agents):
    node_id_counter = 1
    root = HighLevelNode(node_id=node_id_counter)
    node_id_counter += 1
    
    for agent in agents:
        path, cost = get_initial_plan(graph, agent)
        #print(f"initial plan agent{agent.id} path {path}")
        if path is None: return None
        root.paths[agent.id] = path
        root.cost += cost
   
    open_list = []
    heapq.heappush(open_list, root)
    
    while open_list:
        curr_node = heapq.heappop(open_list)
        curr_node.conflict = find_earliest_conflict(agents, curr_node.paths)
    
        if curr_node.conflict is None:
            #print(f"Solution Found at Node {curr_node.node_id} (Cost: {curr_node.cost:.4f})")
            #return curr_node.paths
            #revising the code on 20/02/2026 to return a formatted dictionary
            formatted_solution = {}
            for agent in agents:
                # Transform [(Node, T_in, T_out), ...] -> List of Dicts
                formatted_solution[agent.id] = get_occupancy_intervals(
                    curr_node.paths[agent.id], 
                    agent.length, 
                    agent.speed
                )
            return formatted_solution
        
        conflict = curr_node.conflict
        
        involved = [conflict['agent1'], conflict['agent2']]
        #print(f"Node {curr_node.node_id}: Conflict at {conflict['loc']} (t={conflict['t_min']:.2f}-{conflict['t_max']:.2f}) between {involved}")

        for agent_id in involved:
            child_wait = HighLevelNode(node_id=node_id_counter)
            child_wait.parent_node_id = curr_node.node_id
            node_id_counter += 1
            child_alt = HighLevelNode(node_id=node_id_counter)
            child_alt.parent_node_id = curr_node.node_id
            node_id_counter += 1
            
            child_alt.constraints = copy.deepcopy(curr_node.constraints)
            child_wait.constraints = copy.deepcopy(curr_node.constraints)
            
            if conflict['agent1']== agent_id:
                new_constraint = {
                    'agent_id': agent_id,
                    'loc': conflict['loc'],
                    't_min': conflict['agent2_start'],
                    't_max': conflict['agent2_end']
                }
            else:
                new_constraint = {
                    'agent_id': agent_id,
                    'loc': conflict['loc'],
                    't_min': conflict['agent1_start'],
                    't_max': conflict['agent1_end']
                }    
            child_wait.paths = copy.deepcopy(curr_node.paths)
            child_alt.paths = copy.deepcopy(curr_node.paths)
            
            agent_obj = next(a for a in agents if a.id == agent_id)
            other_agent_id = next(element for element in involved if element != agent_id)
            other_agent = next(a for a in agents if a.id == other_agent_id)
            agent_constraints =[]
            for constraint in child_wait.constraints:
                if constraint.get('agent') == agent_id:
                    agent_constraints.append(constraint)
            wait_plan, _ = generate_plan_with_wait(agent_obj, other_agent,  child_wait.paths[agent_id], new_constraint, agent_constraints)
            alt_plan, _ = generate_alt_plan (graph, agent_obj, new_constraint, agent_constraints)
               
            child_wait.constraints.append(new_constraint)
            child_alt.constraints.append(new_constraint)
            
            if wait_plan:
                child_wait.paths[agent_id] = wait_plan
                child_wait.cost = 0
                for a in agents:
                    child_wait.cost += child_wait.paths[a.id][-1][1]
                #print(f"Parent Node : {child_wait.parent_node_id} Agent {agent_id} Wait plan:  {wait_plan}")

                heapq.heappush(open_list, child_wait)
            if alt_plan:
                child_alt.paths[agent_id] = alt_plan
                child_alt.cost = 0
                for a in agents:
                    child_alt.cost += child_alt.paths[a.id][-1][1]
                #print(f"Parent Node : {child_wait.parent_node_id} Agent {agent_id} Alternate plan {alt_plan}")

                heapq.heappush(open_list, child_alt)

    return None

def build_graph_from_map(filename, connectivity_mode):
    """
    Reads a map file (type octile) and creates a graph.
    STRICT RULE: Only '.' is a valid node. All other characters are obstacles.
    """
    G = nx.Graph()
    grid = []
    
    # --- Step 1: Parse the file ---
    try:
        with open(filename, 'r') as f:
            parsing_map = False
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                
                # The map data usually starts after the line containing "map"
                if stripped == "map":
                    parsing_map = True
                    continue
                
                if parsing_map:
                    grid.append(stripped)
                    
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None

    if not grid:
        return None

    rows = len(grid)
    #print(f"Map Size: {rows} rows")

    # --- Step 2: Define Connectivity Offsets ---
    # We use "forward-looking" offsets to avoid duplicate edges in an undirected graph.
    down = (1, 0)
    right = (0, 1)
    diag_main = (1, 1)    # Down-Right (\)
    diag_anti = (1, -1)   # Down-Left (/)

    offsets = []
    if connectivity_mode == 0:   offsets = [down]
    elif connectivity_mode == 1: offsets = [right]
    elif connectivity_mode == 2: offsets = [down, right]
    elif connectivity_mode == 3: offsets = [down, right, diag_main]
    elif connectivity_mode == 4: offsets = [down, right, diag_anti]
    elif connectivity_mode == 5: offsets = [down, right, diag_main, diag_anti]
    else:
        #print("Invalid connectivity mode. Defaulting to 8-way (Mode 5).")
        offsets = [down, right, diag_main, diag_anti]

    # --- Step 3: Build Nodes and Edges ---
    for r in range(rows):
        # Handle varying line lengths if necessary
        cols = len(grid[r])
        for c in range(cols):
            
            # STRICT CONDITION: Only process if the character is exactly '.'
            if grid[r][c] == '.':
                current_node = (r, c)
                G.add_node(current_node)
                
                # Check neighbors
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    
                    # Boundary check
                    if 0 <= nr < rows and 0 <= nc < len(grid[nr]):
                        
                        # STRICT NEIGHBOR CONDITION: Neighbor must also be '.'
                        if grid[nr][nc] == '.':
                            neighbor = (nr, nc)
                            
                            # Weight: 1.414 for diagonals, 1.0 for straight
                            if dr != 0 and dc != 0:
                                weight = 1.0 #math.sqrt(2)
                            else:
                                weight = 1.0
                            
                            G.add_edge(current_node, neighbor, weight=weight)
                            
    return G

def load_agents_from_scen(filename, start_agent_id, num_agents, default_length=1.0, default_speed=1.0):
    """
    Reads the .scen file and creates Agent objects.
    Format: [bucket] [map] [w] [h] [start_x] [start_y] [goal_x] [goal_y] [dist]
    """
    agents = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
            # Skip version/header line if it exists (usually starts with 'version')
            start_idx = start_agent_id
            #if lines[0].startswith('version'):
            #    start_idx = 1
            
            count = 0
            for line in lines[start_idx:]:
                if count >= num_agents:
                    break
                    
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                
                # Columns (0-indexed based on description):
                # 0: bucket (ignore)
                # 1: map name
                # 2: width
                # 3: height
                # 4: start_x (col)
                # 5: start_y (row)
                # 6: goal_x (col)
                # 7: goal_y (row)
                
                # Conversion: Map logic uses (row, col), Scen uses (x=col, y=row)
                s_x = int(parts[4])
                s_y = int(parts[5])
                g_x = int(parts[6])
                g_y = int(parts[7])
                
                start_node = (s_y, s_x)
                goal_node = (g_y, g_x)
                
                # Create Agent
                agent = Agent(
                    agent_id=count + 1,
                    start=start_node,
                    goal=goal_node,
                    length=default_length,
                    speed=default_speed
                )
                agents.append(agent)
                count += 1
                
    except FileNotFoundError:
        print(f"Error: Scen file '{filename}' not found.")
        return []
        
    return agents

if __name__ == "__main__":
    
    filename = "../maps/empty-32-32.map"
    scen_file = "../scenarios/empty-32-32-random/empty-32-32-random-25.scen"
    
    mode = 2
    graph = build_graph_from_map(filename, mode)
    
    if graph:
    
        try:
            # For automation/demo, setting a fixed number. 
            # In interactive mode, you would use: num = int(input(...))
            num_agents_to_run = 30
            start_agent_id = 0
            print(f"Loading {num_agents_to_run} agents from {scen_file}...")
            
            agents_list = load_agents_from_scen(
                scen_file, 
                start_agent_id,
                num_agents_to_run,
                default_length=1,
                default_speed=1.0
            )
            
            if agents_list:
                #print(f"Loaded {len(agents_list)} agents.")
                #for a in agents_list:
                    #print(f" - {a}: Start {a.start} -> Goal {a.goal}")
               
    
                #print("Starting XCBS Search...")
                solution = xcbs_solve(graph, agents_list)
        
                if solution:
                    #for aid, path in solution.items():
                        #print(f"Agent {aid} Path: {path}")
                    #print("\nStarting Visualization...")

                    # 1. We need the raw map grid (strings) for the background
                    # (Re-reading simply because build_graph didn't return it)
                    raw_grid = []
                    with open(filename, 'r') as f:
                        parsing = False
                        for line in f:
                            if line.strip() == "map": 
                                parsing = True
                                continue
                            if parsing and line.strip(): 
                                raw_grid.append(line.strip())

                    # 2. Configure the visualizer
                    # We pass 'agents_list' so the visualizer knows the specific 
                    # length and speed of each agent for accurate threading.
                    vis = GraphThreadVisualizer(
                        grid_map=raw_grid, 
                        paths=solution, 
                        agents_list=agents_list, 
                        speed_factor=1.0  # Adjust to 0.5 to slow down, 2.0 to speed up
                    )
                    
                    # 3. Run
                    vis.animate()
                else:
                    print("No solution found, skipping visualization.")    
            else:
                print("No agents loaded.")
        except Exception as e:
            print(f"An error occurred: {traceback.print_exc()}")
