import networkx as nx

def prune_to_average_degree(graphml_path, target_avg_degree=64, weight_attr='weight'):
    """
    Reads a GraphML file and removes the weakest links until the 
    average degree drops to the target value.
    
    Args:
        graphml_path (str): Path to the .graphml file.
        target_avg_degree (float): The desired average degree (default 64).
        weight_attr (str): The key in the GraphML used for edge weights.
        
    Returns:
        nx.Graph: The pruned graph object.
    """
    # 1. Read the graph
    # Force undirected if your logic implies simple connectivity. 
    # If your graph is directed, remove force_multigraph/multigraph checks as needed.
    G = nx.read_graphml(graphml_path)
    
    # 2. Basic Stats
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    if num_nodes == 0:
        print("Graph is empty.")
        return G

    # 3. Determine target number of edges
    # Formula for Avg Degree (k) in Undirected Graph: k = 2E / N
    # Therefore, Target Edges (E) = (k * N) / 2
    if G.is_directed():
        # For directed, usually Avg Degree = E / N
        target_num_edges = int(target_avg_degree * num_nodes)
    else:
        target_num_edges = int((target_avg_degree * num_nodes) / 2)
        
    edges_to_remove_count = num_edges - target_num_edges

    # 4. Check if pruning is actually needed
    if edges_to_remove_count <= 0:
        current_degree = (2 * num_edges / num_nodes) if not G.is_directed() else (num_edges / num_nodes)
        print(f"No pruning needed. Current average degree ({current_degree:.2f}) is already <= {target_avg_degree}.")
        return G

    print(f"Pruning {edges_to_remove_count} edges to reach average degree of {target_avg_degree}...")

    # 5. Get all edges with data, sort by weight (ascending), and slice
    # This is O(E log E), much faster than re-calculating degrees inside a loop
    all_edges = sorted(G.edges(data=True), key=lambda x: x[2].get(weight_attr, 0))
    
    # Identify the weakest edges
    weakest_links = all_edges[:edges_to_remove_count]
    
    # 6. Remove the edges
    # We extract just the (u, v) tuples for removal
    G.remove_edges_from([(u, v) for u, v, d in weakest_links])
    
    # Verification
    final_edges = G.number_of_edges()
    final_avg = (2 * final_edges / num_nodes) if not G.is_directed() else (final_edges / num_nodes)
    print(f"Finished. Final Average Degree: {final_avg:.2f}")
    
    return G

# --- Usage Example ---
# G = prune_to_average_degree("my_network.graphml", target_avg_degree=64, weight_attr='weight')
# nx.write_graphml(G, "pruned_network.graphml")