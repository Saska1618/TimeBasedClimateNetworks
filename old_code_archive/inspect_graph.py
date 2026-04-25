import networkx as nx
import os

def inspect_graph(file_path):
    """Reads a GraphML file and prints information about it."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        G = nx.read_graphml(file_path)
        print(f"Successfully loaded graph from: {file_path}")
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")

        # Print info about a sample node
        if G.number_of_nodes() > 0:
            sample_node = list(G.nodes(data=True))[0]
            print(f"\nSample node: {sample_node}")

        # Print info about the first 5 edges
        if G.number_of_edges() > 0:
            print("\nAttributes of the first 5 edges:")
            for i, edge in enumerate(list(G.edges(data=True))[:5]):
                print(f"  Edge {i+1}: from {edge[0]} to {edge[1]}, attributes: {edge[2]}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Path to the graph file relative to the project root
    graph_file = 'networks/global_networks/correct_avg_degree/Bacskatopolya_global_pruned_1960-01_2024-12.graphml'
    inspect_graph(graph_file)
