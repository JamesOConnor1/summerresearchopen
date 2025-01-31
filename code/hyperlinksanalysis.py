import networkx as nx
from sklearn.cluster import SpectralClustering, KMeans
import numpy as np
from pyvis.network import Network
from scipy.sparse.linalg import eigsh
import random

FILE_PATH = 'soc-redditHyperlinks-body.tsv'
SEED = 42  # Set your seed value here

# Set the seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)

# Read the graph data from the text file
def read_graph(file_path):
    graph = nx.DiGraph()
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            parts = line.strip().split('\t')
            if int(parts[4]) == 1:
                node1 = parts[0]
                node2 = parts[1]
                weight = 1
                graph.add_edge(node1, node2, edge=weight)
    return graph

def random_walk_graph(graph, start, iterations=1000):
    unchosen = set()
    divisor = 10
    count = 0
    traverse = [start]
    new_graph = []
    ignored = set()
    while count < iterations:
        adj_nodes = [node for node in list(graph.neighbors(traverse[0])) if node not in unchosen]
        random_nodes = random.sample(adj_nodes, len(adj_nodes) // divisor)
        if len(adj_nodes) > 0 and len(random_nodes) == 0:
            random_nodes = random.sample(adj_nodes, 1)
        ignored.update(set(adj_nodes) - set(random_nodes))
        for node in random_nodes:
            new_graph.append((traverse[0], node))

        traverse.extend(random_nodes)
        traverse.pop(0)
        
        if len(traverse) == 0:
            return new_graph
        unchosen.update(ignored)
        count += 1
    return new_graph

def visualize_graph(graph):
    nt = Network('100%', '100%')
    labels = spectral_clustering(graph, 5)
    colors = ['#FF0000', '#0000FF', '#00FF00', '#FFFF00', '#FF00FF', '#00FFFF']  
    
    # Add nodes with colors based on labels
    for idx, node in enumerate(graph.nodes()):
        color = colors[labels[idx] % len(colors)]  # Cycle through colors
        if node != 'milifans':
            nt.add_node(node, color=color)
    
    # Add edges
    for edge in graph.edges(data=True):
        if edge[1] != 'milifans' and edge[0] != 'milifans':
            nt.add_edge(edge[0], edge[1], weight=edge[2])

    nt.show('hyperlinks_colour_final_cluster.html', notebook=False)


def spectral_clustering(graph, n_clusters):
    adjacency_matrix = nx.to_numpy_array(graph, weight='weight')
    adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(adjacency_matrix, axis=1)))
    laplacian_matrix = np.eye(adjacency_matrix.shape[0]) - d_inv_sqrt @ adjacency_matrix @ d_inv_sqrt
    row_sums = np.sum(adjacency_matrix, axis=1)
    d_inv = np.diag(1.0 / row_sums)
    random_walk_laplacian = np.eye(adjacency_matrix.shape[0]) - d_inv @ adjacency_matrix
    _, eigenvectors = eigsh(random_walk_laplacian, k=n_clusters+1, which='SM')
    eigenvectors=eigenvectors[:,1:] #remove first eigenvector I only use one component graphs
    # use KMeans on eigenvectors
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
    labels = kmeans.fit_predict(eigenvectors)

    return labels


graph = read_graph(FILE_PATH)
print(len(graph.nodes), len(graph.edges))
graph = nx.DiGraph(random_walk_graph(graph, 'askreddit', 1000)) 
print(len(graph.nodes), len(graph.edges))
#askreddit is max degree node by max(graph.degree, key=lambda x: x[1])[0]
print('sampled')

visualize_graph(graph)
print('Done')
