import numpy as np
from sklearn.cluster import KMeans
from pyvis.network import Network
import networkx as nx
from scipy.sparse.linalg import eigsh
import random


SEED = 787 # This is very unstable, several choices will give basically only 1 cluster, and that includes the seed choice I used previously of 42
random.seed(SEED)
# Define the number of clusters
num_clusters = 2

# Function to process each line into a list of integers
def process_line(line):
    line = line.strip()  # Remove leading/trailing whitespace
    line = line[1:-1]  # Remove the square brackets
    return [int(x) for x in line.split(',')]

# Load and process data
with open('reddit_vectors_larger.txt', 'r') as file:
    data = np.array([process_line(line) for line in file], dtype=int)

# Load subreddit names
with open('all_subs_larger.txt', 'r') as file:
    subreddit_names = [line.strip() for line in file]

# Transpose the data to switch users and subreddits
data = data.T  # Now rows are subreddits, and columns are users


# Calculate similarity between subreddits using Jaccard index
n_subreddits = data.shape[0]
similarity_matrix = np.zeros((n_subreddits, n_subreddits))

intersection = np.dot(data, data.T)
union = np.add.outer(np.sum(data, axis=1), np.sum(data, axis=1)) - intersection
similarity_matrix = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)


# Create a graph from the similarity matrix
G = nx.Graph()
for i in range(n_subreddits):
    for j in range(i + 1, n_subreddits):
        if similarity_matrix[i, j] > 0:
            G.add_edge(subreddit_names[i], subreddit_names[j], weight=similarity_matrix[i, j])


def random_walk_graph(graph, start, secondary, iterations=1000):
    unchosen = set()
    divisor = 2
    count = 0
    traverse = [start,secondary]
    new_graph = []
    ignored = set()
    while count < iterations:
        adj_nodes = [node for node in list(graph.neighbors(traverse[0])) if node not in unchosen]
        random_nodes = random.sample(adj_nodes, len(adj_nodes) // divisor)
        if len(adj_nodes) > 0 and len(random_nodes) == 0:
            random_nodes = random.sample(adj_nodes, 1)
        if len(adj_nodes) == 0 and len(traverse) > 1:
            random_nodes = [traverse[1]]
            traverse.pop(1)
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

G = nx.Graph(random_walk_graph(G, max(G.degree, key=lambda x: x[1])[0], 'democrats', 1000))
print(len(G.nodes()))

# Perform spectral clustering on the largest connected component
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

adj_matrix = nx.adjacency_matrix(G)
adj_matrix = adj_matrix.toarray()
labels = spectral_clustering(G, num_clusters)
label_counts = np.bincount(labels)
for i in range(num_clusters):
    print(f"Count of label {i}: {label_counts[i]}")


# Visualize the largest connected component using pyvis
net = Network(height="100%", width="100%", notebook=False)
net.barnes_hut()
# Add nodes to the network with colors based on labels
colors_hex = ['#0000FF','#FF0000']


# Add nodes to the network with colors based on labels
for idx, node in enumerate(G.nodes()):
    color = colors_hex[labels[idx] % len(colors_hex)]  # Cycle through colors
    net.add_node(node, color=color)

# Add edges based on the subgraph
edges = []
for i, node_i in enumerate(G.nodes()):
    for j, node_j in enumerate(G.nodes()):
        if i < j and adj_matrix[i, j] > 0:
            edges.append((node_i, node_j, int(adj_matrix[i, j])))  # Convert to int

# Add edges in bulk to optimize performance
net.add_edges(edges)    

net.show("test2.html", notebook=False)

# Get the number of users for each subreddit
subreddit_user_counts = np.sum(data, axis=1)

# Get the indices of subreddits that are in the graph G
subreddits_in_graph = [subreddit_names.index(node) for node in G.nodes()]

# Get the top 10 subreddits from each cluster
subreddits_in_graph_cluster_0 = [subreddit_index for subreddit_index, label in zip(subreddits_in_graph, labels) if label == 0]
subreddits_in_graph_cluster_1 = [subreddit_index for subreddit_index, label in zip(subreddits_in_graph, labels) if label == 1]

# Get the user counts for these subreddits
user_counts_in_graph_cluster_0 = subreddit_user_counts[subreddits_in_graph_cluster_0]
user_counts_in_graph_cluster_1 = subreddit_user_counts[subreddits_in_graph_cluster_1]

# Get the indices of the top 10 subreddits in each cluster
top_10_indices_in_graph_cluster_0 = np.argsort(user_counts_in_graph_cluster_0)[-10:]
top_10_indices_in_graph_cluster_1 = np.argsort(user_counts_in_graph_cluster_1)[-10:]

# Map these indices back to the original subreddit indices
top_10_indices_cluster_0 = [subreddits_in_graph_cluster_0[i] for i in top_10_indices_in_graph_cluster_0]
top_10_indices_cluster_1 = [subreddits_in_graph_cluster_1[i] for i in top_10_indices_in_graph_cluster_1]

# Get the top 10 subreddit names, their user counts, and their labels
top_10_subreddit_names_cluster_0 = [subreddit_names[i] for i in top_10_indices_cluster_0]
top_10_user_counts_cluster_0 = subreddit_user_counts[top_10_indices_cluster_0]

top_10_subreddit_names_cluster_1 = [subreddit_names[i] for i in top_10_indices_cluster_1]
top_10_user_counts_cluster_1 = subreddit_user_counts[top_10_indices_cluster_1]

print("Top 10 subreddits in cluster 0:")
for name, user_count in zip(top_10_subreddit_names_cluster_0, top_10_user_counts_cluster_0):
    print(f"{name}: {user_count} users")

print("Top 10 subreddits in cluster 1:")
for name, user_count in zip(top_10_subreddit_names_cluster_1, top_10_user_counts_cluster_1):
    print(f"{name}: {user_count} users")

