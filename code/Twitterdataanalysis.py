import json
import numpy as np
import networkx as nx
from sklearn.cluster import AgglomerativeClustering, KMeans
from pyvis.network import Network
from scipy.sparse.linalg import eigsh
import pandas as pd



# Load the data
with open('congress_network_data.json') as f:
    data = json.load(f)


# Data description
# inList[i]: List of nodes sending connections TO node i
# inWeight[i]: Weights (connection strengths) for inList[i]
# outList[i]: List of nodes receiving connections FROM node i
# outWeight[i]: Weights (connection strengths) for outList[i]
# usernameList[i]: Twitter username corresponding to node i


inList = data[0]['inList']
inWeight = data[0]['inWeight']
outList = data[0]['outList']
outWeight = data[0]['outWeight']
usernameList = data[0]['usernameList']

CLUSTERS = 2  # Number of clusters (e.g., political parties)
SEED = 42


num_people = len(inList)  # Total number of nodes in the network

# Initialize an adjacency matrix with zeros
adjacency_matrix = np.zeros((num_people, num_people))

# Fill the adjacency matrix based on outList and outWeight
for sending_node, receiving_node_list in enumerate(outList):
    for num, receiving_node in enumerate(receiving_node_list):
        adjacency_matrix[sending_node,
                         receiving_node] = outWeight[sending_node][num]

G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

def spectral_clustering(graph, n_clusters):
    adjacency_matrix = nx.to_numpy_array(graph, weight='weight')
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



# Visualize the graph
labels = spectral_clustering(G, CLUSTERS)
dist_matrix = 1/(adjacency_matrix + 1e-10)
labels = AgglomerativeClustering(n_clusters=CLUSTERS,linkage='complete').fit_predict(dist_matrix)

# Create Pyvis interactive network
net = Network(height='100%', width='100%', notebook=False)

# Add nodes to Pyvis 
for node_id in G.nodes():
    cluster_label = labels[node_id]  # Get the cluster label for the node
    color = 'blue' if cluster_label == 0 else 'red'
    net.add_node(
        node_id, label=usernameList[node_id], title=usernameList[node_id], color=color)


# Add edges to Pyvis with weights
for u, v, data in G.edges(data=True):
    # Default to 1 if no weight
    weight = data['weight'] if 'weight' in data else 1
    net.add_edge(u, v, value=weight)

# Set layout for physics-based placement
net.force_atlas_2based() 

# Show the interactive graph in an HTML file
# net.show("with_clustering.html", notebook=False)
# Read from congress_twitter_116th.xlsx to grab usernames and party affiliations

# Load the Excel file
df = pd.read_excel('congress_twitter_117th.xlsx',engine='openpyxl')
# Extract usernames and party affiliations
#strip spaces from the usernames and parties columns
df['Link'] = df['Link'].str.strip()
df['Party'] = df['Party'].str.strip()
usernames = df['Link'].tolist()
parties = df['Party'].tolist()


# Create a dictionary for quick lookup of party by username
party_dict = dict(zip(usernames, parties))

# Create a new dataframe of congresspeople and cluster labels
congresspeople = pd.DataFrame({
    'username': usernameList,
    'cluster': labels,
    'party': [party_dict[username] for username in usernameList]
})

# Check how many in cluster 0 are republican and democrat
cluster0 = congresspeople[congresspeople['cluster'] == 0]
cluster1 = congresspeople[congresspeople['cluster'] == 1]


# Count the number of democrats and republicans in cluster 0
democrats_cluster0 = cluster0[cluster0['party'] == 'D'].shape[0]
republicans_cluster0 = cluster0[cluster0['party'] == 'R'].shape[0]

# Count the number of democrats and republicans in cluster 1
democrats_cluster1 = cluster1[cluster1['party'] == 'D'].shape[0]
republicans_cluster1 = cluster1[cluster1['party'] == 'R'].shape[0]

print(f"Cluster 0 - Democrats: {democrats_cluster0}, Republicans: {republicans_cluster0}")
print(f"Cluster 1 - Democrats: {democrats_cluster1}, Republicans: {republicans_cluster1}")
