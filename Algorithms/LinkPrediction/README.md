# Link Prediction 

### 1. Neighbour-based measures: Jaccard coefficient, and preferential attachment measure
#### 1.1 Jaccard coefficientf

#### 1.2 Preferential attachment measure

### 2. Inverse Distance Score

### 3. Random, in which the link is given a random score in [0, 1]



### Explanation of How Everything Works Together in `link_prediction_lab.ipynb`

1. **Import Libraries and Initialize Random Seed**
   - Import necessary libraries: `networkx` for graph operations, `matplotlib.pyplot` for visualization, and `random` for generating random scores.
   - Initialize the random seed for reproducibility.

2. **Define the Graph**
   - Read the edge list from the file `karate` to create the graph `G`.

3. **Define Link Prediction Functions**
   - **Common Neighbors**: Computes the number of common neighbors between two nodes.
   - **Jaccard Coefficient**: Computes the Jaccard coefficient between two nodes.
   - **Preferential Attachment**: Computes the preferential attachment score between two nodes.
   - **Inverse Distance**: Computes the inverse of the shortest path distance between two nodes.
   - **Random Score**: Assigns a random score to each potential link.

4. **Define Helper Functions**
   - **link_list**: Generates a sorted list of potential links and their scores using a specified scoring function.
   - **visualize_links**: Visualizes the top 5 predicted links on the graph and prints their scores.

5. **Visualize the Graph**
   - Draw the initial graph using `networkx` and `matplotlib`.

6. **Compute and Visualize Link Predictions**
   - For each link prediction function (Common Neighbors, Jaccard Coefficient, Preferential Attachment, Inverse Distance, Random Score):
     - Compute the link scores using `link_list`.
     - Visualize the top 5 predicted links using `visualize_links`.

### Code Structure

```python
# Import necessary libraries
import networkx as nx
import matplotlib.pyplot as plt
import random as rnd

rnd.seed()

# Define the graph
G = nx.read_edgelist("karate")

# Define link prediction functions
def common_neighbors(G, i, j):
    return len(set(G.neighbors(i)) & set(G.neighbors(j)))

def jaccard_coefficient(G, i, j):
    neighbors_i = set(G.neighbors(i))
    neighbors_j = set(G.neighbors(j))
    intersection = len(neighbors_i & neighbors_j)
    union = len(neighbors_i | neighbors_j)
    return intersection / union if union != 0 else 0

def preferential_attachment(G, i, j):
    return len(list(G.neighbors(i))) * len(list(G.neighbors(j)))

def inverse_distance(G, i, j):
    try:
        distance = nx.shortest_path_length(G, source=i, target=j)
        return 1 / distance if distance > 0 else 0
    except nx.NetworkXNoPath:
        return 0

def random_score(G, i, j):
    return rnd.random()

# Define helper functions
def link_list(G, i, score_func):
    links = []
    for j in G.nodes():
        if not G.has_edge(i, j):
            e = (i, j)
            sc = score_func(G, i, j)
            links.append([e, sc])
    links.sort(key=lambda x: x[1], reverse=True)
    return links

def visualize_links(G, node, score_func, title):
    links = link_list(G, node, score_func)
    top_links = links[:5]

    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')

    for link, score in top_links:
        nx.draw_networkx_edges(G, pos, edgelist=[link], width=2.5, alpha=0.6, edge_color='r')
        print(f"Link: {link}, Score: {score}")

    plt.title(title)
    plt.show()

# Visualize the initial graph
nx.draw(G, with_labels=True)
plt.show()

# Compute and visualize link predictions
visualize_links(G, '31', common_neighbors, "Common Neighbors")
visualize_links(G, '31', jaccard_coefficient, "Jaccard Coefficient")
visualize_links(G, '31', preferential_attachment, "Preferential Attachment")
visualize_links(G, '31', inverse_distance, "Inverse Distance")
visualize_links(G, '31', random_score, "Random Score")
```

This code structure ensures that the graph is read, link prediction functions are defined, and the results are visualized in a coherent manner.