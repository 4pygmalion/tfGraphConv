#### Graph Convolution Layer
- implemented with TensorFlow 2.x

#### Requriment 
- tensorflow 2.x
- networkx
- numpy 


#### Usage
```python3
import numpy as np
import networkx as nx
import tensorflow as tf

G = nx.karate_club_graph()
X = np.random.rand(34, 5).astype(np.float32)
edge_weight = np.ones(shape=(len(G.edges)), dtype=np.int32)
print('We have %d nodes.' % G.number_of_nodes())
print('We have %d edges.' % G.number_of_edges())
# Node info: 34 nodes with 5 features

from GraphConvNet import GraphConvolution
gc_layer = GraphConvolution(input_dim=5, output_dim=10)
gc_layer(X, edge=tuple(G.edges), weight=edge_weight).shape

# TensorShape([34, 10])
```

#### Install
``` bash
$ git clone https://github.com/4pygmalion/tfGraphConv.git
```