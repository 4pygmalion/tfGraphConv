import tensorflow as tf
import networkx as nx
import numpy as np

class GraphConvolution(tf.keras.layers.Layer):

    def __init__(self, input_dim, output_dim, use_bais=False, **kwargs):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_intializer = 'glorot_uniform'
        self.bias_initializer = 'glorot_uniform'
        self.use_bias = use_bais

    def _get_A_from_graph(self, edges, weight):
        '''Generate the adjancy matrix from edge, weight

        Parameters
        ---------
        edge: nested tuple or list, eg. (v1, v2)
        weight: float

        Return
        ------
        np.ndarray with shape (n, n)
        '''
        items = [(*edge, w) for edge, w in zip(edges, weight)]
        G = nx.Graph()
        G.add_weighted_edges_from(items)
        return nx.adjacency_matrix(G).astype(np.float32)


    def _get_degree_matrix(self, A):
        '''Element-wise inverse. D^{-1}'''
        inv_D = np.zeros(shape=(A.shape))  # (n, n)
        for i in range(len(inv_D)):
            inv_D[i, i] = tf.math.reciprocal(A.sum(axis=-1))[i]
        return inv_D

    def build(self, input_shape):
        '''Add trainable weight W'''
        self.kernel = self.add_weight(shape=(input_shape[-1], self.output_dim), name='W', initializer=self.kernel_intializer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,), initializer=self.bias_initializer)
        else:
            self.bias = None
        self.built = True


    def call(self, X, edge, weight):
        '''TO generate callable object. main operation'''
        A = self._get_A_from_graph(edge, weight)
        inv_D = self._get_degree_matrix(A)
        return inv_D @ A @ X @ self.kernel

