import tensorflow as tf
import networkx as nx

class GraphConvolution(tf.keras.layers.Layer):

    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_intializer = 'glorot_uniform'


    def _get_A_from_graph(self, edge, weight):
        '''Generate the adjancy matrix from edge, weight

        Parameters
        ---------
        edge: nested tuple or list, eg. (v1, v2)
        weight: float

        Return
        ------
        np.ndarray
        '''
        items = [(pari, w) for pair, w in zip(edge, weight)]
        G = nx.Graph()
        G.add_weighted_edges_from(items)
        return nx.adjacency_matrix(G).astype(np.float32)


    def _get_degree_matrix(self, A):
        '''Element-wise inverse. D^{-1}'''
        return tf.math.reciprocal(inv_D)


    def build(self, input_shapes):
        '''Add trainable weight W'''
        self.kernel = self.add_weight(shape=(None, output_dim), name='W', initializer=self.kernel_intializer)
        self.bias = None
        self.built = True


    def call(self, X, edge, weight):
        '''TO generate callable object. main operation'''
        A = self._get_A_from_graph(edge, weight)
        inv_D = self._get_degree_matrix(A)
        return inv_D @ A @ X @ self.W

