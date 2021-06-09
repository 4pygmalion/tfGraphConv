import tensorflow as tf

class GraphConvolution(tf.keras.layers.Layer):

    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_intializer = 'glorot_uniform'


    def _get_A_from_graph(self, edge, weight):
        return A

    def _get_degree_matrix(self, A):
        return inv_D

    def build(self, input_shapes):
        self.kernel = self.add_weight(shape=(None, output_dim), name='W', initializer=self.kernel_intializer)
        self.bias = None
        self.built = True

    def call(self, X):
        A = self._get_A_from_graph()
        inv_D = self._get_degree_matrix(A)
        return inv_D @ A @ X @ self.W


class GCN(object):

    def __init__(self, n_layer):
        self._layers = [GraphConvolution() for _ in range(n_layer)]

    def __call__(self, x):
        h_0 = x.copy()
        for layer in self._layers:
            h_j = layer(h_j)

        return h_j