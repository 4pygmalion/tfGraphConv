import tensorflow as tf

class GCN(object):

    def __init__(self, n_layer):
        self._layers = [GraphConvolution() for _ in range(n_layer)]

    def __call__(self, x):
        h_0 = x.copy()
        for layer in self._layers:
            h_j = layer(h_j)

        return h_j