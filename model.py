import chainer
from chainer import functions, reporter
from chainer_chemistry.links import GraphMLP

from ggnn_gwm_custom import ggnn_gwm


class pair_matrix_model(chainer.Chain):
    def __init__(self, label_type, gnn_dim, n_layers, nn_dim):
        super(pair_matrix_model, self).__init__()
        with self.init_scope():
            self.ggnn_gwm = ggnn_gwm(hidden_channels=gnn_dim, n_update_layers=n_layers)
            self.nn1 = GraphMLP(channels=[nn_dim, 5])
            self.nn2 = GraphMLP(channels=[nn_dim, 2])
        self.label_type = label_type

    def __call__(self, atom_feature, adjs, supernode_feature, label, ind):
        loss, acc = 0.0, 0.0
        h = self.ggnn_gwm(atom_feature, adjs, supernode_feature)
        h1 = self.nn1(h)
        h2 = self.nn2(h)

        # TODO
        if self.label_type == 'sigmoid':
            pass
        elif self.label_type == 'softmax':
            pass

if __name__ == '__main__':
    import numpy as np

    model = pair_matrix_model(label_type='softmax', gnn_dim=300, n_layers=3, nn_dim=100)
    atom_feature = np.random.randn(16, 44, 151).astype(np.float32)
    adjs = np.random.randn(16, 6, 44, 44).astype(np.float32)
    supernode_feature = np.random.randn(16, 248).astype(np.float32)
    label = np.zeros((16, 44, 44)).astype(np.int32)
    ind = 3
    h1, h2 = model(atom_feature, adjs, supernode_feature, label, ind)
