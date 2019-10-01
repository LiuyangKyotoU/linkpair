import chainer
from chainer import functions, reporter
from chainer_chemistry.links import GraphMLP

from ggnn_gwm_custom import ggnn_gwm

import numpy as np


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
        batch_size = h.shape[0]
        hidden_size = h.shape[2]
        h = h.reshape(batch_size, 1, -1, hidden_size) + h.reshape(batch_size, -1, 1, hidden_size)
        h = h.reshape(batch_size, -1, hidden_size)
        h1 = self.nn1(h)
        h2 = self.nn2(h)
        # return h, h1, h2
        # TODO
        if self.label_type == 'sigmoid':
            pass

        elif self.label_type == 'softmax':
            l1 = label[:, 0, :, :].reshape(-1, )
            h1 = h1.reshape(-1, 5)
            loss1 = functions.softmax_cross_entropy(h1, l1, ignore_label=-1) / 2

            l2 = label[:, 1, :, :].reshape(-1, )
            h2 = h2.reshape(-1, 2)
            loss2 = functions.softmax_cross_entropy(h2, l2, ignore_label=-1) / 2
            acc = functions.accuracy(h2, l2, ignore_label=-1)

            loss = loss1 / 2 + loss2 / 2

        reporter.report({
            'loss': loss,
            'acc': acc
        }, self)


if __name__ == '__main__':
    import numpy as np

    model = pair_matrix_model(label_type='softmax', gnn_dim=300, n_layers=3, nn_dim=100)
    atom_feature = np.random.randn(16, 44, 151).astype(np.float32)
    adjs = np.random.randn(16, 6, 44, 44).astype(np.float32)
    supernode_feature = np.random.randn(16, 248).astype(np.float32)
    label = np.random.randint(2, size=(16, 2, 44, 44))
    ind = 3
    h = model(atom_feature, adjs, supernode_feature, label, ind)
