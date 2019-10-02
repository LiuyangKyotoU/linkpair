import chainer
from chainer import links

from chainer_chemistry.models.gwm.gwm import GWM
from chainer_chemistry.links.update.ggnn_update import GGNNUpdate
from chainer_chemistry.links import GraphLinear


class ggnn_gwm(chainer.Chain):
    def __init__(self, hidden_channels=300, update_layer=GGNNUpdate,
                 n_update_layers=3, super_node_dim=248):
        super(ggnn_gwm, self).__init__()

        hidden_channels = [hidden_channels for _ in range(n_update_layers + 1)]

        in_channels_list = hidden_channels[:-1]
        out_channels_list = hidden_channels[1:]
        assert len(in_channels_list) == n_update_layers
        assert len(out_channels_list) == n_update_layers

        with self.init_scope():
            self.embed = GraphLinear(None, out_size=hidden_channels[0])
            self.update_layers = chainer.ChainList(
                *[update_layer(in_channels=in_channels_list[i],
                               out_channels=out_channels_list[i],
                               n_edge_types=6)
                  for i in range(n_update_layers)])
            self.gwm = GWM(hidden_dim=hidden_channels[0],
                           hidden_dim_super=super_node_dim,
                           n_layers=n_update_layers)
            self.embed_super = links.Linear(None, out_size=super_node_dim)

        self.n_update_layers = n_update_layers

    def __call__(self, atom_array, adj, super_node):
        self.reset_state()

        h = self.embed(atom_array)

        h_s = self.embed_super(super_node)

        for step in range(self.n_update_layers):
            h_new = self.update_layers[step](h=h, adj=adj)
            h_new, h_s = self.gwm(h, h_new, h_s, step)
            h = h_new

        return h

    def reset_state(self):
        if hasattr(self.update_layers[0], 'reset_state'):
            [update_layer.reset_state() for update_layer in self.update_layers]

        self.gwm.reset_state()


if __name__ == '__main__':
    import uspto_pre
    from chainer.iterators import SerialIterator
    from chainer_chemistry.dataset.converters import concat_mols

    train_raw = uspto_pre.read_data('../train.txt.proc')
    train_dataset = uspto_pre.USPTO_pre(train_raw[:100], 'softmax')
    train_iter = SerialIterator(train_dataset, 3)

    model = ggnn_gwm()

    for b in train_iter:
        atom_feature, adjs, supernode_feature, label, ind = concat_mols(b, padding=-1)
        print(model(atom_feature, adjs, supernode_feature))