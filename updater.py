import chainer


class MyUpdater(chainer.training.updaters.StandardUpdater):
    def __init__(self, iterator, optimizer, converter, device):
        super(MyUpdater, self).__init__(iterator, optimizer, converter, device)

    def update_core(self):
        iterator = self._iterators['main']
        batch = iterator.next()
        in_arrays = self.converter(batch, self.device, padding=-1)
        # return in_arrays
        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        if isinstance(in_arrays, tuple):
            optimizer.update(loss_func, *in_arrays)
        elif isinstance(in_arrays, dict):
            optimizer.update(loss_func, **in_arrays)
        else:
            optimizer.update(loss_func, in_arrays)

        if self.auto_new_epoch and iterator.is_new_epoch:
            optimizer.new_epoch(auto=True)


if __name__ == '__main__':
    import uspto_pre
    from model import pair_matrix_model
    from chainer.iterators import SerialIterator
    from chainer_chemistry.dataset.converters import concat_mols

    train_raw = uspto_pre.read_data('../train.txt.proc')
    train_dataset = uspto_pre.USPTO_pre(train_raw[:100], 'softmax')
    train_iter = SerialIterator(train_dataset, 3)

    model = pair_matrix_model(label_type='softmax',
                              gnn_dim=300, n_layers=3, nn_dim=100)
    optimizer = chainer.optimizers.Adam(alpha=1e-4)
    optimizer.setup(model)

    updater = MyUpdater(iterator=train_iter, optimizer=optimizer, device=-1, converter=concat_mols)

    atom_feature, adjs, supernode_feature, label, ind = updater.update_core()
