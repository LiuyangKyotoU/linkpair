import chainer


class MyUpdater(chainer.training.updaters.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop('models')
        super(MyUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        optimizer = self.get_optimizer('main')

        batch = self.get_iterator('main').next()

        in_arrays = self.converter(batch=batch, device=self.device, padding=-1)

        # optimizer.update(self.model, in_arrays)

        return in_arrays


if __name__ == '__main__':
    import uspto_pre
    from model import pair_matrix_model
    from chainer.iterators import SerialIterator
    from chainer_chemistry.dataset.converters import concat_mols

    train_raw = uspto_pre.read_data('../train.txt.proc')
    train_dataset = uspto_pre.USPTO_pre(train_raw[:100], 'sigmoid')
    train_iter = SerialIterator(train_dataset, 3)

    model = pair_matrix_model(label_type='sigmoid',
                              gnn_dim=300, n_layers=3, nn_dim=100)
    optimizer = chainer.optimizers.Adam(alpha=1e-4)
    optimizer.setup(model)

    updater = MyUpdater(models=model, iterator=train_iter,
                        optimizer=optimizer, device=-1, converter=concat_mols)

    a = updater.update_core()
