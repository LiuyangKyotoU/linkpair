import chainer
from chainer import training
from chainer.iterators import SerialIterator
from chainer_chemistry.dataset.converters import concat_mols
from chainer.training import extensions, StandardUpdater

import chainermn

import logging
import argparse
from distutils.util import strtobool

from model import pair_matrix_model
import uspto_pre
from updater import MyUpdater
from evaluator import MyEvaluator

from rdkit import RDLogger
rdl = RDLogger.logger()
rdl.setLevel(RDLogger.CRITICAL)

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--out', default='result_debug')
    parser.add_argument('--frequency', type=int, default=-1)
    parser.add_argument('--decay_iter', type=int, default=40000)
    parser.add_argument('--gnn_dim', type=int, default=300)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--nn_dim', type=int, default=100)
    parser.add_argument('--train_path', default='../train.txt.proc')
    parser.add_argument('--valid_path', default='../test.txt.proc')
    parser.add_argument('--communicator', type=str, default='pure_nccl')
    parser.add_argument('--type', default='debug')
    parser.add_argument('--rich', type=strtobool, default='false')

    args = parser.parse_args()

    assert args.type in ['debug', 'all']

    if args.gpu:
        comm = chainermn.create_communicator(args.communicator)
        device = comm.intra_rank
    else:
        comm = chainermn.create_communicator('naive')
        device = -1

    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        if args.gpu:
            print('Using GPUs')
        print('Using {} communicator'.format(args.communicator))
        print('Num Layers: {}'.format(args.n_layers))
        print('Num Hidden-dim: {}'.format(args.gnn_dim))
        print('Num Minibatch-size: {}'.format(args.batch_size))
        print('Num epoch: {}'.format(args.epoch))
        print('==========================================')

    model = pair_matrix_model(gnn_dim=args.gnn_dim, n_layers=args.n_layers, nn_dim=args.nn_dim)
    if device > 0:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()

    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.Adam(alpha=args.lr), comm)
    optimizer.setup(model)

    train_raw = uspto_pre.read_data(args.train_path)
    valid_raw = uspto_pre.read_data(args.valid_path)

    if comm.rank == 0:
        if args.type == 'debug':
            train_dataset = uspto_pre.USPTO_pre(train_raw[:100], args.rich)
            valid_dataset = uspto_pre.USPTO_pre(valid_raw[:40], args.rich)
        elif args.type == 'all':
            train_dataset = uspto_pre.USPTO_pre(train_raw, args.rich)
            valid_dataset = uspto_pre.USPTO_pre(valid_raw, args.rich)
    else:
        train_dataset, valid_dataset = None, None

    train_dataset = chainermn.scatter_dataset(train_dataset, comm, shuffle=True)
    valid_dataset = chainermn.scatter_dataset(valid_dataset, comm, shuffle=True)

    train_iter = SerialIterator(train_dataset, args.batch_size)
    valid_iter = SerialIterator(valid_dataset, args.batch_size, repeat=False, shuffle=False)

    updater = MyUpdater(iterator=train_iter, optimizer=optimizer, device=device, converter=concat_mols)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    evaluator = MyEvaluator(iterator=valid_iter, target=model, device=device, converter=concat_mols)
    trainer.extend(evaluator)

    trainer.extend(extensions.observe_value('alpha', lambda t: optimizer.alpha))
    trainer.extend(extensions.ExponentialShift('alpha', 0.9, optimizer=optimizer),
                   trigger=(args.decay_iter, 'iteration'))

    if comm.rank == 0:
        frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
        trainer.extend(extensions.snapshot_object(model, 'snapshot_{.updater.iteration}'),
                       trigger=(frequency, 'epoch'))

        trainer.extend(extensions.LogReport())

        trainer.extend(extensions.PrintReport(
            ['epoch', 'alpha',
             'main/loss', 'validation/main/loss',
             # 'main/acc', 'validation/main/acc',
             'elapsed_time']
        ))
        # trainer.extend(extensions.PlotReport(
        #     ['main/acc', 'validation/main/acc'],
        #     'epoch', file_name='acc.png'
        # ))
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'],
            'epoch', file_name='loss.png'
        ))
        trainer.extend(extensions.ProgressBar())

    trainer.run()


if __name__ == '__main__':
    main()

    import numpy as np

    np.triu(np.ones((16, 16)), 1)