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

import glob


def inference():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--epoch', type=int, default=10)
    # parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--out', default='result_debug/')
    # parser.add_argument('--frequency', type=int, default=-1)
    # parser.add_argument('--decay_iter', type=int, default=40000)
    parser.add_argument('--gnn_dim', type=int, default=300)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--nn_dim', type=int, default=100)
    # parser.add_argument('--train_path', default='../train.txt.proc')
    parser.add_argument('--valid_path', default='../test.txt.proc')
    parser.add_argument('--communicator', type=str, default='pure_nccl')
    parser.add_argument('--type', default='debug')
    parser.add_argument('--rich', type=strtobool, default='false')

    parser.add_argument('--snapshot')

    args = parser.parse_args()

    assert args.type in ['debug', 'all']

    if args.gpu:
        comm = chainermn.create_communicator(args.communicator)
        device = comm.intra_rank
    else:
        comm = chainermn.create_communicator('naive')
        device = -1

    if comm.rank == 0:
        print(glob.glob(args.out + 'snapshot_*'))

    model = pair_matrix_model(gnn_dim=args.gnn_dim, n_layers=args.n_layers, nn_dim=args.nn_dim)
    chainer.serializers.load_npz(args.out + args.snapshot, model)
    if device > 0:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()

    valid_raw = uspto_pre.read_data(args.valid_path)
    if comm.rank == 0:
        if args.type == 'debug':
            valid_dataset = uspto_pre.USPTO_pre(valid_raw[:40], args.rich)
        elif args.type == 'all':
            valid_dataset = uspto_pre.USPTO_pre(valid_raw, args.rich)
    else:
        valid_dataset = None, None
    valid_dataset = chainermn.scatter_dataset(valid_dataset, comm, shuffle=False)
    valid_iter = SerialIterator(valid_dataset, args.batch_size, repeat=False, shuffle=False)

    for batch in valid_iter:
        in_arrays = concat_mols(batch=batch, device=device, padding=-1)
        assert isinstance(in_arrays, tuple)

        with chainer.using_config('train', False):
            h = model.ggnn_gwm(in_arrays[0], in_arrays[1], in_arrays[2])

            ind = in_arrays[4]

            batch_size = h.shape[0]
            graph_size = h.shape[1]
            hidden_size = h.shape[2]

            h = h.reshape(batch_size, 1, -1, hidden_size) + h.reshape(batch_size, -1, 1, hidden_size)
            h = h.reshape(batch_size, -1, hidden_size)
            h2 = model.nn2(h)
            h2 = h2.reshape(batch_size, graph_size, graph_size, 2)

            for x in range(batch_size):
                index = ind[x]
                action_pred = []
                for i in range(graph_size):
                    for j in range(i + 1, graph_size):
                        if h2[x, i, j, 1].data > h2[x, i, j, 0].data:
                            action_pred.append(str(i) + '-' + str(j))
                with open(args.out + 'inf_' + args.snapshot + '.txt', 'a') as file:
                    file.write(str(index))
                    for a_p in action_pred:
                        file.write('\t' + a_p)
                    file.write('\n')


if __name__ == '__main__':
    inference()
