import chainer
from chainer import training
from chainer.iterators import SerialIterator
from chainer_chemistry.dataset.converters import concat_mols
from chainer.training import extensions, StandardUpdater

import chainermn

import logging
import argparse
from distutils.util import strtobool
import glob

from model import pair_matrix_model
import uspto_pre
from updater import MyUpdater
from evaluator import MyEvaluator

from rdkit import RDLogger

rdl = RDLogger.logger()
rdl.setLevel(RDLogger.CRITICAL)


def inference():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--epoch', type=int, default=10)
    # parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', action='store_true')
    # parser.add_argument('--out', default='result_debug')
    # parser.add_argument('--frequency', type=int, default=-1)
    # parser.add_argument('--decay_iter', type=int, default=40000)
    parser.add_argument('--gnn_dim', type=int, default=300)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--nn_dim', type=int, default=100)
    # parser.add_argument('--train_path', default='../train.txt.proc')
    parser.add_argument('--valid_path', default='../test.txt.proc')
    parser.add_argument('--communicator', type=str, default='pure_nccl')
    # parser.add_argument('--type', default='debug')
    parser.add_argument('--rich', type=strtobool, default='false')

    args = parser.parse_args()




if __name__ == '__main__':
    inference()
