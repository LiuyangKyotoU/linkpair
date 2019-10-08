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
from rdkit import Chem
from tqdm import tqdm


def read_inference(inference):
    l = {}
    with open(inference, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            l[int(line.split()[0])] = line.split()[1:]
    return l


def constrcut_atom_MapNum_Num_list(mol):
    size = mol.GetNumAtoms()
    MapNum_Num = {x + 1: None for x in range(size)}
    for i in range(size):
        MapNum_Num[mol.GetAtomWithIdx(i).GetAtomMapNum()] = i
    return MapNum_Num


def get_reagents_ind(reaction):
    b = []
    for a in reaction[2].split(';'):
        b.append(int(a.split('-')[0]))
        b.append(int(a.split('-')[1]))
    b = list(set(b))

    mapnum_num = constrcut_atom_MapNum_Num_list(Chem.MolFromSmiles(reaction[0]))

    c = []
    for molecular in reaction[0].split('.'):
        consider = False
        for i in b:
            if ':' + str(i) + ']' in molecular:
                consider = True
                break
        if consider is False:
            mol_tmp = Chem.MolFromSmiles(molecular)
            for atom_tmp in mol_tmp.GetAtoms():
                c.append(atom_tmp.GetAtomMapNum())
    reagent_ind = [mapnum_num[x] for x in c]

    return reagent_ind


def get_label_ind(reaction):
    mapnum_num = constrcut_atom_MapNum_Num_list(Chem.MolFromSmiles(reaction[0]))
    l = []
    for a in reaction[2].split(';'):
        ind1 = mapnum_num[int(a.split('-')[0])]
        ind2 = mapnum_num[int(a.split('-')[1])]
        l.append(str(min(ind1, ind2)) + '-' + str(max(ind1, ind2)) + '-' + str(int(float(a.split('-')[2]))))
    return l


def loss1_acc(valid_raw, out):
    test_data = {x: {} for x in range(len(valid_raw))}
    for i in tqdm(range(len(valid_raw))):
        test_data[i]['reagent_ind'] = get_reagents_ind(valid_raw[i])
        test_data[i]['label'] = get_label_ind(valid_raw[i])

    for inference in glob.glob(out + 'inf_*'):
        print('Test ' + inference + 'accuracy: ================')
        inf = read_inference(inference)

        cor = 0
        for i in tqdm(range(len(valid_raw))):
            ignore = test_data[i]['reagent_ind']
            pred = inf[i]
            pred_consider = []

            for p in pred:
                if int(p.split('-')[0]) in ignore or int(p.split('-')[1]) in ignore:
                    continue
                else:
                    pred_consider.append(p)

            label = test_data[i]['label']
            if len(label) == len(pred_consider):
                correct = True
                for p in pred_consider:
                    if p not in label:
                        correct = False
                        break
                if correct:
                    cor += 1
        print(cor / len(valid_raw))


def loss2_acc(valid_raw, out1, out2):
    test_data = {x: {} for x in range(len(valid_raw))}
    for i in tqdm(range(len(valid_raw))):
        test_data[i]['reagent_ind'] = get_reagents_ind(valid_raw[i])
        test_data[i]['label'] = get_label_ind(valid_raw[i])

    for inference in glob.glob(out1 + 'inf_*'):
        print('Test ' + inference + 'accuracy: ================')
        inf_bt = read_inference(inference)
        inf_kc = read_inference(out2)

        cor_kc, cor_all = 0, 0

        for i in tqdm(range(len(valid_raw))):
            ignore = test_data[i]['reagent_ind']
            pred_kc = inf_kc[i]
            pred_kc_consider = []
            for p in pred_kc:
                if int(p.split('-')[0]) in ignore or int(p.split('-')[1]) in ignore:
                    continue
                else:
                    pred_kc_consider.append(p)

            pred_bt = inf_bt[i]
            pred_kc_bt_consider = []
            for p in pred_bt:
                if int(p.split('-')[0]) in ignore or int(p.split('-')[1]) in ignore:
                    continue
                elif pred_kc_consider == []:
                    pred_kc_bt_consider.append(p)
                elif p.split('-')[0] + '-' + p.split('-')[1] in inf_kc[i]:
                    pred_kc_bt_consider.append(p)

            label = test_data[i]['label']

            if len(label) == len(pred_kc_consider):
                correct = True
                for l in label:
                    if l.split('-')[0] + '-' + l.split('-')[1] not in pred_kc_consider:
                        correct = False
                        break
                if correct:
                    cor_kc += 1

            if len(label) == len(pred_kc_bt_consider):
                correct = True
                for p in pred_kc_bt_consider:
                    if p not in label:
                        correct = False
                        break
                if correct:
                    cor_all += 1

        print('kc acc: {};  kc_bt acc: {}'.format(cor_kc / len(valid_raw), cor_all / len(valid_raw)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--valid_path', default='../test.txt.proc')
    parser.add_argument('--out')
    parser.add_argument('--mode', default='loss1')

    args = parser.parse_args()

    assert args.mode in ['loss1', 'loss12']

    valid_raw = uspto_pre.read_data(args.valid_path)

    if args.mode == 'loss1':
        loss1_acc(valid_raw, args.out)
    elif args.mode == 'loss12':
        loss2_acc(valid_raw, args.out, 'inf_loss2.txt')
