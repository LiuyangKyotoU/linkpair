import numpy as np
from rdkit import Chem
import chainer_chemistry
from chainer_chemistry.config import MAX_ATOMIC_NUM


def construct_atomic_number_array(mol):
    return np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)


def construct_atom_feature_matrix(mol, whether_rich=True):
    atom_simple_features = [list(map(lambda s: a.GetAtomicNum() == s, list(range(MAX_ATOMIC_NUM)))) + \
                            list(map(lambda s: a.GetDegree() == s, [0, 1, 2, 3, 4, 5])) + \
                            list(map(lambda s: a.GetExplicitValence() == s, [1, 2, 3, 4, 5, 6])) + \
                            list(map(lambda s: a.GetImplicitValence() == s, [0, 1, 2, 3, 4, 5])) + \
                            [a.GetIsAromatic()] for a in mol.GetAtoms()]
    atom_simple_features = np.array(atom_simple_features, dtype=np.float32)

    if not whether_rich:
        return atom_simple_features

    atom_rich_features = [
        [a.GetIsAromatic() == False and any([neighbor.GetIsAromatic() for neighbor in a.GetNeighbors()])] + \
        [a.IsInRing()] + \
        [a.GetAtomicNum() in [9, 17, 35, 53, 85, 117]] + \
        [a.GetAtomicNum() in [8, 16, 34, 52, 84, 116]] + \
        [a.GetAtomicNum() in [7, 15, 33, 51, 83]] + \
        [a.GetAtomicNum() in [3, 11, 19, 37, 55, 87]] + \
        [a.GetAtomicNum() in [4, 12, 20, 38, 56, 88]] + \
        [a.GetAtomicNum() in [13, 22, 24, 25, 26, 27, 28, 29, 30, 33, 42, 44, 45, 46, 47, 48, 49, 50, 78, 80, 82]] \
        for a in mol.GetAtoms()]
    atom_rich_features = np.array(atom_rich_features, dtype=np.float32)

    return np.concatenate((atom_simple_features, atom_rich_features), axis=1)


def construct_discrete_edge_matrix(mol):
    size = mol.GetNumAtoms()
    adjs = np.zeros((6, size, size), dtype=np.float32)

    bond_type_to_channel = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3
    }

    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        ch = bond_type_to_channel[bond_type]

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjs[ch, i, j] = 1.0
        adjs[ch, j, i] = 1.0

        # add another feature
        if bond.GetIsConjugated():
            adjs[-2, i, j] = 1.0
            adjs[-2, j, i] = 1.0
        if bond.IsInRing():
            adjs[-1, i, j] = 1.0
            adjs[-1, j, i] = 1.0

    return adjs


def construct_supdernode_feature_array(mol, atom_array, adj):
    return chainer_chemistry.dataset.preprocessors.common.construct_supernode_feature(mol, atom_array, adj)


def constrcut_atom_MapNum_Num_list(mol):
    size = mol.GetNumAtoms()
    MapNum_Num = {x + 1: None for x in range(size)}
    for i in range(size):
        MapNum_Num[mol.GetAtomWithIdx(i).GetAtomMapNum()] = i
    return MapNum_Num


# TODO: 好像不对: keep的1更多，change的1很少，导致inference的时候基本都是keep，但是只保留一个的话就必须要threshod或者top-10
def construct_sigmoid_label(mol, adjs, actions):
    '''
    - reactants: [CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]
    :param mol:
    :param actions: 12-13-0.0;12-15-1.0
    :return: 5(none, single, double, triple, aromatic) + 2(keep, change), Note !! : use 2 mlp but cal only 1 sigmoid loss!
    '''

    d = constrcut_atom_MapNum_Num_list(mol)
    size = mol.GetNumAtoms()
    p_label = np.zeros((7, size, size)).astype(np.float32)
    p_label[0, :, :] = 1 - adjs[:4, :, :].sum(axis=0)
    p_label[1:5, :, :] = adjs[:4, :, :]
    p_label[5, :, :] = 1.0

    for a in actions.split(';'):
        tmp = a.split('-')
        i = d[int(tmp[0])]
        j = d[int(tmp[1])]
        l = int(float(tmp[2]))
        p_label[:, i, j] = 0.0
        p_label[:, j, i] = 0.0
        p_label[6, i, j] = 1.0
        p_label[6, j, i] = 1.0
        p_label[l, i, j] = 1.0
        p_label[l, j, i] = 1.0
    return p_label


def construct_softmax_label(mol, adjs, actions):
    '''
    - reactants: [CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]
    :param mol:
    :param actions: 12-13-0.0;12-15-1.0
    :return: 1(none/single/double/triple/aromatic), 1(keep/change), Note !! : use 2 mlp but cal 2 softmax losses!
    '''
    d = constrcut_atom_MapNum_Num_list(mol)
    size = mol.GetNumAtoms()
    s_label = np.zeros((2, size, size)).astype(np.int32)
    bond_type_to_channel_2 = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4
    }
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        ch = bond_type_to_channel_2[bond_type]
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        s_label[0, i, j] = ch
        s_label[0, j, i] = ch

    for a in actions.split(';'):
        tmp = a.split('-')
        i = d[int(tmp[0])]
        j = d[int(tmp[1])]
        l = int(float(tmp[2]))
        assert s_label[0, i, j] != l
        assert s_label[0, j, i] != l
        s_label[0, i, j] = l
        s_label[0, j, i] = l
        s_label[1, i, j] = 1
        s_label[1, j, i] = 1

    mask = np.triu(np.ones((2, size, size)).astype(np.int32), 1) - 1
    s_label = np.triu(s_label, 1) + mask

    return s_label
