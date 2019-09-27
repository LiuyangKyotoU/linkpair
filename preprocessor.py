import numpy as np
from rdkit import Chem
from chainer_chemistry.config import MAX_ATOMIC_NUM

s = '[CH2:1]([CH3:2])[n:3]1[cH:4][c:5]([C:22](=[O:23])[OH:24])[c:6](=[O:21])[c:7]2[cH:8][c:9]([F:20])[c:10](-[c:13]3[cH:14][cH:15][c:16]([NH2:19])[cH:17][cH:18]3)[cH:11][c:12]12.[CH:25](=[O:26])[OH:27]'
m = Chem.MolFromSmiles(s)


def construct_atomic_type_matrix(mol):
    size = mol.GetNumAtoms()
    atom_type = np.zeros((size, MAX_ATOMIC_NUM), dtype=np.float32)

    atom_list = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)
    atom_type[np.arange(size), atom_list] = 1.0

    return atom_type
# TODO: other atom features


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
