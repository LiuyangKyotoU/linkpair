import numpy as np
from rdkit import Chem
import chainer_chemistry
from chainer_chemistry.config import MAX_ATOMIC_NUM


def construct_atomic_number_array(mol):
    return np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32)


def assign_atomic_rich_Properties(mol):
    from rdkit.Chem import rdMolDescriptors, EState, rdPartialCharges

    for (i, x) in enumerate(rdMolDescriptors._CalcCrippenContribs(mol)):
        mol.GetAtomWithIdx(i).SetDoubleProp('crippen_logp', x[0])
        mol.GetAtomWithIdx(i).SetDoubleProp('crippen_mr', x[1])
    for (i, x) in enumerate(rdMolDescriptors._CalcTPSAContribs(mol)):
        mol.GetAtomWithIdx(i).SetDoubleProp('tpsa', x)
    for (i, x) in enumerate(rdMolDescriptors._CalcLabuteASAContribs(mol)[0]):
        mol.GetAtomWithIdx(i).SetDoubleProp('asa', x)
    for (i, x) in enumerate(EState.EStateIndices(mol)):
        mol.GetAtomWithIdx(i).SetDoubleProp('estate', x)
    rdPartialCharges.ComputeGasteigerCharges(mol)


def construct_atom_feature_matrix(mol, whether_rich=True):
    atom_simple_features = [list(map(lambda s: a.GetAtomicNum() == s, list(range(MAX_ATOMIC_NUM)))) + \
                            list(map(lambda s: a.GetDegree() == s, [0, 1, 2, 3, 4, 5])) + \
                            list(map(lambda s: a.GetExplicitValence() == s, [1, 2, 3, 4, 5, 6])) + \
                            list(map(lambda s: a.GetImplicitValence() == s, [0, 1, 2, 3, 4, 5])) + \
                            [a.GetIsAromatic()] for a in mol.GetAtoms()]
    atom_simple_features = np.array(atom_simple_features, dtype=np.float32)

    if not whether_rich:
        return atom_simple_features

    assign_atomic_rich_Properties(mol)
    atom_rich_features = [
        [a.GetIsAromatic() == False and any([neighbor.GetIsAromatic() for neighbor in a.GetNeighbors()])] + \
        [a.IsInRing()] + \
        [a.GetAtomicNum() in [9, 17, 35, 53, 85, 117]] + \
        [a.GetAtomicNum() in [8, 16, 34, 52, 84, 116]] + \
        [a.GetAtomicNum() in [7, 15, 33, 51, 83]] + \
        [a.GetAtomicNum() in [3, 11, 19, 37, 55, 87]] + \
        [a.GetAtomicNum() in [4, 12, 20, 38, 56, 88]] + \
        [a.GetAtomicNum() in [13, 22, 24, 25, 26, 27, 28, 29, 30, 33, 42, 44, 45, 46, 47, 48, 49, 50, 78, 80, 82]] + \
        [a.GetDoubleProp('crippen_logp'), a.GetDoubleProp('crippen_mr'),
         a.GetDoubleProp('tpsa'),
         a.GetDoubleProp('asa'), a.GetDoubleProp('estate'),
         a.GetDoubleProp('_GasteigerCharge'), a.GetDoubleProp('_GasteigerHCharge')] for a in mol.GetAtoms()]
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


def construct_label(mol, actions):
    pass



if __name__ == '__main__':
    s = '[CH2:1]([CH3:2])[n:3]1[cH:4][c:5]([C:22](=[O:23])[OH:24])[c:6](=[O:21])[c:7]2[cH:8][c:9]([F:20])[c:10](-[c:13]3[cH:14][cH:15][c:16]([NH2:19])[cH:17][cH:18]3)[cH:11][c:12]12.[CH:25](=[O:26])[OH:27]'
    m = Chem.MolFromSmiles(s)