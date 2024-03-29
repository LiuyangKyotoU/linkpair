import chainer
from rdkit import Chem

from preprocessor import construct_atomic_number_array, \
    construct_atom_feature_matrix, construct_discrete_edge_matrix, construct_supdernode_feature_array, \
    construct_sigmoid_label, construct_softmax_label


def read_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            r, action = line.strip('\r\n ').split()
            if len(r.split('>')) != 3 or r.split('>')[1] != '': raise ValueError('invalid line:', r)
            react = r.split('>')[0]
            product = r.split('>')[-1]
            data.append([react, product, action])
    return data


def pre_data(ind, reaction, whether_rich=True):
    '''
    :param reaction: [CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]>>[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[NH:15][CH3:14] 12-13-0.0;12-15-1.0

        - reactants: [CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]
        - actions: 12-13-0.0;12-15-1.0

    :return:
        - atom_feature
        - adjs
        - supernode_feature
    '''

    reactants = reaction[0]
    actions = reaction[2]

    mol = Chem.MolFromSmiles(reactants)

    atom_array = construct_atomic_number_array(mol)
    atom_feature = construct_atom_feature_matrix(mol, whether_rich)
    adjs = construct_discrete_edge_matrix(mol)
    supernode_feature = construct_supdernode_feature_array(mol, atom_array, adjs)

    label = construct_softmax_label(mol, adjs, actions)

    return atom_feature, adjs, supernode_feature, label, ind


class USPTO_pre(chainer.dataset.DatasetMixin):
    def __init__(self, reaction_list, whether_rich):
        self.reaction_list = reaction_list
        self.pre_list = [None for _ in range(len(self.reaction_list))]
        self.whether_rich = whether_rich

    def __len__(self):
        return len(self.reaction_list)

    def get_example(self, i):
        if self.pre_list[i] is None:
            self.pre_list[i] = pre_data(i, self.reaction_list[i], whether_rich=self.whether_rich)
        return self.pre_list[i]


if __name__ == '__main__':
    data_raw = read_data('../train.txt.proc')
    USPTO_pre = USPTO_pre(data_raw, whether_rich=True)
    sample = USPTO_pre[1]
