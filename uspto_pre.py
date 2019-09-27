from rdkit import Chem
from rdkit.Chem import rdmolops
from chainer_chemistry.config import MAX_ATOMIC_NUM
import numpy as np

import chainer
import chainer_chemistry

# from chainer_chemistry.dataset.preprocessors.common \
#     import construct_atomic_number_array, construct_discrete_edge_matrix, construct_supernode_feature
from chainer_chemistry.dataset.preprocessors import gwm_preprocessor

s = '[B:22]([O:23][CH:24]([CH3:25])[CH3:26])([O:27][CH:28]([CH3:29])[CH3:30])[O:31][CH:32]([CH3:33])[CH3:34]'
m = Chem.MolFromSmiles(s)
preprocessor = gwm_preprocessor.GGNNGWMPreprocessor()
atom_array, adj_array, super_node_x = preprocessor.get_input_features(m)

atom_array = construct_atomic_number_array(m)
adj_array = construct_discrete_edge_matrix(m)
super_node = construct_supernode_feature(m, atom_array, adj_array)


def pre_data(reaction, ind):
    '''
    :param reaction: [CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]>>[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[NH:15][CH3:14] 12-13-0.0;12-15-1.0

        - reactants: [CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]
        - actions: 12-13-0.0;12-15-1.0

    :return:
    '''

    reactants = reaction[0]
    actions = reaction[2]






class USPTO_dataset(chainer.dataset.DatasetMixin):
    def __init__(self, reaction_list):
        self.reaction_list = reaction_list
        self.pre_list = [None for _ in range(len(self.reaction_list))]

    def __len__(self):
        return len(self.reaction_list)

    def get_example(self, i):
        if self.pre_list[i] is None:
            self.pre_list[i] = pre_data(self.reaction_list[i], i)
        return self.pre_list[i]
