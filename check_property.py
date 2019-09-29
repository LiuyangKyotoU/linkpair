from rdkit import Chem


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


def check_bondtype_change(reactions):
    '''
    bond 可能有断开、合上（三种）、变更（三种），这里验证变更
    '''
    from rdkit import RDLogger
    rdl = RDLogger.logger()
    rdl.setLevel(RDLogger.CRITICAL)

    reactants = reactions[0]
    actions = reactions[2]

    mol = Chem.MolFromSmiles(reactants)

    bond_type_to_channel = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3
    }
    actions_dict = {}
    for a in actions.split(';'):
        tmp = a.split('-')
        actions_dict[str(min(int(tmp[0]), int(tmp[1]))) + '-' + str(max(int(tmp[0]), int(tmp[1])))] = int(
            float(tmp[2]) - 1)
    for bond in mol.GetBonds():
        ch = bond_type_to_channel[bond.GetBondType()]
        i = bond.GetBeginAtom().GetAtomMapNum()
        j = bond.GetEndAtom().GetAtomMapNum()
        key = str(min(i, j)) + '-' + str(max(i, j))
        if key in list(actions_dict.keys()) and actions_dict[key] != -1:
            return reactions

    return ['', '', '']


if __name__ == '__main__':
    data_raw = read_data('../train.txt.proc')
    for i in data_raw:
        a = check_bondtype_change(i)
        if a[2] != '':
            print(i)
            break