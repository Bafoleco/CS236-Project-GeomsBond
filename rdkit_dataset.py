import json
import os
import numpy as np
import random
from tqdm import tqdm
import pickle
import rdkit
from rdkit.Chem.rdchem import Mol, HybridizationType, BondType
import torch
from torch.nn.utils.rnn import pad_sequence

def get_rdkit_datafiles(dataset):
    base_path = "./data/rdkit_folder/"
    if dataset == 'qm9':
        return {'train': os.path.join(base_path, 'prepped', 'qm9_train.npz'),
                'valid': os.path.join(base_path, 'prepped', 'qm9_valid.npz'),
                'test': os.path.join(base_path, 'prepped', 'qm9_test.npz')}

## Adapted from ContGF
def get_rdkit_dataloader(args, seed=None, stack=True):
    """
    base_path: directory that contains GEOM dataset
    dataset_name: dataset name in [qm9, drugs]
    conf_per_mol: keep mol that has at least conf_per_mol confs, and sampling the most probable conf_per_mol confs
    train_size ratio, val = test = (1-train_size) / 2
    tot_mol_size: max num of mols. The total number of final confs should be tot_mol_size * conf_per_mol
    seed: rand seed for RNG
    """
    
    # harcoded params for now
    base_path = "./data/rdkit_folder/"
    dataset_name = "qm9"
    conf_per_mol = 1
    train_size = 0.8
    tot_mol_size = 30000

    # set random seed
    if seed is None:
        seed = 2021
    np.random.seed(seed)
    random.seed(seed)
    
    # read summary file
    assert dataset_name in ['qm9', 'drugs']
    summary_path = os.path.join(base_path, 'summary_%s.json' % dataset_name)
    with open(summary_path, 'r') as f:
        summ = json.load(f)

    # filter valid pickle path
    smiles_list = []
    pickle_path_list = []
    num_mols = 0    
    num_confs = 0    
    for smiles, meta_mol in tqdm(summ.items()):
        u_conf = meta_mol.get('uniqueconfs')
        if u_conf is None:
            continue
        pickle_path = meta_mol.get('pickle_path')
        if pickle_path is None:
            continue
        if u_conf < conf_per_mol:
            continue
        num_mols += 1
        num_confs += conf_per_mol
        smiles_list.append(smiles)
        pickle_path_list.append(pickle_path)
    
    random.shuffle(pickle_path_list)

    assert len(pickle_path_list) >= tot_mol_size, 'the length of all available mols is %d, which is smaller than tot mol size %d' % (len(pickle_path_list), tot_mol_size)
    pickle_path_list = pickle_path_list[:tot_mol_size]

    train_data, val_data, test_data = [], [], []
    val_size = test_size = (1. - train_size) / 2

    # generate train, val, test split indexes
    split_indexes = list(range(tot_mol_size))
    random.shuffle(split_indexes)
    index2split = {}
    for i in range(0, int(len(split_indexes) * train_size)):
        index2split[split_indexes[i]] = 'train'
    for i in range(int(len(split_indexes) * train_size), int(len(split_indexes) * (train_size + val_size))):
        index2split[split_indexes[i]] = 'val'
    for i in range(int(len(split_indexes) * (train_size + val_size)), len(split_indexes)):
        index2split[split_indexes[i]] = 'test'        


    num_mols = np.zeros(4, dtype=int) # (tot, train, val, test)
    num_confs = np.zeros(4, dtype=int) # (tot, train, val, test)

    bad_case = 0
    for i in tqdm(range(len(pickle_path_list))):
        
        with open(os.path.join(base_path, pickle_path_list[i]), 'rb') as fin:
            mol = pickle.load(fin)
        
        if mol.get('uniqueconfs') > len(mol.get('conformers')):
            bad_case += 1
            continue
        if mol.get('uniqueconfs') <= 0:
            bad_case += 1
            continue

        datas = []
        smiles = mol.get('smiles')

        if mol.get('uniqueconfs') == conf_per_mol:
            # use all confs
            conf_ids = np.arange(mol.get('uniqueconfs'))
        else:
            # filter the most probable 'conf_per_mol' confs
            all_weights = np.array([_.get('boltzmannweight', -1.) for _ in mol.get('conformers')])
            descend_conf_id = (-all_weights).argsort()
            conf_ids = descend_conf_id[:conf_per_mol]

        for conf_id in conf_ids:
            conf_meta = mol.get('conformers')[conf_id]
            data = rdmol_to_data(conf_meta.get('rd_mol'), smiles=smiles)
            labels = {
                'totalenergy': conf_meta['totalenergy'],
                'boltzmannweight': conf_meta['boltzmannweight'],
            }
            for k, v in labels.items():
                data[k] = torch.tensor(v, dtype=torch.float32)
            data['idx'] = torch.tensor(i, dtype=torch.long)
            datas.append(data)
        assert len(datas) == conf_per_mol

        if index2split[i] == 'train':
            train_data.extend(datas)
            num_mols += [1, 1, 0, 0]
            num_confs += [len(datas), len(datas), 0, 0]
        elif index2split[i] == 'val':    
            val_data.extend(datas)
            num_mols += [1, 0, 1, 0]
            num_confs += [len(datas), 0, len(datas), 0]
        elif index2split[i] == 'test': 
            test_data.extend(datas)
            num_mols += [1, 0, 0, 1]
            num_confs += [len(datas), 0, 0, len(datas)] 
        else:
            raise ValueError('unknown index2split value.')    

    # Check that all molecules have the same set of items in their dictionary:
    props = train_data[0].keys()

    splits = [('train', train_data), ('valid', val_data), ('test', test_data)]
              
    for split, data in splits:
        # Convert list-of-dicts to dict-of-lists
        molecules = {prop: [mol[prop] for mol in data] for prop in props}

        # If stacking is desireable, pad and then stack.
        if stack:
            stacked_molecules = {}
            for key, val in molecules.items():
                print("key", key, val[0].shape, val[1].shape)
                # if key == "bonds":
                #     continue
                if val[0].dim() > 0:
                    stacked_molecules[key] = pad_sequence(val, batch_first=True)
                else:
                    stacked_molecules[key] = torch.stack(val)

            # molecules = {key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 else torch.stack(val) for key, val in molecules.items()}

        savedir = os.path.join(base_path, 'prepped', f'qm9_{split}.npz')
        np.savez_compressed(savedir, **stacked_molecules)

    print('post-filter: find %d molecules with %d confs' % (num_mols[0], num_confs[0]))    
    print('train size: %d molecules with %d confs' % (num_mols[1], num_confs[1]))    
    print('val size: %d molecules with %d confs' % (num_mols[2], num_confs[2]))    
    print('test size: %d molecules with %d confs' % (num_mols[3], num_confs[3]))    
    print('bad case: %d' % bad_case)
    print('done!')

    return train_data, val_data, test_data, index2split

charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
all_species = torch.tensor([0, 1, 6, 7, 8, 9])

def bond_type_to_int(bond_type):
    type_map = {BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3, BondType.AROMATIC: 4}
    if bond_type not in type_map:
        print("uknown bond type: " + bond_type)
        exit(-1)
    return type_map[bond_type]

def rdmol_to_data(mol:Mol, smiles=None):
    assert mol.GetNumConformers() == 1
    N = mol.GetNumAtoms()

    ## Attempting to match the properties of the GeoLDM dataset but from a different source
    ## Doesn't handle hydrogen removal logic
    pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)

    data = {}

    # print("molecule has: ", len(mol.GetBonds()), " bonds and ", mol.GetNumAtoms(), " atoms.")

    # for bond in mol.GetBonds():
    #     print(bond.GetBondType())

    data['positions'] = pos
    data['charges'] = torch.tensor([charge_dict[atom.GetSymbol()] for atom in mol.GetAtoms()])
    data['bonds'] = torch.tensor([(bond_type_to_int(bond.GetBondType()), bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) 
              for bond in mol.GetBonds()])

    data['one_hot'] = data['charges'].unsqueeze(-1) == all_species.unsqueeze(0)

    # print(data['one_hot'])
    return data


if __name__ == "__main__": 
    get_rdkit_dataloader(None)
