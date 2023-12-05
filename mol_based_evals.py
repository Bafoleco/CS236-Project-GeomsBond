import json
import os
import pickle
from bond_helpers import is_mol_stable
from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.rdkit_functions import mol2smiles
from rdkit import Chem
from qm9.rdkit_functions import mol2smiles

def get_qm9_smiles(dataset_name):
    print("\Getting QM9 SMILES ...")

    remove_h = False
    class StaticArgs:
        def __init__(self, dataset, remove_h):
            self.dataset = dataset
            self.batch_size = 1
            self.num_workers = 1
            self.filter_n_atoms = None
            self.datadir = 'qm9/temp'
            self.remove_h = remove_h
            self.include_charges = True
            self.rdkit = True
    args_dataset = StaticArgs(dataset_name, remove_h)
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args_dataset)
    dataset_info = get_dataset_info(args_dataset.dataset, args_dataset.remove_h)
    n_types = 4 if remove_h else 5
    mols_smiles = []

    # smiles map
    base_path = "./data/rdkit_folder/"
    path = os.path.join(base_path, 'inv_smiles_map.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            inv_smiles_map = json.load(f)
    else:
        print("please run build_smiles_map.py first")
        exit(-1)

    for i, data in enumerate(dataloaders['train']):
        smiles_id = data['smiles'][0].item()
        mols_smiles.append(inv_smiles_map[str(smiles_id)])

        if i % 1000 == 0:
            print("\tConverting QM9 dataset to SMILES {0:.2%}".format(float(i)/len(dataloaders['train'])))
    return mols_smiles


def retrieve_qm9_smiles(dataset_info):
    dataset_name = dataset_info['name']
    assert dataset_info['with_h']
    file_name = 'qm9/temp/%s_rdkit_smiles.pickle' % dataset_name
    try:
        with open(file_name, 'rb') as f:
            qm9_smiles = pickle.load(f)
        return qm9_smiles
    except OSError:
        try:
            os.makedirs('qm9/temp')
        except:
            pass
        qm9_smiles = get_qm9_smiles(dataset_name)
        with open(file_name, 'wb') as f:
            pickle.dump(qm9_smiles, f)
        return qm9_smiles

class BasicMolBasedMetrics(object):
    def __init__(self, dataset_info, dataset_smiles_list=None):
        self.atom_decoder = dataset_info['atom_decoder']
        self.dataset_smiles_list = dataset_smiles_list
        self.dataset_info = dataset_info

        # Retrieve dataset smiles only for qm9 currently.
        if dataset_smiles_list is None and 'qm9' in dataset_info['name']:
            self.dataset_smiles_list = retrieve_qm9_smiles(self.dataset_info)

    def compute_validity(self, generated):
        valid = []

        for mol in generated:
            smiles = mol2smiles(mol)
            if smiles is not None:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)
                valid.append(smiles)

        return valid, len(valid) / len(generated)

    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)
    
    def compute_molecular_stability(self, generated):
        stable = []
        for mol in generated:
            stable.append(is_mol_stable(mol))
        return stable, len(stable) / len(generated)

    def evaluate(self, generated):
        valid, validity = self.compute_validity(generated)
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = 0.0
        else:
            novelty = 0.0
            uniqueness = 0.0
            unique = None

        # new eval
        stable, stability = self.compute_molecular_stability(generated)
        print(f"Stability over {len(generated)} molecules: {stability * 100 :.2f}%")

        return [validity, uniqueness, novelty, stability], unique
