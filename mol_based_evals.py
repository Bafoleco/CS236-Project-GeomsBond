import json
import os
import pickle

from tqdm import tqdm
from bond_helpers import is_mol_stable
from qm9.rdkit_functions import mol2smiles
from rdkit import Chem
from qm9.rdkit_functions import mol2smiles

def get_qm9_smiles(dataset_name):
    print("\Getting QM9 SMILES ...")

    base_path = "./data/rdkit_folder/"
    dataset_name = "qm9"
    conf_per_mol = 1

    # read summary file
    assert dataset_name in ['qm9', 'drugs']
    summary_path = os.path.join(base_path, 'summary_%s.json' % dataset_name)
    with open(summary_path, 'r') as f:
        summ = json.load(f)

    rdkit_mol_smiles = []

    for _, meta_mol in tqdm(summ.items()):
        u_conf = meta_mol.get('uniqueconfs')
        if u_conf is None:
            continue
        pickle_path = meta_mol.get('pickle_path')
        if pickle_path is None:
            continue
        if u_conf < conf_per_mol:
            continue

        with open(os.path.join(base_path, pickle_path), 'rb') as fin:
            mol = pickle.load(fin)
            mol = mol.get('conformers')[0].get('rd_mol')
            rdkit_mol_smiles.append(mol2smiles(mol))

    return rdkit_mol_smiles

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
            if mol is None:
                continue
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
            if is_mol_stable(mol):
                stable.append(mol)
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
