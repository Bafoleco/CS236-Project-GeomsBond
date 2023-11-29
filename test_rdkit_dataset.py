from collections import defaultdict
import json
import os
import random

import torch
from qm9 import dataset

class Args:
    def __init__(self, rdkit) -> None:
        self.rdkit = rdkit
        self.dataset = 'qm9'
        self.batch_size = 64
        self.num_workers = 1
        self.filter_n_atoms = None
        self.datadir = 'qm9/temp'
        self.remove_h = False
        self.include_charges = True

orig_args = Args(rdkit=False)
rdkit_args = Args(rdkit=True)

base_path = "./data/rdkit_folder/"

# load smiles map
path = os.path.join(base_path, 'smiles_map.json')
if os.path.exists(path):
    with open(path, 'r') as f:
        smiles_map = json.load(f)
else:
    print("please run build_smiles_map.py first")
    exit(-1)

# load inverse smiles map
path = os.path.join(base_path, 'inverse_smiles_map.json')
if os.path.exists(path):
    with open(path, 'r') as f:
        inverse_smiles_map = json.load(f)
else:
    print("please run build_smiles_map.py first")
    exit(-1)

print(len(inverse_smiles_map))
print(list(inverse_smiles_map.keys())[10])
print(list(inverse_smiles_map.keys())[11])
print(type(list(inverse_smiles_map.keys())[11]))

original_dataloaders, charge_scale = dataset.retrieve_dataloaders(orig_args)
original_loader = original_dataloaders['train']

rdkit_dataloaders, charge_scale = dataset.retrieve_dataloaders(rdkit_args)
rdkit_loader = rdkit_dataloaders['train']

smiles_to_data = defaultdict(lambda: {})

print("about to load rdkit data")
rdkit_mols = 0
for i, batch in enumerate(rdkit_loader):
    for i, smiles in enumerate(batch['smiles']):
        smiles_to_data[str(smiles.item())]["rdkit"] = {key: batch[key][i] for key in batch.keys()}
        rdkit_mols += 1

        # check bonds
        num_atoms = batch['num_atoms'][i]
        for bond in batch['bonds'][i]:
            bond_type = bond[0].item()
            atom1 = bond[1].item()
            atom2 = bond[2].item()

            if atom1 >= num_atoms or atom2 >= num_atoms:
                assert bond_type == 0
            if bond_type != 1 and bond_type != 0:
                atom_type1 = batch['charges'][i][atom1].item()
                atom_type2 = batch['charges'][i][atom2].item()
                assert atom_type1 != 1 and atom_type2 != 1
                assert atom_type1 != 9 and atom_type2 != 9

print(f"Loaded {rdkit_mols} molecules")

original_mols = 0
print("about to load original data")
for i, batch in enumerate(original_loader):
    for i, smiles in enumerate(batch['smiles']):
        smiles_to_data[str(smiles.item())]["orig"] = {key: batch[key][i] for key in batch.keys()}
        original_mols += 1

print(f"Loaded {original_mols} molecules")

test_count = 0
for smiles, data in smiles_to_data.items():
    if smiles == -1:
        continue

    if "orig" not in data:
        # print(f"Missing original data for {smiles}")
        continue
    if "rdkit" not in data:
        # print(f"Missing rdkit data for {smiles}")
        continue

    test_count += 1

    print(inverse_smiles_map[smiles])

    orig_data = data["orig"]
    rdkit_data = data["rdkit"]

    assert orig_data["num_atoms"] == rdkit_data["num_atoms"]
    assert orig_data["charges"].sum() == rdkit_data["charges"].sum()

    # print(orig_data["positions"].sum())
    # print(rdkit_data["positions"].sum())

    print(orig_data["positions"].sum(dim=0))
    print(rdkit_data["positions"].sum(dim=0))

    # assert orig_data["positions"].sum() == rdkit_data["positions"].sum() - this fails

    # trim padding off charges
    # trimmed_orig_charges = orig_data["charges"][:orig_data["num_atoms"]]
    # trimmed_rdkit_charges = rdkit_data["charges"][:rdkit_data["num_atoms"]]
    # assert torch.allclose(trimmed_orig_charges, trimmed_rdkit_charges) - this fails :(

print(f"Tested {test_count} molecules")
