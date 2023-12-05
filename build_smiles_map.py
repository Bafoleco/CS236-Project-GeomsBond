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

# harcoded params for now
base_path = "./data/rdkit_folder/"
dataset_name = "qm9"

# read summary file
assert dataset_name in ['qm9', 'drugs']
summary_path = os.path.join(base_path, 'summary_%s.json' % dataset_name)
with open(summary_path, 'r') as f:
    summ = json.load(f)

smiles_map = {}
inverse_smiles_map = {}

for i, tuple in enumerate(tqdm(summ.items())):
    smiles, meta_mol = tuple
    print(smiles)
    smiles_map[smiles] = i
    inverse_smiles_map[i] = smiles

# save smiles map at base_path
with open(os.path.join(base_path, 'smiles_map.json'), 'w') as f:
    json.dump(smiles_map, f)

# save inverse smiles map at base_path
with open(os.path.join(base_path, 'inverse_smiles_map.json'), 'w') as f:
    json.dump(inverse_smiles_map, f)