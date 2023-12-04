import time
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType, BondType

type_map = {BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3, BondType.AROMATIC: 4}
inv_type_map = {1: BondType.SINGLE, 2: BondType.DOUBLE, 3: BondType.TRIPLE, 4: BondType.AROMATIC}

def get_mol(adj, charges, n_atoms):
    adj = adj.argmax(dim=-1)[:n_atoms, :n_atoms]
    if charges.min() < 0 or charges.max() > 9:
        return None
    
    if charges[charges == 2].sum() > 0:
        return None
    
    if charges[charges == 3].sum() > 0:
        return None
    
    if charges[charges == 4].sum() > 0:
        return None
    
    if charges[charges == 5].sum() > 0:
        return None
    
    charges = charges[:n_atoms]

    mol = Chem.RWMol()
    for i in range(n_atoms):
        mol.AddAtom(Chem.Atom(int(charges[i])))

    for i in range(n_atoms):
        for j in range(n_atoms):
            if adj[i, j] == 0:
                continue
            bond_type = int(adj[i, j])
            if bond_type != 0 and i < j:
                mol.AddBond(i, j, Chem.BondType(inv_type_map[bond_type]))

    mol.UpdatePropertyCache(strict=False)
    return mol

def get_one_hot_bonds(bonds, n_nodes, n_bond_orders):
    batch_size, max_batch_num_edges, _ = bonds.shape

    adj = torch.zeros((batch_size, n_nodes, n_nodes), dtype=bonds.dtype, device=bonds.device)

    bond_types = bonds[:, :, 0]
    source_nodes = bonds[:, :, 1]
    dest_nodes = bonds[:, :, 2]

    batch_indices = torch.arange(batch_size, device=bonds.device).view(-1, 1).expand(-1, max_batch_num_edges)
    adj[batch_indices, source_nodes, dest_nodes] = bond_types
    adj[batch_indices, dest_nodes, source_nodes] = bond_types

    one_hot_adj = torch.nn.functional.one_hot(adj, num_classes=n_bond_orders)
    return one_hot_adj

def get_bond_edge_attr(bonds, n_nodes, n_bond_orders):
    batch_size, num_edges, bond_params = bonds.shape
    one_hot_adj = get_one_hot_bonds(bonds, n_nodes, n_bond_orders)
    # assert check_one_hot_bonds(one_hot_adj, bonds)
    bond_edge_attr = one_hot_adj.reshape(batch_size * n_nodes * n_nodes, n_bond_orders)
    return bond_edge_attr

def bond_accuracy(bond_rec, bonds_tensor, edge_mask):
    batch_size, n_nodes, n_nodes, _ = bond_rec.shape

    edge_mask = edge_mask.reshape(batch_size, n_nodes, n_nodes)

    # incorrect_bond_predictions = (bond_rec.argmax(dim=-1) != bonds_tensor).float() * edge_mask

    # print("edge mask: ", edge_mask[0])

    assert edge_mask[0][7][7] == 0

    predicted_bonds = bond_rec.argmax(dim=-1)
    actual_bonds = bonds_tensor.argmax(dim=-1)

    no_bonds_mask = (actual_bonds == 0) * edge_mask
    single_bond_mask = (actual_bonds == 1) * edge_mask
    double_bond_mask = (actual_bonds == 2) * edge_mask

    incorrect_bond_predictions = (predicted_bonds != actual_bonds).float() * edge_mask

    print("no bond error: ", (incorrect_bond_predictions * no_bonds_mask).sum() / no_bonds_mask.sum())
    print("single bond error: ", (incorrect_bond_predictions * single_bond_mask).sum() / single_bond_mask.sum())
    print("double bond error: ", (incorrect_bond_predictions * double_bond_mask).sum() / double_bond_mask.sum())

    return incorrect_bond_predictions.sum() / edge_mask.sum()

def get_molecular_stability(bond_rec, charges):
    start = time.time()
    batch_size, _, _, _ = bond_rec.shape

    octet_rule_violations = 0
    for i in range(batch_size):
        # index of last non-zero element
        n_atoms = (charges[i] != 0).sum()

        mol = get_mol(bond_rec[i], charges[i], n_atoms)

        if mol is None:
            octet_rule_violations += 1
            continue

        if len(Chem.GetMolFrags(mol)) > 1:
            octet_rule_violations += 1
            continue

        for atom in mol.GetAtoms():
            # print("type: ", atom.GetSymbol())
            # print("valence: ", atom.GetExplicitValence())
            if atom.GetSymbol() == "N" and atom.GetExplicitValence() != 3:
                octet_rule_violations += 1
                break
            if atom.GetSymbol() == "O" and atom.GetExplicitValence() != 2:
                octet_rule_violations += 1
                break
            if atom.GetSymbol() == "C" and atom.GetExplicitValence() != 4:
                octet_rule_violations += 1
                break
            if atom.GetSymbol() == "F" and atom.GetExplicitValence() != 1:
                octet_rule_violations += 1
                break
            if atom.GetSymbol() == "H" and atom.GetExplicitValence() != 1:
                octet_rule_violations += 1
                break

            # doesn't work for some reason
            # if atom.GetFormalCharge() != 0:
            #     octet_rule_violations += 1
            #     break

    end = time.time()
    # print("octet rule violations time: ", end - start)
    return (batch_size - octet_rule_violations) / batch_size

def valid_fraction(bond_rec, charges):
    start = time.time()
    batch_size, n_nodes, n_nodes, _ = bond_rec.shape

    octet_rule_violations = 0
    for i in range(batch_size):
        # index of last non-zero element
        n_atoms = (charges[i] != 0).sum()

        mol = get_mol(bond_rec[i], charges[i], n_atoms)

        if len(Chem.GetMolFrags(mol)) > 1:
            octet_rule_violations += 1
            continue

        try:
            Chem.SanitizeMol(mol)
        except ValueError:  
            print("octet rule violation")
            octet_rule_violations += 1

    end = time.time()
    print("octet rule violations time: ", end - start)
    return (batch_size - octet_rule_violations) / batch_size

def octet_rule_violations_old(bond_rec, charges):
    predicted_bonds = bond_rec.argmax(dim=-1).float()
    # print("predicted bonds: ", predicted_bonds[0])

    predicted_bonds[predicted_bonds == 4] = 1.5
    predicted_bonds = predicted_bonds.sum(dim=-1).unsqueeze(-1)

    # print("predicted bonds: ", predicted_bonds)
    # print("charges: ", charges)

    hydrogens = (charges == 1)
    carbons = (charges == 6)
    nitrogens = (charges == 7)
    oxygens = (charges == 8)
    fluorines = (charges == 9)

    correct_h = (predicted_bonds[hydrogens] == 1).sum()
    correct_c = (predicted_bonds[carbons] == 4).sum()
    correct_n = (predicted_bonds[nitrogens] == 3).sum()
    correct_o = (predicted_bonds[oxygens] == 2).sum()
    correct_f = (predicted_bonds[fluorines] == 1).sum()

    total_h = hydrogens.sum()
    total_c = carbons.sum()
    total_n = nitrogens.sum()
    total_o = oxygens.sum()
    total_f = fluorines.sum()

    print("hydrogen octet rule accuracy: ", correct_h / total_h)
    print("carbon octet rule accuracy: ", correct_c / total_c)
    print("nitrogen octet rule accuracy: ", correct_n / total_n)
    print("oxygen octet rule accuracy: ", correct_o / total_o)
    print("fluorine octet rule accuracy: ", correct_f / total_f)

    total_correct = correct_h + correct_c + correct_n + correct_o + correct_f
    total = total_h + total_c + total_n + total_o + total_f

    print("total octet rule accuracy: ", total_correct / total)

    return total_correct / total


def check_adj_bonds(adj, bonds):
    batch_size, n_edges, _ = bonds.shape
    for i in range(batch_size):
        for j in range(n_edges):
            bond_type = bonds[i, j, 0]
            source_node = bonds[i, j, 1]
            dest_node = bonds[i, j, 2]

            if adj[i, source_node, dest_node] != bond_type:
                print(f"Error at {i}, {j}")
                print(f"s: {source_node}, d: {dest_node}")  
                print(f"Bond type: {bond_type}")
                print(f"Adj: {adj[i, source_node, dest_node]}")
                return False
    return True

def check_one_hot_bonds(adj, bonds):
    batch_size, n_edges, _ = bonds.shape

    for i in range(batch_size):
        for j in range(n_edges):
            bond_type = bonds[i, j, 0]
            source_node = bonds[i, j, 1]
            dest_node = bonds[i, j, 2]

            if adj[i, source_node, dest_node].argmax() != bond_type:
                print(f"Error at {i}, {j}")
                print(f"Bond type: {bond_type}")
                print(f"Adj: {adj[i, source_node, dest_node]}")
                print(f"Adj argmax: {adj[i, source_node, dest_node].argmax()}")
                return False
    return True

# just for reference
def process_bonds(self, bonds, n_nodes, n_bond_orders):
    # assuming we have bonds in batch_size x max_num_bonds x bond_desc form
    # TODO we need proper masking as we are currently padding the edge set
    # TODO this can definitely be vectorized somehow
    # TODO we may want to test this

    # print("processing bonds: ", bonds.shape)

    batch_size, num_edges, bond_params = bonds.shape
    total_n_edges = n_nodes * (n_nodes - 1) // 2

    # print(type(total_n_edges))

    # for batch_idx in range(batch_size):
    #     for i in range(n_nodes):
    #         for j in range(n_nodes):
    #             rows.append(i + batch_idx * n_nodes)
    #             cols.append(j + batch_idx * n_nodes)

    ## 1 1 1 1 2 2 2 2 3 3 3 3
    ## 1 2 3 4 1 2 3 4 1 2 3 4
    print(bonds.device)
    bond_edge_attr = torch.zeros((batch_size * n_nodes * n_nodes, n_bond_orders), device=bonds.device, dtype=bonds.dtype)
    # bond_tensor = torch.zeros((batch_size, total_n_edges, n_bond_orders), device=bonds.device)
    # default unbonded
    # bond_edge_attr[:, 0] = 1
    for batch_idx in range(batch_size):
        # I'm kind of convinced we don't need this loop over batch_size, and can
        # just have a first dim corresponding to batch size and then reshape later?
        for i in range(num_edges):
            bond = bonds[batch_idx, i]
            data = bond[0] # Assuming this is just an int with bond order? -- DW
            start = bond[1]
            end = bond[2]

            if data == 0:
                continue

            assert data < n_bond_orders and data >= 0,'If this fails, just index into data to get the bond order. We can then try including more bond parameters later!'

            # Modified this to be one-hot over bond order
            bond_edge_attr[start * n_nodes + end + batch_idx * n_nodes * n_nodes, data] = 1
            # bond_edge_attr[start * n_nodes + end + batch_idx * n_nodes * n_nodes, 0] = 0

            # bond_edge_attr[end * n_nodes + start + batch_idx * n_nodes * n_nodes, data] = 1

        # This part can also probably be done faster using some vectorized torch.triu_indices
        # triu_index_to_ij = [(i,j) for i in range(1,n_nodes) for j in range(i)]
        # print(n_nodes * n_nodes)
        # print(len(triu_index_to_ij))
        # print(total_n_edges)
        # assert len(triu_index_to_ij) == total_n_edges
        # for triu_idx in range(total_n_edges):
        #     i,j = triu_index_to_ij[triu_idx]
        #     bond_tensor[batch_idx,triu_idx] = bond_edge_attr[i * n_nodes + j + batch_idx * n_nodes * n_nodes]
        # Should correspond to 
        # bond_tensor = torch.tensor([bond_matrix[:,i,j] for i in range(1,n_nodes) for j in range(i)])
    
    # print("bond tensor: ", bond_tensor)

    # vectorized approach
    one_hot_adj = get_one_hot_bonds(bonds, n_nodes, n_bond_orders)
    assert check_one_hot_bonds(one_hot_adj, bonds)
    bond_edge_attr_vec = one_hot_adj.reshape(batch_size * n_nodes * n_nodes, n_bond_orders)

    assert bond_edge_attr_vec.shape == bond_edge_attr.shape

    # print("bond edge attr: ", bond_edge_attr[0:50])
    # print("bond edge attr vec: ", bond_edge_attr_vec[0:50])


    # for i in range(batch_size * n_nodes * n_nodes):
    #     print(i)
    #     print(bond_edge_attr[i])
    #     print(bond_edge_attr_vec[i])
    #     assert bond_edge_attr[i].allclose(bond_edge_attr_vec[i])

    assert bond_edge_attr_vec.allclose(bond_edge_attr)

    return bond_edge_attr, None
