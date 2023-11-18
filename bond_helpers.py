import torch

def get_one_hot_bonds(bonds, n_nodes, n_bond_orders):
    batch_size, max_batch_num_edges, _ = bonds.shape

    adj = torch.zeros((batch_size, n_nodes, n_nodes), dtype=bonds.dtype, device=bonds.device)

    bond_types = bonds[:, :, 0]
    source_nodes = bonds[:, :, 1]
    dest_nodes = bonds[:, :, 2]

    batch_indices = torch.arange(batch_size, device=bonds.device).view(-1, 1).expand(-1, max_batch_num_edges)
    adj[batch_indices, source_nodes, dest_nodes] = bond_types

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
