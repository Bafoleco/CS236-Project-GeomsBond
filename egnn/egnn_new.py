from torch import nn
import torch
import math

class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edge_model_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_model_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        # print("In edge_model, edge_attr has shape", edge_attr.shape)
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)

        mij = self.edge_mlp(out)
            

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        # print("In GCL forward, edge_attr has shape", edge_attr.shape, "and h has shape", h.shape)
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        # return h, mij # Not sure why they would return mij instead of edge_feat
            # when the latter is just the former possibly with attention
            # TODO: bay what do you think?
        return h,edge_feat


class EquivariantUpdate(nn.Module):
    def __init__(self, input_nf, hidden_nf, normalization_factor, aggregation_method,
                 act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        # TODO: Understand why edges_in_d=1 instead of 0. Should correspond
        # to the case when edge_attr is None but gets concatted in coord_model input_tensor?
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        # input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.coord_mlp = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            # nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    # got rid of default 2 on edge_in_nf
    def __init__(self, hidden_nf, edge_in_nf, n_bond_orders, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum', bonds_in=False, bonds_out=False):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.bonds_in = bonds_in # Is this one even necessary?? 
        # TODO remove it (and the kwarg) if its not
        self.bonds_out = bonds_out

        gcl_input_nf = self.hidden_nf
        base_edge_model_in_d = 2 * gcl_input_nf + edge_in_nf
        # input_edge = 256, input_nf = 128 (bc its a hidden size somewhere??), edges_in_d = 2 (why??)
        edge_feat_nf = base_edge_model_in_d + self.hidden_nf

        for i in range(0, n_layers):
            # TODO: modify edge_feat_nf. Right now it is just 2 (hardcoded in original code)
            # corresponding to orig distances + curr distancs. Add ... how many? 
            if i == 0:
                # if self.bonds_in:
                #     edge_model_in_d = base_edge_model_in_d + n_bond_orders
                # else:
                edge_model_in_d = base_edge_model_in_d
                print('WARNING: EBlock hardcoding no bond input for first GCL')
            else:
                edge_model_in_d = edge_feat_nf # GCLs other than first also take hidden_nf-dim edge_feat
                
            self.add_module("gcl_%d" % i, GCL(gcl_input_nf, self.hidden_nf, self.hidden_nf, edge_model_in_d=edge_model_in_d,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(edge_feat_nf, hidden_nf, # edges_in_d=self.hidden_nf, # was edge_feat_nf, TODO do we want this?
                                                       act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))
        
        self.bond_mlp = nn.Sequential(
            nn.Linear(1 + 2 * gcl_input_nf + self.hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, n_bond_orders),
            # nn.Softmax(dim=-1)
        ) # TODO: A way to make this the same across all EquivBlocks? init it higher up and pass it around?
        # or does that happen automatically?

        self.to(self.device)

    def forward(self, h, x, bonds, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        # print('\n\nIn EBlock forward, h.shape =', h.shape,', x.shape =', x.shape, 
        #       'bonds.shape =', bonds.shape if bonds is not None else "it's None!")
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
            
        univ_edge_attr = torch.cat([distances, edge_attr], dim=1) 
        # Universal edge attributes to be passed to ALL GCLs
        # Here, edge_attr is the ORIGINAL distances

        # print('In EBlock forward, univ_edge_attr shape after cat:',univ_edge_attr.shape)
        
        # if self.n_layers == 1:
            # print('Why do we only have a single GCL in our EBlocks???')

        if self.bonds_in and bonds is not None:
            # assert bonds is not None
            edge_attr = torch.cat([univ_edge_attr, bonds],dim=1)
        else:
            edge_attr = univ_edge_attr

        for i in range(0, self.n_layers):
            # print(f'\nAbout to call forward on EBlock GCL layer {i} of {self.n_layers}')
            h, edge_feat = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
            # print(f'in EBlock forward after layer {i} of {self.n_layers}, edge_feat has shape', edge_feat.shape)

            if self.bonds_out or self.bonds_in: # TODO: Check we also want this for bonds in 
                # use the output edge features from the edge model
                # as the edge 
                # just need to play with shapes somewhere and have different shapes for different GCLs in the stack
                edge_attr = torch.cat([univ_edge_attr, edge_feat], dim=1)
                # print(f'after reassignment of edge_feat (in EBlock forward layer {i} of {self.n_layers}), edge_attr has shape', edge_attr.shape)

        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)
        # TODO: make sure edge_attr works here the way we want it to (check shapes!)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask

        final_distances, _ = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            final_distances = self.sin_embedding(final_distances)
        row, col = edge_index
        bond_input = torch.cat([final_distances, h[row], h[col], edge_feat], dim=1)
        bonds = self.bond_mlp(bond_input)

        # print('EBlock return edge_attr has shape', edge_attr.shape)
        return h, x, bonds


class EGNN(nn.Module):
    def __init__(self, in_node_nf, n_bond_orders, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum', 
                 bonds_in=False, bonds_out=False):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers) if n_layers > 0 else float(coords_range)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        # print("n bond orders = ", n_bond_orders)

        # These are probably unnecessary to save as attributes, 
        # TODO remove them (rn I am using for debugging) 
        self.bonds_in = bonds_in 
        self.bonds_out = bonds_out

        if sin_embedding:
            if bonds_in or bonds_out: print('WARNING! Bonds have not been tested with sin embedding!')
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_in_nf=edge_feat_nf, 
                                                               n_bond_orders=n_bond_orders, device=device,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method,
                                                               bonds_in=bonds_in, bonds_out=bonds_out))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, bonds=None):
        # if self.bonds_in and not self.bonds_out:
            # print("\nIn the EGNN for the encoder!")
        # elif not self.bonds_in and self.bonds_out:
            # print("\nIn the EGNN for the decoder!")
        # else:
            # print('\n\nWARNING: NOT USING BONDS!!!\n')
            # raise ValueError(f'I did something wrong with bonds_in/out... in = {self.bonds_in}, out = {self.bonds_out}')

        # print('EGNN input bonds has shape', "its None!" if bonds is None else bonds.shape)
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)

        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, bonds = self._modules["e_block_%d" % i](h, x, bonds, edge_index, node_mask=node_mask, edge_mask=edge_mask, edge_attr=distances)
            # print(f'EBlock {i} bonds has shape',bonds.shape)

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask

        # print('EGNN forward returns bonds with shape', bonds.shape)
        if self.bonds_out:
            return h, x, bonds
        return h,x


class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', device='cpu',
                 act_fn=nn.SiLU(), n_layers=4, attention=False,
                 normalization_factor=1, out_node_nf=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, act_fn=act_fn,
                attention=attention))
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result
