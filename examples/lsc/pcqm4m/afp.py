import torch
import torch.nn.functional as F
from torch.nn import Linear, GRUCell
from torch_geometric.nn import global_add_pool, GATConv
from torch_geometric.nn.models.attentive_fp import GATEConv
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder


class AttentiveFP(torch.nn.Module):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.

    Args:
        emb_dim (int): Hidden node feature dimensionality.
        num_tasks (int): Size of each output sample.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        drop_ratio (float, optional): Dropout probability. (default: :obj:`0.0`)

    """
    def __init__(self, num_timesteps=4, emb_dim = 300, num_layers = 5, drop_ratio = 0, num_tasks = 1, **args):
        super(AttentiveFP, self).__init__()

        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.drop_ratio = drop_ratio

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

        conv = GATEConv(emb_dim, emb_dim, emb_dim, drop_ratio)
        gru = GRUCell(emb_dim, emb_dim)
        self.atom_convs = torch.nn.ModuleList([conv])
        self.atom_grus = torch.nn.ModuleList([gru])
        for _ in range(num_layers - 1):
            conv = GATConv(emb_dim, emb_dim, dropout=drop_ratio,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(emb_dim, emb_dim))

        self.mol_conv = GATConv(emb_dim, emb_dim,
                                dropout=drop_ratio, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_gru = GRUCell(emb_dim, emb_dim)

        self.graph_pred_linear = Linear(emb_dim, num_tasks)

        self.reset_parameters()

    def reset_parameters(self):
        # self.atom_encoder.reset_parameters() # reset in init()
        # self.bond_encoder.reset_parameters() # reset in init()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.graph_pred_linear.reset_parameters()

    def forward(self, batched_data):
        """"""
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        # Atom Embedding:
        x = F.leaky_relu_(self.atom_encoder(x))
        edge_attr = self.bond_encoder(edge_attr)

        h = F.elu_(self.atom_convs[0](x, edge_index, edge_attr))
        h = F.dropout(h, p=self.drop_ratio, training=self.training)
        x = self.atom_grus[0](h, x).relu_()

        for conv, gru in zip(self.atom_convs[1:], self.atom_grus[1:]):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.drop_ratio, training=self.training)
            x = gru(h, x).relu_()

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.drop_ratio, training=self.training)
            out = self.mol_gru(h, out).relu_()

        # Predictor:
        out = F.dropout(out, p=self.drop_ratio, training=self.training)
        return self.graph_pred_linear(out)
