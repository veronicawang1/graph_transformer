# import libraries and modules
import utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math

# identical to EvolveGCN-o except GCN weight matrices are evolved over time by a
# Transformer encoder instead of a first-order recurrent cell. The GCN 
# backbone, features, and output dimensions are not touched, so this is a clean swap.
class EGCN(torch.nn.Module):
    def __init__(self, args, activation, device='cpu', skipfeats=False):
        super().__init__()
        feats = [args.feats_per_node,
                 args.layer_1_feats,
                 args.layer_2_feats]
        self.device = device
        self.skipfeats = skipfeats

        # hyperparameters
        t_args = u.Namespace({
            'window':   getattr(args, 'egcn_t_window', 5),
            'heads':    getattr(args, 'egcn_t_heads', 1),
            'layers':   getattr(args, 'egcn_t_layers', 1),
            'ff':       getattr(args, 'egcn_t_ff', None),
            'dropout':  getattr(args, 'egcn_t_dropout', 0.0),
            'residual': getattr(args, 'egcn_t_residual', True),
        })

        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        for i in range(1, len(feats)):
            GRCU_args = u.Namespace({'in_feats':  feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': activation,
                                     't_args': t_args})
            grcu_i = GRCU(GRCU_args)
            self.GRCU_layers.append(grcu_i.to(self.device))
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

    def parameters(self):
        return self._parameters

    def forward(self, A_list, Nodes_list, nodes_mask_list):
        node_feats = Nodes_list[-1]

        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list, Nodes_list)

        out = Nodes_list[-1]
        if self.skipfeats:
            out = torch.cat((out, node_feats), dim=1)
        return out


class GRCU(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.activation = self.args.activation

        # weight matrix
        self.GCN_init_weights = Parameter(torch.Tensor(args.in_feats, args.out_feats))
        self.reset_param(self.GCN_init_weights)

        # temporal aggregator of weight matrices
        self.evolve_weights = mat_Transformer_cell(rows=args.in_feats,
                                                   cols=args.out_feats,
                                                   t_args=args.t_args)

    def reset_param(self, t):
        # initialization
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, A_list, node_embs_list):
        # history of weight matrices
        weight_hist = [self.GCN_init_weights]
        window = self.args.t_args.window
        out_seq = []
        for t, Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            GCN_weights = self.evolve_weights(weight_hist[-window:])
            weight_hist.append(GCN_weights)
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))
            out_seq.append(node_embs)

        return out_seq

# Transformer cell that evolves GCN weight matrices over time. Each weight matrix
# is # of cols tokens of dimension # of rows.
class mat_Transformer_cell(torch.nn.Module):
    def __init__(self, rows, cols, t_args):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.window = t_args.window
        self.residual = t_args.residual

        # head count must divide cols for multihead attention, defaults is 1
        heads = t_args.heads
        if rows % heads != 0:
            heads = 1
        ff = t_args.ff if t_args.ff is not None else max(2 * rows, 16)

        encoder_layer = nn.TransformerEncoderLayer(d_model=rows,
                                                   nhead=heads,
                                                   dim_feedforward=ff,
                                                   dropout=t_args.dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=t_args.layers)

        # learned positional encoding for up to 'window' steps, added to token embeddings 
        # before transformer
        self.pos = Parameter(torch.zeros(self.window, 1, rows))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, weight_hist):
        # weight_hist
        L = len(weight_hist)
        # stack into tensor of shape [L, rows, cols], then permute to [cols, L, rows] 
        # for transformer (tokens are cols, sequence length is L, embedding dim is rows)
        seq = torch.stack(weight_hist, dim=0)
        seq = seq.permute(2, 0, 1)

        # add positional encoding to token embeddings. pos is [window, 1, rows], 
        # so it broadcasts over batch (cols) dimension
        pos = self.pos[:L].permute(1, 0, 2)
        seq = seq + pos

        # transformer expects input of shape [batch, seq_len, embed_dim], returns same shape
        # we take output at last step in sequence as new weight matrix, of shape [cols, rows], 
        # and transpose it
        out = self.transformer(seq)
        new_W = out[:, -1, :].t()

        if self.residual:
            new_W = new_W + weight_hist[-1]

        return new_W