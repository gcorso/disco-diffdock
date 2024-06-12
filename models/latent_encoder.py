import copy
import torch
from e3nn import o3
from torch import nn
from torch_cluster import radius_graph, radius
from torch_geometric.nn import TransformerConv
from torch_scatter import scatter_mean
import torch.nn.functional as F

from datasets_utils.process_mols import lig_feature_dims, rec_residue_feature_dims
from models.layers import AtomEncoder, gumbel_softmax, FCBlock
from models.model_classes import GenericEncoder
from models.tensor_layers import GaussianSmearing, TensorProductConvLayer, get_irrep_seq


class TPEncoder(GenericEncoder):
    def __init__(self, device, latent_dim, latent_vocab, in_lig_edge_features=4, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 use_second_order_repr=False, batch_norm=True, dropout=0.0, lm_embedding_type=None,
                 latent_no_batchnorm=False, latent_dropout=0.0, latent_hidden_dim=128,
                 use_oracle=True, input_latent_dim=0, apply_gumbel_softmax=True, latent_virtual_nodes=False,
                 latent_nodes_residual=False):
        super(TPEncoder, self).__init__()

        self.in_lig_edge_features = in_lig_edge_features
        sigma_embed_dim = 0
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.device = device
        self.no_torsion = no_torsion
        self.num_conv_layers = num_conv_layers
        self.latent_dim = latent_dim
        self.latent_vocab = latent_vocab
        self.latent_temperature = 1.0   # this gets reset by the different training and evaluation routines
        self.use_oracle = use_oracle
        self.input_latent_dim = input_latent_dim
        self.apply_gumbel_softmax = apply_gumbel_softmax
        self.latent_virtual_nodes = latent_virtual_nodes
        self.latent_nodes_residual = latent_nodes_residual

        latent_factor = 0 if latent_virtual_nodes else 1

        self.lig_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim,
                                              latent_dim=input_latent_dim * (latent_vocab + 1) * latent_factor)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim +
                                                          input_latent_dim * (max(latent_vocab, 2) + 1) * latent_factor, ns),
                                                nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.rec_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=sigma_embed_dim,
                                              lm_embedding_type=lm_embedding_type, latent_dim=input_latent_dim * (latent_vocab + 1) * latent_factor)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim + input_latent_dim * (max(latent_vocab, 2) + 1) * latent_factor,
                                                          ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.cross_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim + input_latent_dim * (max(latent_vocab, 2) + 1) * latent_factor,
                                                            ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        irrep_seq = get_irrep_seq(ns, nv, use_second_order_repr)

        conv_layers = []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            layer = TensorProductConvLayer(
                in_irreps=in_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=out_irreps,
                n_edge_features=3 * ns,
                hidden_features=3 * ns,
                residual=True,
                batch_norm=batch_norm,
                dropout=dropout,
                faster=sh_lmax == 1 and not use_second_order_repr,
                edge_groups=4,
            )
            conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(conv_layers)

        if self.latent_virtual_nodes:
            self.initial_virtual_node_attr = nn.Parameter(torch.zeros(max(self.latent_dim, input_latent_dim), self.ns), requires_grad=True)
            nn.init.xavier_uniform_(self.initial_virtual_node_attr)
            self.virtual_edge_attr = nn.Parameter(torch.zeros(max(self.latent_dim, input_latent_dim), self.ns), requires_grad=True)
            nn.init.xavier_uniform_(self.virtual_edge_attr)
            self.complex_edge_attr = nn.Parameter(torch.zeros(2, self.ns), requires_grad=True)
            nn.init.xavier_uniform_(self.complex_edge_attr)

            if self.input_latent_dim > 0:
                self.latent_node_embedding = FCBlock(input_latent_dim + latent_vocab, self.ns, self.ns, 2, dropout=dropout)
                self.latent_edge_embedding = FCBlock(input_latent_dim + latent_vocab, self.ns, self.ns, 2, dropout=dropout)
                self.active_virtual_node_attr = nn.Parameter(torch.zeros(max(self.latent_dim, input_latent_dim), self.ns), requires_grad=True)
                nn.init.xavier_uniform_(self.active_virtual_node_attr)
                self.active_edge_attr = nn.Parameter(torch.zeros(max(self.latent_dim, input_latent_dim), self.ns),requires_grad=True)
                nn.init.xavier_uniform_(self.active_edge_attr)

            transformer_layers = []
            for i in range(num_conv_layers):
                transformer_layer = TransformerConv(
                    in_channels=self.ns if i < 2 else 2*self.ns,
                    out_channels=self.ns if i < 1 else 2*self.ns,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                    edge_dim=2 * self.ns,
                )
                transformer_layers.append(transformer_layer)
            self.transformer_layers = nn.ModuleList(transformer_layers)

            if latent_nodes_residual:
                residual_node_layers = []
                residual_latent_layers = []
                for i in range(num_conv_layers):
                    residual_node_layers.append(FCBlock(self.ns if i < 1 else 2*self.ns, 2*self.ns, self.ns if i < 2 else 2*self.ns, 2, dropout=dropout))
                    residual_latent_layers.append(FCBlock(self.ns if i < 1 else 2*self.ns, 2*self.ns, self.ns if i < 1 else 2*self.ns, 2, dropout=dropout))
                self.residual_node_layers = nn.ModuleList(residual_node_layers)
                self.residual_latent_layers = nn.ModuleList(residual_latent_layers)

            dim = max(latent_dim, input_latent_dim)
            self.virtual_node_predictor = nn.ModuleList([
                FCBlock(self.ns if num_conv_layers < 2 else 2*self.ns, latent_hidden_dim, latent_vocab, 3,
                        dropout=latent_dropout, batchnorm=not latent_no_batchnorm)
                for _ in range(dim)
            ])
        else:

            if self.latent_vocab > 1:
                self.latent_predictor = nn.Sequential(
                    nn.Linear(4*self.ns if num_conv_layers >= 3 else 2*self.ns, latent_hidden_dim),
                    nn.BatchNorm1d(latent_hidden_dim) if not latent_no_batchnorm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(latent_dropout),
                    nn.Linear(latent_hidden_dim, latent_hidden_dim),
                    nn.BatchNorm1d(latent_hidden_dim) if not latent_no_batchnorm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(latent_dropout),
                    nn.Linear(latent_hidden_dim, latent_dim * latent_vocab)
                )
            else:
                self.latent_s_predictor = nn.Sequential(
                    nn.Linear(2*self.ns if num_conv_layers >= 3 else self.ns, latent_hidden_dim),
                    nn.BatchNorm1d(latent_hidden_dim) if not latent_no_batchnorm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(latent_dropout),
                    nn.Linear(latent_hidden_dim, latent_hidden_dim),
                    nn.BatchNorm1d(latent_hidden_dim) if not latent_no_batchnorm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(latent_dropout),
                    nn.Linear(latent_hidden_dim, latent_dim)
                )
                self.latent_r_predictor = nn.Sequential(
                    nn.Linear(2*self.ns if num_conv_layers >= 3 else self.ns, latent_hidden_dim),
                    nn.BatchNorm1d(latent_hidden_dim) if not latent_no_batchnorm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(latent_dropout),
                    nn.Linear(latent_hidden_dim, latent_hidden_dim),
                    nn.BatchNorm1d(latent_hidden_dim) if not latent_no_batchnorm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(latent_dropout),
                    nn.Linear(latent_hidden_dim, latent_dim)
                )

    def forward(self, data):
        if self.input_latent_dim > 0:
            decoding_idx = data.decoding_idx
            if self.latent_vocab > 1:
                input_latent = data.input_latent
                B, _, _ = input_latent.shape
                input_latent = input_latent.reshape(B, -1)
            else:
                input_latent = data['ligand'].input_latent, data['receptor'].input_latent
        else:
            input_latent = None
            decoding_idx = -1

        if self.input_latent_dim > 0:
            assert input_latent is not None
            assert torch.all(decoding_idx >= 0)
            assert self.use_oracle is False

        if self.use_oracle:
            data['ligand'].pos_e = data['ligand'].orig_pos_torch
        else:
            data['ligand'].pos_e = data['ligand'].pos

        if self.latent_virtual_nodes:
            latent_attr = input_latent
            input_latent = None

        # build ligand graph
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data, input_latent, decoding_idx)
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        # build receptor graph
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data, input_latent, decoding_idx)
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

        # build cross graph
        cross_cutoff = self.cross_max_distance
        lr_edge_index, lr_edge_attr, lr_edge_sh = self.build_cross_conv_graph(data, cross_cutoff, input_latent, decoding_idx)
        lr_edge_attr = self.cross_edge_embedding(lr_edge_attr)

        # combine the graphs
        node_attr = torch.cat([lig_node_attr, rec_node_attr], dim=0)
        lr_edge_index[1] = lr_edge_index[1] + len(lig_node_attr)
        edge_index = torch.cat([lig_edge_index, lr_edge_index, rec_edge_index + len(lig_node_attr),
                                torch.flip(lr_edge_index, dims=[0])], dim=1)
        edge_attr = torch.cat([lig_edge_attr, lr_edge_attr, rec_edge_attr, lr_edge_attr], dim=0)
        edge_sh = torch.cat([lig_edge_sh, lr_edge_sh, rec_edge_sh, lr_edge_sh], dim=0)
        s1, s2, s3 = len(lig_edge_index[0]), len(lig_edge_index[0]) + len(lr_edge_index[0]), len(
            lig_edge_index[0]) + len(lr_edge_index[0]) + len(rec_edge_index[0])

        if self.latent_virtual_nodes:
            B = data.num_graphs
            dim = max(self.latent_dim, self.input_latent_dim)
            offset = len(lig_node_attr) + len(rec_node_attr)

            latent_node_attr = self.initial_virtual_node_attr.unsqueeze(0).repeat(B, 1, 1)
            latent_edge_attr = self.virtual_edge_attr.unsqueeze(0).repeat(B, 1, 1)

            if self.input_latent_dim > 0:
                decoding_onehot = torch.zeros((B, self.input_latent_dim), device=data['ligand'].x.device)
                decoding_onehot[torch.arange(0, B).long(), decoding_idx] = 1

                latent_attr = torch.cat([latent_attr, decoding_onehot], dim=-1)
                latent_node_attr = latent_node_attr + self.latent_node_embedding(latent_attr)
                latent_edge_attr = latent_edge_attr + self.latent_edge_embedding(latent_attr)

                latent_node_attr[torch.arange(0, B).long(), decoding_idx] = \
                    latent_node_attr[torch.arange(0, B).long(), decoding_idx] + self.active_virtual_node_attr[decoding_idx]

                latent_edge_attr[torch.arange(0, B).long(), decoding_idx] = \
                    latent_edge_attr[torch.arange(0, B).long(), decoding_idx] + self.active_edge_attr[decoding_idx]

            latent_node_attr = latent_node_attr.reshape(B * dim, self.ns)

            # compute edges and edge attributes for the transformer layers
            trans_edge_index = ([], [])
            trans_edge_attr = ([], [])
            batch = torch.cat([data['ligand'].batch, data['receptor'].batch], dim=0)
            for i in range(dim):
                trans_edge_index[0].append(torch.arange(0, offset).to(batch.device))
                trans_edge_index[1].append(batch * dim + offset + i)
                trans_edge_attr[0].append(self.complex_edge_attr[0:1].repeat(len(lig_node_attr), 1))
                trans_edge_attr[0].append(self.complex_edge_attr[1:2].repeat(len(rec_node_attr), 1))
                trans_edge_attr[1].append(latent_edge_attr[batch, i])

            for i in range(B):
                pairs = torch.combinations(torch.arange(dim), r=2).to(batch.device)
                trans_edge_index[0].append(offset + pairs[:, 0] + i * dim)
                trans_edge_index[1].append(offset + pairs[:, 1] + i * dim)
                trans_edge_attr[0].append(latent_edge_attr[i, pairs[:, 0]])
                trans_edge_attr[1].append(latent_edge_attr[i, pairs[:, 1]])

            trans_edge_index = ((trans_edge_index[0] + trans_edge_index[1]), (trans_edge_index[1] + trans_edge_index[0]))
            trans_edge_attr = ((trans_edge_attr[0] + trans_edge_attr[1]), (trans_edge_attr[1] + trans_edge_attr[0]))

            trans_edge_index = torch.stack([torch.cat(trans_edge_index[0]), torch.cat(trans_edge_index[1])])
            trans_edge_attr = torch.cat([torch.cat(trans_edge_attr[0]), torch.cat(trans_edge_attr[1])], dim=1)

        for l in range(len(self.conv_layers)):
            edge_attr_ = torch.cat([edge_attr, node_attr[edge_index[0], :self.ns], node_attr[edge_index[1], :self.ns]], -1)
            edge_attr_ = [edge_attr_[:s1], edge_attr_[s1:s2], edge_attr_[s2:s3], edge_attr_[s3:]]
            node_attr = self.conv_layers[l](node_attr, edge_index, edge_attr_, edge_sh)

            if self.latent_virtual_nodes:
                # apply transformer cross layers
                node_scalar_attr = torch.cat([node_attr[:,:self.ns], node_attr[:,-self.ns:]], dim=1) if l >= 2 else node_attr[:,:self.ns]
                transformer_node_attr = torch.cat([node_scalar_attr, latent_node_attr], dim=0)
                transformer_node_attr = self.transformer_layers[l](transformer_node_attr, trans_edge_index, trans_edge_attr)

                # reassign the scalar attributes
                if self.latent_nodes_residual:
                    res_node_attr = self.residual_node_layers[l](transformer_node_attr[:len(node_attr)])
                    res_latent_attr = self.residual_latent_layers[l](transformer_node_attr[len(node_attr):])
                    node_attr[:, :self.ns] = node_attr[:, :self.ns] + res_node_attr[:, :self.ns]
                    if l >= 2: node_attr[:, -self.ns:] = node_attr[:, -self.ns:] + res_node_attr[:, -self.ns:]
                    if latent_node_attr.shape[-1] < res_latent_attr.shape[-1]:
                        latent_node_attr = F.pad(latent_node_attr, (0, res_latent_attr.shape[-1] - latent_node_attr.shape[-1]))
                    latent_node_attr = latent_node_attr + res_latent_attr
                else:
                    node_attr[:, :self.ns] = transformer_node_attr[:len(node_attr), :self.ns]
                    if l >= 2: node_attr[:, -self.ns:] = transformer_node_attr[:len(node_attr), -self.ns:]
                    latent_node_attr = transformer_node_attr[len(node_attr):]

        # compute latent
        lig_node_attr, rec_node_attr = node_attr[:len(lig_node_attr)], node_attr[len(lig_node_attr):]
        scalar_lig_attr = torch.cat([lig_node_attr[:,:self.ns],lig_node_attr[:,-self.ns:] ], dim=1) if self.num_conv_layers >= 3 else lig_node_attr[:,:self.ns]
        scalar_rec_attr = torch.cat([rec_node_attr[:,:self.ns],rec_node_attr[:,-self.ns:] ], dim=1) if self.num_conv_layers >= 3 else rec_node_attr[:,:self.ns]

        if self.latent_virtual_nodes:
            latent_node_attr = latent_node_attr.reshape(B, dim, -1)
            latent = torch.zeros(B, dim, self.latent_vocab).to(latent_node_attr.device)
            for i in range(dim):
                latent[:, i] = self.virtual_node_predictor[i](latent_node_attr[:, i])

            if self.apply_gumbel_softmax:
                return gumbel_softmax(latent, self.latent_temperature)
            else:
                return latent

        elif self.latent_vocab > 1:
            aggr_scalar_attr = torch.cat([scatter_mean(scalar_lig_attr, data['ligand'].batch, dim=0), scatter_mean(scalar_rec_attr, data['receptor'].batch, dim=0)], dim=1)
            latent = self.latent_predictor(aggr_scalar_attr)
            latent = latent.reshape(-1, self.latent_dim, self.latent_vocab)     # (bs, latent_dim, vocab_size)
            if self.apply_gumbel_softmax:
                return gumbel_softmax(latent, self.latent_temperature)
            else:
                return latent
        else:
            assert self.latent_vocab == 1
            scalar_lig_attr = self.latent_s_predictor(scalar_lig_attr)
            scalar_rec_attr = self.latent_r_predictor(scalar_rec_attr)

            latent_l = torch.zeros(scalar_lig_attr.shape[0], self.latent_dim, device=scalar_lig_attr.device)
            latent_r = torch.zeros(scalar_rec_attr.shape[0], self.latent_dim, device=scalar_rec_attr.device)
            latents = []
            for i in range(data.num_graphs):
                l_idx = (data['ligand'].batch == i)
                r_idx = (data['receptor'].batch == i)
                lat = torch.transpose(torch.cat([scalar_lig_attr[l_idx], scalar_rec_attr[r_idx]], dim=0), 0, 1).unsqueeze(0)
                if self.apply_gumbel_softmax:
                    lat = gumbel_softmax(lat, self.latent_temperature)
                    latent_l[l_idx] = torch.transpose(lat[0, :, :l_idx.sum()], 0, 1)
                    latent_r[r_idx] = torch.transpose(lat[0, :, l_idx.sum():], 0, 1)
                else:
                    # used for training autoregressive model
                    latents.append(lat)

            if self.apply_gumbel_softmax:
                return latent_l, latent_r
            else:
                return latents

    def build_lig_conv_graph(self, data, input_latent, decoding_idx):
        # builds the ligand graph edges and initial node and edge features

        # compute edges
        radius_edges = radius_graph(data['ligand'].pos_e, self.lig_max_radius, data['ligand'].batch)
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device)
        ], 0)

        src, dst = edge_index
        edge_vec = data['ligand'].pos_e[dst.long()] - data['ligand'].pos_e[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        if input_latent is not None:
            if self.latent_vocab > 1:
                node_latent = input_latent[data['ligand'].batch]
                edge_latent = node_latent[edge_index[0].long()]
            else:
                node_latent = input_latent[0]
                edge_latent = torch.cat([node_latent[edge_index[0].long()], node_latent[edge_index[1].long()]], 1)

            B = data.num_graphs
            decoding_onehot = torch.zeros((B, self.input_latent_dim), device=data['ligand'].x.device)
            decoding_onehot[torch.arange(0, B).long(), decoding_idx] = 1

            edge_attr = torch.cat([edge_attr, edge_length_emb, edge_latent, decoding_onehot[data['ligand'].batch[src.long()]]], 1)
            node_attr = torch.cat([data['ligand'].x, node_latent, decoding_onehot[data['ligand'].batch]], 1)
        else:
            edge_attr = torch.cat([edge_attr, edge_length_emb], 1)
            node_attr = data['ligand'].x

        return node_attr, edge_index, edge_attr, edge_sh

    def build_rec_conv_graph(self, data, input_latent, decoding_idx):
        # builds the receptor initial node and edge embeddings
        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]

        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        if input_latent is not None:
            if self.latent_vocab > 1:
                node_latent = input_latent[data['receptor'].batch]
                edge_latent = node_latent[edge_index[0].long()]
            else:
                node_latent = input_latent[1]
                edge_latent = torch.cat([node_latent[edge_index[0].long()], node_latent[edge_index[1].long()]], 1)

            B = data.num_graphs
            decoding_onehot = torch.zeros((B, self.input_latent_dim), device=data['ligand'].x.device)
            decoding_onehot[torch.arange(0, B).long(), decoding_idx] = 1

            edge_attr = torch.cat(
                [edge_length_emb, edge_latent, decoding_onehot[data['receptor'].batch[src.long()]]], 1)
            node_attr = torch.cat([data['receptor'].x, node_latent, decoding_onehot[data['receptor'].batch]], 1)
        else:
            edge_attr = edge_length_emb
            node_attr = data['receptor'].x

        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, cross_distance_cutoff, input_latent, decoding_idx):
        # builds the cross edges between ligand and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos_e / cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:
            edge_index = radius(data['receptor'].pos, data['ligand'].pos_e, cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['ligand'].pos_e[src.long()]

        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        if input_latent is not None:
            if self.latent_vocab > 1:
                node_latent = input_latent[data['ligand'].batch]
                edge_latent = node_latent[src.long()]
            else:
                l_latent = input_latent[0]
                r_latent = input_latent[1]
                edge_latent = torch.cat([l_latent[src.long()], r_latent[dst.long()]], 1)

            B = data.num_graphs
            decoding_onehot = torch.zeros((B, self.input_latent_dim), device=data['ligand'].x.device)
            decoding_onehot[torch.arange(0, B).long(), decoding_idx] = 1

            edge_attr = torch.cat([edge_length_emb, edge_latent, decoding_onehot[data['ligand'].batch[src.long()]]], 1)
        else:
            edge_attr = edge_length_emb

        return edge_index, edge_attr, edge_sh

