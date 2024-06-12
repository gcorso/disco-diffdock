from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter_mean

from models.layers import OldAtomEncoder, AtomEncoder, FCBlock, CrossAttention
from models.tensor_layers import TensorProductConvLayer, GaussianSmearing, get_irrep_seq
from utils import so3, torus
from datasets_utils.process_mols import lig_feature_dims, rec_residue_feature_dims


class TensorProductScoreModel(torch.nn.Module):
    def __init__(self, t_to_sigma, device, timestep_emb_func, in_lig_edge_features=4, sigma_embed_dim=32, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, lm_embedding_type=None, confidence_mode=False,
                 confidence_dropout=0, confidence_no_batchnorm=False, num_confidence_outputs=1, use_old_atom_encoder=False,
                 latent_dim=0, latent_vocab=32, latent_cross_attention=False, new_cross_attention=False,
                 cross_attention_heads=2, cross_attention_dim=16, latent_droprate=0.0):
        super(TensorProductScoreModel, self).__init__()
        assert not (new_cross_attention and not latent_cross_attention)
        self.t_to_sigma = t_to_sigma
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.device = device
        self.no_torsion = no_torsion
        self.timestep_emb_func = timestep_emb_func
        self.confidence_mode = confidence_mode
        self.num_conv_layers = num_conv_layers
        self.latent_dim = latent_dim
        self.latent_vocab = latent_vocab
        self.latent_cross_attention = latent_cross_attention
        self.new_cross_attention = new_cross_attention
        self.latent_droprate = latent_droprate

        atom_encoder_class = OldAtomEncoder if use_old_atom_encoder else AtomEncoder
        self.lig_node_embedding = atom_encoder_class(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim, latent_dim=latent_dim * latent_vocab)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim + latent_dim * max(latent_vocab, 2), ns),nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.rec_node_embedding = atom_encoder_class(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=sigma_embed_dim, lm_embedding_type=lm_embedding_type, latent_dim=latent_dim * latent_vocab)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim + latent_dim * max(latent_vocab, 2), ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.cross_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim + latent_dim * max(latent_vocab, 2), ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        if latent_droprate > 0:
            self.lig_node_unconditional_embedding = nn.Parameter(torch.zeros(1, ns))
            self.rec_node_unconditional_embedding = nn.Parameter(torch.zeros(1, ns))
            self.lig_edge_unconditional_embedding = nn.Parameter(torch.zeros(1, ns))
            self.rec_edge_unconditional_embedding = nn.Parameter(torch.zeros(1, ns))
            self.cross_edge_unconditional_embedding = nn.Parameter(torch.zeros(1, ns))

        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        if latent_cross_attention:
            self.latent_embedding_layers = []
            self.latent_residual_layers = []
            self.cross_attention_layers = []
            for i in range(latent_dim):
                self.tot_scalar_size = num_conv_layers * ns + max(0, num_conv_layers - 2) * ns
                self.latent_embedding_layers.append(FCBlock(latent_vocab, self.tot_scalar_size, self.tot_scalar_size, 2, dropout=dropout))

            for i in range(num_conv_layers):
                t_ns = ns if i < 2 else 2*ns
                self.latent_residual_layers.append(FCBlock(t_ns, t_ns, t_ns, 2, dropout=dropout))

                if new_cross_attention:
                    self.cross_attention_layers.append(CrossAttention(t_ns, heads=cross_attention_heads,
                                                                      dim_head=cross_attention_dim, dropout=dropout))

            self.latent_embedding_layers = nn.ModuleList(self.latent_embedding_layers)
            self.latent_residual_layers = nn.ModuleList(self.latent_residual_layers)
            self.cross_attention_layers = nn.ModuleList(self.cross_attention_layers)

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
                edge_groups=4
            )
            conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(conv_layers)

        if self.confidence_mode:
            self.confidence_predictor = nn.Sequential(
                nn.Linear(2*self.ns if num_conv_layers >= 3 else self.ns,ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, num_confidence_outputs)
            )
        else:
            # center of mass translation and rotation components
            self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
            self.center_edge_embedding = nn.Sequential(
                nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ns, ns)
            )

            self.final_conv = TensorProductConvLayer(
                in_irreps=self.conv_layers[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=f'2x1o + 2x1e',
                n_edge_features=2 * ns,
                residual=False,
                dropout=dropout,
                batch_norm=batch_norm
            )
            self.tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
            self.rot_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))

            if not no_torsion:
                # torsion angles components
                self.final_edge_embedding = nn.Sequential(
                    nn.Linear(distance_embed_dim, ns),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, ns)
                )
                self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
                self.tor_bond_conv = TensorProductConvLayer(
                    in_irreps=self.conv_layers[-1].out_irreps,
                    sh_irreps=self.final_tp_tor.irreps_out,
                    out_irreps=f'{ns}x0o + {ns}x0e',
                    n_edge_features=3 * ns,
                    residual=False,
                    dropout=dropout,
                    batch_norm=batch_norm
                )
                self.tor_final_layer = nn.Sequential(
                    nn.Linear(2 * ns, ns, bias=False),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, 1, bias=False)
                )

    def embed(self, data):
        if self.latent_dim > 0:
            if self.latent_vocab > 1:
                latent_h = data.latent_h
                B, dim, _ = latent_h.shape

                if self.latent_cross_attention:
                    latent_embedding = torch.zeros(B, dim, self.tot_scalar_size).to(latent_h.device)
                    for i in range(dim):
                        latent_embedding[:, i, :] = self.latent_embedding_layers[i](latent_h[:, i, :])

                latent_h = latent_h.reshape(B, -1)
            else:
                latent_h = data['ligand'].latent_h, data['receptor'].latent_h
        else:
            latent_h = None

        if not self.confidence_mode:
            tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(*[data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']])
        else:
            tr_sigma, rot_sigma, tor_sigma = [data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']]

        # build ligand graph
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data, latent_h)
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        # build receptor graph
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data, latent_h)
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

        # build cross graph
        if self.dynamic_max_cross:
            cross_cutoff = (tr_sigma * 3 + 20).unsqueeze(1)
        else:
            cross_cutoff = self.cross_max_distance
        lr_edge_index, lr_edge_attr, lr_edge_sh = self.build_cross_conv_graph(data, cross_cutoff, latent_h)
        lr_edge_attr = self.cross_edge_embedding(lr_edge_attr)
        
        if self.latent_droprate > 0:
            assert self.latent_vocab == 1, "Only implemented for equivariant latents"
            lig_node_attr = lig_node_attr + data['ligand'].unconditional * self.lig_node_unconditional_embedding
            rec_node_attr = rec_node_attr + data['receptor'].unconditional * self.rec_node_unconditional_embedding
            lig_edge_attr = lig_edge_attr + data['ligand'].unconditional[lig_edge_index[0]] * self.lig_edge_unconditional_embedding
            rec_edge_attr = rec_edge_attr + data['receptor'].unconditional[rec_edge_index[0]] * self.rec_edge_unconditional_embedding
            lr_edge_attr = lr_edge_attr + data['ligand'].unconditional[lr_edge_index[0]] * self.cross_edge_unconditional_embedding
            
        # combine the graphs
        node_attr = torch.cat([lig_node_attr, rec_node_attr], dim=0)
        lr_edge_index[1] = lr_edge_index[1] + len(lig_node_attr)
        edge_index = torch.cat([lig_edge_index, lr_edge_index, rec_edge_index + len(lig_node_attr),
                                torch.flip(lr_edge_index, dims=[0])], dim=1)
        edge_attr = torch.cat([lig_edge_attr, lr_edge_attr, rec_edge_attr, lr_edge_attr], dim=0)
        edge_sh = torch.cat([lig_edge_sh, lr_edge_sh, rec_edge_sh, lr_edge_sh], dim=0)
        s1, s2, s3 = len(lig_edge_index[0]), len(lig_edge_index[0]) + len(lr_edge_index[0]), len(
            lig_edge_index[0]) + len(lr_edge_index[0]) + len(rec_edge_index[0])

        for l in range(len(self.conv_layers)):
            edge_attr_ = torch.cat([edge_attr, node_attr[edge_index[0], :self.ns], node_attr[edge_index[1], :self.ns]], -1)
            edge_attr_ = [edge_attr_[:s1], edge_attr_[s1:s2], edge_attr_[s2:s3], edge_attr_[s3:]]
            node_attr = self.conv_layers[l](node_attr, edge_index, edge_attr_, edge_sh)

            if self.latent_cross_attention:
                node_scalar_attr = torch.cat([node_attr[:,:self.ns], node_attr[:,-self.ns:]], dim=1) if l >= 2 else node_attr[:,:self.ns]
                cur_idx, cur_len = (l * self.ns + max(0, l - 2) * self.ns, self.ns if l < 2 else 2 * self.ns)
                cur_latent_embedding = latent_embedding[:, :, cur_idx:cur_idx + cur_len]
                batch = torch.cat([data['ligand'].batch, data['receptor'].batch], dim=0)
                batched_cur_latent_embedding = cur_latent_embedding[batch]

                if self.new_cross_attention:
                    res_cross_attention = self.cross_attention_layers[l](node_scalar_attr.unsqueeze(dim=1), batched_cur_latent_embedding)
                    res_cross_attention = res_cross_attention.squeeze(1) + node_scalar_attr
                    res_cross_attention = self.latent_residual_layers[l](res_cross_attention) + res_cross_attention
                    res_cross_attention = res_cross_attention - node_scalar_attr # remove as it will be added back right after
                else:
                    res_cross_attention = torch.bmm(torch.bmm(node_scalar_attr.unsqueeze(dim=1), batched_cur_latent_embedding.transpose(1,2)),
                                                    batched_cur_latent_embedding).squeeze(1)
                    res_cross_attention = self.latent_residual_layers[l](res_cross_attention)

                if l < 2:
                    res_cross_attention = F.pad(res_cross_attention, (0, node_attr.shape[1] - res_cross_attention.shape[1]))
                else:
                    res_cross_attention = torch.cat([F.pad(res_cross_attention[:, :self.ns], (0, node_attr.shape[1] - res_cross_attention.shape[1])), res_cross_attention[:, self.ns:]], dim=1)

                node_attr = node_attr + res_cross_attention

        lig_node_attr, rec_node_attr = node_attr[:len(lig_node_attr)], node_attr[len(lig_node_attr):]
        return lig_node_attr, rec_node_attr, tr_sigma, rot_sigma, tor_sigma

    def forward(self, data):
        lig_node_attr, rec_node_attr, tr_sigma, rot_sigma, tor_sigma = self.embed(data)

        # compute confidence score
        if self.confidence_mode:
            scalar_lig_attr = torch.cat([lig_node_attr[:,:self.ns],lig_node_attr[:,-self.ns:] ], dim=1) if self.num_conv_layers >= 3 else lig_node_attr[:,:self.ns]
            confidence = self.confidence_predictor(scatter_mean(scalar_lig_attr, data['ligand'].batch, dim=0)).squeeze(dim=-1)
            return confidence

        # compute translational and rotational score vectors
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1)
        global_pred = self.final_conv(lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)

        tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
        data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['tr'])

        # fix the magnitude of translational and rotational score vectors
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, data.graph_sigma_emb], dim=1))
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, data.graph_sigma_emb], dim=1))

        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = rot_pred * so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(data['ligand'].x.device)

        if self.no_torsion or data['ligand'].edge_mask.sum() == 0: return tr_pred, rot_pred, torch.empty(0, device=self.device)

        # torsional components
        tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh = self.build_bond_conv_graph(data)
        tor_bond_vec = data['ligand'].pos[tor_bonds[1]] - data['ligand'].pos[tor_bonds[0]]
        tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
        tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_index[0]])

        tor_edge_attr = torch.cat([tor_edge_attr, lig_node_attr[tor_edge_index[1], :self.ns],
                                   tor_bond_attr[tor_edge_index[0], :self.ns]], -1)
        tor_pred = self.tor_bond_conv(lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
                                  out_nodes=data['ligand'].edge_mask.sum(), reduce='mean')
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        edge_sigma = tor_sigma[data['ligand'].batch][data['ligand', 'ligand'].edge_index[0]][data['ligand'].edge_mask]

        if self.scale_by_sigma:
            tor_pred = tor_pred * torch.sqrt(torch.tensor(torus.score_norm(edge_sigma.cpu().numpy())).float()
                                             .to(data['ligand'].x.device))
        return tr_pred, rot_pred, tor_pred

    def build_lig_conv_graph(self, data, latent_h):
        # builds the ligand graph edges and initial node and edge features
        data['ligand'].node_sigma_emb = self.timestep_emb_func(data['ligand'].node_t['tr'])

        # compute edges
        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch)
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device)
        ], 0)

        # compute initial features
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[0].long()]

        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        if latent_h is not None:
            if self.latent_vocab > 1:
                node_latent = latent_h[data['ligand'].batch]
                edge_latent = node_latent[edge_index[0].long()]
            else:
                node_latent = latent_h[0]
                edge_latent = torch.cat([node_latent[edge_index[0].long()], node_latent[edge_index[1].long()]], 1)
            edge_attr = torch.cat([edge_attr, edge_sigma_emb, edge_length_emb, edge_latent], 1)
            node_attr = torch.cat([data['ligand'].x, data['ligand'].node_sigma_emb, node_latent], 1)
        else:
            edge_attr = torch.cat([edge_attr, edge_sigma_emb, edge_length_emb], 1)
            node_attr = torch.cat([data['ligand'].x, data['ligand'].node_sigma_emb], 1)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh

    def build_rec_conv_graph(self, data, latent_h):
        # builds the receptor initial node and edge embeddings
        data['receptor'].node_sigma_emb = self.timestep_emb_func(data['receptor'].node_t['tr']) # tr rot and tor noise is all the same

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]

        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['receptor'].node_sigma_emb[edge_index[0].long()]

        if latent_h is not None:
            if self.latent_vocab > 1:
                node_latent = latent_h[data['receptor'].batch]
                edge_latent = node_latent[edge_index[0].long()]
            else:
                node_latent = latent_h[1]
                edge_latent = torch.cat([node_latent[edge_index[0].long()], node_latent[edge_index[1].long()]], 1)
            node_attr = torch.cat([data['receptor'].x, data['receptor'].node_sigma_emb, node_latent], 1)
            edge_attr = torch.cat([edge_sigma_emb, edge_length_emb, edge_latent], 1)
        else:
            node_attr = torch.cat([data['receptor'].x, data['receptor'].node_sigma_emb], 1)
            edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, cross_distance_cutoff, latent_h):
        # builds the cross edges between ligand and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:
            edge_index = radius(data['receptor'].pos, data['ligand'].pos, cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['ligand'].pos[src.long()]

        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[src.long()]

        if latent_h is not None:
            if self.latent_vocab > 1:
                node_latent = latent_h[data['ligand'].batch]
                edge_latent = node_latent[src]
            else:
                l_latent = latent_h[0]
                r_latent = latent_h[1]
                edge_latent = torch.cat([l_latent[src.long()], r_latent[dst.long()]], 1)

            edge_latent = torch.zeros(edge_latent.shape).to(edge_latent.device)
            edge_attr = torch.cat([edge_sigma_emb, edge_length_emb, edge_latent], 1)
        else:
            edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh

    def build_center_conv_graph(self, data):
        # builds the filter and edges for the convolution generating translational and rotational scores
        edge_index = torch.cat([data['ligand'].batch.unsqueeze(0), torch.arange(len(data['ligand'].batch)).to(data['ligand'].x.device).unsqueeze(0)], dim=0)

        center_pos, count = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device), torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
        center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)

        edge_vec = data['ligand'].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh

    def build_bond_conv_graph(self, data):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        bonds = data['ligand', 'ligand'].edge_index[:, data['ligand'].edge_mask].long()
        bond_pos = (data['ligand'].pos[bonds[0]] + data['ligand'].pos[bonds[1]]) / 2
        bond_batch = data['ligand'].batch[bonds[0]]
        edge_index = radius(data['ligand'].pos, bond_pos, self.lig_max_radius, batch_x=data['ligand'].batch, batch_y=bond_batch)

        edge_vec = data['ligand'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh
