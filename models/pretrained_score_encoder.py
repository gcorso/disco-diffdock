import torch
from torch import nn
from models.layers import gumbel_softmax
from models.model_classes import GenericEncoder
from utils.diffusion_utils import t_to_sigma, set_time


class PretrainedScoreEncoder(GenericEncoder):
    def __init__(self, pretrained_score_model, ns, latent_dim, latent_vocab, latent_no_batchnorm=False, latent_dropout=0.0, latent_hidden_dim=128,
                 input_latent_dim=0, apply_gumbel_softmax=True):
        super(PretrainedScoreEncoder, self).__init__()

        assert input_latent_dim > 0
        self.ns = ns
        self.latent_dim = latent_dim
        self.latent_vocab = latent_vocab
        self.latent_temperature = 1.0   # this gets reset by the different training and evaluation routines
        self.input_latent_dim = input_latent_dim
        self.apply_gumbel_softmax = apply_gumbel_softmax

        self.pretrained_score_model = pretrained_score_model

        self.latent_s_predictor = nn.Sequential(
            nn.Linear(2*ns if self.pretrained_score_model.num_conv_layers >= 3 else ns, latent_hidden_dim),
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
            nn.Linear(2*ns if self.pretrained_score_model.num_conv_layers >= 3 else ns, latent_hidden_dim),
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
        decoding_idx = data.decoding_idx
        if self.latent_vocab > 1:
            data.latent_h = data.input_latent
        else:
            data['ligand'].latent_h, data['receptor'].latent_h = data['ligand'].input_latent, data['receptor'].input_latent

        assert torch.all(decoding_idx >= 0)

        device = data['ligand'].pos.device
        set_time(data, 1, 1, 1, data.num_graphs, False, device)
        data['ligand'].unconditional = torch.ones((len(data['ligand'].pos), 1)).to(device)
        data['receptor'].unconditional = torch.ones((len(data['receptor'].pos), 1)).to(device)

        lig_node_attr, rec_node_attr, tr_sigma, rot_sigma, tor_sigma = self.pretrained_score_model.embed(data)

        scalar_lig_attr = torch.cat([lig_node_attr[:,:self.ns],lig_node_attr[:,-self.ns:] ], dim=1) \
            if self.pretrained_score_model.num_conv_layers >= 3 else lig_node_attr[:,:self.ns]
        scalar_rec_attr = torch.cat([rec_node_attr[:,:self.ns],rec_node_attr[:,-self.ns:] ], dim=1) \
            if self.pretrained_score_model.num_conv_layers >= 3 else rec_node_attr[:,:self.ns]

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
