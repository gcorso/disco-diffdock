import torch
import copy


class GenericEncoder(torch.nn.Module):
    def __init__(self):
        super(GenericEncoder, self).__init__()

    def encode_ar(self, data, sampling_temperature=1.0):
        # assumes graphs of the same complex as input
        B = data.num_graphs
        if self.latent_vocab > 1:
            latent = torch.zeros(B, self.input_latent_dim, self.latent_vocab).to(data['ligand'].pos.device)
            for decoding_idx in range(self.input_latent_dim):
                data.input_latent = latent
                data.decoding_idx = torch.zeros(B).to(data['ligand'].pos.device).long() + decoding_idx
                latent[:, decoding_idx:decoding_idx+1, :] = self.forward(copy.deepcopy(data))
        else:
            temp_apply_gumbel_softmax = self.apply_gumbel_softmax
            self.apply_gumbel_softmax = False
            len_lig = len(data['ligand'].pos) // B
            len_rec = len(data['receptor'].pos) // B

            latent_l = torch.zeros(len(data['ligand'].pos), self.input_latent_dim).to(data['ligand'].pos.device)
            latent_r = torch.zeros(len(data['receptor'].pos), self.input_latent_dim).to(data['ligand'].pos.device)
            for decoding_idx in range(self.input_latent_dim):
                data['ligand'].input_latent, data['receptor'].input_latent = (latent_l, latent_r)
                data.decoding_idx = torch.zeros(B).to(data['ligand'].pos.device).long() + decoding_idx

                lat = self.forward(copy.deepcopy(data))
                lat = torch.cat(lat, dim=0)[:, 0, :] * sampling_temperature
                assert lat.shape == (B, len_lig + len_rec)
                if sampling_temperature >= 100:
                    lat_choice = torch.argmax(lat, 1, keepdim=True)
                else:
                    p = torch.exp(lat)
                    if torch.any(torch.isnan(p)) or torch.any(torch.isinf(p)):
                        print("Warning: NaNs or INF in AR setting them to 0")
                        p = torch.nan_to_num(p)
                    lat_choice = torch.multinomial(p, 1)

                for i in range(B):
                    if lat_choice[i, 0] < len_lig:
                        latent_l[i * len_lig + lat_choice[i, 0], decoding_idx] = 1
                    else:
                        latent_r[i * len_rec + lat_choice[i, 0] - len_lig, decoding_idx] = 1
            latent = (latent_l, latent_r)
            self.apply_gumbel_softmax = temp_apply_gumbel_softmax
        return latent



class ModelWrapper(torch.nn.Module):
    def __init__(self, encoder, score_model, training_latent_temperature, device, latent_droprate):
        super(ModelWrapper, self).__init__()
        self.encoder = encoder
        self.score_model = score_model
        self.training_latent_temperature = training_latent_temperature
        self.device = device
        self.latent_droprate = latent_droprate

    def forward(self, data):
        if self.encoder is not None:
            self.encoder.latent_temperature = self.training_latent_temperature
            latent_h = self.encoder(data)

            if isinstance(latent_h, tuple):
                data['ligand'].latent_h, data['receptor'].latent_h = latent_h

                if self.latent_droprate > 0:
                    B = data.num_graphs
                    mask = torch.bernoulli(torch.full((B, 1), 1 - self.latent_droprate)).to(data['ligand'].latent_h.device)
                    data['ligand'].unconditional = 1 - mask[data['ligand'].batch]
                    data['receptor'].unconditional = 1 - mask[data['receptor'].batch]
                    data['ligand'].latent_h = data['ligand'].latent_h * mask[data['ligand'].batch]
                    data['receptor'].latent_h = data['receptor'].latent_h * mask[data['receptor'].batch]

            else:
                B, _, _ = latent_h.shape
                mask = torch.bernoulli(torch.full((B, 1, 1), 1 - self.latent_droprate)).to(latent_h.device)
                latent_h = latent_h * mask
                data.latent_h = latent_h
                data.unconditional = 1 - mask[data.batch]

        return self.score_model(data)
