import os
import pickle
import random
import copy
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from utils.utils import get_args_and_cache_path, ListDataset
from datasets_utils.pdbbind import PDBBind
from models.layers import gumbel_softmax
from utils.sampling import randomize_position
from utils.model_utils import get_model


class AutoregressiveDataset(Dataset):
    def __init__(self, cache_path, original_model_dir, split, device, limit_complexes,
                 all_atoms, args, model_ckpt, tr_sigma_max, multiplicity=1, no_randomness=False, no_sampling=False):
        super(AutoregressiveDataset, self).__init__()

        self.device = device
        self.limit_complexes = limit_complexes
        self.all_atoms = all_atoms
        self.original_model_dir = original_model_dir
        self.model_ckpt = model_ckpt
        self.original_model_args, self.original_model_cache = get_args_and_cache_path(original_model_dir, split)
        self.sampling_latent_temperature = args.sampling_latent_temperature
        self.tr_sigma_max = tr_sigma_max
        self.args = args
        self.split = split
        self.multiplicity = multiplicity
        self.no_randomness = no_randomness
        self.no_sampling = no_sampling

        self.full_cache_path = os.path.join(cache_path,
                                            f'model_{os.path.splitext(os.path.basename(original_model_dir))[0]}'
                                            f'_split_{split}_limit_{limit_complexes}'
                                            f'{"_nosampling" if no_sampling else ""}')

        if not os.path.exists(os.path.join(self.full_cache_path, "latent_labels.pkl")):
            os.makedirs(self.full_cache_path, exist_ok=True)
            self.preprocessing()

        # load the graphs that the confidence model will use
        self.dataset = PDBBind(transform=None, root=args.data_dir, limit_complexes=args.limit_complexes,
                               receptor_radius=args.receptor_radius,
                               cache_path=args.cache_path,
                               split_path=args.split_val if split == 'val' else args.split_train,
                               remove_hs=args.remove_hs, max_lig_size=None,
                               c_alpha_max_neighbors=args.c_alpha_max_neighbors,
                               matching=not args.no_torsion, keep_original=True,
                               popsize=args.matching_popsize,
                               maxiter=args.matching_maxiter,
                               all_atoms=args.all_atoms,
                               atom_radius=args.atom_radius,
                               atom_max_neighbors=args.atom_max_neighbors,
                               esm_embeddings_path=args.esm_embeddings_path,
                               require_ligand=True,
                               require_rdkit=no_randomness)

        # load the latent labels
        with open(os.path.join(self.full_cache_path, "latent_labels.pkl"), 'rb') as f:
            self.latent_labels = pickle.load(f)

        print("number of labels", len(self.latent_labels), '/', len(self.dataset))

    def len(self):
        return len(self.dataset) * self.multiplicity

    def get(self, idx):
        idx = idx % len(self.dataset)
        complex_graph = copy.deepcopy(self.dataset.get(idx))
        while complex_graph.name not in self.latent_labels:
            idx = random.randint(0, len(self.dataset) - 1)
            complex_graph = copy.deepcopy(self.dataset.get(idx))
            print('HAPPENING | complex not in latent labels, trying another one')
        input_latent = copy.deepcopy(self.latent_labels[complex_graph.name])

        decoding_idx = random.randint(0, self.original_model_args.latent_dim - 1)
        complex_graph.decoding_idx = decoding_idx
        if self.no_sampling:
            B, L, T = input_latent[0].shape
            complex_graph.true_latent = torch.transpose(F.one_hot(torch.argmax(input_latent[0][0], dim=-1), T), 0, 1).numpy()
        else:
            complex_graph.true_latent = np.concatenate([input_latent[0].numpy(), input_latent[1].numpy()], axis=0)

        if self.no_randomness:
            # set the positions to the original rdkit ones
            complex_graph['ligand'].pos = torch.from_numpy(complex_graph['ligand'].orig_rdkit_pos).float()
        randomize_position([complex_graph], no_torsion=self.no_randomness, no_random=self.no_randomness,
                           tr_sigma_max=self.tr_sigma_max, unbatched=True)

        if self.no_sampling:
            assert isinstance(input_latent, list), 'not implemented for invariant latents'
            lat = input_latent[0] # 1 x latent_dim x L+R
            B, L, T = lat.shape
            assert B == 1 and T == (len(complex_graph['ligand'].pos) + len(complex_graph['receptor'].pos)) and L == self.original_model_args.latent_dim

            complex_graph.latent_label = torch.softmax(lat[0, decoding_idx, :], dim=0).numpy()

            lat = F.one_hot(torch.argmax(gumbel_softmax(lat, 0.01, 'cpu'), dim=-1), T).float()[0]
            lat[decoding_idx:, :] = 0.0
            complex_graph['ligand'].input_latent = torch.transpose(lat[:, :len(complex_graph['ligand'].pos)], 0, 1)
            complex_graph['receptor'].input_latent = torch.transpose(lat[:, len(complex_graph['ligand'].pos):], 0, 1)

        else:
            if isinstance(input_latent, tuple):
                complex_graph.latent_label = torch.argmax(
                    torch.cat([input_latent[0][:, decoding_idx], input_latent[1][:, decoding_idx]], dim=0), dim=0)

                complex_graph['ligand'].input_latent, complex_graph['receptor'].input_latent = input_latent

                # mask the latent vectors after the decoding index
                complex_graph['ligand'].input_latent[:, decoding_idx:] = 0
                complex_graph['receptor'].input_latent[:, decoding_idx:] = 0

            else:
                complex_graph.latent_label = torch.argmax(input_latent[0, decoding_idx])
                complex_graph.input_latent = input_latent
                complex_graph.input_latent[decoding_idx:] = 0

        return complex_graph

    def preprocessing(self):
        model = get_model(self.original_model_args, self.device, t_to_sigma=None)
        model = model.module if isinstance(model, nn.DataParallel) else model
        state_dict = torch.load(f'{self.original_model_dir}/{self.model_ckpt}', map_location=self.device)
        model.load_state_dict(state_dict, strict=True)
        encoder = model.encoder
        encoder = encoder.to(self.device)
        encoder.eval()
        encoder.latent_temperature = self.sampling_latent_temperature
        encoder.apply_gumbel_softmax = not self.no_sampling

        dataset = PDBBind(transform=None, root=self.args.data_dir, limit_complexes=self.args.limit_complexes,
                          receptor_radius=self.original_model_args.receptor_radius,
                          cache_path=self.args.cache_path,
                          split_path=self.args.split_val if self.split == 'val' else self.args.split_train,
                          remove_hs=self.original_model_args.remove_hs, max_lig_size=None,
                          c_alpha_max_neighbors=self.original_model_args.c_alpha_max_neighbors,
                          matching=not self.original_model_args.no_torsion, keep_original=True,
                          popsize=self.original_model_args.matching_popsize,
                          maxiter=self.original_model_args.matching_maxiter,
                          all_atoms=self.original_model_args.all_atoms,
                          atom_radius=self.original_model_args.atom_radius,
                          atom_max_neighbors=self.original_model_args.atom_max_neighbors,
                          esm_embeddings_path=self.args.esm_embeddings_path,
                          require_ligand=True,
                          num_workers=1)
        complex_graphs = [dataset.get(i) for i in range(len(dataset))]
        dataset = ListDataset(complex_graphs)
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        latent_labels = {}
        print('dataset of size', len(dataset))
        for idx, complex_graph in tqdm(enumerate(loader)):
            complex_graph['ligand'].orig_pos_torch = torch.from_numpy(
                complex_graph['ligand'].orig_pos - complex_graph.original_center.cpu().numpy()).float()[0]
            complex_graph = complex_graph.to(self.device)

            latent = encoder(complex_graph)
            if isinstance(latent, tuple):
                latent = (latent[0].detach().cpu(), latent[1].detach().cpu())
            elif isinstance(latent, list):
                latent = [l.detach().cpu() for l in latent]
            else:
                latent = latent.detach().cpu()
            latent_labels[complex_graph.name[0]] = latent

        with open(os.path.join(self.full_cache_path, f"latent_labels.pkl"), 'wb') as f:
            pickle.dump(latent_labels, f)
