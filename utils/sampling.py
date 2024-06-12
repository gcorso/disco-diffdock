import copy

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from utils.diffusion_utils import modify_conformer, set_time, modify_conformer_batch
from utils.torsion import modify_conformer_torsion_angles
from scipy.spatial.transform import Rotation as R


def randomize_position(data_list, no_torsion, no_random, tr_sigma_max, unbatched=False, ar_args=None):

    # in place modification of the list
    if not no_torsion:
        # randomize torsion angles
        for complex_graph in data_list:
            torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=complex_graph['ligand'].edge_mask.sum())
            complex_graph['ligand'].pos = \
                modify_conformer_torsion_angles(complex_graph['ligand'].pos,
                                                complex_graph['ligand', 'ligand'].edge_index.T[complex_graph['ligand'].edge_mask],
                                                complex_graph['ligand'].mask_rotate[0] if not unbatched else complex_graph['ligand'].mask_rotate,
                                                torsion_updates)

    for complex_graph in data_list:
        # randomize position
        molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(R.random().as_matrix()).float()
        complex_graph['ligand'].pos = (complex_graph['ligand'].pos - molecule_center) @ random_rotation.T
        # base_rmsd = np.sqrt(np.sum((complex_graph['ligand'].pos.cpu().numpy() - orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean())

        if not no_random:  # note for now the torsion angles are still randomised
            tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
            complex_graph['ligand'].pos += tr_update

    if ar_args is not None:
        if ar_args.no_randomness:
            # set the positions to the original rdkit ones
            for complex_graph in data_list:
                complex_graph['ligand'].ar_pos = torch.from_numpy(complex_graph['ligand'].orig_rdkit_pos[0]).float()
                molecule_center = torch.mean(complex_graph['ligand'].ar_pos, dim=0, keepdim=True)
                random_rotation = torch.from_numpy(R.random().as_matrix()).float()
                complex_graph['ligand'].ar_pos = (complex_graph['ligand'].ar_pos - molecule_center) @ random_rotation.T
        else:
            for complex_graph in data_list:
                complex_graph['ligand'].ar_pos = copy.deepcopy(complex_graph['ligand'].pos)


def sampling(data_list, model, inference_steps, tr_schedule, rot_schedule, tor_schedule, device, t_to_sigma, model_args,
             no_random=False, ode=False, visualization_list=None, confidence_model=None, confidence_data_list=None,
             confidence_model_args=None, batch_size=32, no_final_step_noise=False, use_latent=True,
             gumbel_latent_temperature=0.01, ar_model=None, ar_args=None, temp_sampling=1.0, temp_psi=0.0, temp_sigma_data=0.5,
             classifier_free_guidance_weight=0.0, softmax_latent_temperature=1.0, cfg_start=1.0, cfg_end=0.0,
             compute_ar_accuracy=False):
    N = len(data_list)
    loader = DataLoader(data_list, batch_size=batch_size)
    mask_rotate = torch.from_numpy(data_list[0]['ligand'].mask_rotate[0]).to(device)

    confidence = None
    if confidence_model is not None:
        confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
        confidence = []

    with torch.no_grad():
        for batch_id, complex_graph_batch in enumerate(loader):
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)

            if use_latent and hasattr(model_args, 'latent_dim') and model_args.latent_dim > 0:
                if ar_model is None or compute_ar_accuracy:
                    if hasattr(model, 'module'):
                        model.module.encoder.latent_temperature = gumbel_latent_temperature
                    else:
                        model.encoder.latent_temperature = gumbel_latent_temperature

                    latent_h = model.module.encoder(complex_graph_batch) if hasattr(model, 'module') else model.encoder(complex_graph_batch)
                    if compute_ar_accuracy:
                        true_latent_h = latent_h

                if ar_model is not None:
                    # replace the positions with the ar ones to allow for no randomness option
                    temp_lig_pos = complex_graph_batch['ligand'].pos
                    complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].ar_pos
                    latent_h = ar_model.encode_ar(complex_graph_batch, softmax_latent_temperature)
                    complex_graph_batch['ligand'].pos = temp_lig_pos

                if isinstance(latent_h, tuple):
                    complex_graph_batch['ligand'].latent_h, complex_graph_batch['receptor'].latent_h = latent_h

                    if compute_ar_accuracy:
                        latent_h = torch.cat(latent_h, dim=0)
                        true_latent_h = torch.cat(true_latent_h, dim=0)
                        latent_h = (latent_h > 0.5).float()
                        true_latent_h = (true_latent_h > 0.5).float()
                        latent_h = latent_h[:, 0]   # only interested in first latent
                        true_latent_h = true_latent_h[:, 0]
                        assert latent_h.sum() == true_latent_h.sum() and true_latent_h.sum() == b
                        ar_accuracy = (latent_h * true_latent_h).sum() / b
                        print('AR accuracy: {}'.format(ar_accuracy))
                        for idx in range(b):
                            data_list[batch_id * batch_size + idx].ar_accuracy = ar_accuracy.item()
                else:
                    complex_graph_batch.latent_h = latent_h

            for t_idx in range(inference_steps):
                t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
                dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
                dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
                dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]

                tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t_tr, t_rot, t_tor)

                set_time(complex_graph_batch, t_tr, t_rot, t_tor, b, 'all_atoms' in model_args and model_args.all_atoms, device)
                complex_graph_batch['ligand'].unconditional = torch.zeros((len(complex_graph_batch['ligand'].pos), 1)).to(device)
                complex_graph_batch['receptor'].unconditional = torch.zeros((len(complex_graph_batch['receptor'].pos), 1)).to(device)
                tr_score, rot_score, tor_score = model.module.score_model(complex_graph_batch)[:3] if hasattr(model, 'module') \
                    else model.score_model(complex_graph_batch)[:3]

                if classifier_free_guidance_weight != 0.0 and t_tr <= cfg_start and t_tr >= cfg_end:
                    # set unconditional to 1 and latent to 0
                    complex_graph_batch['ligand'].unconditional = torch.ones((len(complex_graph_batch['ligand'].pos), 1)).to(device)
                    complex_graph_batch['receptor'].unconditional = torch.ones((len(complex_graph_batch['receptor'].pos), 1)).to(device)
                    temp_latent = complex_graph_batch['ligand'].latent_h, complex_graph_batch['receptor'].latent_h
                    complex_graph_batch['ligand'].latent_h = 0 * complex_graph_batch['ligand'].latent_h
                    complex_graph_batch['receptor'].latent_h = 0 * complex_graph_batch['receptor'].latent_h

                    # compute unconditional scores and apply guidance
                    uncond_tr_score, uncond_rot_score, uncond_tor_score = model.module.score_model(complex_graph_batch)[:3] if hasattr(model, 'module') \
                        else model.score_model(complex_graph_batch)[:3]
                    tr_score = tr_score + classifier_free_guidance_weight * (tr_score - uncond_tr_score)
                    rot_score = rot_score + classifier_free_guidance_weight * (rot_score - uncond_rot_score)
                    tor_score = tor_score + classifier_free_guidance_weight * (tor_score - uncond_tor_score)

                    # reset latent
                    complex_graph_batch['ligand'].latent_h, complex_graph_batch['receptor'].latent_h = temp_latent

                tr_g = tr_sigma * torch.sqrt(
                    torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
                rot_g = rot_sigma * torch.sqrt(
                    torch.tensor(2 * np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))

                if ode:
                    tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score)
                    rot_perturb = (0.5 * rot_score * dt_rot * rot_g ** 2)
                else:
                    tr_z = torch.zeros((min(batch_size, N), 3), device=device) if no_random or (
                                no_final_step_noise and t_idx == inference_steps - 1) \
                        else torch.normal(mean=0, std=1, size=(min(batch_size, N), 3), device=device)
                    tr_perturb = (tr_g ** 2 * dt_tr * tr_score + tr_g * np.sqrt(dt_tr) * tr_z)

                    rot_z = torch.zeros((min(batch_size, N), 3), device=device) if no_random or (
                                no_final_step_noise and t_idx == inference_steps - 1) \
                        else torch.normal(mean=0, std=1, size=(min(batch_size, N), 3), device=device)
                    rot_perturb = (rot_score * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z)

                if not model_args.no_torsion:
                    tor_g = tor_sigma * torch.sqrt(
                        torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
                    if ode:
                        tor_perturb = (0.5 * tor_g ** 2 * dt_tor * tor_score)
                    else:
                        tor_z = torch.zeros(tor_score.shape, device=device) if no_random or (
                                    no_final_step_noise and t_idx == inference_steps - 1) \
                            else torch.normal(mean=0, std=1, size=tor_score.shape, device=device)
                        tor_perturb = (tor_g ** 2 * dt_tor * tor_score + tor_g * np.sqrt(dt_tor) * tor_z)
                    torsions_per_molecule = tor_perturb.shape[0] // b
                else:
                    tor_perturb = None


                if not is_iterable(temp_sampling): temp_sampling = [temp_sampling] * 3
                if not is_iterable(temp_psi): temp_psi = [temp_psi] * 3
                if not is_iterable(temp_sigma_data): temp_sigma_data = [temp_sigma_data] * 3

                assert len(temp_sampling) == 3
                assert len(temp_psi) == 3
                assert len(temp_sigma_data) == 3

                if temp_sampling[0] != 1.0:
                    tr_sigma_data = np.exp(temp_sigma_data[0] * np.log(model_args.tr_sigma_max) + (1 - temp_sigma_data[0]) * np.log(model_args.tr_sigma_min))
                    lambda_tr = (tr_sigma_data + tr_sigma) / (tr_sigma_data + tr_sigma / temp_sampling[0])
                    tr_perturb = (tr_g ** 2 * dt_tr * (lambda_tr + temp_sampling[0] * temp_psi[0] / 2) * tr_score + tr_g * np.sqrt(dt_tr * (1 + temp_psi[0])) * tr_z)

                if temp_sampling[1] != 1.0:
                    rot_sigma_data = np.exp(temp_sigma_data[1] * np.log(model_args.rot_sigma_max) + (1 - temp_sigma_data[1]) * np.log(model_args.rot_sigma_min))
                    lambda_rot = (rot_sigma_data + rot_sigma) / (rot_sigma_data + rot_sigma / temp_sampling[1])
                    rot_perturb = (rot_g ** 2 * dt_rot * (lambda_rot + temp_sampling[1] * temp_psi[1] / 2) * rot_score + rot_g * np.sqrt(dt_rot * (1 + temp_psi[1])) * rot_z)

                if temp_sampling[2] != 1.0:
                    tor_sigma_data = np.exp(temp_sigma_data[2] * np.log(model_args.tor_sigma_max) + (1 - temp_sigma_data[2]) * np.log(model_args.tor_sigma_min))
                    lambda_tor = (tor_sigma_data + tor_sigma) / (tor_sigma_data + tor_sigma / temp_sampling[2])
                    tor_perturb = (tor_g ** 2 * dt_tor * (lambda_tor + temp_sampling[2] * temp_psi[2] / 2) * tor_score + tor_g * np.sqrt(dt_tor * (1 + temp_psi[2])) * tor_z)

                # Apply noise
                complex_graph_batch['ligand'].pos = \
                    modify_conformer_batch(complex_graph_batch['ligand'].pos, complex_graph_batch, tr_perturb,
                                           rot_perturb,
                                           tor_perturb if not model_args.no_torsion else None, mask_rotate)

            len_lig = len(complex_graph_batch['ligand'].pos) // b
            len_rec = len(complex_graph_batch['receptor'].pos) // b
            for i in range(b):
                data_list[batch_id * batch_size + i]['ligand'].pos = complex_graph_batch['ligand'].pos[i * len_lig:len_lig * (i + 1)]

                if use_latent and hasattr(model_args, 'latent_dim') and model_args.latent_dim > 0:
                    lig_lat = complex_graph_batch['ligand'].latent_h[i * len_lig:len_lig * (i + 1)]
                    rec_lat = complex_graph_batch['receptor'].latent_h[i * len_rec:len_rec * (i + 1)]
                    lat_str = ""
                    lat_pos = []
                    for j in range(model_args.latent_dim):
                        assert torch.sum(lig_lat[:, j]) + torch.sum(rec_lat[:, j]) == 1
                        if torch.sum(lig_lat[:, j]) == 1:
                            id = torch.argmax(lig_lat[:, j]).detach().cpu().item()
                            lat_str += ('L' + str(id))
                            lat_pos.append(data_list[batch_id * batch_size + i]['ligand'].pos[id: id + 1].detach().cpu() + data_list[batch_id * batch_size + i].original_center.detach().cpu())
                        else:
                            id = torch.argmax(rec_lat[:, j]).detach().cpu().item()
                            lat_str += ('R' + str(id))
                            lat_pos.append(data_list[batch_id * batch_size + i]['receptor'].pos[id: id + 1].detach().cpu() + data_list[batch_id * batch_size + i].original_center.detach().cpu())
                    data_list[batch_id * batch_size + i].latent_str = lat_str
                    data_list[batch_id * batch_size + i].latent_pos = torch.cat(lat_pos, dim=0)


            if visualization_list is not None:
                for idx, visualization in enumerate(visualization_list):
                    visualization.add(
                        (data_list[idx]['ligand'].pos.detach().cpu() + data_list[idx].original_center.detach().cpu()),
                        part=1, order=2)

            if confidence_model is not None:
                if confidence_data_list is not None:
                    confidence_complex_graph_batch = next(confidence_loader)
                    confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos.cpu()

                    confidence_complex_graph_batch = confidence_complex_graph_batch.to(device)
                    set_time(confidence_complex_graph_batch, 0, 0, 0, b, confidence_model_args.all_atoms, device)
                    out = confidence_model(confidence_complex_graph_batch)
                else:
                    out = confidence_model(complex_graph_batch)

                if type(out) is tuple:
                    out = out[0]
                confidence.append(out)

    if confidence_model is not None:
        confidence = torch.cat(confidence, dim=0)
        confidence = torch.nan_to_num(confidence, nan=-1000)

    return data_list, confidence



def is_iterable(arr):
    try:
        some_object_iterator = iter(arr)
        return True
    except TypeError as te:
        return False

