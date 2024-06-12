import copy
import os
import torch
import time
from argparse import ArgumentParser, Namespace, FileType
from datetime import datetime
from functools import partial
import numpy as np
import wandb
from biopandas.pdb import PandasPdb
from rdkit import RDLogger, Chem
from torch_geometric.loader import DataLoader

from datasets_utils.pdbbind import PDBBind, read_mol
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import get_symmetry_rmsd, remove_all_hs, read_strings_from_txt
from utils.model_utils import get_model, remove_data_parallel, get_ar_model
from utils.visualise import PDBFile
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
import yaml

cache_name = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f')
parser = ArgumentParser()
parser.add_argument('--config', type=FileType(mode='r'), default=None)
parser.add_argument('--model_dir', type=str, default='workdir', help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--ckpt_encoder', type=str, default='best_inference_epoch_model.pt', help='Checkpoint to use inside the folder')
parser.add_argument('--ckpt_score', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use inside the folder')
parser.add_argument('--ar_model_dir', type=str, default=None, help='Path to folder with trained ar model and hyperparameters')
parser.add_argument('--ar_ckpt', type=str, default='best_model.pt', help='Checkpoint to use inside the folder')
parser.add_argument('--confidence_model_dir', type=str, default=None, help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--confidence_ckpt', type=str, default='best_model.pt', help='Checkpoint to use inside the folder')
parser.add_argument('--num_cpu', type=int, default=None, help='if this is a number instead of none, the max number of cpus used by torch will be set to this.')
parser.add_argument('--run_name', type=str, default='test', help='')
parser.add_argument('--project', type=str, default='ligbind_inf', help='')
parser.add_argument('--out_dir', type=str, default=None, help='Where to save results to')
parser.add_argument('--batch_size', type=int, default=10, help='Number of poses to sample in parallel')
parser.add_argument('--cache_path', type=str, default='data/cacheNew', help='Folder from where to load/restore cached dataset')
parser.add_argument('--data_dir', type=str, default='data/PDBBind_processed/', help='Folder containing original structures')
parser.add_argument('--split_path', type=str, default='data/splits/timesplit_no_lig_overlap_val', help='Path of file defining the split')
parser.add_argument('--no_overlap_names_path', type=str, default='data/splits/timesplit_test_no_rec_overlap', help='Path text file with the folder names in the test set that have no receptor overlap with the train set')
parser.add_argument('--no_model', action='store_true', default=False, help='Whether to return seed conformer without running model')
parser.add_argument('--no_random', action='store_true', default=False, help='Whether to add randomness in diffusion steps')
parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Whether to add noise after the final step')
parser.add_argument('--ode', action='store_true', default=False, help='Whether to run the probability flow ODE')
parser.add_argument('--wandb', action='store_true', default=False, help='')
parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
parser.add_argument('--limit_complexes', type=int, default=0, help='Limit to the number of complexes')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for dataset creation')
parser.add_argument('--tqdm', action='store_true', default=False, help='Whether to show progress bar')
parser.add_argument('--save_visualisation', action='store_true', default=False, help='Whether to save visualizations')
parser.add_argument('--samples_per_complex', type=int, default=1, help='Number of poses to sample for each complex')
parser.add_argument('--actual_steps', type=int, default=None, help='')
parser.add_argument('--oracle', action='store_true', default=False, help='')
parser.add_argument('--limit_failures', type=float, default=3, help='')
parser.add_argument('--gumbel_latent_temperature', type=float, default=0.01, help='Temperature for the Gumbel')
parser.add_argument('--log_softmax_latent_temperature', type=float, default=0.0, help='Log temperature for the Gumbel softmax')
parser.add_argument('--esm_embeddings_path', type=str, default=None, help='')
parser.add_argument('--classifier_free_guidance_weight', type=float, default=0.0, help='')
parser.add_argument('--cfg_start', type=float, default=None, help='')
parser.add_argument('--cfg_end', type=float, default=None, help='')
parser.add_argument('--compute_ar_accuracy', action='store_true', default=False, help='')

parser.add_argument('--overwrite_no_final_step_noise', default=False, help='This sets wandb to True if it is True. It exists for wandb sweeps.')
parser.add_argument('--overwrite_wandb', default=False, help='This sets wandb to True if it is True. It exists for wandb sweeps.')
parser.add_argument('--overwrite_oracle', default=False, help='This sets wandb to True if it is True. It exists for wandb sweeps.')

parser.add_argument('--temp_sampling_tr', type=float, default=1.0)
parser.add_argument('--temp_psi_tr', type=float, default=0.0)
parser.add_argument('--temp_sigma_data_tr', type=float, default=0.5)
parser.add_argument('--temp_sampling_rot', type=float, default=1.0)
parser.add_argument('--temp_psi_rot', type=float, default=0.0)
parser.add_argument('--temp_sigma_data_rot', type=float, default=0.5)
parser.add_argument('--temp_sampling_tor', type=float, default=1.0)
parser.add_argument('--temp_psi_tor', type=float, default=0.0)
parser.add_argument('--temp_sigma_data_tor', type=float, default=0.5)

args = parser.parse_args()

if args.cfg_start is None:
    args.cfg_start = 1.0
if args.cfg_end is None:
    args.cfg_end = 0.0
if args.cfg_start < args.cfg_end:
    if args.wandb:
        run = wandb.init(
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=args
        )
        wandb.log({'filtered_rmsds_below_2': 0.0})

    raise Exception('cfg_start must be greater than cfg_end')

if args.overwrite_wandb:  # This sets wandb to True if it is True. It exists for wandb sweeps.
    args.wandb = True
if args.overwrite_no_final_step_noise:  # This sets wandb to True if it is True. It exists for wandb sweeps.
    args.no_final_step_noise = True
if args.overwrite_oracle:  # This sets wandb to True if it is True. It exists for wandb sweeps.
    args.oracle = True

if args.out_dir is None: args.out_dir = f'inference_out_dir_not_specified/{args.run_name}'
os.makedirs(args.out_dir, exist_ok=True)
with open(f'{args.model_dir}/model_parameters.yml') as f:
    score_model_args = Namespace(**yaml.full_load(f))

ar_args = None
if args.ar_model_dir is not None:
    with open(f'{args.ar_model_dir}/model_parameters.yml') as f:
        ar_args = Namespace(**yaml.full_load(f))

if args.confidence_model_dir is not None:
    with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
        confidence_args = Namespace(**yaml.full_load(f))
    if not os.path.exists(confidence_args.original_model_dir):
        print("Path does not exist: ", confidence_args.original_model_dir)
        confidence_args.original_model_dir = os.path.join(*confidence_args.original_model_dir.split('/')[-2:])
        print('instead trying path: ', confidence_args.original_model_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_dataset = PDBBind(transform=None, root=args.data_dir, limit_complexes=args.limit_complexes,
                       receptor_radius=score_model_args.receptor_radius,
                       cache_path=args.cache_path, split_path=args.split_path,
                       remove_hs=score_model_args.remove_hs, max_lig_size=None,
                       c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
                       matching=not score_model_args.no_torsion, keep_original=True,
                       popsize=score_model_args.matching_popsize,
                       maxiter=score_model_args.matching_maxiter,
                       all_atoms=score_model_args.all_atoms,
                       atom_radius=score_model_args.atom_radius,
                       atom_max_neighbors=score_model_args.atom_max_neighbors,
                       esm_embeddings_path=args.esm_embeddings_path,
                       require_ligand=True,
                       num_workers=args.num_workers)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

if args.confidence_model_dir is not None:
    if not (confidence_args.use_original_model_cache or confidence_args.transfer_weights):
        # if the confidence model uses the same type of data as the original model then we do not need this dataset
        print('Loading data for the confidence model')
        confidence_test_dataset = PDBBind(transform=None, root=args.data_dir, limit_complexes=args.limit_complexes,
                                          receptor_radius=confidence_args.receptor_radius,
                                          cache_path=args.cache_path, split_path=args.split_path,
                                          remove_hs=confidence_args.remove_hs, max_lig_size=None,
                                          c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                                          matching=not confidence_args.no_torsion, keep_original=True,
                                          popsize=confidence_args.matching_popsize,
                                          maxiter=confidence_args.matching_maxiter,
                                          all_atoms=confidence_args.all_atoms,
                                          atom_radius=confidence_args.atom_radius,
                                          atom_max_neighbors=confidence_args.atom_max_neighbors,
                                          esm_embeddings_path=args.esm_embeddings_path, require_ligand=True,
                                          num_workers=args.num_workers)
        confidence_complex_dict = {d.name: d for d in confidence_test_dataset}

t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

if not args.no_model:
    model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True)

    if args.oracle or args.ckpt_score is None or args.compute_ar_accuracy:
        encoder_state_dict = torch.load(f'{args.model_dir}/{args.ckpt_encoder}', map_location=torch.device('cpu'))
        encoder_state_dict = remove_data_parallel(encoder_state_dict)
        model.load_state_dict(encoder_state_dict, strict=True)

    if args.ckpt_score is not None:
        score_state_dict = torch.load(f'{args.model_dir}/{args.ckpt_score}', map_location=torch.device('cpu'))
        model.score_model.load_state_dict(score_state_dict, strict=True)

    model = model.to(device)
    model.eval()

    if not args.oracle:
        assert args.ar_model_dir is not None, 'AR model directory must be specified if not using oracle'
        ar_model = get_ar_model(ar_args, score_model_args, device, training=False)
        ar_state_dict = torch.load(f'{args.ar_model_dir}/{args.ar_ckpt}', map_location=torch.device('cpu'))
        ar_model.load_state_dict(ar_state_dict, strict=True)
        ar_model.eval()

    if args.confidence_model_dir is not None:
        if confidence_args.transfer_weights:
            with open(f'{confidence_args.original_model_dir}/model_parameters.yml') as f:
                confidence_model_args = Namespace(**yaml.full_load(f))
        else:
            confidence_model_args = confidence_args

        confidence_model = get_model(confidence_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True,
                                     confidence_mode=True)
        state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location=torch.device('cpu'))
        confidence_model.load_state_dict(state_dict, strict=True)
        confidence_model = confidence_model.to(device)
        confidence_model.eval()
    else:
        confidence_model = None
        confidence_args = None
        confidence_model_args = None

if args.wandb:
    run = wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project=args.project,
        name=args.run_name,
        config=args
    )

tr_schedule = get_t_schedule(inference_steps=args.inference_steps)
rot_schedule = tr_schedule
tor_schedule = tr_schedule
print('t schedule', tr_schedule)

rmsds_list, obrmsds, centroid_distances_list, failures, skipped, min_cross_distances_list, base_min_cross_distances_list, confidences_list, names_list = [], [], [], 0, 0, [], [], [], []
run_times, min_self_distances_list, without_rec_overlap_list = [], [], []
N = args.samples_per_complex
names_no_rec_overlap = read_strings_from_txt(args.no_overlap_names_path)
ar_accuracy = []
print('Size of test dataset: ', len(test_dataset))

for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
    if confidence_model is not None and not (confidence_args.use_original_model_cache or confidence_args.transfer_weights) \
            and orig_complex_graph.name[0] not in confidence_complex_dict.keys():
        skipped += 1
        print(f"The confidence dataset did not contain {orig_complex_graph.name[0]}. We are skipping this complex.")
        continue

    success = 0
    bs = args.batch_size
    while 0 >= success > -args.limit_failures:
        try:
            data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
            randomize_position(data_list, score_model_args.no_torsion, args.no_random, score_model_args.tr_sigma_max, ar_args=ar_args)

            pdb = None
            if args.save_visualisation:
                visualization_list = []
                for idx, graph in enumerate(data_list):
                    lig = read_mol(args.data_dir, graph['name'][0], remove_hs=score_model_args.remove_hs)
                    pdb = PDBFile(lig)
                    pdb.add(lig, 0, 0)
                    pdb.add((orig_complex_graph['ligand'].pos + orig_complex_graph.original_center).detach().cpu(), 1,
                            0)
                    pdb.add((graph['ligand'].pos + graph.original_center).detach().cpu(), part=1, order=1)
                    visualization_list.append(pdb)
            else:
                visualization_list = None

            rec_path = os.path.join(args.data_dir, data_list[0]["name"][0],
                                    f'{data_list[0]["name"][0]}_protein_processed.pdb')
            if not os.path.exists(rec_path):
                rec_path = os.path.join(args.data_dir, data_list[0]["name"][0],
                                        f'{data_list[0]["name"][0]}_protein_obabel_reduce.pdb')
            rec = PandasPdb().read_pdb(rec_path)
            rec_df = rec.df['ATOM']
            receptor_pos = rec_df[['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(
                np.float32) - orig_complex_graph.original_center.cpu().numpy()
            receptor_pos = np.tile(receptor_pos, (N, 1, 1))
            start_time = time.time()
            if not args.no_model:
                if confidence_model is not None and not (
                        confidence_args.use_original_model_cache or confidence_args.transfer_weights):
                    confidence_data_list = [copy.deepcopy(confidence_complex_dict[orig_complex_graph.name[0]]) for _ in
                                            range(N)]
                else:
                    confidence_data_list = None

                data_list, confidence = sampling(data_list=data_list, model=model,
                                                 inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                                 tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                 tor_schedule=tor_schedule,
                                                 device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                                                 no_random=args.no_random,
                                                 ode=args.ode, visualization_list=visualization_list,
                                                 confidence_model=confidence_model,
                                                 confidence_data_list=confidence_data_list,
                                                 confidence_model_args=confidence_model_args,
                                                 batch_size=bs,
                                                 no_final_step_noise=args.no_final_step_noise,
                                                 gumbel_latent_temperature=args.gumbel_latent_temperature,
                                                 ar_model=None if args.oracle else ar_model,
                                                 ar_args=ar_args,
                                                 temp_sampling=[args.temp_sampling_tr, args.temp_sampling_rot, args.temp_sampling_tor],
                                                 temp_psi=[args.temp_psi_tr, args.temp_psi_rot, args.temp_psi_tor],
                                                 temp_sigma_data=[args.temp_sigma_data_tr, args.temp_sigma_data_rot, args.temp_sigma_data_tor],
                                                 classifier_free_guidance_weight=args.classifier_free_guidance_weight,
                                                 softmax_latent_temperature=np.exp(args.log_softmax_latent_temperature),
                                                 cfg_start=args.cfg_start,
                                                 cfg_end=args.cfg_end,
                                                 compute_ar_accuracy=args.compute_ar_accuracy,
                                                 )

            run_times.append(time.time() - start_time)
            if score_model_args.no_torsion: orig_complex_graph['ligand'].orig_pos = \
                (orig_complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

            filterHs = torch.not_equal(data_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

            if isinstance(orig_complex_graph['ligand'].orig_pos, list):
                orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

            ligand_pos = np.asarray(
                [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in data_list])
            orig_ligand_pos = np.expand_dims(
                orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy(),
                axis=0)

            try:
                mol = remove_all_hs(orig_complex_graph.mol[0])
                rmsd = get_symmetry_rmsd(mol, orig_ligand_pos[0], [l for l in ligand_pos])
            except Exception as e:
                print("Using non corrected RMSD because of the error", e)
                rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
            rmsds_list.append(rmsd)
            centroid_distance = np.linalg.norm(ligand_pos.mean(axis=1) - orig_ligand_pos.mean(axis=1), axis=1)
            lat_strs = np.asarray([complex_graph.latent_str for complex_graph in data_list]) if hasattr(score_model_args, 'latent_dim') and score_model_args.latent_dim > 0 else np.zeros(len(data_list))
            if confidence is not None and isinstance(confidence_args.rmsd_classification_cutoff, list):
                confidence = confidence[:, 0]
            if confidence is not None:
                confidence = confidence.cpu().numpy()
                re_order = np.argsort(confidence)[::-1]
                print(orig_complex_graph['name'], ' rmsd', np.around(rmsd, 1)[re_order], ' centroid distance',
                      np.around(centroid_distance, 1)[re_order], ' confidences ', np.around(confidence, 4)[re_order],
                      ' latents', lat_strs[re_order])
                confidences_list.append(confidence)
            else:
                print(orig_complex_graph['name'], ' rmsd', np.around(rmsd, 1), ' centroid distance',
                      np.around(centroid_distance, 1), ' latents', lat_strs)
            centroid_distances_list.append(centroid_distance)

            cross_distances = np.linalg.norm(receptor_pos[:, :, None, :] - ligand_pos[:, None, :, :], axis=-1)
            min_cross_distances_list.append(np.min(cross_distances, axis=(1, 2)))
            self_distances = np.linalg.norm(ligand_pos[:, :, None, :] - ligand_pos[:, None, :, :], axis=-1)
            self_distances = np.where(np.eye(self_distances.shape[2]), np.inf, self_distances)
            min_self_distances_list.append(np.min(self_distances, axis=(1, 2)))

            base_cross_distances = np.linalg.norm(receptor_pos[:, :, None, :] - orig_ligand_pos[:, None, :, :], axis=-1)
            base_min_cross_distances_list.append(np.min(base_cross_distances, axis=(1, 2)))

            if args.compute_ar_accuracy:
                ar_acc = np.asarray([complex_graph.ar_accuracy for complex_graph in data_list])
                ar_accuracy.append(ar_acc.mean())

            if args.save_visualisation:
                os.mkdir(f'{args.out_dir}/{data_list[0]["name"][0]}')
                if confidence is not None:
                    for rank, batch_idx in enumerate(re_order):

                        extra = ""
                        if hasattr(score_model_args, 'latent_dim') and score_model_args.latent_dim > 0:
                            extra = '_lat' + data_list[batch_idx].latent_str

                            for l in range(len(data_list[batch_idx].latent_pos)):
                                mol = Chem.RWMol()
                                mol.AddAtom(Chem.Atom(30))
                                conf = Chem.Conformer(1)
                                conf.SetAtomPosition(0, (data_list[batch_idx].latent_pos[l, 0].item(),
                                                         data_list[batch_idx].latent_pos[l, 1].item(),
                                                         data_list[batch_idx].latent_pos[l, 2].item()))
                                mol.AddConformer(conf)
                                mol = mol.GetMol()

                                pdb = PDBFile(mol)
                                pdb.add(mol, 0, 0)
                                pdb.write(f'{args.out_dir}/{data_list[batch_idx]["name"][0]}/{rank + 1}_latent{l}.pdb')

                        visualization_list[batch_idx].write(
                            f'{args.out_dir}/{data_list[batch_idx]["name"][0]}/{rank + 1}_{rmsd[batch_idx]:.1f}_{(confidence)[batch_idx]:.1f}{extra}.pdb')
                else:
                    for rank, batch_idx in enumerate(np.argsort(rmsd)):
                        extra = ""
                        if hasattr(score_model_args, 'latent_dim') and score_model_args.latent_dim > 0:
                            extra = '_lat' + data_list[batch_idx].latent_str

                            for l in range(len(data_list[batch_idx].latent_pos)):
                                mol = Chem.RWMol()
                                mol.AddAtom(Chem.Atom(30))
                                conf = Chem.Conformer(1)
                                conf.SetAtomPosition(0, (data_list[batch_idx].latent_pos[l, 0].item(),
                                                         data_list[batch_idx].latent_pos[l, 1].item(),
                                                         data_list[batch_idx].latent_pos[l, 2].item()))
                                mol.AddConformer(conf)
                                mol = mol.GetMol()

                                pdb = PDBFile(mol)
                                pdb.add(mol, 0, 0)
                                pdb.write(f'{args.out_dir}/{data_list[batch_idx]["name"][0]}/{rank + 1}_latent{l}.pdb')

                        visualization_list[batch_idx].write(
                            f'{args.out_dir}/{data_list[batch_idx]["name"][0]}/{rank + 1}_{rmsd[batch_idx]:.1f}{extra}.pdb')
            without_rec_overlap_list.append(1 if orig_complex_graph.name[0] in names_no_rec_overlap else 0)
            names_list.append(orig_complex_graph.name[0])
            success = 1
        except Exception as e:
            print("Failed on", orig_complex_graph["name"], e)
            success -= 1
            if bs > 1:
                bs = bs // 2

    if success != 1:
        rmsds_list.append(np.zeros(args.samples_per_complex) + 10000)
        if confidence_model_args is not None:
            confidences_list.append(np.zeros(args.samples_per_complex) - 10000)
        centroid_distances_list.append(np.zeros(args.samples_per_complex) + 10000)
        min_self_distances_list.append(np.zeros(args.samples_per_complex) + 10000)
        without_rec_overlap_list.append(1 if orig_complex_graph.name[0] in names_no_rec_overlap else 0)
        names_list.append(orig_complex_graph.name[0])
        failures += 1

print('Performance without hydrogens included in the loss')
print(failures, "failures due to exceptions")
print(skipped, ' skipped because complex was not in confidence dataset')

performance_metrics = {}

if args.compute_ar_accuracy:
    performance_metrics['ar_accuracy'] = np.mean(np.asarray(ar_accuracy))
    print('average ar_accuracy', performance_metrics['ar_accuracy'])

for overlap in ['', 'no_overlap_']:
    if 'no_overlap_' == overlap:
        without_rec_overlap = np.array(without_rec_overlap_list, dtype=bool)
        if without_rec_overlap.sum() == 0: continue
        rmsds = np.array(rmsds_list)[without_rec_overlap]
        min_self_distances = np.array(min_self_distances_list)[without_rec_overlap]
        centroid_distances = np.array(centroid_distances_list)[without_rec_overlap]
        confidences = np.array(confidences_list)[without_rec_overlap]
        min_cross_distances = np.array(min_cross_distances_list)[without_rec_overlap]
        base_min_cross_distances = np.array(base_min_cross_distances_list)[without_rec_overlap]
        names = np.array(names_list)[without_rec_overlap]
    else:
        rmsds = np.array(rmsds_list)
        min_self_distances = np.array(min_self_distances_list)
        centroid_distances = np.array(centroid_distances_list)
        confidences = np.array(confidences_list)
        min_cross_distances = np.array(min_cross_distances_list)
        base_min_cross_distances = np.array(base_min_cross_distances_list)
        names = np.array(names_list)

    run_times = np.array(run_times)
    np.save(f'{args.out_dir}/{overlap}min_cross_distances.npy', min_cross_distances)
    np.save(f'{args.out_dir}/{overlap}min_self_distances.npy', min_self_distances)
    np.save(f'{args.out_dir}/{overlap}base_min_cross_distances.npy', base_min_cross_distances)
    np.save(f'{args.out_dir}/{overlap}rmsds.npy', rmsds)
    np.save(f'{args.out_dir}/{overlap}centroid_distances.npy', centroid_distances)
    np.save(f'{args.out_dir}/{overlap}confidences.npy', confidences)
    np.save(f'{args.out_dir}/{overlap}run_times.npy', run_times)
    np.save(f'{args.out_dir}/{overlap}complex_names.npy', np.array(names))

    performance_metrics.update({
        f'{overlap}run_times_std': run_times.std().__round__(2),
        f'{overlap}run_times_mean': run_times.mean().__round__(2),
        f'{overlap}steric_clash_fraction': (
                100 * (min_cross_distances < 0.4).sum() / len(min_cross_distances) / N).__round__(2),
        f'{overlap}self_intersect_fraction': (
                100 * (min_self_distances < 0.4).sum() / len(min_self_distances) / N).__round__(2),
        f'{overlap}mean_rmsd': rmsds.mean(),
        f'{overlap}rmsds_below_2': (100 * (rmsds < 2).sum() / len(rmsds) / N),
        f'{overlap}rmsds_below_5': (100 * (rmsds < 5).sum() / len(rmsds) / N),
        f'{overlap}rmsds_percentile_25': np.percentile(rmsds, 25).round(2),
        f'{overlap}rmsds_percentile_50': np.percentile(rmsds, 50).round(2),
        f'{overlap}rmsds_percentile_75': np.percentile(rmsds, 75).round(2),

        f'{overlap}mean_centroid': centroid_distances.mean().__round__(2),
        f'{overlap}centroid_below_2': (100 * (centroid_distances < 2).sum() / len(centroid_distances) / N).__round__(2),
        f'{overlap}centroid_below_5': (100 * (centroid_distances < 5).sum() / len(centroid_distances) / N).__round__(2),
        f'{overlap}centroid_percentile_25': np.percentile(centroid_distances, 25).round(2),
        f'{overlap}centroid_percentile_50': np.percentile(centroid_distances, 50).round(2),
        f'{overlap}centroid_percentile_75': np.percentile(centroid_distances, 75).round(2),
    })

    if N >= 5:
        top5_rmsds = np.min(rmsds[:, :5], axis=1)
        top5_centroid_distances = centroid_distances[
                                      np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :5], axis=1)][:, 0]
        performance_metrics.update({
            f'{overlap}top5_rmsds_below_2': (100 * (top5_rmsds < 2).sum() / len(top5_rmsds)).__round__(2),
            f'{overlap}top5_rmsds_below_5': (100 * (top5_rmsds < 5).sum() / len(top5_rmsds)).__round__(2),
            f'{overlap}top5_rmsds_percentile_25': np.percentile(top5_rmsds, 25).round(2),
            f'{overlap}top5_rmsds_percentile_50': np.percentile(top5_rmsds, 50).round(2),
            f'{overlap}top5_rmsds_percentile_75': np.percentile(top5_rmsds, 75).round(2),

            f'{overlap}top5_centroid_below_2': (
                    100 * (top5_centroid_distances < 2).sum() / len(top5_centroid_distances)).__round__(2),
            f'{overlap}top5_centroid_below_5': (
                    100 * (top5_centroid_distances < 5).sum() / len(top5_centroid_distances)).__round__(2),
            f'{overlap}top5_centroid_percentile_25': np.percentile(top5_centroid_distances, 25).round(2),
            f'{overlap}top5_centroid_percentile_50': np.percentile(top5_centroid_distances, 50).round(2),
            f'{overlap}top5_centroid_percentile_75': np.percentile(top5_centroid_distances, 75).round(2),
        })

    if N >= 10:
        top10_rmsds = np.min(rmsds[:, :10], axis=1)
        top10_centroid_distances = centroid_distances[
                                       np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :10], axis=1)][:, 0]
        performance_metrics.update({
            f'{overlap}top10_rmsds_below_2': (100 * (top10_rmsds < 2).sum() / len(top10_rmsds)).__round__(2),
            f'{overlap}top10_rmsds_below_5': (100 * (top10_rmsds < 5).sum() / len(top10_rmsds)).__round__(2),
            f'{overlap}top10_rmsds_percentile_25': np.percentile(top10_rmsds, 25).round(2),
            f'{overlap}top10_rmsds_percentile_50': np.percentile(top10_rmsds, 50).round(2),
            f'{overlap}top10_rmsds_percentile_75': np.percentile(top10_rmsds, 75).round(2),

            f'{overlap}top10_centroid_below_2': (
                    100 * (top10_centroid_distances < 2).sum() / len(top10_centroid_distances)).__round__(2),
            f'{overlap}top10_centroid_below_5': (
                    100 * (top10_centroid_distances < 5).sum() / len(top10_centroid_distances)).__round__(2),
            f'{overlap}top10_centroid_percentile_25': np.percentile(top10_centroid_distances, 25).round(2),
            f'{overlap}top10_centroid_percentile_50': np.percentile(top10_centroid_distances, 50).round(2),
            f'{overlap}top10_centroid_percentile_75': np.percentile(top10_centroid_distances, 75).round(2),
        })

    if confidence_model is not None:
        confidence_ordering = np.argsort(confidences, axis=1)[:, ::-1]

        filtered_rmsds = rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, 0]
        filtered_centroid_distances = centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, 0]
        performance_metrics.update({
            f'{overlap}filtered_rmsds_below_2': (100 * (filtered_rmsds < 2).sum() / len(filtered_rmsds)).__round__(2),
            f'{overlap}filtered_rmsds_below_5': (100 * (filtered_rmsds < 5).sum() / len(filtered_rmsds)).__round__(2),
            f'{overlap}filtered_rmsds_percentile_25': np.percentile(filtered_rmsds, 25).round(2),
            f'{overlap}filtered_rmsds_percentile_50': np.percentile(filtered_rmsds, 50).round(2),
            f'{overlap}filtered_rmsds_percentile_75': np.percentile(filtered_rmsds, 75).round(2),

            f'{overlap}filtered_centroid_below_2': (
                    100 * (filtered_centroid_distances < 2).sum() / len(filtered_centroid_distances)).__round__(2),
            f'{overlap}filtered_centroid_below_5': (
                    100 * (filtered_centroid_distances < 5).sum() / len(filtered_centroid_distances)).__round__(2),
            f'{overlap}filtered_centroid_percentile_25': np.percentile(filtered_centroid_distances, 25).round(2),
            f'{overlap}filtered_centroid_percentile_50': np.percentile(filtered_centroid_distances, 50).round(2),
            f'{overlap}filtered_centroid_percentile_75': np.percentile(filtered_centroid_distances, 75).round(2),
        })

        if N >= 5:
            top5_filtered_rmsds = np.min(rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5], axis=1)
            top5_filtered_centroid_distances = \
                centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5][
                    np.arange(rmsds.shape[0])[:, None], np.argsort(
                        rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5], axis=1)][:, 0]
            performance_metrics.update({
                f'{overlap}top5_filtered_rmsds_below_2': (
                        100 * (top5_filtered_rmsds < 2).sum() / len(top5_filtered_rmsds)).__round__(2),
                f'{overlap}top5_filtered_rmsds_below_5': (
                        100 * (top5_filtered_rmsds < 5).sum() / len(top5_filtered_rmsds)).__round__(2),
                f'{overlap}top5_filtered_rmsds_percentile_25': np.percentile(top5_filtered_rmsds, 25).round(2),
                f'{overlap}top5_filtered_rmsds_percentile_50': np.percentile(top5_filtered_rmsds, 50).round(2),
                f'{overlap}top5_filtered_rmsds_percentile_75': np.percentile(top5_filtered_rmsds, 75).round(2),

                f'{overlap}top5_filtered_centroid_below_2': (100 * (top5_filtered_centroid_distances < 2).sum() / len(
                    top5_filtered_centroid_distances)).__round__(2),
                f'{overlap}top5_filtered_centroid_below_5': (100 * (top5_filtered_centroid_distances < 5).sum() / len(
                    top5_filtered_centroid_distances)).__round__(2),
                f'{overlap}top5_filtered_centroid_percentile_25': np.percentile(top5_filtered_centroid_distances,
                                                                                25).round(2),
                f'{overlap}top5_filtered_centroid_percentile_50': np.percentile(top5_filtered_centroid_distances,
                                                                                50).round(2),
                f'{overlap}top5_filtered_centroid_percentile_75': np.percentile(top5_filtered_centroid_distances,
                                                                                75).round(2),
            })
        if N >= 10:
            top10_filtered_rmsds = np.min(rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10],
                                          axis=1)
            top10_filtered_centroid_distances = \
                centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10][
                    np.arange(rmsds.shape[0])[:, None], np.argsort(
                        rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10], axis=1)][:, 0]
            performance_metrics.update({
                f'{overlap}top10_filtered_rmsds_below_2': (
                        100 * (top10_filtered_rmsds < 2).sum() / len(top10_filtered_rmsds)).__round__(2),
                f'{overlap}top10_filtered_rmsds_below_5': (
                        100 * (top10_filtered_rmsds < 5).sum() / len(top10_filtered_rmsds)).__round__(2),
                f'{overlap}top10_filtered_rmsds_percentile_25': np.percentile(top10_filtered_rmsds, 25).round(2),
                f'{overlap}top10_filtered_rmsds_percentile_50': np.percentile(top10_filtered_rmsds, 50).round(2),
                f'{overlap}top10_filtered_rmsds_percentile_75': np.percentile(top10_filtered_rmsds, 75).round(2),

                f'{overlap}top10_filtered_centroid_below_2': (100 * (top10_filtered_centroid_distances < 2).sum() / len(
                    top10_filtered_centroid_distances)).__round__(2),
                f'{overlap}top10_filtered_centroid_below_5': (100 * (top10_filtered_centroid_distances < 5).sum() / len(
                    top10_filtered_centroid_distances)).__round__(2),
                f'{overlap}top10_filtered_centroid_percentile_25': np.percentile(top10_filtered_centroid_distances,
                                                                                 25).round(2),
                f'{overlap}top10_filtered_centroid_percentile_50': np.percentile(top10_filtered_centroid_distances,
                                                                                 50).round(2),
                f'{overlap}top10_filtered_centroid_percentile_75': np.percentile(top10_filtered_centroid_distances,
                                                                                 75).round(2),
            })

for k in performance_metrics:
    print(k, performance_metrics[k])

if args.wandb:
    wandb.log(performance_metrics)
    histogram_metrics_list = [('rmsd', rmsds[:, 0]),
                              ('centroid_distance', centroid_distances[:, 0]),
                              ('mean_rmsd', rmsds.mean(axis=1)),
                              ('mean_centroid_distance', centroid_distances.mean(axis=1))]
    if N >= 5:
        histogram_metrics_list.append(('top5_rmsds', top5_rmsds))
        histogram_metrics_list.append(('top5_centroid_distances', top5_centroid_distances))
    if N >= 10:
        histogram_metrics_list.append(('top10_rmsds', top10_rmsds))
        histogram_metrics_list.append(('top10_centroid_distances', top10_centroid_distances))
    if confidence_model is not None:
        histogram_metrics_list.append(('filtered_rmsd', filtered_rmsds))
        histogram_metrics_list.append(('filtered_centroid_distance', filtered_centroid_distances))
        if N >= 5:
            histogram_metrics_list.append(('top5_filtered_rmsds', top5_filtered_rmsds))
            histogram_metrics_list.append(('top5_filtered_centroid_distances', top5_filtered_centroid_distances))
        if N >= 10:
            histogram_metrics_list.append(('top10_filtered_rmsds', top10_filtered_rmsds))
            histogram_metrics_list.append(('top10_filtered_centroid_distances', top10_filtered_centroid_distances))
