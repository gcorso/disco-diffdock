import gc
import os
import shutil
from argparse import Namespace, ArgumentParser
import torch.nn.functional as F
import wandb
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from autoregressive.dataset_ar import AutoregressiveDataset
from utils.training import AverageMeter

torch.multiprocessing.set_sharing_strategy('file_system')

import yaml
from utils.utils import save_yaml_file, get_optimizer_and_scheduler
from utils.model_utils import get_ar_model

parser = ArgumentParser()
parser.add_argument('--original_model_dir', type=str, default='workdir',
                    help='Path to folder with trained model and hyperparameters')
parser.add_argument('--restart_dir', type=str, default=None, help='')
parser.add_argument('--data_dir', type=str, default='data/PDBBind_processed/',
                    help='Folder containing original structures')
parser.add_argument('--ckpt', type=str, default='best_model.pt', help='Checkpoint to use inside the folder')
parser.add_argument('--model_save_frequency', type=int, default=0,
                    help='Frequency with which to save the last model. If 0, then only the early stopping criterion best model is saved and overwritten.')
parser.add_argument('--best_model_save_frequency', type=int, default=0,
                    help='Frequency with which to save the best model. If 0, then only the early stopping criterion best model is saved and overwritten.')
parser.add_argument('--run_name', type=str, default='test_autoregressive', help='')
parser.add_argument('--project', type=str, default='latent_ar', help='')
parser.add_argument('--split_train', type=str, default='data/splits/timesplit_no_lig_overlap_train',
                    help='Path of file defining the split')
parser.add_argument('--split_val', type=str, default='data/splits/timesplit_no_lig_overlap_val',
                    help='Path of file defining the split')
parser.add_argument('--split_test', type=str, default='data/splits/timesplit_test',
                    help='Path of file defining the split')
parser.add_argument('--cudnn_benchmark', action='store_true', default=False, help='CUDA optimization parameter for faster training')

# Inference parameters for creating the positions and rmsds that the confidence predictor will be trained on.
parser.add_argument('--cache_path', type=str, default='data/cache',
                    help='Folder from where to load/restore cached dataset')
parser.add_argument('--wandb', action='store_true', default=False, help='')
parser.add_argument('--log_dir', type=str, default='workdir', help='')
parser.add_argument('--main_metric', type=str, default='accuracy',
                    help='Metric to track for early stopping. Mostly [loss, accuracy, ROC AUC]')
parser.add_argument('--main_metric_goal', type=str, default='max', help='Can be [min, max]')
parser.add_argument('--batch_size', type=int, default=5, help='')
parser.add_argument('--lr', type=float, default=1e-3, help='')
parser.add_argument('--w_decay', type=float, default=0.0, help='')
parser.add_argument('--scheduler', type=str, default='plateau', help='')
parser.add_argument('--scheduler_patience', type=int, default=20, help='')
parser.add_argument('--n_epochs', type=int, default=500, help='')
parser.add_argument('--num_accumulation_steps', type=int, default=1, help='')

# Dataset
parser.add_argument('--limit_complexes', type=int, default=0, help='')
parser.add_argument('--all_atoms', action='store_true', default=False, help='')
parser.add_argument('--train_multiplicity', type=int, default=1, help='')
parser.add_argument('--val_multiplicity', type=int, default=1, help='')
parser.add_argument('--chain_cutoff', type=float, default=10, help='')
parser.add_argument('--receptor_radius', type=float, default=30, help='')
parser.add_argument('--c_alpha_max_neighbors', type=int, default=10, help='')
parser.add_argument('--atom_radius', type=float, default=5, help='')
parser.add_argument('--atom_max_neighbors', type=int, default=8, help='')
parser.add_argument('--matching_popsize', type=int, default=20, help='')
parser.add_argument('--matching_maxiter', type=int, default=20, help='')
parser.add_argument('--max_lig_size', type=int, default=None, help='Maximum number of heavy atoms')
parser.add_argument('--remove_hs', action='store_true', default=False, help='remove Hs')
parser.add_argument('--num_conformers', type=int, default=1, help='')
parser.add_argument('--esm_embeddings_path', type=str, default=None,
                    help='If this is set then the LM embeddings at that path will be used for the receptor features')
parser.add_argument('--no_torsion', action='store_true', default=False, help='')
parser.add_argument('--no_sampling', action='store_true', default=False, help='')

# Model
parser.add_argument('--num_conv_layers', type=int, default=2, help='Number of interaction layers')
parser.add_argument('--max_radius', type=float, default=5.0, help='Radius cutoff for geometric graph')
parser.add_argument('--ns', type=int, default=16, help='Number of hidden features per node of order 0')
parser.add_argument('--nv', type=int, default=4, help='Number of hidden features per node of order >0')
parser.add_argument('--distance_embed_dim', type=int, default=32, help='')
parser.add_argument('--cross_distance_embed_dim', type=int, default=32, help='')
parser.add_argument('--no_batch_norm', action='store_true', default=False, help='If set, it removes the batch norm')
parser.add_argument('--sh_lmax', type=int, default=2, help='')
parser.add_argument('--use_second_order_repr', action='store_true', default=False,
                    help='Whether to use only up to first order representations or also second')
parser.add_argument('--cross_max_distance', type=float, default=80, help='')
parser.add_argument('--dropout', type=float, default=0.0, help='MLP dropout')
parser.add_argument('--latent_no_batchnorm', action='store_true', default=False,
                    help='If set, it removes the batch norm')
parser.add_argument('--latent_dropout', type=float, default=0.0, help='MLP dropout')
parser.add_argument('--latent_hidden_dim', type=int, default=128, help='Size of the hidden layer in the latent space')
parser.add_argument('--sampling_latent_temperature', type=float, default=0.01, help='')
parser.add_argument('--no_randomness', action='store_true', default=False, help='')
parser.add_argument('--overfit', action='store_true', default=False, help='')
parser.add_argument('--latent_virtual_nodes', action='store_true', default=False, help='')
parser.add_argument('--latent_nodes_residual', action='store_true', default=False, help='')
parser.add_argument('--use_pretrained_score', action='store_true', default=False, help='')
parser.add_argument('--warmup_epochs', type=int, default=0, help='')
parser.add_argument('--compute_ar_accuracy', action='store_true', default=False, help='')

args = parser.parse_args()
assert (args.main_metric_goal == 'max' or args.main_metric_goal == 'min')


def train_epoch(model, loader, optimizer, device, no_sampling, num_accumulation_steps=1):
    model.train()
    meter = AverageMeter(['ar_loss'])

    for idx, data in tqdm(enumerate(loader), total=len(loader)):
        try:
            data = data.to(device)
            pred = model(data)

            if not isinstance(pred, list):
                latent_loss = F.cross_entropy(pred[:, 0], data.latent_label)
            else:
                latent_loss = 0
                for i in range(len(pred)):
                    if no_sampling:
                        probs = torch.from_numpy(data.latent_label[i]).to(device)
                        latent_loss = latent_loss + F.cross_entropy(pred[i][0, 0].unsqueeze(0), probs.unsqueeze(0))
                    else:
                        latent_loss = latent_loss + F.cross_entropy(pred[i][0, 0], data.latent_label[i])

                latent_loss = latent_loss / len(pred)

            latent_loss = latent_loss / num_accumulation_steps
            latent_loss.backward()

            if ((idx + 1) % num_accumulation_steps == 0) or (idx + 1 == len(loader)):
                optimizer.step()
                optimizer.zero_grad()

            meter.add([latent_loss.cpu().detach()])
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e

    return meter.summary()


def test_epoch(model, loader, device, no_sampling, num_latents):
    model.eval()
    meter = AverageMeter(['ar_loss', 'accuracy'] + [f'accuracy{i}' for i in range(num_latents)], unpooled_metrics=True)
    meter_all = AverageMeter( ['ar_loss', 'accuracy'], unpooled_metrics=True, intervals=num_latents)

    for data in tqdm(loader, total=len(loader)):
        try:
            data = data.to(device)
            with torch.no_grad():
                pred = model(data)

            if not isinstance(pred, list):
                latent_loss = F.cross_entropy(pred[:, 0], data.latent_label)
                accuracy = torch.mean((data.latent_label == torch.argmax(pred[:, 0], dim=-1)).float())
                raise NotImplementedError

            else:
                latent_loss, accuracy = torch.zeros(len(pred)), torch.zeros(len(pred))
                for i in range(len(pred)):
                    if no_sampling:
                        probs = torch.from_numpy(data.latent_label[i]).to(device)
                        latent_loss[i] = F.cross_entropy(pred[i][0, 0].unsqueeze(0), probs.unsqueeze(0)).cpu().detach()
                        accuracy[i] = 1 if torch.argmax(pred[i][0, 0]) == torch.argmax(probs) else 0
                    else:
                        latent_loss[i] = F.cross_entropy(pred[i][0, 0], data.latent_label[i])
                        accuracy[i] = 1 if torch.argmax(pred[i][0, 0]) == data.latent_label[i] else 0

            meter.add([latent_loss, accuracy])
            meter_all.add([latent_loss, accuracy], interval_idx=[data.decoding_idx.cpu().detach(), data.decoding_idx.cpu().detach()])

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    out = meter.summary()
    out.update(meter_all.summary())

    if args.compute_ar_accuracy:
        new_loader = DataLoader(loader.dataset, batch_size=1, shuffle=False, num_workers=0)
        ar_accuracy = 0
        for data in tqdm(new_loader, total=len(new_loader)):
            with torch.no_grad():
                predicted_latent = model.encode_ar(data)

            predicted_latent = torch.cat(predicted_latent, dim=0).cpu().detach()
            true_latent = torch.from_numpy(data.true_latent[0])

            # take only first latent
            predicted_latent = predicted_latent[:, 0]
            true_latent = true_latent[:, 0]

            ar_accuracy += 1 if torch.argmax(true_latent) == torch.argmax(predicted_latent) else 0

        ar_accuracy = ar_accuracy / len(new_loader)
        print("AR accuracy: ", ar_accuracy)
        out.update({'ar_accuracy': ar_accuracy})

    return out


def train(args, model, optimizer, scheduler, train_loader, val_loader, run_dir, latent_dim):
    best_val_accuracy = 0
    best_val_loss = 100000
    best_epoch = 0

    print("Starting training...")
    for epoch in range(args.n_epochs):
        logs = {}
        train_metrics = train_epoch(model, train_loader, optimizer, args.device, args.no_sampling, args.num_accumulation_steps)
        print("Epoch {}: Training loss {:.4f}".format(epoch, train_metrics['ar_loss']))

        val_metrics = test_epoch(model, val_loader, args.device, args.no_sampling, latent_dim)
        print("Epoch {}: Validation loss {:.4f}  accuracy {:.4f}".format(epoch, val_metrics['ar_loss'], val_metrics['accuracy']))

        if args.wandb:
            logs.update({'val_' + k: v for k, v in val_metrics.items()}, step=epoch + 1)
            logs.update({'train_' + k: v for k, v in train_metrics.items()}, step=epoch + 1)
            logs.update({'current_lr': optimizer.param_groups[0]['lr']})
            wandb.log(logs, step=epoch + 1)

        if scheduler:
            scheduler.step(val_metrics[args.main_metric])

        if args.use_pretrained_score and args.warmup_epochs == epoch+1:
            print("Unfreezing the score model weights and reinitializing optimizer")
            for param in model.pretrained_score_model.parameters():
                param.requires_grad = True
            optimizer, scheduler = get_optimizer_and_scheduler(args, model, scheduler_mode=args.main_metric_goal)

        state_dict = model.state_dict()

        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            best_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_model_accuracy.pt'))

        if val_metrics['ar_loss'] < best_val_loss:
            best_val_loss = val_metrics['ar_loss']
            torch.save(state_dict, os.path.join(run_dir, 'best_model_loss.pt'))

        if args.model_save_frequency > 0 and (epoch + 1) % args.model_save_frequency == 0:
            torch.save(state_dict, os.path.join(run_dir, f'model_epoch{epoch + 1}.pt'))
        if args.best_model_save_frequency > 0 and (epoch + 1) % args.best_model_save_frequency == 0:
            shutil.copyfile(os.path.join(run_dir, 'best_model_accuracy.pt'), os.path.join(run_dir, f'best_model_accuracy_epoch{epoch + 1}.pt'))
            shutil.copyfile(os.path.join(run_dir, 'best_model_loss.pt'), os.path.join(run_dir, f'best_model_loss_epoch{epoch + 1}.pt'))

        torch.save({
            'epoch': epoch,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(run_dir, 'last_model.pt'))

    print("Best Validation accuracy {} on Epoch {}".format(best_val_accuracy, best_epoch))


def construct_loader_ar(args, device, tr_sigma_max, no_randomness):
    common_args = {'cache_path': args.cache_path, 'original_model_dir': args.original_model_dir, 'device': device,
                   'limit_complexes': args.limit_complexes, 'all_atoms': args.all_atoms, "model_ckpt": args.ckpt,
                   'tr_sigma_max': tr_sigma_max, 'args': args, 'no_randomness': no_randomness, 'no_sampling': args.no_sampling}
    loader_class = DataLoader

    train_dataset = AutoregressiveDataset(split="train", multiplicity=args.train_multiplicity, **common_args)
    train_loader = loader_class(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = AutoregressiveDataset(split="train" if args.overfit else "val", multiplicity=args.val_multiplicity, **common_args)
    val_loader = loader_class(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(f'{args.original_model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True


    # construct loader
    train_loader, val_loader = construct_loader_ar(args, device, score_model_args.tr_sigma_max, args.no_randomness)
    model = get_ar_model(args, score_model_args, device, training=True)
    optimizer, scheduler = get_optimizer_and_scheduler(args, model, scheduler_mode=args.main_metric_goal)

    if args.restart_dir:
        dict = torch.load(f'{args.restart_dir}/last_model.pt', map_location=torch.device('cpu'))
        model.load_state_dict(dict['model'], strict=True)
        optimizer.load_state_dict(dict['optimizer'])
        print("Restarting from epoch", dict['epoch'])

    elif args.use_pretrained_score and args.warmup_epochs > 0:
        print("Freezing the score model weights for {} epochs".format(args.warmup_epochs))
        for param in model.pretrained_score_model.parameters():
            param.requires_grad = False
        optimizer, scheduler = get_optimizer_and_scheduler(args, model, scheduler_mode=args.main_metric_goal)

    numel = sum([p.numel() for p in model.parameters()])
    print('Model with', numel, 'parameters')

    if args.wandb:
        wandb.init(
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=args
        )
        wandb.log({'numel': numel})

    # record parameters
    run_dir = os.path.join(args.log_dir, args.run_name)
    yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, args.__dict__)
    args.device = device

    train(args, model, optimizer, scheduler, train_loader, val_loader, run_dir, score_model_args.latent_dim)
